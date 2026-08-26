import pytest
import torch
import numpy as np
import pandas as pd
from src.data.dataset import ISICMultimodalDataset
from src.data.transforms import get_image_transforms, build_tabular_pipeline
from src.models.baselines import TabularLGBMBaseline, ResNetVisionBaseline
from src.models.multimodal_fusion import MultimodalFusionNet
from src.models.uncertainty import estimate_epistemic_uncertainty
from src.explainability.gradcam import GradCAMPlusPlus
from src.explainability.shap_wrapper import TabularSHAPExplainer
from src.explainability.fidelity_eval import (
    compute_pixel_deletion_curve,
    compute_pixel_insertion_curve,
)
from src.models.multimodal_fusion import MultimodalFusionNet
from src.utils.metrics import compute_medical_metrics, find_threshold_at_sensitivity
from src.evaluation.calibration import TemperatureScaler, compute_ece
from src.evaluation.fairness import audit_demographic_fairness

import io
from PIL import Image
from fastapi.testclient import TestClient
from deployment.api import app


def test_multimodal_dataset_shapes(tmp_path):
    df = pd.DataFrame(
        {
            "isic_id": [f"ISIC_000{i}" for i in range(10)],
            "patient_id": ["P1", "P1", "P2", "P2", "P3", "P3", "P4", "P4", "P5", "P5"],
            "age_approx": np.random.randint(20, 80, 10),
            "sex": ["male", "female"] * 5,
            "anatom_site_general": ["torso", "head/neck"] * 5,
            "target": [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        }
    )

    num_cols = ["age_approx"]
    cat_cols = ["sex", "anatom_site_general"]

    pipeline = build_tabular_pipeline(num_cols, cat_cols)
    X_tab = pipeline.fit_transform(df)

    train_transform, _ = get_image_transforms(img_size=224)

    dataset = ISICMultimodalDataset(
        df=df,
        image_dir_or_hdf5=str(tmp_path),
        tabular_features=X_tab,
        transform=train_transform,
        is_hdf5=False,
    )

    sample = dataset[0]

    assert sample["image"].shape == (
        3,
        224,
        224,
    ), f"Unexpected image shape: {sample['image'].shape}"
    assert sample["tabular"].shape == (
        X_tab.shape[1],
    ), f"Unexpected tabular shape: {sample['tabular'].shape}"
    assert sample["target"].shape == (
        1,
    ), f"Unexpected target shape: {sample['target'].shape}"


def test_tabular_lgbm_baseline():
    X_train = np.random.randn(100, 5)
    y_train = np.random.randint(0, 2, 100)
    X_val = np.random.randn(20, 5)
    y_val = np.random.randint(0, 2, 20)

    lgbm = TabularLGBMBaseline()
    lgbm.fit(X_train, y_train, X_val, y_val, num_boost_round=10)

    preds = lgbm.predict_proba(X_val)
    assert preds.shape == (20,), f"Expected shape (20,), got {preds.shape}"
    assert np.all((preds >= 0) & (preds <= 1)), "Probabilities out of bounds [0, 1]"


def test_resnet_vision_baseline_shape():
    model = ResNetVisionBaseline(pretrained=False)
    dummy_images = torch.randn(4, 3, 224, 224)

    with torch.no_grad():
        logits = model(dummy_images)

    assert logits.shape == (4, 1), f"Expected shape (4, 1), got {logits.shape}"


def test_multimodal_fusion_net_shape():
    batch_size = 4
    tabular_dim = 6
    model = MultimodalFusionNet(tabular_dim=tabular_dim, pretrained=False)

    dummy_images = torch.randn(batch_size, 3, 224, 224)
    dummy_tabular = torch.randn(batch_size, tabular_dim)

    model.eval()
    with torch.no_grad():
        logits = model(dummy_images, dummy_tabular)

    assert logits.shape == (
        batch_size,
        1,
    ), f"Expected shape ({batch_size}, 1), got {logits.shape}"


def test_mc_dropout_uncertainty_variance():
    batch_size = 2
    tabular_dim = 6
    model = MultimodalFusionNet(
        tabular_dim=tabular_dim, dropout_rate=0.5, pretrained=False
    )

    dummy_images = torch.randn(batch_size, 3, 224, 224)
    dummy_tabular = torch.randn(batch_size, tabular_dim)

    means, variances = estimate_epistemic_uncertainty(
        model, dummy_images, dummy_tabular, num_samples=15, device="cpu"
    )

    assert means.shape == (
        batch_size,
    ), f"Expected shape ({batch_size},), got {means.shape}"
    assert variances.shape == (
        batch_size,
    ), f"Expected shape ({batch_size},), got {variances.shape}"
    assert np.all(variances >= 0), "Variance cannot be negative"


def test_gradcam_plus_plus_execution():
    batch_size = 1
    tabular_dim = 6
    model = MultimodalFusionNet(tabular_dim=tabular_dim, pretrained=False)

    # Target final 4D convolutional layer of EfficientNet backbone
    target_layer = model.vision_backbone.conv_head
    gradcam = GradCAMPlusPlus(model=model, target_layer=target_layer)

    dummy_image = torch.randn(batch_size, 3, 224, 224)
    dummy_tabular = torch.randn(batch_size, tabular_dim)

    heatmap = gradcam.generate_heatmap(dummy_image, dummy_tabular)

    assert heatmap.shape == (
        224,
        224,
    ), f"Expected shape (224, 224), got {heatmap.shape}"
    assert np.all(
        (heatmap >= 0.0) & (heatmap <= 1.0)
    ), "Heatmap values out of range [0, 1]"


def test_pixel_deletion_insertion_fidelity():
    batch_size = 1
    tabular_dim = 6
    model = MultimodalFusionNet(tabular_dim=tabular_dim, pretrained=False)

    dummy_image = torch.randn(batch_size, 3, 224, 224)
    dummy_tabular = torch.randn(batch_size, tabular_dim)
    dummy_heatmap = np.random.rand(224, 224)

    del_probs, del_auc = compute_pixel_deletion_curve(
        model, dummy_image, dummy_tabular, dummy_heatmap, step_percent=0.25
    )
    ins_probs, ins_auc = compute_pixel_insertion_curve(
        model, dummy_image, dummy_tabular, dummy_heatmap, step_percent=0.25
    )

    assert len(del_probs) > 0 and len(ins_probs) > 0, "Empty probability list returned"
    assert isinstance(del_auc, float) and isinstance(
        ins_auc, float
    ), "AUC output must be a float"


def test_medical_metrics_calculation():
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 0])
    y_prob = np.array([0.1, 0.2, 0.4, 0.8, 0.9, 0.7, 0.3, 0.1, 0.85, 0.25])

    metrics = compute_medical_metrics(y_true, y_prob, threshold=0.5)

    assert 0.0 <= metrics["sensitivity"] <= 1.0
    assert 0.0 <= metrics["specificity"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert metrics["tp"] + metrics["fp"] + metrics["tn"] + metrics["fn"] == len(y_true)


def test_temperature_scaling_and_ece():
    val_logits = torch.randn(50, 1)
    val_targets = torch.randint(0, 2, (50, 1)).float()

    scaler = TemperatureScaler()
    scaler.fit(val_logits, val_targets, max_iter=10)

    calibrated_logits = scaler(val_logits)
    calibrated_probs = torch.sigmoid(calibrated_logits).detach().numpy().flatten()

    ece = compute_ece(val_targets.numpy().flatten(), calibrated_probs, n_bins=5)

    assert scaler.temperature.item() > 0.0
    assert 0.0 <= ece <= 1.0


def test_demographic_fairness_auditing():
    df = pd.DataFrame(
        {
            "target": [0, 1, 0, 1, 0, 1, 0, 0],
            "prob": [0.1, 0.9, 0.2, 0.85, 0.15, 0.75, 0.3, 0.2],
            "sex": [
                "male",
                "male",
                "male",
                "male",
                "female",
                "female",
                "female",
                "female",
            ],
        }
    )

    fairness_df = audit_demographic_fairness(
        df, y_true_col="target", y_prob_col="prob", sensitive_col="sex"
    )

    assert len(fairness_df) == 2
    assert "disparate_impact" in fairness_df.columns
    assert "sensitivity" in fairness_df.columns


client = TestClient(app)


def test_fastapi_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_fastapi_prediction_endpoint():
    # Create synthetic test image
    img = Image.new("RGB", (224, 224), color="red")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_bytes = img_byte_arr.getvalue()

    files = {"image_file": ("test.jpg", img_bytes, "image/jpeg")}
    data = {
        "age": "45",
        "clin_size_mm": "6.5",
        "sex_male": "1",
        "sex_female": "0",
        "site_torso": "1",
        "site_head_neck": "0",
    }

    response = client.post("/predict", files=files, data=data)
    assert response.status_code == 200, f"Response failed: {response.text}"

    res_json = response.json()
    assert "prediction" in res_json
    assert "probability" in res_json["prediction"]
    assert "epistemic_variance" in res_json["prediction"]
    assert "heatmap_overlay_base64" in res_json["explanations"]
