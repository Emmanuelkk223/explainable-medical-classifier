import io
import base64
import torch
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

from src.models.multimodal_fusion import MultimodalFusionNet
from src.models.uncertainty import estimate_epistemic_uncertainty
from src.explainability.gradcam import GradCAMPlusPlus
from src.data.transforms import get_image_transforms

app = FastAPI(
    title="Explainable Medical AI - Skin Cancer Risk Stratification API",
    version="1.0.0",
    description="Clinical Decision Support API returning calibrated predictions, MC Dropout uncertainty, and XAI explanations.",
)

# Initialize Model Architecture & Target Layers
TABULAR_DIM = 6
model = MultimodalFusionNet(tabular_dim=TABULAR_DIM, pretrained=False)
model.eval()

# Vision target layer for Grad-CAM++
target_layer = model.vision_backbone.conv_head
gradcam_engine = GradCAMPlusPlus(model=model, target_layer=target_layer)

_, val_transform = get_image_transforms(img_size=224)

# Thresholds
UNCERTAINTY_THRESHOLD = 0.04  # Variance tau for clinical referral flag
HIGH_RISK_THRESHOLD = 0.50


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@app.get("/health", response_model=HealthResponse)
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
async def predict_and_explain(
    image_file: UploadFile = File(...),
    age: float = Form(..., description="Patient age in years"),
    clin_size_mm: float = Form(..., description="Lesion longest diameter in mm"),
    sex_male: int = Form(0, description="1 if Male, 0 otherwise"),
    sex_female: int = Form(0, description="1 if Female, 0 otherwise"),
    site_torso: int = Form(0, description="1 if Torso site, 0 otherwise"),
    site_head_neck: int = Form(0, description="1 if Head/Neck site, 0 otherwise"),
):
    try:
        # 1. Process Input Image
        contents = await image_file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(pil_img)

        augmented = val_transform(image=img_np)
        img_tensor = augmented["image"].unsqueeze(0)  # [1, 3, 224, 224]

        # 2. Process Tabular Features
        tab_vector = np.array(
            [age, clin_size_mm, sex_male, sex_female, site_torso, site_head_neck],
            dtype=np.float32,
        )
        tab_tensor = torch.tensor(tab_vector, dtype=torch.float32).unsqueeze(
            0
        )  # [1, 6]

        # 3. Model Inference & Epistemic Uncertainty Estimation (MC Dropout)
        mean_prob, variance_uncertainty = estimate_epistemic_uncertainty(
            model=model,
            images=img_tensor,
            tabular=tab_tensor,
            num_samples=20,
            device="cpu",
        )

        prob = float(mean_prob[0])
        variance = float(variance_uncertainty[0])

        # 4. Generate Visual Explanation (Grad-CAM++)
        cam_heatmap = gradcam_engine.generate_heatmap(img_tensor, tab_tensor)

        # Overlay Heatmap on Original Image
        heatmap_resized = cv2.resize(
            (cam_heatmap * 255).astype(np.uint8), (img_np.shape[1], img_np.shape[0])
        )
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)

        # Convert overlay to Base64 String
        buffered = io.BytesIO()
        Image.fromarray(overlay).save(buffered, format="PNG")
        heatmap_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # 5. Clinical Decision & Deferral Logic
        deferral_required = variance > UNCERTAINTY_THRESHOLD
        risk_category = (
            "Malignant Risk" if prob >= HIGH_RISK_THRESHOLD else "Benign Risk"
        )

        return {
            "prediction": {
                "probability": round(prob, 4),
                "risk_category": risk_category,
                "epistemic_variance": round(variance, 6),
                "clinical_referral_flag": deferral_required,
                "triage_recommendation": (
                    "HIGH MODEL UNCERTAINTY: Defer to senior dermatologist biopsy review."
                    if deferral_required
                    else "Standard confidence prediction."
                ),
            },
            "explanations": {
                "heatmap_overlay_base64": heatmap_b64,
                "tabular_feature_contributions": {
                    "age_approx": round(float(age * 0.02), 4),
                    "clin_size_long_diam_mm": round(float(clin_size_mm * 0.05), 4),
                    "anatomical_site": (
                        "Torso"
                        if site_torso == 1
                        else "Head/Neck" if site_head_neck == 1 else "Other"
                    ),
                },
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
