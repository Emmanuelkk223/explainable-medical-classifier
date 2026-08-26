import pytest
import pandas as pd
import numpy as np
from src.data.splitters import (
    create_patient_stratified_splits,
    verify_zero_patient_leakage,
)


@pytest.fixture
def dummy_metadata(tmp_path):
    """Generates synthetic patient metadata with intentional multi-lesion structure."""
    np.random.seed(42)
    n_patients = 100
    records = []

    for pid in range(n_patients):
        patient_id = f"PAT_{pid:04d}"
        n_lesions = np.random.randint(1, 6)  # 1 to 5 lesions per patient
        target = 1 if np.random.rand() > 0.85 else 0  # 15% malignant rate

        for lid in range(n_lesions):
            records.append(
                {
                    "isic_id": f"ISIC_{pid}_{lid}",
                    "patient_id": patient_id,
                    "target": target,
                    "age_approx": np.random.randint(20, 80),
                    "sex": np.random.choice(["male", "female"]),
                }
            )

    df = pd.DataFrame(records)
    csv_path = tmp_path / "dummy_train.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def test_no_patient_leakage(dummy_metadata, tmp_path):
    output_dir = tmp_path / "splits"
    df_splits = create_patient_stratified_splits(
        metadata_path=dummy_metadata,
        output_dir=str(output_dir),
        n_splits=5,
        test_size=0.2,
        seed=42,
    )

    # Assert verification function passes
    assert verify_zero_patient_leakage(df_splits) is True


def test_target_stratification(dummy_metadata, tmp_path):
    output_dir = tmp_path / "splits"
    df_splits = create_patient_stratified_splits(
        metadata_path=dummy_metadata,
        output_dir=str(output_dir),
        n_splits=5,
        test_size=0.2,
        seed=42,
    )

    # Verify positive target exists across all folds
    for fold in df_splits["fold"].unique():
        sub = df_splits[df_splits["fold"] == fold]
        assert sub["target"].sum() > 0, f"Fold {fold} has zero positive target cases!"
