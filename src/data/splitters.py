import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from typing import Tuple, Dict


def create_patient_stratified_splits(
    metadata_path: str,
    output_dir: str,
    n_splits: int = 5,
    test_size: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Creates leakage-free Train/Val/Test splits grouped strictly by patient_id
    and stratified by binary target (malignancy).
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")

    df = pd.read_csv(metadata_path)

    # Check essential columns
    required_cols = {"isic_id", "patient_id", "target"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Metadata must contain columns: {required_cols}")

    # Handle missing patient IDs by treating missing entries as unique individual patients
    df["patient_id"] = df["patient_id"].fillna(df["isic_id"])

    # Step 1: Hold out a dedicated Test Set using StratifiedGroupKFold
    test_sgkf = StratifiedGroupKFold(
        n_splits=int(1 / test_size), shuffle=True, random_state=seed
    )
    train_val_idx, test_idx = next(
        test_sgkf.split(df, df["target"], groups=df["patient_id"])
    )

    df["fold"] = -1
    df.loc[test_idx, "fold"] = -2  # -2 indicates test set

    # Step 2: Split remaining train_val set into N-fold CV
    df_train_val = df.iloc[train_val_idx].reset_index()
    cv_sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (_, val_split_idx) in enumerate(
        cv_sgkf.split(
            df_train_val, df_train_val["target"], groups=df_train_val["patient_id"]
        )
    ):
        original_indices = df_train_val.iloc[val_split_idx]["index"].values
        df.loc[original_indices, "fold"] = fold

    # Save to output metadata directory
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "metadata_splits.csv")
    df.to_csv(out_file, index=False)

    print_split_summary(df)
    return df


def verify_zero_patient_leakage(df: pd.DataFrame) -> bool:
    """
    Strictly verifies no patient_id overlaps across cross-validation folds and test set.
    """
    folds = df["fold"].unique()
    patient_sets: Dict[int, set] = {}

    for f in folds:
        patient_sets[f] = set(df[df["fold"] == f]["patient_id"])

    for f1 in folds:
        for f2 in folds:
            if f1 < f2:
                intersection = patient_sets[f1].intersection(patient_sets[f2])
                if len(intersection) > 0:
                    raise AssertionError(
                        f"DATA LEAKAGE DETECTED between fold {f1} and fold {f2}: {len(intersection)} shared patients!"
                    )

    print("SUCCESS: 0 patient overlap detected across all splits.")
    return True


def print_split_summary(df: pd.DataFrame):
    """Prints diagnostic table of target balance and patient counts per split."""
    print("\n--- DATASET SPLIT DIAGNOSTICS ---")
    summary = []
    for f in sorted(df["fold"].unique()):
        name = "Test Set" if f == -2 else f"Fold {f}"
        sub = df[df["fold"] == f]
        pos_cases = sub["target"].sum()
        total_cases = len(sub)
        n_patients = sub["patient_id"].nunique()
        pos_rate = (pos_cases / total_cases) * 100
        summary.append(
            {
                "Split": name,
                "Total Samples": total_cases,
                "Unique Patients": n_patients,
                "Malignant (1)": pos_cases,
                "Malignancy Rate (%)": f"{pos_rate:.2f}%",
            }
        )
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    raw_csv = "data/raw/train-metadata.csv"
    out_dir = "data/metadata_splits"
    if os.path.exists(raw_csv):
        df_splits = create_patient_stratified_splits(raw_csv, out_dir)
        verify_zero_patient_leakage(df_splits)
