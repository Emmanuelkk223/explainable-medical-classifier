import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    brier_score_loss,
    fbeta_score,
)
from typing import Dict, Any, Tuple


def compute_medical_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Computes medical classification metrics for binary malignancy diagnosis.
    Handles severe class imbalance and calculates both ROC-AUC and PR-AUC.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn + 1e-8)  # Recall
    specificity = tn / (tn + fp + 1e-8)  # True Negative Rate
    precision = tp / (tp + fp + 1e-8)  # Positive Predictive Value

    f1 = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-8)
    f2 = fbeta_score(
        y_true, y_pred, beta=2, zero_division=0
    )  # Weights Recall 2x over Precision

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc_auc = 0.5

    prec_array, rec_array, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = float(auc(rec_array, prec_array))
    brier = float(brier_score_loss(y_true, y_prob))

    return {
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1_score": float(f1),
        "f2_score": float(f2),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "threshold": float(threshold),
    }


def find_threshold_at_sensitivity(
    y_true: np.ndarray, y_prob: np.ndarray, target_sensitivity: float = 0.95
) -> float:
    """
    Finds the optimal decision threshold that guarantees a minimum target Sensitivity (e.g., 95%),
    minimizing false negatives in clinical triage.
    """
    prec_array, rec_array, thresholds = precision_recall_curve(y_true, y_prob)
    # Filter thresholds where recall >= target_sensitivity
    valid_indices = np.where(rec_array >= target_sensitivity)[0]

    if len(valid_indices) == 0:
        return 0.5

    # Pick the threshold yielding highest precision among valid candidates
    best_idx = valid_indices[np.argmax(prec_array[valid_indices])]
    if best_idx < len(thresholds):
        return float(thresholds[best_idx])
    return 0.5
