import pandas as pd
import numpy as np
from typing import Dict, List
from src.utils.metrics import compute_medical_metrics


def audit_demographic_fairness(
    df: pd.DataFrame,
    y_true_col: str,
    y_prob_col: str,
    sensitive_col: str,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Audits model performance, Selection Rates, and Disparate Impact across demographic subgroups.
    """
    results = []
    groups = df[sensitive_col].fillna("Unknown").unique()

    overall_selection_rate = float(np.mean(df[y_prob_col] >= threshold))

    for group in sorted(groups):
        sub_df = df[df[sensitive_col].fillna("Unknown") == group]
        if len(sub_df) == 0:
            continue

        metrics = compute_medical_metrics(
            sub_df[y_true_col].values, sub_df[y_prob_col].values, threshold=threshold
        )

        selection_rate = float(np.mean(sub_df[y_prob_col] >= threshold))
        disparate_impact = selection_rate / (overall_selection_rate + 1e-8)

        metrics.update(
            {
                "group": str(group),
                "sample_size": len(sub_df),
                "selection_rate": selection_rate,
                "disparate_impact": disparate_impact,
            }
        )
        results.append(metrics)

    return pd.DataFrame(results)
