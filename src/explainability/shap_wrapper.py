import shap
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import List, Union, Dict, Any, Optional


class TabularSHAPExplainer:
    """
    SHAP feature attribution wrapper supporting TreeSHAP (for LightGBM/XGBoost)
    and KernelSHAP (for deep neural network tabular branches).
    """

    def __init__(
        self,
        model: Any,
        background_data: np.ndarray,
        feature_names: List[str],
        is_tree_model: bool = False,
    ):
        self.model = model
        self.background_data = background_data
        self.feature_names = feature_names
        self.is_tree_model = is_tree_model

        if self.is_tree_model:
            self.explainer = shap.TreeExplainer(self.model)
        else:
            # Model prediction wrapper for deep tabular inputs
            def predict_fn(x: np.ndarray) -> np.ndarray:
                tensor_x = torch.tensor(x, dtype=torch.float32)
                with torch.no_grad():
                    if hasattr(self.model, "predict_proba"):
                        return self.model.predict_proba(tensor_x)
                    elif isinstance(self.model, nn.Module):
                        logits = self.model(tensor_x)
                        return torch.sigmoid(logits).cpu().numpy()
                return np.zeros((x.shape[0], 1))

            self.explainer = shap.KernelExplainer(predict_fn, self.background_data)

    def explain_instance(self, instance: np.ndarray) -> pd.DataFrame:
        """
        Computes SHAP feature importance values for a single sample instance.
        """
        if instance.ndim == 1:
            instance = instance.reshape(1, -1)

        shap_values = self.explainer.shap_values(instance)

        if isinstance(shap_values, list):
            shap_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            shap_vals = shap_values

        shap_vals = shap_vals.squeeze()

        df_importance = (
            pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "shap_value": shap_vals,
                    "absolute_impact": np.abs(shap_vals),
                }
            )
            .sort_values(by="absolute_impact", ascending=False)
            .reset_index(drop=True)
        )

        return df_importance
