import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple


class TemperatureScaler(nn.Module):
    """
    Post-hoc logit calibration using Temperature Scaling.
    Learns a single scalar T > 0 on validation set logits to align output confidence with empirical accuracy:
        P_calibrated = Sigmoid(logit / T)
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(
        self, val_logits: torch.Tensor, val_targets: torch.Tensor, max_iter: int = 50
    ):
        """Optimizes temperature parameter T on validation set via L-BFGS."""
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)
        criterion = nn.BCEWithLogitsLoss()

        def eval_loss():
            optimizer.zero_grad()
            loss = criterion(self.forward(val_logits), val_targets)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        with torch.no_grad():
            self.temperature.clamp_(min=0.1, max=10.0)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE):
        ECE = sum_{b=1}^B (|B_b| / N) * |acc(B_b) - conf(B_b)|
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(y_prob)

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            avg_confidence = np.mean(y_prob[in_bin])
            avg_accuracy = np.mean(y_true[in_bin])
            ece += (bin_size / n_samples) * np.abs(avg_accuracy - avg_confidence)

    return float(ece)
