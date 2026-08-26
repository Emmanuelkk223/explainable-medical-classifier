import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List, Optional

# Compatibility wrapper for NumPy 2.0+
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


def compute_pixel_deletion_curve(
    model: nn.Module,
    image: torch.Tensor,
    tabular: Optional[torch.Tensor],
    heatmap: np.ndarray,
    step_percent: float = 0.1,
    device: str = "cpu",
) -> Tuple[List[float], float]:
    """
    Iteratively masks the most important pixels according to the heatmap
    and measures the drop in target prediction probability.
    """
    model.eval()
    model.to(device)

    img = image.clone().to(device)
    if img.ndim == 3:
        img = img.unsqueeze(0)

    tab = tabular.clone().to(device) if tabular is not None else None

    H, W = heatmap.shape
    total_pixels = H * W
    flat_heatmap_indices = np.argsort(heatmap.flatten())[::-1]

    step_size = int(total_pixels * step_percent)
    probabilities = []

    with torch.no_grad():
        for step in range(0, total_pixels + step_size, step_size):
            masked_img = img.clone()
            mask_indices = flat_heatmap_indices[: min(step, total_pixels)]

            for idx in mask_indices:
                r, c = divmod(idx, W)
                masked_img[0, :, r, c] = 0.0

            if tab is not None:
                logits = model(masked_img, tab)
            else:
                logits = model(masked_img)

            prob = torch.sigmoid(logits).item()
            probabilities.append(prob)

    auc_score = float(_trapz(probabilities, dx=step_percent))
    return probabilities, auc_score


def compute_pixel_insertion_curve(
    model: nn.Module,
    image: torch.Tensor,
    tabular: Optional[torch.Tensor],
    heatmap: np.ndarray,
    step_percent: float = 0.1,
    device: str = "cpu",
) -> Tuple[List[float], float]:
    """
    Starts with a zeroed baseline image and iteratively restores top-ranked pixels.
    """
    model.eval()
    model.to(device)

    img = image.clone().to(device)
    if img.ndim == 3:
        img = img.unsqueeze(0)

    tab = tabular.clone().to(device) if tabular is not None else None

    H, W = heatmap.shape
    total_pixels = H * W
    flat_heatmap_indices = np.argsort(heatmap.flatten())[::-1]

    step_size = int(total_pixels * step_percent)
    probabilities = []

    baseline_img = torch.zeros_like(img)

    with torch.no_grad():
        for step in range(0, total_pixels + step_size, step_size):
            restored_img = baseline_img.clone()
            restore_indices = flat_heatmap_indices[: min(step, total_pixels)]

            for idx in restore_indices:
                r, c = divmod(idx, W)
                restored_img[0, :, r, c] = img[0, :, r, c]

            if tab is not None:
                logits = model(restored_img, tab)
            else:
                logits = model(restored_img)

            prob = torch.sigmoid(logits).item()
            probabilities.append(prob)

    auc_score = float(_trapz(probabilities, dx=step_percent))
    return probabilities, auc_score
