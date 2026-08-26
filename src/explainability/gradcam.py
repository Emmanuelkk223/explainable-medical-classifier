import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any


class GradCAMPlusPlus:
    """
    Grad-CAM++ visual explanation extractor tailored for convolutional vision backbones.
    Computes fine-grained localized saliency maps for single-modality or multimodal networks.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        # Register PyTorch hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(
        self, module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor
    ):
        self.activations = output.detach()

    def _backward_hook(
        self,
        module: nn.Module,
        grad_input: Tuple[torch.Tensor],
        grad_output: Tuple[torch.Tensor],
    ):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(
        self,
        images: torch.Tensor,
        tabular: Optional[torch.Tensor] = None,
        target_class: int = 0,
    ) -> np.ndarray:
        """
        Generates a 2D spatial attribution heatmap normalized to [0, 1].
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        if tabular is not None:
            logits = self.model(images, tabular)
        else:
            logits = self.model(images)

        score = logits[0, target_class] if logits.ndim > 1 else logits[0]
        score.backward(retain_graph=True)

        gradients = self.gradients
        activations = self.activations

        # Ensure 4D shape [B, C, H, W] for spatial feature maps
        if activations.ndim == 2:
            activations = activations.unsqueeze(-1).unsqueeze(-1)
            gradients = gradients.unsqueeze(-1).unsqueeze(-1)

        # Grad-CAM++ weighting calculation
        grad_power_2 = gradients.pow(2)
        grad_power_3 = gradients.pow(3)

        sum_activations = activations.sum(dim=(2, 3), keepdim=True)
        aij = grad_power_2 / (2 * grad_power_2 + sum_activations * grad_power_3 + 1e-8)
        aij = torch.where(gradients != 0, aij, torch.zeros_like(aij))

        weights = (aij * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Upsample heatmap to target image dimensions [H, W]
        cam = F.interpolate(
            cam, size=images.shape[2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()

        # Min-max normalization
        norm_cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return norm_cam
