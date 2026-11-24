"""Grad-CAM sobre la última capa convolucional de cualquier modelo de timm.

La API lo usa para mostrarle al médico en qué zona de la retina se fijó el modelo (contrato 4.9,
campo `heatmap_png_base64`). No depende de la arquitectura (EfficientNet-B3 en producción, pero
funciona con cualquier `timm.create_model(...)`): encuentra sola la última `nn.Conv2d`.

Importante: a diferencia de la inferencia normal, esto necesita que los gradientes estén
habilitados (no llamar dentro de `torch.no_grad()`), porque Grad-CAM se calcula con un
`backward()` sobre el logit de la clase elegida.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import nn


def find_target_layer(model: nn.Module) -> nn.Module:
    """Última capa Conv2d del modelo (para EfficientNet de timm, `conv_head`)."""
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise ValueError("El modelo no tiene capas Conv2d: no se puede calcular Grad-CAM.")
    return convs[-1]


class _ActivationsAndGradients:
    """Engancha una capa para guardar su activación y el gradiente que le llega al hacer backward."""

    def __init__(self, target_layer: nn.Module):
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inputs, output) -> None:
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def remove(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()


def compute_cam(
    model: nn.Module, input_tensor: torch.Tensor, target_class: int, target_layer: nn.Module | None = None
) -> np.ndarray:
    """input_tensor: (3, H, W) normalizado (salida de to_tensor()), sin dimensión de batch.

    Regresa el mapa de activación en [0, 1], con el tamaño espacial de target_layer (más chico
    que la imagen de entrada; overlay_heatmap() lo escala de vuelta)."""
    layer = target_layer or find_target_layer(model)
    hooks = _ActivationsAndGradients(layer)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(input_tensor.unsqueeze(0))
        logits[0, target_class].backward()
        activations = hooks.activations[0]  # (C, H', W')
        gradients = hooks.gradients[0]  # (C, H', W')
    finally:
        hooks.remove()

    weights = gradients.mean(dim=(1, 2))  # promedio global del gradiente por canal
    cam = torch.relu((weights[:, None, None] * activations).sum(dim=0))
    cam = cam.cpu().numpy()
    peak = cam.max()
    return cam / peak if peak > 0 else cam


def overlay_heatmap(cam: np.ndarray, original_img: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """cam: mapa en [0, 1]. original_img: RGB uint8 (H, W, 3), ya recortada (crop_retina).

    Regresa RGB uint8 del mismo tamaño que original_img, con el mapa de calor encimado."""
    h, w = original_img.shape[:2]
    resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_bgr = cv2.applyColorMap((resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return (alpha * heatmap_rgb + (1 - alpha) * original_img).astype(np.uint8)


def generate_heatmap(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    original_img: np.ndarray,
    target_layer: nn.Module | None = None,
    alpha: float = 0.4,
) -> np.ndarray:
    """Atajo: compute_cam() + overlay_heatmap(). Regresa RGB uint8 listo para codificar a PNG."""
    cam = compute_cam(model, input_tensor, target_class, target_layer)
    return overlay_heatmap(cam, original_img, alpha)
