"""Pruebas de Grad-CAM con un modelo timm chico, sin pesos preentrenados (rápido)."""

import numpy as np
import pytest
import timm
import torch
from torch import nn

from src.common.gradcam import compute_cam, find_target_layer, generate_heatmap

IMAGE_SIZE = 224


@pytest.fixture(scope="module")
def model():
    m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=5)
    m.eval()
    return m


def test_find_target_layer_is_conv2d(model):
    assert isinstance(find_target_layer(model), nn.Conv2d)


def test_find_target_layer_raises_without_conv2d():
    with pytest.raises(ValueError):
        find_target_layer(nn.Sequential(nn.Linear(10, 10)))


def test_compute_cam_is_normalized_and_smaller_than_input(model):
    cam = compute_cam(model, torch.randn(3, IMAGE_SIZE, IMAGE_SIZE), target_class=2)
    assert cam.min() >= 0.0
    assert cam.max() <= 1.0 + 1e-6
    assert cam.shape[0] < IMAGE_SIZE and cam.shape[1] < IMAGE_SIZE


def test_generate_heatmap_matches_original_image(model):
    original = (np.random.rand(IMAGE_SIZE, IMAGE_SIZE, 3) * 255).astype(np.uint8)
    overlay = generate_heatmap(
        model, torch.randn(3, IMAGE_SIZE, IMAGE_SIZE), target_class=0, original_img=original
    )
    assert overlay.shape == original.shape
    assert overlay.dtype == np.uint8


def test_generate_heatmap_respects_original_image_size(model):
    # overlay_heatmap debe escalar el mapa de vuelta al tamaño de la foto, no al de la red
    original = (np.random.rand(500, 350, 3) * 255).astype(np.uint8)
    overlay = generate_heatmap(
        model, torch.randn(3, IMAGE_SIZE, IMAGE_SIZE), target_class=1, original_img=original
    )
    assert overlay.shape == original.shape


def test_model_still_usable_for_plain_inference_after_gradcam(model):
    tensor = torch.randn(3, IMAGE_SIZE, IMAGE_SIZE)
    generate_heatmap(model, tensor, target_class=1, original_img=np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8))
    with torch.no_grad():
        probs = torch.softmax(model(tensor.unsqueeze(0)), dim=1)
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-4)


def test_can_run_gradcam_several_times_in_a_row(model):
    tensor = torch.randn(3, IMAGE_SIZE, IMAGE_SIZE)
    first = compute_cam(model, tensor, target_class=0)
    second = compute_cam(model, tensor, target_class=0)
    assert first.shape == second.shape
    assert np.allclose(first, second, atol=1e-4)
