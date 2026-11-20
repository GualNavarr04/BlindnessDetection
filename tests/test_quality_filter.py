"""Pruebas del filtro de calidad con retinas sintéticas (círculo con "vasos" y textura)."""

import cv2
import numpy as np
import pytest

from src.common.quality_filter import QualityResult, check_quality, quality_scores


def synthetic_retina(seed=0, size=800):
    """Círculo naranja con líneas oscuras (vasos) y textura fina, sobre fondo negro."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    c, r = size // 2, int(size * 0.45)
    cv2.circle(img, (c, c), r, (190, 90, 40), thickness=-1)
    for _ in range(25):
        p1 = tuple(int(v) for v in rng.integers(c - r, c + r, 2))
        p2 = tuple(int(v) for v in rng.integers(c - r, c + r, 2))
        cv2.line(img, p1, p2, (110, 40, 20), thickness=int(rng.integers(2, 6)))
    noise = rng.normal(0, 6, img.shape)
    inside = np.zeros((size, size), np.uint8)
    cv2.circle(inside, (c, c), r, 1, thickness=-1)
    img = np.clip(img + noise * inside[..., None], 0, 255).astype(np.uint8)
    return img


def darken(img, factor):
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def blur(img, sigma=6):
    return cv2.GaussianBlur(img, (0, 0), sigma)


def test_scores_keys_and_ranges():
    scores = quality_scores(synthetic_retina())
    assert set(scores) == {"brightness", "sharpness"}
    assert 50 < scores["brightness"] < 200
    assert scores["sharpness"] > 0


def test_darkening_does_not_look_blurry():
    # La razón del cambio respecto al EDA: oscurecer no debe bajar la nitidez
    img = synthetic_retina()
    sharp = quality_scores(img)["sharpness"]
    dark = quality_scores(darken(img, 0.45))["sharpness"]
    assert dark == pytest.approx(sharp, rel=0.25)


def test_blur_lowers_sharpness_a_lot():
    img = synthetic_retina()
    assert quality_scores(blur(img))["sharpness"] < 0.2 * quality_scores(img)["sharpness"]


def test_check_quality_reasons():
    img = synthetic_retina()
    sharp = quality_scores(img)["sharpness"]
    thresholds = {"brightness_threshold": 30, "sharpness_threshold": sharp / 2}

    good = check_quality(img, **thresholds)
    assert isinstance(good, QualityResult) and good.ok and good.reason is None

    assert check_quality(blur(img), **thresholds).reason == "borrosa"
    assert check_quality(darken(img, 0.15), **thresholds).reason == "oscura"
    # oscura y borrosa a la vez: el motivo es "oscura"
    assert check_quality(blur(darken(img, 0.15)), **thresholds).reason == "oscura"


def test_black_image_is_dark():
    result = check_quality(np.zeros((300, 400, 3), dtype=np.uint8))
    assert not result.ok and result.reason == "oscura"
    assert result.scores == {"brightness": 0.0, "sharpness": 0.0}


def test_default_thresholds_accept_a_good_image():
    assert check_quality(synthetic_retina()).ok
