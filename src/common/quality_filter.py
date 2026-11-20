"""Filtro de calidad de la foto de fondo de ojo (contrato 4.8 de docs/plan.md).

Antes de pasar la foto al modelo, la API revisa que no esté demasiado oscura ni borrosa. Si no
pasa, pide repetir la toma en lugar de dar un resultado poco confiable.

Dos mediciones, sobre la retina recortada y llevada a QUALITY_SIZE × QUALITY_SIZE en gris:
- brillo:  gris promedio dentro de la retina (0-255).
- nitidez: varianza del Laplaciano dividida entre el brillo (× 10). En el EDA vimos que la varianza
  del Laplaciano sola también baja en fotos oscuras aunque estén enfocadas. Primero probamos dividir
  entre el brillo², pero se pasaba al otro lado: las fotos oscuras salían *más* nítidas y las muy
  claras parecían borrosas. Dividiendo entre el brillo la nitidez casi no depende de él, y el
  desenfoque la sigue bajando mucho más que oscurecer la foto.

Las dos se miden dejando fuera una orilla de EDGE_MARGIN pixeles: el borde del círculo contra el
fondo negro es un "borde" muy fuerte que no dice nada del enfoque.

Los umbrales salen de notebooks/quality.ipynb: el dataset casi no tiene fotos malas, así que se
eligen con fotos buenas degradadas a propósito (desenfoque y oscuridad).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from src.common.preprocess import BACKGROUND_THRESHOLD, crop_retina

QUALITY_SIZE = 512
EDGE_MARGIN = 8           # pixeles (a QUALITY_SIZE) que se quitan de la orilla de la retina
MIN_RETINA_FRACTION = 0.05  # si la "retina" ocupa menos que esto, la foto es prácticamente negra

# PROVISIONALES (probados con 150 fotos): se confirman con notebooks/quality.ipynb
BRIGHTNESS_THRESHOLD = 25.0   # gris promedio de la retina; menor = oscura
SHARPNESS_THRESHOLD = 0.5     # nitidez normalizada; menor = borrosa


@dataclass
class QualityResult:
    ok: bool
    reason: str | None  # "oscura", "borrosa" o None
    scores: dict = field(default_factory=dict)  # {"brightness": float, "sharpness": float}


def quality_scores(img: np.ndarray, cropped: bool = False) -> dict:
    """Mide brillo y nitidez. img: RGB uint8 (de load_image); cropped=True si ya viene recortada."""
    retina = img if cropped else crop_retina(img)
    small = cv2.resize(retina, (QUALITY_SIZE, QUALITY_SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)

    mask = (gray > BACKGROUND_THRESHOLD).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * EDGE_MARGIN + 1, 2 * EDGE_MARGIN + 1))
    inside = cv2.erode(mask, kernel).astype(bool)
    if inside.mean() < MIN_RETINA_FRACTION:
        return {"brightness": 0.0, "sharpness": 0.0}

    brightness = float(gray[inside].mean())
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    sharpness = float(laplacian[inside].var() / brightness * 10)
    return {"brightness": round(brightness, 2), "sharpness": round(sharpness, 4)}


def check_quality(
    img: np.ndarray,
    brightness_threshold: float | None = None,
    sharpness_threshold: float | None = None,
) -> QualityResult:
    """Decide si la foto sirve. img: RGB uint8 de load_image(), sin recortar.

    Los umbrales por defecto son las constantes de este archivo; se pueden pasar otros para probar.
    Si la foto es oscura y borrosa a la vez, el motivo es "oscura" (lo primero que hay que corregir).
    """
    b_thr = BRIGHTNESS_THRESHOLD if brightness_threshold is None else brightness_threshold
    s_thr = SHARPNESS_THRESHOLD if sharpness_threshold is None else sharpness_threshold

    scores = quality_scores(img)
    if scores["brightness"] < b_thr:
        return QualityResult(False, "oscura", scores)
    if scores["sharpness"] < s_thr:
        return QualityResult(False, "borrosa", scores)
    return QualityResult(True, None, scores)
