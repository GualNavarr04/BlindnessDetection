"""Preprocesamiento de fotos de fondo de ojo (contrato 4.2 de docs/plan.md).

Lo usan el entrenamiento (train.py) y la API, para que el modelo vea las fotos igual en los dos
lados. Los aumentos de datos (rotaciones, brillo) NO van aquí: solo los usa train.py.

Flujo: load_image() -> crop_retina() -> to_tensor()   (o todo junto con preprocess()).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

IMAGE_SIZE = 384
MEAN = (0.485, 0.456, 0.406)  # ImageNet
STD = (0.229, 0.224, 0.225)

# Un pixel con gris <= este valor se considera fondo negro alrededor de la retina.
BACKGROUND_THRESHOLD = 7


def load_image(source: str | Path | bytes) -> np.ndarray:
    """Lee una foto desde una ruta o desde bytes (lo que sube la página web).

    Regresa RGB uint8 con forma (H, W, 3). Lanza ValueError si no es una imagen válida.
    """
    if isinstance(source, (bytes, bytearray, memoryview)):
        data = np.frombuffer(source, dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    else:
        # imdecode en lugar de imread: imread falla con rutas con acentos en Windows.
        data = np.fromfile(str(source), dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("No se pudo leer la imagen: formato no soportado o archivo dañado.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def crop_retina(img: np.ndarray, threshold: int = BACKGROUND_THRESHOLD) -> np.ndarray:
    """Recorta el borde negro alrededor de la retina y deja la imagen cuadrada.

    Las fotos de APTOS vienen de cámaras distintas: unas con mucho borde negro, otras con la
    retina cortada arriba y abajo. Primero se recorta al rectángulo que contiene la retina y
    luego se rellena con negro hasta hacerla cuadrada (centrada), para que al cambiar el tamaño
    la retina no se deforme. Si la foto es casi toda negra, se regresa sin recortar.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    rows = np.where((gray > threshold).any(axis=1))[0]
    cols = np.where((gray > threshold).any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return img
    cropped = img[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]

    h, w = cropped.shape[:2]
    side = max(h, w)
    top = (side - h) // 2
    left = (side - w) // 2
    return cv2.copyMakeBorder(
        cropped, top, side - h - top, left, side - w - left, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )


def to_tensor(img: np.ndarray, size: int = IMAGE_SIZE) -> torch.Tensor:
    """Cambia el tamaño a (size, size) y normaliza con MEAN/STD de ImageNet.

    Recibe RGB uint8 (H, W, 3) y regresa un tensor float32 (3, size, size).
    """
    interpolation = cv2.INTER_AREA if min(img.shape[:2]) > size else cv2.INTER_LINEAR
    resized = cv2.resize(img, (size, size), interpolation=interpolation)
    arr = resized.astype(np.float32) / 255.0
    arr = (arr - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())


def preprocess(source: str | Path | bytes, size: int = IMAGE_SIZE) -> torch.Tensor:
    """Atajo: load_image -> crop_retina -> to_tensor. Regresa (3, size, size)."""
    return to_tensor(crop_retina(load_image(source)), size)
