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

# Un pixel con gris <= este valor (después de suavizar) se considera fondo negro.
BACKGROUND_THRESHOLD = 15
# Fracción mínima de pixeles de retina para que una fila o columna cuente como retina.
MIN_RETINA_FRACTION = 0.02
# Lado mayor de la copia reducida sobre la que se calcula la máscara de la retina.
MASK_SIZE = 512


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

    La orilla de la retina es oscura y difusa, y el ruido de JPEG en el fondo puede pasar el
    umbral. Para que el recorte sea estable, la máscara se calcula sobre una copia reducida y
    suavizada, y una fila o columna solo cuenta como retina si al menos MIN_RETINA_FRACTION de
    sus pixeles pasan el umbral.
    """
    scale = MASK_SIZE / max(img.shape[:2])
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    mask = gray > threshold
    rows = np.where(mask.mean(axis=1) > MIN_RETINA_FRACTION)[0]
    cols = np.where(mask.mean(axis=0) > MIN_RETINA_FRACTION)[0]
    if rows.size == 0 or cols.size == 0:
        return img

    # De la máscara reducida a coordenadas de la imagen original
    fy = img.shape[0] / small.shape[0]
    fx = img.shape[1] / small.shape[1]
    top_r, bottom_r = int(rows[0] * fy), min(img.shape[0], int(np.ceil((rows[-1] + 1) * fy)))
    left_c, right_c = int(cols[0] * fx), min(img.shape[1], int(np.ceil((cols[-1] + 1) * fx)))
    cropped = img[top_r:bottom_r, left_c:right_c]

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
