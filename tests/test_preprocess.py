import cv2
import numpy as np
import pytest
import torch

from src.common.preprocess import (
    IMAGE_SIZE,
    MEAN,
    STD,
    crop_retina,
    load_image,
    preprocess,
    to_tensor,
)


def fake_fundus(h=300, w=400, radius=120, color=(200, 80, 40)):
    """Imagen RGB con un círculo de color (la "retina") sobre fondo negro."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.circle(img, (w // 2, h // 2), radius, color, thickness=-1)
    return img


def png_bytes(img_rgb):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    assert ok
    return buf.tobytes()


def test_load_image_from_bytes_is_rgb():
    img = fake_fundus()
    loaded = load_image(png_bytes(img))
    assert loaded.dtype == np.uint8
    assert loaded.shape == img.shape
    # el centro debe seguir siendo (R, G, B) = (200, 80, 40), no BGR
    assert tuple(loaded[150, 200]) == (200, 80, 40)


def test_load_image_from_path(tmp_path):
    img = fake_fundus()
    path = tmp_path / "retina_ñ.png"  # con acento, como las carpetas de OneDrive
    path.write_bytes(png_bytes(img))
    assert load_image(path).shape == img.shape
    assert load_image(str(path)).shape == img.shape


def test_load_image_invalid_bytes():
    with pytest.raises(ValueError):
        load_image(b"esto no es una imagen")


def test_crop_retina_removes_black_border_and_is_square():
    img = fake_fundus(h=300, w=400, radius=120)
    cropped = crop_retina(img)
    h, w = cropped.shape[:2]
    assert h == w
    assert abs(h - 241) <= 2  # diámetro del círculo


def test_crop_retina_pads_cut_retina_to_square():
    # retina cortada arriba y abajo: el círculo es más alto que la imagen
    img = fake_fundus(h=200, w=400, radius=150)
    cropped = crop_retina(img)
    assert cropped.shape[0] == cropped.shape[1]
    assert cropped.shape[1] >= 299


def test_crop_retina_stable_with_jpeg_and_dim_edge():
    # retina con orilla oscura (viñeteo) y guardada como JPEG: el recorte no debe cambiar de escala
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    yy, xx = np.mgrid[:600, :800]
    r = np.hypot(yy - 300, xx - 400)
    inside = r <= 250
    img[inside] = (np.clip(180 - 150 * (r[inside] / 250) ** 4, 0, 255)[:, None] * [1, 0.45, 0.2]).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    jpeg = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    assert abs(crop_retina(img).shape[0] - crop_retina(jpeg).shape[0]) <= 6


def test_crop_retina_all_black_returns_same():
    img = np.zeros((100, 120, 3), dtype=np.uint8)
    assert crop_retina(img) is img


def test_to_tensor_shape_and_normalization():
    img = np.full((500, 700, 3), 255, dtype=np.uint8)
    t = to_tensor(img)
    assert isinstance(t, torch.Tensor)
    assert t.dtype == torch.float32
    assert t.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    expected = [(1.0 - m) / s for m, s in zip(MEAN, STD)]
    assert torch.allclose(t[:, 0, 0], torch.tensor(expected), atol=1e-5)


def test_to_tensor_custom_size():
    assert to_tensor(fake_fundus(), size=224).shape == (3, 224, 224)


def test_preprocess_end_to_end():
    t = preprocess(png_bytes(fake_fundus()))
    assert t.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
