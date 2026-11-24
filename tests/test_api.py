"""Pruebas de la API (contrato 4.9) con un modelo falso en memoria, sin tocar AWS."""

import base64

import cv2
import numpy as np
import pytest
import timm
from fastapi.testclient import TestClient

from src.api import main as api_main
from src.api.main import _traffic_light, create_app
from src.api.model import ModelBundle

IMAGE_SIZE = 224


def fake_bundle(confidence_threshold=0.05, referable_threshold=0.1):
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=5)
    model.eval()
    metadata = {
        "run_id": "test-run",
        "architecture": "efficientnet_b0",
        "num_classes": 5,
        "image_size": IMAGE_SIZE,
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "decision": {
            "referable_threshold": referable_threshold,
            "confidence_threshold": confidence_threshold,
            "traffic_light": {"verde": [0, 1], "amarillo": [2], "rojo": [3, 4]},
        },
    }
    return ModelBundle(model=model, metadata=metadata)


def synthetic_retina(size=300, gray=140, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), gray, dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), size // 2 - 5, (gray + 40, gray + 20, gray), -1)
    noisy = img.astype(np.int16) + rng.integers(-60, 60, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def png_bytes(img):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    assert ok
    return buf.tobytes()


@pytest.fixture(autouse=True)
def no_aws_calls(monkeypatch):
    # El registro a S3/CloudWatch ya tiene sus propias pruebas de "nunca tronar"; aquí solo
    # evitamos que cada prueba intente de verdad una llamada de red a AWS.
    monkeypatch.setattr(api_main.monitoring, "record_prediction", lambda *a, **k: None)
    monkeypatch.setattr(api_main.monitoring, "record_error", lambda *a, **k: None)


@pytest.fixture
def client():
    return TestClient(create_app(bundle=fake_bundle()))


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["run_id"] == "test-run"


def test_static_page_served_at_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_predict_rejects_dark_photo(client):
    dark = np.zeros((300, 300, 3), dtype=np.uint8)
    r = client.post("/predict", files={"file": ("dark.png", png_bytes(dark), "image/png")})
    assert r.status_code == 200
    assert r.json() == {
        "quality_ok": False,
        "quality_reason": "oscura",
        "predicted_class": None,
        "class_name": None,
        "probabilities": None,
        "traffic_light": None,
        "recommendation": None,
        "low_confidence": None,
        "heatmap_png_base64": None,
        "run_id": "test-run",
    }


def test_predict_invalid_file_returns_400(client):
    r = client.post("/predict", files={"file": ("bad.png", b"no es una imagen", "image/png")})
    assert r.status_code == 400
    assert "detail" in r.json()


def test_predict_normal_photo_returns_full_contract(client):
    r = client.post("/predict", files={"file": ("ok.png", png_bytes(synthetic_retina()), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["quality_ok"] is True
    assert body["run_id"] == "test-run"
    assert len(body["probabilities"]) == 5
    assert abs(sum(body["probabilities"]) - 1.0) < 1e-3
    assert body["class_name"] in ["Sin retinopatía", "Leve", "Moderada", "Severa", "Proliferativa"]
    assert body["heatmap_png_base64"]
    assert base64.b64decode(body["heatmap_png_base64"])[:8] == b"\x89PNG\r\n\x1a\n"


def test_predict_low_confidence_has_no_traffic_light():
    # Umbral de confianza casi imposible de superar -> siempre cae en revisión humana
    client = TestClient(create_app(bundle=fake_bundle(confidence_threshold=0.99)))
    r = client.post("/predict", files={"file": ("ok.png", png_bytes(synthetic_retina()), "image/png")})
    body = r.json()
    assert body["low_confidence"] is True
    assert body["traffic_light"] is None
    assert body["recommendation"] == "Revisión humana"


# --- _traffic_light: la regla de negocio exacta (contrato 4.3), probada aparte de la API ---


def test_traffic_light_verde_below_referable_threshold():
    probs = np.array([0.70, 0.20, 0.05, 0.03, 0.02])
    assert _traffic_light(probs, {"referable_threshold": 0.3}) == "verde"


def test_traffic_light_gate_overrides_argmax():
    # El argmax es la clase 1 ("Leve"), pero P(clase>=2)=0.18 ya supera el umbral: debe ser referible.
    probs = np.array([0.40, 0.42, 0.15, 0.02, 0.01])
    assert probs.argmax() == 1
    assert _traffic_light(probs, {"referable_threshold": 0.10}) == "amarillo"


def test_traffic_light_rojo_when_severe_mass_dominates():
    probs = np.array([0.05, 0.05, 0.10, 0.40, 0.40])
    assert _traffic_light(probs, {"referable_threshold": 0.3}) == "rojo"


def test_traffic_light_boundary_equal_to_threshold_counts_as_referable():
    probs = np.array([0.80, 0.10, 0.05, 0.03, 0.02])  # P(clase>=2) = 0.10, igual al umbral
    assert _traffic_light(probs, {"referable_threshold": 0.10}) != "verde"
