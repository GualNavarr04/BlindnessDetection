"""API de inferencia (contrato 4.9 de docs/plan.md).

Flujo de /predict: calidad -> modelo -> semáforo o baja confianza -> Grad-CAM.

El umbral `referable_threshold` decide el color (amarillo/rojo) comparando contra
P(clase>=2), no contra la clase más probable (argmax) por separado: así no se deja pasar un
caso donde la evidencia de retinopatía está repartida entre varias clases sin que ninguna
domine por sí sola. `predicted_class`/`class_name` siguen mostrando el argmax real de las 5
clases, aunque a veces no coincida con el grupo del semáforo (ver docs/plan.md, contrato 4.3).
"""

from __future__ import annotations

import base64
import time

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from src.api import monitoring
from src.api.model import ModelBundle, load_current_model
from src.common.gradcam import generate_heatmap
from src.common.preprocess import crop_retina, load_image, to_tensor
from src.common.quality_filter import check_quality

CLASS_NAMES = ["Sin retinopatía", "Leve", "Moderada", "Severa", "Proliferativa"]
RECOMMENDATION = {
    "verde": "Control anual con fondo de ojo",
    "amarillo": "Referir a oftalmología",
    "rojo": "Referir a oftalmología pronto",
}
LOW_CONFIDENCE_RECOMMENDATION = "Revisión humana"
REFERABLE_CLASS = 2
HEATMAP_OUTPUT_SIZE = 512


def _traffic_light(probs: np.ndarray, decision: dict) -> str:
    referable_prob = float(probs[REFERABLE_CLASS:].sum())
    if referable_prob < decision["referable_threshold"]:
        return "verde"
    amarillo_prob = float(probs[REFERABLE_CLASS])
    rojo_prob = float(probs[REFERABLE_CLASS + 1 :].sum())
    return "amarillo" if amarillo_prob >= rojo_prob else "rojo"


def _encode_png(img: np.ndarray) -> str:
    ok, png = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("No se pudo codificar el mapa de calor a PNG")
    return base64.b64encode(png.tobytes()).decode("ascii")


def _resize_for_output(img: np.ndarray, size: int = HEATMAP_OUTPUT_SIZE) -> np.ndarray:
    """Reduce el mapa de calor a lo sumo `size` px de lado (no agranda fotos chicas)."""
    if max(img.shape[:2]) <= size:
        return img
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def _analyze(bundle: ModelBundle, data: bytes) -> dict:
    """Todo el trabajo pesado (CPU): calidad -> modelo -> semáforo/baja confianza -> Grad-CAM.

    Es una función normal (no async) para poder correrla en threadpool y no bloquear el event
    loop mientras el modelo procesa una foto (contrato 4.9, servicio de EC2)."""
    run_id = bundle.metadata["run_id"]
    try:
        img = load_image(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    quality = check_quality(img)
    if not quality.ok:
        return {
            "quality_ok": False,
            "quality_reason": quality.reason,
            "predicted_class": None,
            "class_name": None,
            "probabilities": None,
            "traffic_light": None,
            "recommendation": None,
            "low_confidence": None,
            "heatmap_png_base64": None,
            "run_id": run_id,
        }

    cropped = crop_retina(img)
    tensor = to_tensor(cropped, bundle.metadata["image_size"])

    with torch.no_grad():
        logits = bundle.model(tensor.unsqueeze(0))
        probs = torch.softmax(logits, dim=1)[0].numpy()

    predicted_class = int(probs.argmax())
    confidence = float(probs.max())
    decision = bundle.metadata["decision"]
    low_confidence = confidence < decision["confidence_threshold"]

    if low_confidence:
        traffic_light = None
        recommendation = LOW_CONFIDENCE_RECOMMENDATION
    else:
        traffic_light = _traffic_light(probs, decision)
        recommendation = RECOMMENDATION[traffic_light]

    heatmap = _resize_for_output(generate_heatmap(bundle.model, tensor, predicted_class, cropped))

    return {
        "quality_ok": True,
        "quality_reason": None,
        "predicted_class": predicted_class,
        "class_name": CLASS_NAMES[predicted_class],
        "probabilities": [round(float(p), 5) for p in probs],
        "traffic_light": traffic_light,
        "recommendation": recommendation,
        "low_confidence": low_confidence,
        "heatmap_png_base64": _encode_png(heatmap),
        "run_id": run_id,
    }


def create_app(bundle: ModelBundle | None = None) -> FastAPI:
    """bundle=None carga el modelo real de S3 (producción); en pruebas se inyecta uno falso."""
    bundle = bundle or load_current_model()
    app = FastAPI(title="Detección de Retinopatía Diabética")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "run_id": bundle.metadata["run_id"]}

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)) -> dict:
        started = time.perf_counter()
        data = await file.read()
        try:
            response = await run_in_threadpool(_analyze, bundle, data)
        except HTTPException:
            monitoring.record_error()
            raise
        monitoring.record_prediction(response, (time.perf_counter() - started) * 1000)
        return response

    # Al final: si se montara antes, taparía /health y /predict (contrato 4.9).
    app.mount("/", StaticFiles(directory="src/api/static", html=True), name="static")
    return app


# Producción: uvicorn src.api.main:create_app --factory
# (--factory para que el modelo real de S3 se cargue al arrancar el servidor, no al importar
# el módulo — así las pruebas pueden importar create_app() sin tocar S3.)
