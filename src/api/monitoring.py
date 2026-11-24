"""Registro de predicciones (contrato 4.4) y métricas a CloudWatch.

Nada de esto debe tumbar la respuesta al médico: si S3 o CloudWatch fallan (red, permisos,
límite de la cuenta), se registra el error en el log de la aplicación y la API sigue respondiendo
con normalidad — por eso cada función externa atrapa sus propias excepciones.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone

import boto3

logger = logging.getLogger("blindness_detection.api")

_s3 = None
_cloudwatch = None


def _s3_client():
    global _s3
    if _s3 is None:
        _s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION"))
    return _s3


def _cloudwatch_client():
    global _cloudwatch
    if _cloudwatch is None:
        _cloudwatch = boto3.client("cloudwatch", region_name=os.environ.get("AWS_REGION"))
    return _cloudwatch


def _upload_log(record: dict, bucket: str | None = None) -> None:
    bucket = bucket or os.environ.get("S3_BUCKET_PREDICTIONS")
    if not bucket:
        logger.warning("S3_BUCKET_PREDICTIONS no configurado; no se guardó el registro de predicción")
        return
    now = datetime.now(timezone.utc)
    key = f"logs/predictions/{now:%Y/%m/%d}/{uuid.uuid4()}.json"
    try:
        _s3_client().put_object(
            Bucket=bucket,
            Key=key,
            ContentType="application/json",
            Body=json.dumps(record, ensure_ascii=False).encode("utf-8"),
        )
    except Exception:
        logger.exception("No se pudo guardar el registro de predicción en s3://%s/%s", bucket, key)


def _put_metrics(metric_data: list[dict], namespace: str | None = None) -> None:
    namespace = namespace or os.environ.get("CLOUDWATCH_NAMESPACE", "BlindnessDetection")
    try:
        _cloudwatch_client().put_metric_data(Namespace=namespace, MetricData=metric_data)
    except Exception:
        logger.exception("No se pudieron publicar métricas a CloudWatch (namespace=%s)", namespace)


def record_prediction(response: dict, latency_ms: float) -> None:
    """response: el mismo dict que regresa POST /predict. Sube el log (contrato 4.4) y las
    métricas: conteo de predicciones, latencia, rechazos por calidad y distribución por clase."""
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "run_id": response["run_id"],
        "quality_ok": response["quality_ok"],
        "quality_reason": response["quality_reason"],
        "predicted_class": response["predicted_class"],
        "probabilities": response["probabilities"],
        "traffic_light": response["traffic_light"],
        "low_confidence": response["low_confidence"],
        "latency_ms": round(latency_ms, 1),
    }
    _upload_log(log_entry)

    metric_data = [
        {"MetricName": "Predictions", "Value": 1, "Unit": "Count"},
        {"MetricName": "Latency", "Value": latency_ms, "Unit": "Milliseconds"},
    ]
    if not response["quality_ok"]:
        metric_data.append(
            {
                "MetricName": "QualityRejections",
                "Value": 1,
                "Unit": "Count",
                "Dimensions": [{"Name": "Reason", "Value": response["quality_reason"]}],
            }
        )
    else:
        metric_data.append(
            {
                "MetricName": "ClassDistribution",
                "Value": 1,
                "Unit": "Count",
                "Dimensions": [{"Name": "Class", "Value": str(response["predicted_class"])}],
            }
        )
        if response["low_confidence"]:
            metric_data.append({"MetricName": "LowConfidence", "Value": 1, "Unit": "Count"})
    _put_metrics(metric_data)


def record_error() -> None:
    _put_metrics([{"MetricName": "Errors", "Value": 1, "Unit": "Count"}])
