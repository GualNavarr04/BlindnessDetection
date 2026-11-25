"""Vuelve a escoger los umbrales de un modelo ya entrenado, sin reentrenarlo.

    python -m src.training.rethreshold --run-id 2026-11-23-effb3-162536 --dry-run   # solo muestra el cambio
    python -m src.training.rethreshold --run-id 2026-11-23-effb3-162536             # actualiza S3

Lee models/<run_id>/predictions.csv (probabilidades de validación y prueba que dejó train.py),
escoge los umbrales en VALIDACIÓN con --min-sensitivity y mide en PRUEBA, igual que train.py.
Reescribe la parte de decisión y de métricas de models/<run_id>/metadata.json; antes guarda el
original como metadata.before-rethreshold.json (solo la primera vez).

Por qué: con min_sensitivity 0.90 el umbral de referible quedó en 0.57 y en prueba se escapaban 27
de 202 referibles. Para un tamizaje conviene más sensibilidad aunque se refieran unos casos de más.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src.data.metrics import choose_thresholds, compute_metrics  # noqa: E402

NUM_CLASSES = 5
DEFAULT_MIN_SENSITIVITY = 0.95
METRIC_KEYS = ("qwk", "sensitivity_referable", "specificity_referable")


def evaluate(val_labels, val_probs, test_labels, test_probs, min_sensitivity: float) -> dict:
    """Umbrales escogidos en validación y métricas en prueba. Lo usan train.py y este script."""
    thresholds = choose_thresholds(val_labels, val_probs, min_sensitivity=min_sensitivity)
    thresholds = {k: round(float(v), 4) for k, v in thresholds.items()}
    val_metrics = compute_metrics(val_labels, val_probs, thresholds["referable_threshold"])
    test_metrics = compute_metrics(test_labels, test_probs, thresholds["referable_threshold"])
    test_probs = np.asarray(test_probs)
    low_confidence = float((test_probs.max(axis=1) < thresholds["confidence_threshold"]).mean())
    return {
        "thresholds": thresholds,
        # Métricas en PRUEBA, con los umbrales elegidos en validación
        "metrics": {**{k: test_metrics[k] for k in METRIC_KEYS},
                    "accuracy": float((test_probs.argmax(axis=1) == np.asarray(test_labels)).mean()),
                    "low_confidence_rate": round(low_confidence, 4),
                    "confusion_matrix": test_metrics["confusion_matrix"], "n": int(len(test_labels))},
        "validation_metrics": {**{k: val_metrics[k] for k in METRIC_KEYS}, "n": int(len(val_labels))},
        "threshold_selection": {"chosen_on": "val", "min_sensitivity": min_sensitivity,
                                "min_confidence_accuracy": 0.9},
    }


def split_arrays(predictions, split: str):
    part = predictions[predictions["split"] == split]
    return part["diagnosis"].to_numpy(int), part[[f"p{c}" for c in range(NUM_CLASSES)]].to_numpy(float)


def updated_metadata(metadata: dict, predictions, min_sensitivity: float) -> dict:
    result = evaluate(*split_arrays(predictions, "val"), *split_arrays(predictions, "test"), min_sensitivity)
    new = dict(metadata)
    new["decision"] = {**metadata["decision"], **result["thresholds"]}
    new["metrics"] = result["metrics"]
    new["validation_metrics"] = result["validation_metrics"]
    new["threshold_selection"] = {
        **result["threshold_selection"],
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    return new


def describe(old: dict, new: dict) -> str:
    def row(name, a, b):
        return f"  {name:<30} {a:>8} -> {b:<8}"

    lines = [row("umbral de referible", old["decision"]["referable_threshold"], new["decision"]["referable_threshold"]),
             row("umbral de confianza", old["decision"]["confidence_threshold"],
                 new["decision"]["confidence_threshold"])]
    for key in METRIC_KEYS:
        lines.append(row(f"prueba {key}", f"{old['metrics'][key]:.3f}", f"{new['metrics'][key]:.3f}"))
        lines.append(row(f"validación {key}", f"{old['validation_metrics'][key]:.3f}",
                         f"{new['validation_metrics'][key]:.3f}"))
    lines.append(row("prueba baja confianza", "-", f"{new['metrics']['low_confidence_rate']:.1%}"))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", required=True)
    p.add_argument("--min-sensitivity", type=float, default=DEFAULT_MIN_SENSITIVITY)
    p.add_argument("--bucket", default=os.environ.get("S3_BUCKET_MODELS"))
    p.add_argument("--dry-run", action="store_true", help="Mostrar el cambio sin escribir en S3")
    args = p.parse_args(argv)
    if not args.bucket:
        p.error("Falta --bucket o la variable S3_BUCKET_MODELS")

    import boto3
    import pandas as pd
    from botocore.exceptions import ClientError

    s3 = boto3.client("s3")
    prefix = f"models/{args.run_id}"
    try:
        with s3.get_object(Bucket=args.bucket, Key=f"{prefix}/metadata.json")["Body"] as body:
            raw = body.read()
        with s3.get_object(Bucket=args.bucket, Key=f"{prefix}/predictions.csv")["Body"] as body:
            predictions = pd.read_csv(io.BytesIO(body.read()))
        metadata = json.loads(raw)
        new = updated_metadata(metadata, predictions, args.min_sensitivity)
        print(f"{args.run_id} · umbrales escogidos en validación con min_sensitivity={args.min_sensitivity}")
        print(describe(metadata, new))
        if args.dry_run:
            print("--dry-run: no se escribió nada en S3.")
            return 0

        backup = f"{prefix}/metadata.before-rethreshold.json"
        try:
            s3.head_object(Bucket=args.bucket, Key=backup)
        except ClientError:
            s3.put_object(Bucket=args.bucket, Key=backup, Body=raw, ContentType="application/json")
            print(f"Original guardado en s3://{args.bucket}/{backup}")
        s3.put_object(Bucket=args.bucket, Key=f"{prefix}/metadata.json", ContentType="application/json",
                      Body=json.dumps(new, indent=2, ensure_ascii=False).encode("utf-8"))
        print(f"Actualizado s3://{args.bucket}/{prefix}/metadata.json")
        return 0
    except ClientError as e:
        print(f"Error de AWS: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
