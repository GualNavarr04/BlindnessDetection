"""Promueve un modelo entrenado (models/<run_id>/) a producción (models/current/) en S3.

Copia model.pth y metadata.json dentro del mismo bucket (server-side copy, no baja ni sube
los archivos) y deja un registro de trazabilidad en models/current/promoted_from.json con el
run_id promovido y la fecha.
"""

import argparse
import json
import os
from datetime import datetime, timezone

import boto3
from dotenv import load_dotenv

load_dotenv()

ARTIFACT_FILES = ("model.pth", "metadata.json")


def promote(bucket: str, run_id: str) -> None:
    s3 = boto3.client("s3")

    for filename in ARTIFACT_FILES:
        src_key = f"models/{run_id}/{filename}"
        s3.head_object(Bucket=bucket, Key=src_key)  # falla rápido si el run_id no existe

    for filename in ARTIFACT_FILES:
        src_key = f"models/{run_id}/{filename}"
        dst_key = f"models/current/{filename}"
        s3.copy_object(Bucket=bucket, CopySource={"Bucket": bucket, "Key": src_key}, Key=dst_key)
        print(f"{src_key} -> {dst_key}")

    record = {"run_id": run_id, "promoted_at": datetime.now(timezone.utc).isoformat()}
    s3.put_object(
        Bucket=bucket,
        Key="models/current/promoted_from.json",
        Body=json.dumps(record, indent=2).encode(),
        ContentType="application/json",
    )
    print(f"Promovido {run_id} -> models/current/ en bucket {bucket}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Carpeta en models/<run_id>/ a promover")
    parser.add_argument(
        "--bucket",
        default=os.environ.get("S3_BUCKET_MODELS"),
        help="Bucket de modelos (default: variable de entorno S3_BUCKET_MODELS)",
    )
    args = parser.parse_args()

    if not args.bucket:
        parser.error("Falta --bucket o la variable de entorno S3_BUCKET_MODELS")

    promote(args.bucket, args.run_id)


if __name__ == "__main__":
    main()
