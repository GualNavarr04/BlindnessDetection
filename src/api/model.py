"""Carga el modelo en producción (models/current/ en S3_BUCKET_MODELS) para la API."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import timm
import torch
from torch import nn


@dataclass
class ModelBundle:
    model: nn.Module
    metadata: dict


def load_current_model(bucket: str | None = None) -> ModelBundle:
    """Descarga models/current/model.pth + metadata.json y arma el modelo listo para inferencia."""
    import boto3

    bucket = bucket or os.environ["S3_BUCKET_MODELS"]
    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION"))

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        s3.download_file(bucket, "models/current/model.pth", str(tmp_path / "model.pth"))
        s3.download_file(bucket, "models/current/metadata.json", str(tmp_path / "metadata.json"))
        metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
        model = timm.create_model(
            metadata["architecture"], pretrained=False, num_classes=metadata["num_classes"]
        )
        state_dict = torch.load(tmp_path / "model.pth", map_location="cpu", weights_only=True)

    model.load_state_dict(state_dict)
    model.eval()
    return ModelBundle(model=model, metadata=metadata)
