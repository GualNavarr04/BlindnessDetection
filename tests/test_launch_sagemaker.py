"""Pruebas de launch_sagemaker.py sin llamar a AWS."""

import io
import json
import re
import tarfile
from datetime import datetime

import pytest

from src.training import launch_sagemaker as ls


def request_for(argv):
    args = ls.parse_args(argv + ["--bucket", "modelos", "--role-arn", "arn:aws:iam::1:role/R"])
    return args, ls.build_request(args, job_name=ls.job_name_for(args.run_id), default_bucket="sm-default",
                                  code_uri="s3://sm-default/code.tar.gz", image_uri="img", commit="abc")


def test_defaults_are_final_model():
    args, req = request_for([])
    hp = {k: json.loads(v) for k, v in req["HyperParameters"].items()}
    assert hp["architecture"] == "tf_efficientnet_b3.ns_jft_in1k"
    assert hp["image-size"] == 512 and hp["epochs"] == 20 and hp["batch-size"] == 32
    assert hp["sagemaker_program"] == "src/training/train.py"
    assert req["ResourceConfig"]["InstanceType"] == "ml.g5.xlarge"
    assert req["EnableManagedSpotTraining"] is True
    assert req["StoppingCondition"] == {"MaxRuntimeInSeconds": 3600, "MaxWaitTimeInSeconds": 7200}


def test_channels_checkpoints_and_env():
    args, req = request_for(["--run-id", "r1"])
    assert req["InputDataConfig"][0]["ChannelName"] == "cache"
    assert req["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"] == "s3://modelos/data/aptos-cache/"
    # el rol solo escribe en el bucket por defecto de SageMaker fuera de models/*
    assert req["CheckpointConfig"]["S3Uri"].startswith("s3://sm-default/") and "r1" in req["CheckpointConfig"]["S3Uri"]
    assert req["OutputDataConfig"]["S3OutputPath"].startswith("s3://sm-default/")
    assert req["Environment"]["S3_BUCKET_MODELS"] == "modelos"
    assert req["Environment"]["GIT_COMMIT"] == "abc"


def test_on_demand_has_no_spot_settings():
    _, req = request_for(["--spot", "0"])
    assert "EnableManagedSpotTraining" not in req and "CheckpointConfig" not in req
    assert "MaxWaitTimeInSeconds" not in req["StoppingCondition"]


def test_metric_regexes_match_train_output():
    line = "epoch=3 train_loss=0.5120 val_loss=0.6010 val_qwk=0.8120 time=41s"
    final = "test_qwk=0.8700 test_sensitivity=0.9100 test_specificity=-0.0100"
    found = {}
    for m in ls.METRIC_DEFINITIONS:
        hit = re.search(m["Regex"], line) or re.search(m["Regex"], final)
        found[m["Name"]] = float(hit.group(1))
    assert found == {"epoch": 3, "train_loss": 0.512, "val_loss": 0.601, "val_qwk": 0.812,
                     "test_qwk": 0.87, "test_sensitivity": 0.91, "test_specificity": -0.01}


def test_run_id_and_job_name():
    now = datetime(2026, 11, 23, 9, 5, 7)
    assert ls.default_run_id("tf_efficientnet_b3.ns_jft_in1k", 0, now) == "2026-11-23-effb3-090507"
    assert ls.default_run_id("efficientnet_b0", 64, now) == "2026-11-23-effb0-prueba-090507"
    name = ls.job_name_for("2026-11-23_effb3.x" + "y" * 80)
    assert re.fullmatch(r"[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}", name)


def test_tarball_has_code_splits_and_training_requirements():
    files = ls.source_files()
    names = set(files)
    assert "src/training/train.py" in names and "src/common/preprocess.py" in names
    assert "data/splits.csv" in names
    assert not any(n.startswith("src/api/") for n in names)  # la API no hace falta para entrenar
    with tarfile.open(fileobj=io.BytesIO(ls.build_tarball(files)), mode="r:gz") as tar:
        req = tar.extractfile("requirements.txt").read().decode()
    assert "timm" in req and "fastapi" not in req.lower()


def test_max_wait_must_cover_max_run():
    with pytest.raises(SystemExit):
        ls.parse_args(["--max-run", "3600", "--max-wait", "60"])
