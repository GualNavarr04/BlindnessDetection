"""Lanza train.py como trabajo de entrenamiento de SageMaker (contrato 4.7 de docs/plan.md).

    python -m src.training.launch_sagemaker --dry-run                    # solo muestra lo que lanzaría
    python -m src.training.launch_sagemaker --subset 64 --epochs 1       # corrida corta de prueba
    python -m src.training.launch_sagemaker                              # modelo final (B3 a 512, 20 épocas)
    python -m src.training.launch_sagemaker --no-wait                    # lanzar y regresar la terminal

Qué hace:
1. Empaqueta el código que necesita train.py (src/ sin la API, data/splits.csv) y pone
   src/training/requirements.txt como requirements.txt en la raíz, para que el contenedor de PyTorch
   instale solo eso. Lo sube al bucket por defecto de SageMaker.
2. Crea el trabajo con boto3 (sin el SDK `sagemaker`): contenedor de PyTorch con Python 3.11,
   ml.g5.xlarge en Spot, canal `cache` = s3://<S3_BUCKET_MODELS>/data/aptos-cache/, checkpoints en S3
   para reanudar si AWS interrumpe la instancia, y métricas por época leídas de los logs.
3. Espera a que termine e imprime las métricas y los segundos que se cobraron.

train.py sube model.pth, metadata.json y predictions.csv a s3://<S3_BUCKET_MODELS>/models/<run_id>/.

El rol de SageMaker solo puede leer y escribir models/* y leer data/aptos-cache/* en el bucket de
modelos, y usar el bucket por defecto de SageMaker (sagemaker-<región>-<cuenta>). Por eso el código,
los checkpoints y la salida de SageMaker van en ese bucket por defecto.

Variables de entorno (.env): S3_BUCKET_MODELS, SAGEMAKER_ROLE_ARN, AWS_REGION, KAGGLE_TOKEN_PARAM.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

# Contenedor oficial de PyTorch para entrenamiento (Python 3.11). Se puede cambiar con --image-uri.
DLC_ACCOUNT = "763104351884"
DLC_TAG = "2.5.1-gpu-py311-cu124-ubuntu22.04-sagemaker"
PREFIX = "blindness-detection"  # carpeta dentro del bucket por defecto de SageMaker

# train.py imprime "epoch=3 train_loss=0.512 val_loss=0.601 val_qwk=0.812" y al final
# "test_qwk=… test_sensitivity=… test_specificity=…"; SageMaker las lee con estas expresiones.
NUMBER = r"(-?[0-9]+(?:\.[0-9]+)?)"
METRIC_DEFINITIONS = [
    {"Name": "epoch", "Regex": rf"epoch={NUMBER} "},
    {"Name": "train_loss", "Regex": rf"train_loss={NUMBER}"},
    {"Name": "val_loss", "Regex": rf"val_loss={NUMBER}"},
    {"Name": "val_qwk", "Regex": rf"val_qwk={NUMBER}"},
    {"Name": "test_qwk", "Regex": rf"test_qwk={NUMBER}"},
    {"Name": "test_sensitivity", "Regex": rf"test_sensitivity={NUMBER}"},
    {"Name": "test_specificity", "Regex": rf"test_specificity={NUMBER}"},
]


# ---------------------------------------------------------------------------- argumentos


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("hiperparámetros de train.py (modelo final por defecto, ver plan.md)")
    g.add_argument("--architecture", default="tf_efficientnet_b3.ns_jft_in1k")
    g.add_argument("--image-size", type=int, default=512)
    g.add_argument("--epochs", type=int, default=20)
    g.add_argument("--batch-size", type=int, default=32)
    g.add_argument("--lr", type=float, default=3e-4)
    # Los del modelo en producción (2026-11-23-effb3-162536). Se probó weight decay 0.05 con label
    # smoothing 0.1 (2026-11-23-effb3-172449) y salió peor: 40 % de casos a revisión humana.
    g.add_argument("--weight-decay", type=float, default=1e-4)
    g.add_argument("--label-smoothing", type=float, default=0.0)
    g.add_argument("--min-sensitivity", type=float, default=0.95, help="Meta de sensibilidad al escoger el umbral")
    g.add_argument("--subset", type=int, default=0, help="N fotos de entrenamiento (0 = todas)")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--run-id", default=None, help="Carpeta models/<run_id>/ (por defecto: fecha-modelo-hora)")

    a = p.add_argument_group("AWS")
    a.add_argument("--instance-type", default="ml.g5.xlarge")
    a.add_argument("--spot", type=int, default=1, help="1 = instancia Spot (más barata, se puede interrumpir)")
    a.add_argument("--max-run", type=int, default=3600, help="Segundos máximos entrenando")
    a.add_argument("--max-wait", type=int, default=7200, help="Segundos máximos contando la espera de Spot")
    a.add_argument("--volume-gb", type=int, default=30)
    a.add_argument("--image-uri", default=None, help=f"Por defecto: pytorch-training:{DLC_TAG}")
    a.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    a.add_argument("--role-arn", default=os.environ.get("SAGEMAKER_ROLE_ARN"))
    a.add_argument("--bucket", default=os.environ.get("S3_BUCKET_MODELS"), help="Bucket de modelos")
    a.add_argument("--kaggle-token-param",
                   default=os.environ.get("KAGGLE_TOKEN_PARAM", "/blindness-detection/kaggle-token"))

    p.add_argument("--dry-run", action="store_true", help="Armar el paquete y mostrar la petición sin lanzar nada")
    p.add_argument("--no-wait", action="store_true", help="Lanzar y salir sin esperar a que termine")
    args = p.parse_args(argv)

    if args.run_id is None:
        args.run_id = default_run_id(args.architecture, args.subset)
    if args.max_wait < args.max_run:
        p.error("--max-wait debe ser mayor o igual que --max-run")
    return args


def short_name(architecture: str) -> str:
    """tf_efficientnet_b3.ns_jft_in1k -> effb3; cualquier otro nombre se limpia a letras y números."""
    match = re.search(r"efficientnet_?(b\d|v2_\w)", architecture)
    if match:
        return "eff" + match.group(1).replace("_", "")
    return re.sub(r"[^a-z0-9]", "", architecture.split(".")[0].lower())[:12]


def default_run_id(architecture: str, subset: int, now: datetime | None = None) -> str:
    now = now or datetime.now()
    tag = f"{short_name(architecture)}-prueba" if subset else short_name(architecture)
    return f"{now:%Y-%m-%d}-{tag}-{now:%H%M%S}"


def job_name_for(run_id: str) -> str:
    """Nombre del trabajo: letras, números y guiones, máximo 63 caracteres."""
    name = re.sub(r"[^a-zA-Z0-9-]", "-", f"bd-{run_id}")
    return re.sub(r"-+", "-", name).strip("-")[:63]


# ---------------------------------------------------------------------------- paquete de código


def source_files(root: Path = ROOT) -> dict[str, Path]:
    """Archivos que lleva el paquete: {ruta dentro del tar: archivo local}.

    Solo lo que usa train.py: src/ sin la API (ni la página web) y data/splits.csv. El
    requirements.txt de la raíz del repo trae FastAPI y más; el contenedor instala el de
    src/training/, que va como requirements.txt en la raíz del paquete.
    """
    files: dict[str, Path] = {}
    for path in sorted((root / "src").rglob("*.py")):
        rel = path.relative_to(root)
        # .ipynb_checkpoints: copias que Jupyter deja en Studio
        if {"__pycache__", ".ipynb_checkpoints"} & set(rel.parts) or rel.parts[:2] == ("src", "api"):
            continue
        files[rel.as_posix()] = path
    files["data/splits.csv"] = root / "data" / "splits.csv"
    files["requirements.txt"] = root / "src" / "training" / "requirements.txt"
    missing = [str(p) for p in files.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Faltan archivos para el paquete: {missing}")
    return files


def build_tarball(files: dict[str, Path]) -> bytes:
    """sourcedir.tar.gz en memoria, con fechas fijas para que el mismo código dé el mismo archivo."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for arcname, path in sorted(files.items()):
            data = path.read_bytes()
            info = tarfile.TarInfo(arcname)
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def git_state(root: Path = ROOT) -> tuple[str, bool]:
    """(commit, hay cambios sin commit en src/ o data/splits.csv)."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "data/splits.csv"], cwd=root,
                               capture_output=True, text=True, check=True).stdout.strip()
        return commit, bool(dirty)
    except (OSError, subprocess.CalledProcessError):
        return "desconocido", False


# ---------------------------------------------------------------------------- petición


def build_request(args: argparse.Namespace, *, job_name: str, default_bucket: str, code_uri: str,
                  image_uri: str, commit: str) -> dict:
    """Parámetros de create_training_job. Sin llamadas a AWS, para poder probarla."""
    base = f"s3://{default_bucket}/{PREFIX}"
    hyperparameters = {
        "architecture": args.architecture,
        "image-size": args.image_size,
        "epochs": args.epochs,
        "batch-size": args.batch_size,
        "lr": args.lr,
        "weight-decay": args.weight_decay,
        "label-smoothing": args.label_smoothing,
        "min-sensitivity": args.min_sensitivity,
        "subset": args.subset,
        "seed": args.seed,
        "run-id": args.run_id,
        # Lo que necesita el contenedor de PyTorch para correr nuestro script
        "sagemaker_program": "src/training/train.py",
        "sagemaker_submit_directory": code_uri,
        "sagemaker_region": args.region,
        "sagemaker_container_log_level": 20,
    }
    request = {
        "TrainingJobName": job_name,
        "RoleArn": args.role_arn,
        "AlgorithmSpecification": {
            "TrainingImage": image_uri,
            "TrainingInputMode": "File",
            "MetricDefinitions": METRIC_DEFINITIONS,
            "EnableSageMakerMetricsTimeSeries": True,
        },
        # Como el SDK de SageMaker: cada valor en JSON; el contenedor lo decodifica y llama a
        # `python src/training/train.py --architecture … --image-size 512 …`
        "HyperParameters": {k: json.dumps(v) for k, v in hyperparameters.items()},
        "InputDataConfig": [{
            "ChannelName": "cache",
            "DataSource": {"S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": f"s3://{args.bucket}/data/aptos-cache/",
                "S3DataDistributionType": "FullyReplicated",
            }},
        }],
        "OutputDataConfig": {"S3OutputPath": f"{base}/output/"},
        "ResourceConfig": {"InstanceType": args.instance_type, "InstanceCount": 1, "VolumeSizeInGB": args.volume_gb},
        "StoppingCondition": {"MaxRuntimeInSeconds": args.max_run},
        "Environment": {
            "S3_BUCKET_MODELS": args.bucket,
            "KAGGLE_TOKEN_PARAM": args.kaggle_token_param,
            "GIT_COMMIT": commit,
        },
        "Tags": [{"Key": "Project", "Value": "BlindnessDetection"}, {"Key": "RunId", "Value": args.run_id}],
    }
    if args.spot:
        request["EnableManagedSpotTraining"] = True
        request["StoppingCondition"]["MaxWaitTimeInSeconds"] = args.max_wait
        # train.py guarda last.pth y best.pth en /opt/ml/checkpoints; SageMaker los copia aquí y los
        # restaura si AWS interrumpe la instancia Spot
        request["CheckpointConfig"] = {"S3Uri": f"{base}/checkpoints/{args.run_id}/",
                                       "LocalPath": "/opt/ml/checkpoints"}
    return request


# ---------------------------------------------------------------------------- AWS


def wait_for(sm, job_name: str, poll_s: int = 30) -> dict:
    """Imprime cada cambio de estado hasta que el trabajo termina. Regresa describe_training_job."""
    last = None
    while True:
        job = sm.describe_training_job(TrainingJobName=job_name)
        state = (job["TrainingJobStatus"], job.get("SecondaryStatus"))
        if state != last:
            message = job.get("SecondaryStatusTransitions", [{}])[-1].get("StatusMessage", "")
            print(f"[{datetime.now():%H:%M:%S}] {state[0]} · {state[1]} · {message}", flush=True)
            last = state
        if state[0] in ("Completed", "Failed", "Stopped"):
            return job
        time.sleep(poll_s)


def summary(job: dict, bucket: str, run_id: str) -> str:
    lines = [f"Estado: {job['TrainingJobStatus']}"]
    if job.get("FailureReason"):
        lines.append(f"Motivo: {job['FailureReason']}")
    metrics = {m["MetricName"]: m["Value"] for m in job.get("FinalMetricDataList", [])}
    if metrics:
        lines.append("Métricas: " + ", ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items())))
    if job.get("TrainingTimeInSeconds"):
        billable = job.get("BillableTimeInSeconds", job["TrainingTimeInSeconds"])
        lines.append(f"Tiempo entrenando: {job['TrainingTimeInSeconds'] / 60:.1f} min"
                     f" · cobrado: {billable / 60:.1f} min")
    if job["TrainingJobStatus"] == "Completed":
        lines.append(f"Modelo: s3://{bucket}/models/{run_id}/")
        lines.append(f"Para ponerlo en producción: python -m src.training.promote_model --run-id {run_id}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    missing = [name for name, value in (("S3_BUCKET_MODELS / --bucket", args.bucket),
                                        ("SAGEMAKER_ROLE_ARN / --role-arn", args.role_arn)) if not value]
    if missing and not args.dry_run:
        print(f"Falta: {', '.join(missing)} (ponlos en .env)", file=sys.stderr)
        return 2

    files = source_files()
    tarball = build_tarball(files)
    commit, dirty = git_state()
    job_name = job_name_for(args.run_id)
    image_uri = args.image_uri or f"{DLC_ACCOUNT}.dkr.ecr.{args.region}.amazonaws.com/pytorch-training:{DLC_TAG}"
    print(f"Trabajo {job_name} · run_id {args.run_id} · {args.architecture} a {args.image_size} px · "
          f"{args.epochs} épocas · {args.instance_type}{' Spot' if args.spot else ''}")
    print(f"Paquete: {len(files)} archivos, {len(tarball) / 1024:.0f} KB · commit {commit[:7]}")
    if dirty:
        print("Aviso: hay cambios sin commit en src/ o data/splits.csv; se entrena con la copia local y "
              "metadata.json va a decir el último commit, no lo que corrió.", file=sys.stderr)

    if args.dry_run:
        request = build_request(args, job_name=job_name, default_bucket="sagemaker-<región>-<cuenta>",
                                code_uri=f"s3://sagemaker-<región>-<cuenta>/{PREFIX}/code/{job_name}/sourcedir.tar.gz",
                                image_uri=image_uri, commit=commit)
        print(json.dumps(request, indent=2, ensure_ascii=False))
        print("\n--dry-run: no se subió ni se lanzó nada.")
        return 0

    import boto3
    from botocore.exceptions import ClientError

    try:
        session = boto3.Session(region_name=args.region)
        account = session.client("sts").get_caller_identity()["Account"]
        default_bucket = f"sagemaker-{args.region}-{account}"
        code_key = f"{PREFIX}/code/{job_name}/sourcedir.tar.gz"
        session.client("s3").upload_fileobj(io.BytesIO(tarball), default_bucket, code_key)
        request = build_request(args, job_name=job_name, default_bucket=default_bucket,
                                code_uri=f"s3://{default_bucket}/{code_key}", image_uri=image_uri, commit=commit)

        sm = session.client("sagemaker")
        sm.create_training_job(**request)
        print(f"Lanzado. Consola: https://{args.region}.console.aws.amazon.com/sagemaker/home"
              f"?region={args.region}#/jobs/{job_name}")
        print(f"Logs: aws logs tail /aws/sagemaker/TrainingJobs --log-stream-name-prefix {job_name} --follow")
        if args.no_wait:
            return 0

        job = wait_for(sm, job_name)
        print(summary(job, args.bucket, args.run_id))
        return 0 if job["TrainingJobStatus"] == "Completed" else 1
    except KeyboardInterrupt:
        print(f"\nDejaste de esperar, pero el trabajo sigue corriendo. Para detenerlo:\n"
              f"aws sagemaker stop-training-job --training-job-name {job_name}", file=sys.stderr)
        return 130
    except ClientError as e:
        print(f"Error de AWS: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
