"""Entrenamiento del clasificador de retinopatía (contratos 4.3 y 4.7 de docs/plan.md).

Corre igual en SageMaker Studio (para probar) y como trabajo de entrenamiento de SageMaker:

    python -m src.training.train --subset 64 --epochs 1 --upload 0     # prueba rápida en Studio
    python -m src.training.train --image-size 512 --cache-only 1            # solo preparar el caché a 512
    python src/training/train.py --architecture efficientnet_b0 --image-size 384 --epochs 15 \\
        --batch-size 32 --lr 3e-4 --run-id 2026-11-20-effb0-001          # lo que lanza SageMaker

Pasos:
1. Lee data/splits.csv (train / val / test, ya sin duplicados).
2. Recorta la retina y la escala a --image-size UNA vez, y lo guarda en --cache-dir/<size>/. Así
   cada época lee fotos chicas en lugar de las originales de hasta 4,000 px. Es el mismo recorte y
   escalado que hace la API con preprocess(), para que el modelo vea las fotos igual en los dos lados.
   Para las fotos que no están en el caché usa las originales de --data-dir, y si tampoco están, las
   baja de Kaggle (contrato 4.1). Si el caché ya está completo, no necesita las originales.
3. Entrena con pesos por clase y aumentos de datos (solo en entrenamiento). Cada época imprime
   `epoch=3 train_loss=0.512 val_loss=0.601 val_qwk=0.812` (lo lee SageMaker) y guarda un
   checkpoint en --checkpoint-dir: si la instancia Spot se interrumpe, al reiniciar sigue de ahí.
4. Con la mejor época (mayor kappa en validación) elige los umbrales en validación y mide en prueba.
5. Escribe model.pth, metadata.json y predictions.csv en --output-dir y los sube a
   s3://<S3_BUCKET_MODELS>/models/<run_id>/.

Los hiperparámetros son `--nombre valor` porque así los pasa SageMaker; por eso los interruptores
son 0/1 (`--upload 0`) y no banderas sin valor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# En SageMaker se ejecuta como `python src/training/train.py`: hace falta la raíz del repo en el path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # en Studio, DATA_DIR y S3_BUCKET_MODELS vienen del .env; en SageMaker no hay .env ni hace falta
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src.common.preprocess import IMAGE_SIZE, MEAN, STD, crop_retina, load_image, to_tensor  # noqa: E402
from src.data.metrics import choose_thresholds, compute_metrics  # noqa: E402

NUM_CLASSES = 5
TRAFFIC_LIGHT = {"verde": [0, 1], "amarillo": [2], "rojo": [3, 4]}
ON_SAGEMAKER = "SM_MODEL_DIR" in os.environ


# ---------------------------------------------------------------------------- argumentos


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--architecture", default="efficientnet_b0", help="Nombre del modelo en timm")
    p.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--subset", type=int, default=0, help="Usar solo N fotos de entrenamiento (0 = todas)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-id", default=None, help="Carpeta models/<run_id>/ (por defecto: fecha-arquitectura-hora)")
    p.add_argument("--pretrained", type=int, default=1, help="1 = pesos de ImageNet")
    p.add_argument("--upload", type=int, default=1, help="1 = subir los artefactos a S3")
    p.add_argument("--download", type=int, default=1, help="1 = bajar de Kaggle las fotos que falten")
    p.add_argument("--workers", type=int, default=-1, help="Procesos del DataLoader (-1 = automático)")
    p.add_argument("--cache-only", type=int, default=0, help="1 = solo preparar el caché de --image-size y salir")
    p.add_argument("--bucket", default=os.environ.get("S3_BUCKET_MODELS"))
    p.add_argument("--splits", default=str(ROOT / "data" / "splits.csv"))
    p.add_argument("--data-dir", default=None, help="Carpeta con train_images/ (por defecto: canal de SageMaker o DATA_DIR)")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--checkpoint-dir", default=None)
    args = p.parse_args(argv)

    if args.run_id is None:
        now = datetime.now()
        args.run_id = f"{now:%Y-%m-%d}-{args.architecture.replace('_', '')}-{now:%H%M%S}"
    if args.data_dir is None:
        args.data_dir = os.environ.get("SM_CHANNEL_TRAIN") or os.environ.get("DATA_DIR") or str(ROOT / "data" / "aptos")
    data_dir = Path(args.data_dir).expanduser()
    if args.cache_dir is None:
        # En SageMaker: el canal "cache" (fotos ya recortadas, subidas a S3) si se pasó; si no, se arma en /tmp.
        # En Studio el caché queda junto al dataset y se reutiliza entre corridas.
        args.cache_dir = os.environ.get("SM_CHANNEL_CACHE") or ("/tmp/aptos-cache" if ON_SAGEMAKER else str(data_dir / "cache"))
    if args.output_dir is None:
        args.output_dir = os.environ.get("SM_MODEL_DIR") or str(ROOT / "outputs" / args.run_id)
    if args.checkpoint_dir is None:
        # SageMaker sincroniza /opt/ml/checkpoints con S3 (checkpoint_s3_uri) y lo restaura tras una interrupción Spot
        args.checkpoint_dir = "/opt/ml/checkpoints" if ON_SAGEMAKER else str(Path(args.output_dir) / "checkpoints")
    if args.workers < 0:
        args.workers = 0 if os.name == "nt" else min(4, os.cpu_count() or 1)
    return args


# ---------------------------------------------------------------------------- datos


def select_rows(splits: pd.DataFrame, subset: int, seed: int) -> pd.DataFrame:
    """Con subset=N toma N fotos de entrenamiento y N/4 (mínimo 16) de validación y de prueba."""
    if not subset:
        return splits.reset_index(drop=True)
    sizes = {"train": subset, "val": max(16, subset // 4), "test": max(16, subset // 4)}
    parts = [g.sample(min(sizes[name], len(g)), random_state=seed) for name, g in splits.groupby("split")]
    return pd.concat(parts).sort_values(["split", "id_code"]).reset_index(drop=True)


def ensure_images(data_dir: Path, ids: list[str], download: bool) -> None:
    missing = [i for i in ids if not (data_dir / "train_images" / f"{i}.png").is_file()]
    if not missing:
        return
    if not download:
        raise FileNotFoundError(f"Faltan {len(missing)} fotos en {data_dir / 'train_images'}, p. ej. {missing[:3]}")
    from src.data.download_kaggle import download_aptos

    print(f"[datos] faltan {len(missing)} fotos, se bajan de Kaggle", flush=True)
    download_aptos(data_dir, ids=missing)


def cached_path(cache_dir: Path, size: int, image_id: str) -> Path:
    return cache_dir / str(size) / f"{image_id}.png"


def build_cache(data_dir: Path, cache_dir: Path, ids: list[str], size: int, workers: int = 8) -> None:
    """Guarda cada foto recortada y escalada a size × size (lo mismo que hace preprocess() antes de normalizar)."""
    (cache_dir / str(size)).mkdir(parents=True, exist_ok=True)
    pending = [i for i in ids if not cached_path(cache_dir, size, i).is_file()]
    if not pending:
        return

    def work(image_id: str) -> None:
        img = crop_retina(load_image(data_dir / "train_images" / f"{image_id}.png"))
        interpolation = cv2.INTER_AREA if min(img.shape[:2]) > size else cv2.INTER_LINEAR
        small = cv2.resize(img, (size, size), interpolation=interpolation)
        ok, png = cv2.imencode(".png", cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
        if not ok:
            raise ValueError(f"No se pudo guardar {image_id} en el caché")
        out = cached_path(cache_dir, size, image_id)
        tmp = out.with_suffix(".tmp")
        png.tofile(str(tmp))  # tofile: funciona con rutas con acentos en Windows
        os.replace(tmp, out)

    start = time.time()
    print(f"[datos] preparando {len(pending)} fotos a {size}×{size} en {cache_dir}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, pending))
    print(f"[datos] caché listo en {time.time() - start:.0f} s", flush=True)


def augment(img: np.ndarray) -> np.ndarray:
    """Aumentos solo para entrenamiento. La retina es un círculo: se puede girar en cualquier ángulo.

    El brillo se multiplica (no se suma) para que el fondo siga negro, como en una foto real.
    """
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(0, 360), random.uniform(0.95, 1.10))
    img = cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    if random.random() < 0.5:
        img = img[:, ::-1]
    img = np.clip(img.astype(np.float32) * random.uniform(0.8, 1.2), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


class FundusDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, cache_dir: Path, size: int, train: bool):
        self.ids = rows.id_code.tolist()
        self.labels = rows.diagnosis.astype(int).tolist()
        self.cache_dir, self.size, self.train = cache_dir, size, train

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        img = load_image(cached_path(self.cache_dir, self.size, self.ids[i]))
        if self.train:
            img = augment(img)
        return to_tensor(img, self.size), self.labels[i]


def class_weights(labels: pd.Series) -> torch.Tensor:
    """Peso inversamente proporcional a la frecuencia de cada clase; el promedio de los pesos es 1."""
    counts = np.bincount(labels.astype(int), minlength=NUM_CLASSES).astype(np.float64)
    counts[counts == 0] = 1  # una clase ausente (solo pasa con --subset chico) no debe dividir entre 0
    weights = counts.sum() / (NUM_CLASSES * counts)
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


# ---------------------------------------------------------------------------- entrenamiento


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None, scaler=None):
    """Una pasada por loader. Con optimizer entrena; sin él solo evalúa. Regresa (loss, probs, labels)."""
    train = optimizer is not None
    model.train(train)
    total_loss, all_probs, all_labels = 0.0, [], []
    use_amp = device.type == "cuda"
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            total_loss += loss.item() * len(y)
            all_probs.append(torch.softmax(logits.float(), dim=1).detach().cpu())
            all_labels.append(y.cpu())
    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    return total_loss / len(labels), probs, labels


def git_commit() -> str:
    if os.environ.get("GIT_COMMIT"):
        return os.environ["GIT_COMMIT"]
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "desconocido"


def upload(bucket: str, run_id: str, output_dir: Path, files: list[str]) -> None:
    import boto3

    s3 = boto3.client("s3")
    for name in files:
        key = f"models/{run_id}/{name}"
        s3.upload_file(str(output_dir / name), bucket, key)
        print(f"[s3] s3://{bucket}/{key}", flush=True)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.time()

    data_dir, cache_dir = Path(args.data_dir).expanduser(), Path(args.cache_dir).expanduser()
    output_dir, checkpoint_dir = Path(args.output_dir), Path(args.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.upload and not args.bucket:
        raise SystemExit("Falta el bucket: variable S3_BUCKET_MODELS o --bucket (o --upload 0 para no subir)")

    # --- datos
    splits_path = Path(args.splits)
    splits = pd.read_csv(splits_path)
    rows = select_rows(splits, args.subset, args.seed)
    # Solo hacen falta las fotos originales de las que no están en el caché
    not_cached = [i for i in rows.id_code if not cached_path(cache_dir, args.image_size, i).is_file()]
    ensure_images(data_dir, not_cached, bool(args.download))
    build_cache(data_dir, cache_dir, not_cached, args.image_size)
    if args.cache_only:
        print(f"[datos] caché completo: {len(rows)} fotos en {cache_dir / str(args.image_size)}", flush=True)
        return {}
    parts = {name: rows[rows.split == name] for name in ("train", "val", "test")}
    print(f"[datos] {', '.join(f'{k} {len(v)}' for k, v in parts.items())} · run_id {args.run_id}", flush=True)

    def loader(name: str, train: bool) -> DataLoader:
        dataset = FundusDataset(parts[name], cache_dir, args.image_size, train)
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=train, num_workers=args.workers,
                          pin_memory=torch.cuda.is_available(), drop_last=train and len(dataset) > args.batch_size)

    train_loader, val_loader, test_loader = loader("train", True), loader("val", False), loader("test", False)

    # --- modelo
    import timm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(args.architecture, pretrained=bool(args.pretrained), num_classes=NUM_CLASSES).to(device)
    weights = class_weights(parts["train"].diagnosis)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, pct_start=0.1,
                                                    total_steps=max(1, args.epochs * len(train_loader)))
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    print(f"[modelo] {args.architecture} en {device} · pesos por clase {[round(float(w), 2) for w in weights]}", flush=True)

    # --- reanudar si hay checkpoint de esta misma corrida (interrupción Spot)
    last_ckpt, best_ckpt = checkpoint_dir / "last.pth", checkpoint_dir / "best.pth"
    start_epoch, best_qwk, best_epoch = 1, -1.0, 0
    if last_ckpt.is_file():
        state = torch.load(last_ckpt, map_location=device, weights_only=False)
        if state.get("run_id") == args.run_id:
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            scaler.load_state_dict(state["scaler"])
            start_epoch, best_qwk, best_epoch = state["epoch"] + 1, state["best_qwk"], state["best_epoch"]
            print(f"[checkpoint] Reanudando desde la época {state['epoch']}", flush=True)

    # --- épocas
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, _, _ = run_epoch(model, train_loader, criterion, device, optimizer, scheduler, scaler)
        val_loss, val_probs, val_labels = run_epoch(model, val_loader, criterion, device)
        val_qwk = compute_metrics(val_labels, val_probs)["qwk"]
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_qwk={val_qwk:.4f} "
              f"time={time.time() - t0:.0f}s", flush=True)

        if val_qwk > best_qwk:
            best_qwk, best_epoch = val_qwk, epoch
            torch.save(model.state_dict(), best_ckpt)
        torch.save({"run_id": args.run_id, "epoch": epoch, "best_qwk": best_qwk, "best_epoch": best_epoch,
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict()}, last_ckpt)

    # --- evaluación con la mejor época
    model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
    _, val_probs, val_labels = run_epoch(model, val_loader, criterion, device)
    _, test_probs, test_labels = run_epoch(model, test_loader, criterion, device)
    thresholds = choose_thresholds(val_labels, val_probs)
    val_metrics = compute_metrics(val_labels, val_probs, thresholds["referable_threshold"])
    test_metrics = compute_metrics(test_labels, test_probs, thresholds["referable_threshold"])
    thresholds = {k: round(float(v), 4) for k, v in thresholds.items()}
    print(f"[resultado] mejor época {best_epoch} · umbral de referible {thresholds['referable_threshold']} · "
          f"umbral de confianza {thresholds['confidence_threshold']}", flush=True)
    print(f"test_qwk={test_metrics['qwk']:.4f} test_sensitivity={test_metrics['sensitivity_referable']:.4f} "
          f"test_specificity={test_metrics['specificity_referable']:.4f}", flush=True)

    # --- artefactos (contrato 4.3)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, output_dir / "model.pth")

    predictions = pd.concat([
        pd.DataFrame(probs, columns=[f"p{c}" for c in range(NUM_CLASSES)]).assign(
            id_code=parts[name].id_code.values, split=name, diagnosis=labels)
        for name, probs, labels in [("val", val_probs, val_labels), ("test", test_probs, test_labels)]])
    predictions = predictions[["id_code", "split", "diagnosis"] + [f"p{c}" for c in range(NUM_CLASSES)]]
    predictions.round(5).to_csv(output_dir / "predictions.csv", index=False)

    metadata = {
        "run_id": args.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "architecture": args.architecture,
        "num_classes": NUM_CLASSES,
        "image_size": args.image_size,
        "normalization": {"mean": list(MEAN), "std": list(STD)},
        "decision": {
            "referable_threshold": thresholds["referable_threshold"],
            "confidence_threshold": thresholds["confidence_threshold"],
            "traffic_light": TRAFFIC_LIGHT,
        },
        # Métricas en PRUEBA, con los umbrales elegidos en validación
        "metrics": {**{k: test_metrics[k] for k in ("qwk", "sensitivity_referable", "specificity_referable")},
                    "confusion_matrix": test_metrics["confusion_matrix"], "n": int(len(test_labels))},
        "validation_metrics": {**{k: val_metrics[k] for k in ("qwk", "sensitivity_referable", "specificity_referable")},
                               "n": int(len(val_labels))},
        "training": {
            "epochs": args.epochs, "best_epoch": best_epoch, "batch_size": args.batch_size, "lr": args.lr,
            "weight_decay": args.weight_decay, "subset": args.subset, "seed": args.seed,
            "pretrained": bool(args.pretrained), "class_weights": [round(float(w), 4) for w in weights],
            "n_train": int(len(parts["train"])), "device": str(device),
            "duration_s": round(time.time() - started),
        },
        "split_manifest_sha256": hashlib.sha256(splits_path.read_bytes()).hexdigest(),
        "git_commit": git_commit(),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[artefactos] {output_dir}", flush=True)

    if args.upload:
        upload(args.bucket, args.run_id, output_dir, ["model.pth", "metadata.json", "predictions.csv"])
    return metadata


if __name__ == "__main__":
    main()
