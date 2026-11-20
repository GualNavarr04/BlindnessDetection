"""Quita duplicados y divide APTOS 2019 en entrenamiento, validación y prueba.

Regla de duplicados (los grupos salen de notebooks/eda.ipynb, en DATA_DIR/duplicates.csv):
- Grupo con la misma etiqueta: se queda una imagen (la de mayor resolución) y se quitan las copias.
- Grupo con etiquetas distintas: se quita completo, porque no sabemos cuál etiqueta es la buena.

Después divide 70/15/15, estratificado por clase y con semilla fija, y escribe data/splits.csv
(`id_code,diagnosis,split`). Ese archivo se sube al repo para que todos los entrenamientos usen
exactamente la misma división.

El notebook notebooks/splits.ipynb usa estas funciones y muestra los resultados. Desde la terminal:
    python -m src.data.make_splits
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42
DEFAULT_OUT = Path("data") / "splits.csv"


def drop_duplicates(
    labels: pd.DataFrame,
    duplicates: pd.DataFrame,
    stats: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica la regla de duplicados.

    labels:     `id_code, diagnosis` (train.csv).
    duplicates: `id_code, dup_group` (duplicates.csv); solo las imágenes que están en algún grupo.
    stats:      opcional, `id_code, height, width` (image_stats.csv). Si viene, de cada grupo se queda
                la imagen más grande; si no, la primera por id_code.

    Regresa (kept, removed). `removed` tiene además `dup_group` y `reason`.
    """
    groups = duplicates[["id_code", "dup_group"]].merge(labels[["id_code", "diagnosis"]], on="id_code")
    if len(groups) != len(duplicates):
        raise ValueError("duplicates.csv tiene ids que no están en train.csv")

    groups["conflict"] = groups.groupby("dup_group").diagnosis.transform("nunique") > 1
    if stats is not None:
        size = stats.set_index("id_code").eval("height * width")
        groups["pixels"] = groups.id_code.map(size).fillna(0)
    else:
        groups["pixels"] = 0

    # Dentro de cada grupo sin conflicto, la primera después de ordenar es la que se queda
    ordered = groups.sort_values(["dup_group", "pixels", "id_code"], ascending=[True, False, True])
    keeper = ~ordered.conflict & ~ordered.duplicated("dup_group")

    removed = ordered[~keeper].copy()
    removed["reason"] = removed.conflict.map({True: "etiquetas distintas", False: "copia"})
    removed = removed[["id_code", "diagnosis", "dup_group", "reason"]].reset_index(drop=True)

    kept = labels[~labels.id_code.isin(removed.id_code)].reset_index(drop=True)
    return kept, removed


def split_dataset(labels: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Divide en train/val/test según SPLITS, estratificado por `diagnosis`.

    Regresa `id_code, diagnosis, split`, ordenado por id_code.
    """
    test_size = SPLITS["test"]
    val_size = SPLITS["val"] / (1 - test_size)  # proporción de lo que queda después de test

    rest, test = train_test_split(labels, test_size=test_size, stratify=labels.diagnosis, random_state=seed)
    train, val = train_test_split(rest, test_size=val_size, stratify=rest.diagnosis, random_state=seed)

    out = pd.concat([train.assign(split="train"), val.assign(split="val"), test.assign(split="test")])
    return out[["id_code", "diagnosis", "split"]].sort_values("id_code").reset_index(drop=True)


def make_splits(
    labels: pd.DataFrame,
    duplicates: pd.DataFrame,
    stats: pd.DataFrame | None = None,
    seed: int = SEED,
) -> pd.DataFrame:
    """Quita duplicados y divide. Regresa `id_code, diagnosis, split`."""
    kept, _ = drop_duplicates(labels, duplicates, stats)
    return split_dataset(kept, seed)


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    data_dir = Path(os.environ.get("DATA_DIR", "data/aptos")).expanduser()
    parser = argparse.ArgumentParser(description="Quita duplicados y divide APTOS en train/val/test.")
    parser.add_argument("--data-dir", type=Path, default=data_dir,
                        help="Carpeta con train.csv, duplicates.csv e image_stats.csv (por defecto DATA_DIR).")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Archivo de salida.")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    dup_path = args.data_dir / "duplicates.csv"
    if not dup_path.is_file():
        raise SystemExit(f"No existe {dup_path}: primero corre notebooks/eda.ipynb.")
    stats_path = args.data_dir / "image_stats.csv"

    labels = pd.read_csv(args.data_dir / "train.csv")
    duplicates = pd.read_csv(dup_path)
    stats = pd.read_csv(stats_path) if stats_path.is_file() else None

    splits = make_splits(labels, duplicates, stats, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(args.out, index=False)
    print(f"{len(labels) - len(splits)} imágenes quitadas, {len(splits)} divididas -> {args.out}")
    print(splits.split.value_counts().to_string())


if __name__ == "__main__":
    main()
