"""Pruebas de make_splits.py con un dataset inventado (no necesita las imágenes)."""

import pandas as pd
import pytest

from src.data.make_splits import SPLITS, drop_duplicates, make_splits, split_dataset


def fake_labels(n_per_class=40):
    rows = [(f"img{c}_{i:03d}", c) for c in range(5) for i in range(n_per_class)]
    return pd.DataFrame(rows, columns=["id_code", "diagnosis"])


@pytest.fixture
def data():
    labels = fake_labels()
    duplicates = pd.DataFrame({
        "id_code": ["img0_000", "img0_001", "img0_002",   # grupo 1: misma etiqueta, 3 copias
                    "img1_000", "img2_000"],              # grupo 2: etiquetas distintas
        "dup_group": [1, 1, 1, 2, 2],
    })
    stats = pd.DataFrame({"id_code": labels.id_code, "height": 100, "width": 100})
    stats.loc[stats.id_code == "img0_002", ["height", "width"]] = 2000  # la más grande del grupo 1
    return labels, duplicates, stats


def test_drop_duplicates_keeps_one_per_group(data):
    labels, duplicates, stats = data
    kept, removed = drop_duplicates(labels, duplicates, stats)

    # Grupo 1: se queda la de mayor resolución y se quitan las otras dos
    assert "img0_002" in set(kept.id_code)
    assert not {"img0_000", "img0_001"} & set(kept.id_code)
    # Grupo 2: con etiquetas distintas se quita completo
    assert not {"img1_000", "img2_000"} & set(kept.id_code)

    assert len(kept) == len(labels) - 4
    assert sorted(removed.reason) == ["copia", "copia", "etiquetas distintas", "etiquetas distintas"]


def test_drop_duplicates_without_stats_keeps_first_id(data):
    labels, duplicates, _ = data
    kept, _ = drop_duplicates(labels, duplicates)
    assert "img0_000" in set(kept.id_code)


def test_drop_duplicates_rejects_unknown_ids(data):
    labels, duplicates, _ = data
    bad = pd.concat([duplicates, pd.DataFrame({"id_code": ["no-existe"], "dup_group": [3]})])
    with pytest.raises(ValueError):
        drop_duplicates(labels, bad)


def test_split_proportions_and_stratification():
    labels = fake_labels(n_per_class=100)
    splits = split_dataset(labels)

    assert len(splits) == len(labels)
    assert splits.id_code.is_unique  # ninguna imagen en dos partes
    for name, frac in SPLITS.items():
        part = splits[splits.split == name]
        assert abs(len(part) / len(labels) - frac) < 0.01
        # cada clase conserva su proporción (20%) en cada parte
        assert (part.diagnosis.value_counts(normalize=True) - 0.2).abs().max() < 0.02


def test_split_is_reproducible():
    labels = fake_labels()
    assert split_dataset(labels, seed=1).equals(split_dataset(labels, seed=1))
    assert not split_dataset(labels, seed=1).equals(split_dataset(labels, seed=2))


def test_make_splits_end_to_end(data):
    labels, duplicates, stats = data
    splits = make_splits(labels, duplicates, stats)
    assert list(splits.columns) == ["id_code", "diagnosis", "split"]
    assert len(splits) == len(labels) - 4
    assert set(splits.split) == {"train", "val", "test"}
