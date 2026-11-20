"""Pruebas de train.py con fotos sintéticas, un modelo sin pesos preentrenados y sin internet ni AWS."""

import json

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

timm = pytest.importorskip("timm")

from src.training import train  # noqa: E402


def write_png(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, png = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    png.tofile(str(path))


def fake_retina(rng, label, size=120):
    """Círculo con fondo negro; el color cambia un poco con la clase para que haya algo que aprender."""
    img = np.zeros((size, int(size * 1.3), 3), dtype=np.uint8)
    color = (150 + 20 * label, 80, 40)
    cv2.circle(img, (img.shape[1] // 2, size // 2), int(size * 0.45), color, thickness=-1)
    noise = rng.normal(0, 5, img.shape) * (img > 0)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


@pytest.fixture
def dataset(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for split, per_class in [("train", 4), ("val", 2), ("test", 2)]:
        for label in range(5):
            for k in range(per_class):
                image_id = f"{split}{label}{k}"
                write_png(tmp_path / "data" / "train_images" / f"{image_id}.png", fake_retina(rng, label))
                rows.append({"id_code": image_id, "diagnosis": label, "split": split})
    splits = tmp_path / "splits.csv"
    pd.DataFrame(rows).to_csv(splits, index=False)
    return tmp_path, splits


def args_for(tmp_path, splits, epochs=2, run_id="prueba"):
    return ["--architecture", "efficientnet_b0", "--image-size", "64", "--epochs", str(epochs),
            "--batch-size", "8", "--pretrained", "0", "--upload", "0", "--download", "0", "--workers", "0",
            "--run-id", run_id, "--splits", str(splits), "--data-dir", str(tmp_path / "data"),
            "--cache-dir", str(tmp_path / "cache"), "--output-dir", str(tmp_path / "out")]


def test_class_weights_balance_classes():
    labels = pd.Series([0] * 8 + [1] * 2 + [2] * 4 + [3] * 1 + [4] * 1)
    w = train.class_weights(labels)
    assert w.mean().item() == pytest.approx(1.0)
    # cada clase pesa lo mismo en total: peso × número de fotos es constante
    totals = w.numpy() * np.bincount(labels, minlength=5)
    assert np.allclose(totals, totals[0])


def test_augment_keeps_shape_and_black_background():
    rng = np.random.default_rng(0)
    img = cv2.resize(fake_retina(rng, 2), (64, 64))
    out = train.augment(img)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert out[0, 0].max() == 0  # la esquina sigue negra: el brillo se multiplica, no se suma


def test_select_rows_subset():
    splits = pd.DataFrame({"id_code": [f"i{k}" for k in range(300)], "diagnosis": [k % 5 for k in range(300)],
                           "split": ["train"] * 200 + ["val"] * 50 + ["test"] * 50})
    sub = train.select_rows(splits, 64, seed=0)
    assert sub.split.value_counts().to_dict() == {"train": 64, "val": 16, "test": 16}
    assert train.select_rows(splits, 64, seed=0).equals(sub)  # reproducible
    assert len(train.select_rows(splits, 0, seed=0)) == 300


def test_cache_matches_api_preprocessing(dataset):
    # El caché debe dar el mismo tensor que preprocess() de la API
    from src.common.preprocess import preprocess, to_tensor, load_image

    tmp_path, splits = dataset
    path = tmp_path / "data" / "train_images" / "val00.png"
    train.build_cache(tmp_path / "data", tmp_path / "cache", ["val00"], 64, workers=1)
    cached = to_tensor(load_image(train.cached_path(tmp_path / "cache", 64, "val00")), 64)
    assert torch.allclose(cached, preprocess(str(path), 64), atol=1e-6)


def test_missing_images_without_download_fail(dataset):
    tmp_path, _ = dataset
    with pytest.raises(FileNotFoundError):
        train.ensure_images(tmp_path / "data", ["no_existe"], download=False)


def test_end_to_end_writes_contract_artifacts(dataset, capsys):
    tmp_path, splits = dataset
    metadata = train.main(args_for(tmp_path, splits))
    out = tmp_path / "out"

    # contrato 4.7: una línea por época que lee SageMaker
    log = capsys.readouterr().out
    assert "epoch=1 train_loss=" in log and "epoch=2 " in log and "val_qwk=" in log

    # contrato 4.3: metadata.json
    saved = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert saved == json.loads(json.dumps(metadata))
    for key in ["run_id", "created_at", "architecture", "num_classes", "image_size", "normalization",
                "decision", "metrics", "split_manifest_sha256", "git_commit"]:
        assert key in saved
    assert saved["image_size"] == 64 and saved["num_classes"] == 5
    assert set(saved["decision"]) == {"referable_threshold", "confidence_threshold", "traffic_light"}
    assert {"qwk", "sensitivity_referable", "specificity_referable"} <= set(saved["metrics"])

    # contrato 4.3: model.pth se carga en un modelo timm sin pesos preentrenados
    model = timm.create_model(saved["architecture"], pretrained=False, num_classes=5)
    model.load_state_dict(torch.load(out / "model.pth", weights_only=True))

    predictions = pd.read_csv(out / "predictions.csv")
    assert predictions.split.value_counts().to_dict() == {"val": 10, "test": 10}
    assert np.allclose(predictions[[f"p{c}" for c in range(5)]].sum(axis=1), 1, atol=1e-3)


def test_resumes_from_checkpoint(dataset, capsys, monkeypatch):
    # Simula una interrupción Spot a media época 2; al relanzar la misma corrida sigue en la época 2
    tmp_path, splits = dataset
    real_run_epoch, calls = train.run_epoch, []

    def interrupted(*a, **k):
        calls.append(1)
        if len(calls) == 3:  # 1: train época 1, 2: val época 1, 3: train época 2
            raise RuntimeError("instancia Spot interrumpida")
        return real_run_epoch(*a, **k)

    monkeypatch.setattr(train, "run_epoch", interrupted)
    with pytest.raises(RuntimeError, match="Spot"):
        train.main(args_for(tmp_path, splits, epochs=2))
    monkeypatch.setattr(train, "run_epoch", real_run_epoch)
    capsys.readouterr()

    train.main(args_for(tmp_path, splits, epochs=2))
    log = capsys.readouterr().out
    assert "Reanudando desde la época 1" in log
    assert "epoch=1 " not in log and "epoch=2 " in log


def test_full_cache_does_not_need_originals(dataset):
    # En SageMaker se puede pasar solo el caché (fotos ya recortadas) sin las originales de 10 GB
    import shutil

    tmp_path, splits = dataset
    ids = pd.read_csv(splits).id_code.tolist()
    train.build_cache(tmp_path / "data", tmp_path / "cache", ids, 64, workers=2)
    shutil.rmtree(tmp_path / "data" / "train_images")
    metadata = train.main(args_for(tmp_path, splits, epochs=1))
    assert metadata["training"]["n_train"] == 20
