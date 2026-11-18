"""Pruebas de download_kaggle.py sin internet: se reemplaza la descarga HTTP por una falsa."""

import io
import zipfile

import pytest

from src.data import download_kaggle as dk

TRAIN_CSV = b"id_code,diagnosis\naaa,0\nbbb,2\nccc,4\nddd,1\n"


def zipped(name, content):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(name, content)
    return buf.getvalue()


@pytest.fixture
def fake_kaggle(monkeypatch):
    """Simula Kaggle: train.csv sin comprimir y cada imagen dentro de un zip, como la API real."""
    calls = []

    def fake_get(file_name, token):
        calls.append(file_name)
        if file_name == "train.csv":
            return TRAIN_CSV
        image_id = file_name.split("/")[-1].removesuffix(".png")
        return zipped(f"{image_id}.png", f"png-{image_id}".encode())

    monkeypatch.setattr(dk, "_http_get", fake_get)
    monkeypatch.setenv("KAGGLE_API_TOKEN", "token-de-prueba")
    return calls


def test_downloads_everything(tmp_path, fake_kaggle):
    out = dk.download_aptos(tmp_path, workers=2)
    assert out == str(tmp_path)
    assert (tmp_path / "train.csv").read_bytes() == TRAIN_CSV
    images = sorted(p.name for p in (tmp_path / "train_images").iterdir())
    assert images == ["aaa.png", "bbb.png", "ccc.png", "ddd.png"]
    assert (tmp_path / "train_images" / "bbb.png").read_bytes() == b"png-bbb"  # sin el zip


def test_skips_existing_files(tmp_path, fake_kaggle):
    dk.download_aptos(tmp_path, workers=2)
    fake_kaggle.clear()
    dk.download_aptos(tmp_path, workers=2)
    assert fake_kaggle == []  # no volvió a bajar nada


def test_limit_and_ids(tmp_path, fake_kaggle):
    dk.download_aptos(tmp_path / "a", limit=2, workers=1)
    assert len(list((tmp_path / "a" / "train_images").iterdir())) == 2

    dk.download_aptos(tmp_path / "b", ids=["ccc"], workers=1)
    assert [p.name for p in (tmp_path / "b" / "train_images").iterdir()] == ["ccc.png"]

    with pytest.raises(dk.KaggleError):
        dk.download_aptos(tmp_path / "c", ids=["no-existe"])


def test_token_order(tmp_path, monkeypatch):
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    monkeypatch.delenv("KAGGLE_TOKEN_PARAM", raising=False)
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))

    # Sin variable ni archivo, va a Parameter Store; aquí se simula que falla
    import boto3

    def no_aws(*args, **kwargs):
        raise RuntimeError("sin credenciales")

    monkeypatch.setattr(boto3, "client", no_aws)
    with pytest.raises(dk.KaggleError):
        dk.get_kaggle_token()

    (tmp_path / "access_token").write_text("desde-archivo\n")
    assert dk.get_kaggle_token() == "desde-archivo"

    monkeypatch.setenv("KAGGLE_API_TOKEN", "desde-env")
    assert dk.get_kaggle_token() == "desde-env"
