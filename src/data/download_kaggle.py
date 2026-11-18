"""Descarga del dataset APTOS 2019 desde Kaggle (contrato 4.1 de docs/plan.md).

Baja train.csv y train_images/<id_code>.png a dest_dir, imagen por imagen, con la API REST de
Kaggle. No usa el paquete `kaggle` (los tokens nuevos KGAT_ piden kaggle>=1.8 y Python 3.11+),
solo la biblioteca estándar, así que funciona igual en Studio, en SageMaker y en local.

Bajar imagen por imagen (en vez del zip de 10 GB con todo) permite:
- bajar solo algunas imágenes para pruebas rápidas (`limit` o `ids`), y
- retomar una descarga cortada: las imágenes que ya están no se vuelven a bajar.

El token se busca en este orden:
1. Variable de entorno KAGGLE_API_TOKEN.
2. Archivo access_token en KAGGLE_CONFIG_DIR (por defecto ~/.kaggle/).
3. Parameter Store, en el parámetro KAGGLE_TOKEN_PARAM (así funciona en Studio y en SageMaker).

Uso desde la terminal:
    python -m src.data.download_kaggle                    # todo, a DATA_DIR
    python -m src.data.download_kaggle --limit 64         # solo 64 imágenes, para probar
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

COMPETITION = "aptos2019-blindness-detection"
API_URL = f"https://www.kaggle.com/api/v1/competitions/data/download/{COMPETITION}/"
RETRIES = 4
TIMEOUT_S = 120


class KaggleError(RuntimeError):
    pass


# ---------------------------------------------------------------------------- token


def get_kaggle_token() -> str:
    """Regresa el token de Kaggle según el orden descrito arriba."""
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if token:
        return token

    config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", "~/.kaggle")).expanduser()
    token_file = config_dir / "access_token"
    if token_file.is_file():
        token = token_file.read_text(encoding="utf-8").strip()
        if token:
            return token

    param = os.environ.get("KAGGLE_TOKEN_PARAM", "").strip()
    if param:
        import boto3  # solo hace falta si el token viene de Parameter Store

        ssm = boto3.client("ssm", region_name=os.environ.get("AWS_REGION", "us-east-1"))
        return ssm.get_parameter(Name=param, WithDecryption=True)["Parameter"]["Value"].strip()

    raise KaggleError(
        "No encontré el token de Kaggle. Define KAGGLE_API_TOKEN, pon el archivo "
        f"{token_file} o define KAGGLE_TOKEN_PARAM (Parameter Store)."
    )


# ---------------------------------------------------------------------------- HTTP


def _http_get(file_name: str, token: str) -> bytes:
    """Descarga un archivo de la competencia y regresa su contenido (sin descomprimir)."""
    url = API_URL + urllib.parse.quote(file_name, safe="")
    request = urllib.request.Request(url)
    # "unredirected": Kaggle redirige a Google Cloud Storage con una URL firmada, y esa URL
    # rechaza la petición si lleva nuestro header de autorización.
    request.add_unredirected_header("Authorization", f"Bearer {token}")
    request.add_header("User-Agent", "blindness-detection-downloader")

    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
                return response.read()
        except urllib.error.HTTPError as e:
            if e.code == 401:
                raise KaggleError("Kaggle rechazó el token (401): revisa que sea válido.") from e
            if e.code == 403:
                raise KaggleError(
                    "Kaggle negó el acceso (403): hay que aceptar las reglas de la competencia en "
                    f"https://www.kaggle.com/competitions/{COMPETITION}/rules"
                ) from e
            if e.code == 404:
                raise KaggleError(f"No existe {file_name} en la competencia (404).") from e
            if attempt == RETRIES or e.code not in (429, 500, 502, 503, 504):
                raise KaggleError(f"Error HTTP {e.code} al bajar {file_name}.") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == RETRIES:
                raise KaggleError(f"No se pudo bajar {file_name}: {e}") from e
        time.sleep(2**attempt)  # 2, 4, 8 s
    raise AssertionError("inalcanzable")


def _unwrap(content: bytes) -> bytes:
    """Kaggle a veces regresa el archivo dentro de un zip de una sola entrada."""
    if content[:4] == b"PK\x03\x04":
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            return z.read(z.infolist()[0])
    return content


def _write_atomic(path: Path, content: bytes) -> None:
    """Escribe a un .part y luego renombra: un archivo a medias nunca queda con su nombre final."""
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(content)
    tmp.replace(path)


# ---------------------------------------------------------------------------- descarga


def download_aptos(
    dest_dir: str | Path,
    ids: Iterable[str] | None = None,
    limit: int | None = None,
    workers: int = 16,
    seed: int = 0,
) -> str:
    """Descarga train.csv y train_images/ de APTOS 2019 a dest_dir y regresa la ruta.

    ids:   bajar solo estas imágenes (por id_code).
    limit: bajar solo N imágenes elegidas al azar con `seed` (ignorado si se pasa `ids`).
    Las imágenes que ya existen en dest_dir no se vuelven a bajar.
    """
    dest = Path(dest_dir).expanduser()
    images_dir = dest / "train_images"
    images_dir.mkdir(parents=True, exist_ok=True)
    token = get_kaggle_token()

    csv_path = dest / "train.csv"
    if not csv_path.is_file():
        _write_atomic(csv_path, _unwrap(_http_get("train.csv", token)))
    with csv_path.open(newline="", encoding="utf-8") as f:
        all_ids = [row["id_code"] for row in csv.DictReader(f)]

    if ids is not None:
        wanted = list(dict.fromkeys(ids))
        unknown = set(wanted) - set(all_ids)
        if unknown:
            raise KaggleError(f"{len(unknown)} ids no están en train.csv, p. ej. {sorted(unknown)[:3]}")
    elif limit:
        wanted = random.Random(seed).sample(all_ids, min(limit, len(all_ids)))
    else:
        wanted = all_ids

    pending = [i for i in wanted if not (images_dir / f"{i}.png").is_file()]
    print(f"[download] {len(wanted)} imágenes pedidas, {len(wanted) - len(pending)} ya estaban, "
          f"{len(pending)} por bajar -> {images_dir}", flush=True)

    def fetch(image_id: str) -> None:
        content = _unwrap(_http_get(f"train_images/{image_id}.png", token))
        _write_atomic(images_dir / f"{image_id}.png", content)

    start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, i) for i in pending]
        for done, future in enumerate(as_completed(futures), start=1):
            future.result()  # si una imagen falla después de los reintentos, se detiene todo
            if done % 200 == 0 or done == len(pending):
                print(f"[download] {done}/{len(pending)} ({time.time() - start:.0f} s)", flush=True)

    return str(dest)


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Descarga APTOS 2019 desde Kaggle.")
    parser.add_argument("--dest", default=os.environ.get("DATA_DIR", "data/aptos"),
                        help="Carpeta destino (por defecto DATA_DIR).")
    parser.add_argument("--limit", type=int, default=None, help="Bajar solo N imágenes.")
    parser.add_argument("--workers", type=int, default=16, help="Descargas en paralelo.")
    args = parser.parse_args()
    print(download_aptos(args.dest, limit=args.limit, workers=args.workers))


if __name__ == "__main__":
    main()
