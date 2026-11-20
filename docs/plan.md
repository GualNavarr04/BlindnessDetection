# Plan de trabajo

Estado al 19 de noviembre de 2026. Este documento dice qué ya está hecho, dónde y cómo vamos a
trabajar, qué falta y quién hace qué. Los dos trabajamos en las dos partes del proyecto (datos y
entrenamiento, y servicio y monitoreo). Para no esperarnos, cada pieza que conecta el trabajo de
uno con el del otro tiene un **contrato** (sección 4) que se acuerda una vez al principio.

## 1. Lo que ya está hecho

### Documentación
- [x] README con problema, propuesta, dataset, métricas, servicios y arquitectura.
- [x] Diagrama de arquitectura (`docs/arquitectura.png`) y flujo en `docs/arquitectura.md`.
- [x] Convención de modelos en S3: `models/<run_id>/` (historial) y `models/current/` (producción).
- [x] `.env.example` y `requirements.txt`.

### Código
- [x] `src/training/promote_model.py`: promueve un `run_id` a `models/current/`.
- [x] `src/data/metrics.py` (contrato 4.6): kappa cuadrático, sensibilidad/especificidad de referibles y elección de umbrales.
- [x] `src/common/preprocess.py` (contrato 4.2), `src/data/download_kaggle.py` (contrato 4.1) y
  `src/data/make_splits.py`, con 21 pruebas en `tests/`.
- [x] `src/common/quality_filter.py` (contrato 4.8), con 6 pruebas más (27 en total). Umbrales
  calibrados en `notebooks/quality.ipynb`: brillo < 28 → "oscura", nitidez < 0.5 → "borrosa".
- [x] `src/training/train.py` (contratos 4.3 y 4.7) y `src/training/requirements.txt`, con 8 pruebas
  más (35 en total): artefactos del contrato, reanudar tras una interrupción Spot y que el caché dé el
  mismo tensor que `preprocess()` de la API. Falta la prueba en Studio (`--subset 64 --epochs 1`).

### Datos
- [x] Dataset completo en Studio (`/home/sagemaker-user/data/aptos`, 3,662 imágenes). La descarga imagen
  por imagen se cortó porque Kaggle limitó la cuenta (429 por 72 h); se bajó el zip completo desde la
  página de Kaggle y se extrajo `train.csv` y `train_images/`.
- [x] `notebooks/eda.ipynb` ejecutado con el dataset completo y con conclusiones (sección 7): pesos por
  clase, riesgo de que el modelo aprenda la cámara, 145 grupos de duplicados (39 con etiquetas
  distintas, quedan 3,466 imágenes) y umbrales de calidad por ajustar. Deja `duplicates.csv` e
  `image_stats.csv` en `DATA_DIR`.
- [x] División en `data/splits.csv` con `notebooks/splits.ipynb`: se quitan 196 duplicados y quedan
  3,466 imágenes, train 2,426 / val 520 / test 520, estratificado (semilla 42). Las 8 verificaciones
  del notebook pasan y las cámaras quedan repartidas en las tres partes.
- [x] `notebooks/quality.ipynb` ejecutado y con conclusiones: la nitidez se divide entre el brillo
  (correlación con el brillo 0.01, antes 0.77). Con fotos de prueba degradadas se rechaza el 99% del
  desenfoque fuerte y el 88% de las casi negras; del dataset solo 2 fotos, que se quedan en el
  entrenamiento. Deja `quality_scores.csv` y `quality_degraded.csv` en `DATA_DIR`.
- [ ] Decidir de dónde lee el dataset el entrenamiento en SageMaker: volver a bajarlo de Kaggle puede
  topar con el mismo límite. Propuesta: subir a S3 solo el caché que arma `train.py` (las fotos ya
  recortadas a 384 px, cientos de MB en lugar de 10 GB) y pasarlo como canal `cache`. Hace falta que
  `BlindnessDetection-SageMakerRole` pueda leer ese prefijo.

### AWS (región `us-east-1`)
- [x] Bucket de modelos `blindness-detection-modelos-<account-id>`: versionado, cifrado, sin acceso público.
- [x] Bucket de predicciones `blindness-detection-predicciones-<account-id>`: borra `logs/predictions/` a los 90 días.
- [x] Rol `BlindnessDetection-SageMakerRole`: escribe `models/*` y lee el token de Kaggle.
- [x] Rol `BlindnessDetection-EC2Role` (con perfil de instancia): lee `models/*` y escribe `logs/predictions/*`.
- [x] Rol `BlindnessDetection-StudioRole`: el de los notebooks de Studio (sección 2).
- [x] Token de Kaggle en Parameter Store: `/blindness-detection/kaggle-token` (`SecureString`).
- [x] Grupo de logs `/blindness-detection/api` con retención de 30 días.
- [x] Presupuesto `BlindnessDetection-mensual`: avisos por correo al pasar $10 y $25 (sin descontar créditos).
- [x] Cupo de GPU para entrenamientos: 1 × `ml.g4dn.xlarge` y 1 × `ml.g5.xlarge`, también en Spot.
- [x] Usuario IAM para cada integrante (cada quien con sus propias credenciales).
- [x] SageMaker Studio: dominio, perfiles y espacios (sección 2).

### Kaggle
- [x] Reglas de la competencia APTOS aceptadas y token probado (la descarga funciona).

## 2. Dónde trabajamos: SageMaker Studio

Desarrollamos en **SageMaker Studio** (JupyterLab en el navegador). El entrenamiento con GPU
**no** corre en el notebook: se lanza desde el notebook como *trabajo de entrenamiento* de
SageMaker, que se apaga solo al terminar.

| Pieza | Configuración |
|---|---|
| Dominio | `blindness-detection` (`d-x3mszzb8c5yn`), región `us-east-1`, VPC por defecto con acceso a internet |
| Perfiles de usuario | `kevin` y `eduardo` |
| Espacios | `kevin-jupyterlab` y `eduardo-jupyterlab`: JupyterLab privado, `ml.t3.medium` (sin GPU), disco de 30 GB (se puede subir hasta 50) |
| Apagado automático | A los 60 minutos sin uso (cada quien lo puede subir hasta 4 horas) |
| Rol de los notebooks | `BlindnessDetection-StudioRole`: `AmazonSageMakerFullAccess` + leer/escribir los dos buckets del proyecto, leer el token de Kaggle, lanzar entrenamientos con `BlindnessDetection-SageMakerRole` y publicar métricas de prueba en `BlindnessDetection` |
| Dataset | Se descarga en el disco del espacio: `/home/sagemaker-user/data/aptos` |
| Token de Kaggle | Se lee de Parameter Store; nadie necesita su propio token |

**Costo aproximado:** ~$0.05 por hora que el espacio está encendido, más ~$2.40 al mes por cada
disco de 30 GB. Los entrenamientos se pagan aparte (GPU Spot, solo mientras corren).

**Cómo empieza cada quien:**

1. Entrar a la consola de AWS con su usuario IAM → **Amazon SageMaker AI** → **Studio** →
   elegir su perfil (`kevin` o `eduardo`) → **Open Studio**.
2. En Studio: **JupyterLab** → abrir su espacio (`<nombre>-jupyterlab`) → **Run space** → **Open**.
   Tarda un par de minutos la primera vez.
3. Configurar git una vez, en la terminal de JupyterLab:
   ```bash
   git config --global user.name "Nombre Apellido"
   git config --global user.email "correo@iteso.mx"
   git clone https://github.com/GualNavarr04/BlindnessDetection.git
   ```
   Para hacer push, GitHub pide un *personal access token* **classic** (los *fine-grained* no
   sirven para el repo personal de otra persona, aunque seas colaborador):
   - Antes, ser colaborador del repo (Settings → Collaborators) y aceptar la invitación.
   - En GitHub: foto → **Settings → Developer settings → Personal access tokens → Tokens (classic)
     → Generate new token (classic)**. Scope: solo **`repo`**; vencimiento después de la entrega.
     Copiarlo (`ghp_...`), GitHub lo muestra una sola vez.
   - En la terminal de Studio:
     ```bash
     git config --global credential.helper store
     git push    # Username: tu usuario de GitHub · Password: el token ghp_...
     ```
     Queda guardado en `~/.git-credentials`, en el disco privado del espacio (fuera del repo).
4. `cp .env.example .env` y llenar el número de cuenta.
5. `pip install --user -r requirements.txt`. El `--user` es importante: instala en `~/.local`, que
   está en el disco del espacio y se conserva. Sin él se instala en `/opt/conda`, que se borra cada
   vez que el espacio se apaga (y sale `No module named cv2`).
6. **Al terminar el día:** detener el espacio (*Stop space*) aunque exista el apagado automático.

**Reglas del notebook:**
- El notebook es para explorar y probar. El código final va en archivos `.py` dentro de `src/`,
  que el notebook importa. Así se puede probar con pytest, correr en CI y usar en SageMaker y en
  la API.
- Nunca entrenar el modelo completo dentro del notebook: para eso está `launch_sagemaker.py`.
- No guardar llaves ni tokens en celdas de notebook (los notebooks se suben al repo).
- Limpiar las salidas grandes (imágenes) antes de hacer commit de un notebook.

## 3. Cabos sueltos

| Pendiente | Quién | Estado |
|---|---|---|
| Mandar a Eduardo sus credenciales y borrar el archivo | Kevin | ✅ Hecho |
| Token de Kaggle | Kevin | ✅ Se queda el actual en Parameter Store (decisión del equipo) |
| Retención de `/aws/sagemaker/TrainingJobs` | Kevin | ✅ Se guardan siempre (decisión del equipo) |
| Clonar el repo en Studio | Kevin | ✅ Hecho |
| Entrar a la consola con su usuario, cambiar contraseña y activar MFA | Eduardo | Pendiente |
| Crear un personal access token **classic** de GitHub para hacer push desde Studio | Ambos | Kevin ✅ · Eduardo pendiente. Ver sección 2, paso 3 |
| Revisar la regla del semáforo con un médico | Ambos | Pendiente. El README lo promete; ideal antes de la entrega final |

## 4. Contratos (acordar juntos antes de empezar)

Cada contrato tiene un **dueño** (quien lo implementa) y un **usuario** (quien lo usa). Mientras
el dueño no lo termina, el usuario trabaja con una versión de prueba que respeta el contrato.
Cambiar un contrato se avisa al otro antes de hacer push.

| Contrato | Dueño | Lo usa |
|---|---|---|
| 4.1 Descarga de datos | Kevin | Ambos (EDA, calidad, entrenamiento) |
| 4.2 Preprocesamiento | Kevin | Eduardo (API, Grad-CAM) |
| 4.3 Artefactos del modelo | Kevin (`train.py`) | Eduardo (API, modelo de prueba) |
| 4.4 Registro de predicciones | Eduardo | — (lo leen el monitoreo y la alarma) |
| 4.5 Variables de entorno | Ambos | Ambos |
| 4.6 Métricas | Eduardo | Kevin (`train.py`) |
| 4.7 Entrada de `train.py` | Kevin | Eduardo (`launch_sagemaker.py`) |
| 4.8 Filtro de calidad | Kevin | Eduardo (API) |
| 4.9 Respuesta de la API | Eduardo | Kevin (página web) |

### 4.1 Descarga de datos — `src/data/download_kaggle.py`

```python
def download_aptos(dest_dir: str) -> str:
    """Descarga train.csv y train_images/ de APTOS 2019 a dest_dir y regresa la ruta."""
```

El token se busca en este orden: variable de entorno `KAGGLE_API_TOKEN`, archivo
`~/.kaggle/access_token`, Parameter Store (`KAGGLE_TOKEN_PARAM`). En Studio y en los
entrenamientos se usa Parameter Store.

### 4.2 Preprocesamiento — `src/common/preprocess.py`

```python
IMAGE_SIZE = 384
MEAN = (0.485, 0.456, 0.406)   # ImageNet
STD = (0.229, 0.224, 0.225)

def load_image(source: str | bytes) -> np.ndarray:        # RGB uint8 (H, W, 3)
def crop_retina(img: np.ndarray) -> np.ndarray:           # recorta el borde negro
def to_tensor(img: np.ndarray, size: int = IMAGE_SIZE) -> torch.Tensor:  # (3, size, size) normalizado
```

Lo usan el entrenamiento y la API, para que el modelo vea las fotos igual en los dos lados.
Los aumentos de datos (rotaciones, brillo) **no** van aquí: solo los usa `train.py`.

### 4.3 Artefactos del modelo — `models/<run_id>/`

- `model.pth`: `state_dict` de `timm.create_model(architecture, pretrained=False, num_classes=5)`.
  La salida del modelo son 5 logits; la probabilidad es `softmax`.
- `metadata.json`:

```json
{
  "run_id": "2026-11-20-effb0-001",
  "created_at": "2026-11-20T18:00:00Z",
  "architecture": "efficientnet_b0",
  "num_classes": 5,
  "image_size": 384,
  "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
  "decision": {
    "referable_threshold": 0.5,
    "confidence_threshold": 0.6,
    "traffic_light": {"verde": [0, 1], "amarillo": [2], "rojo": [3, 4]}
  },
  "metrics": {"qwk": 0.0, "sensitivity_referable": 0.0, "specificity_referable": 0.0},
  "split_manifest_sha256": "…",
  "git_commit": "…"
}
```

- `referable_threshold`: si `P(clase ≥ 2)` lo supera, el caso es referible (amarillo o rojo).
- `confidence_threshold`: si la probabilidad máxima queda debajo, no se asigna color y se manda
  a revisión humana.
- Los umbrales del **filtro de calidad** no van aquí (ver 4.8), porque no dependen del modelo.

`train.py` sube estos dos archivos con `boto3` a `models/<run_id>/` al terminar (no basta el
`output_path` de SageMaker, que deja un `model.tar.gz` en otra ruta). La API siempre lee
`models/current/`.

### 4.4 Registro de predicciones — `logs/predictions/AAAA/MM/DD/<uuid>.json`

```json
{"timestamp": "…", "run_id": "…", "quality_ok": true, "quality_reason": null,
 "predicted_class": 2, "probabilities": [0.05, 0.1, 0.6, 0.15, 0.1],
 "traffic_light": "amarillo", "low_confidence": false, "latency_ms": 340}
```

Sin imagen, nombre de archivo ni nada que identifique al paciente.

### 4.5 Variables de entorno (`.env.example`)

`AWS_REGION`, `S3_BUCKET_MODELS`, `S3_BUCKET_PREDICTIONS`, `SAGEMAKER_ROLE_ARN`,
`KAGGLE_TOKEN_PARAM`, `DATA_DIR`, `CLOUDWATCH_LOG_GROUP`, `CLOUDWATCH_NAMESPACE`.

### 4.6 Métricas — `src/data/metrics.py`

```python
def compute_metrics(y_true: np.ndarray, probs: np.ndarray, referable_threshold: float = 0.5) -> dict:
    """y_true: (N,) clases 0-4; probs: (N, 5) softmax.
    Regresa {"qwk", "sensitivity_referable", "specificity_referable", "confusion_matrix"}."""

def choose_thresholds(y_true: np.ndarray, probs: np.ndarray, min_sensitivity: float = 0.9) -> dict:
    """Barre umbrales sobre P(clase >= 2) y elige el de mayor especificidad que cumpla
    min_sensitivity. Regresa {"referable_threshold", "confidence_threshold"}."""
```

Mientras no existe, `train.py` puede calcular solo el kappa con `sklearn` y dejar umbrales fijos.

### 4.7 Entrada de `train.py`

SageMaker le pasa los hiperparámetros como argumentos de línea de comandos:

```bash
python train.py --architecture efficientnet_b0 --image-size 384 --epochs 15 \
                --batch-size 32 --lr 3e-4 --subset 0 --run-id 2026-11-20-effb0-001
```

- `--subset N`: usar solo N imágenes de entrenamiento y N/4 de validación y prueba (0 = todas).
- Interruptores 0/1, porque SageMaker pasa todo como `--nombre valor`: `--pretrained`, `--upload`
  (subir a S3) y `--download` (bajar de Kaggle lo que falte). Los tres son 1 por defecto.
- Lee `KAGGLE_TOKEN_PARAM` y `S3_BUCKET_MODELS` de variables de entorno, y `GIT_COMMIT` si el
  lanzador lo pasa (en el contenedor no hay `.git`).
- Datos: canal `cache` (fotos ya recortadas, `<size>/<id_code>.png`) o canal `train` (originales,
  `train_images/`). Sin ninguno, baja de Kaggle las fotos que falten.
- Guarda checkpoints en `/opt/ml/checkpoints` (para Spot) y al reiniciar sigue de la última época.
- Imprime cada época en este formato, para que `metric_definitions` lo lea:
  `epoch=3 train_loss=0.512 val_loss=0.601 val_qwk=0.812`. Al final: `test_qwk=… test_sensitivity=…
  test_specificity=…`.
- Además de `model.pth` y `metadata.json` sube `predictions.csv` (probabilidades de validación y
  prueba), para evaluar por clase y por cámara sin volver a correr el modelo. En `metadata.json`,
  `metrics` son de **prueba** (con los umbrales elegidos en validación) y `validation_metrics` de
  validación.
- Para el lanzador: `source_dir` es la raíz del repo (usa `src/` y `data/splits.csv`),
  `entry_point="src/training/train.py"`, y `src/training/requirements.txt` debe quedar como
  `requirements.txt` en la raíz del `source_dir` (el de la raíz del repo trae FastAPI y otras cosas
  que el entrenamiento no usa).

Mientras `train.py` no existe, `launch_sagemaker.py` se prueba con un script falso que acepte
estos argumentos y escriba un modelo de prueba.

### 4.8 Filtro de calidad — `src/common/quality_filter.py`

```python
@dataclass
class QualityResult:
    ok: bool
    reason: str | None      # "borrosa", "oscura" o None
    scores: dict            # {"brightness": float, "sharpness": float}

def check_quality(img: np.ndarray) -> QualityResult:   # img RGB uint8 de load_image()
```

Los umbrales viven en el mismo archivo como constantes (`BRIGHTNESS_THRESHOLD = 28.0`,
`SHARPNESS_THRESHOLD = 0.5`) y salen de `notebooks/quality.ipynb`. Si la foto es oscura y borrosa a la
vez, el motivo es "oscura".

### 4.9 Respuesta de la API — `POST /predict`

Recibe la foto como `multipart/form-data` (campo `file`) y regresa:

```json
{
  "quality_ok": true,
  "quality_reason": null,
  "predicted_class": 2,
  "class_name": "Moderada",
  "probabilities": [0.05, 0.1, 0.6, 0.15, 0.1],
  "traffic_light": "amarillo",
  "recommendation": "Referir a oftalmología",
  "low_confidence": false,
  "heatmap_png_base64": "iVBORw0KGgo…",
  "run_id": "2026-11-20-effb0-001"
}
```

Si la foto no pasa el filtro de calidad, regresa `quality_ok: false`, el motivo y los demás
campos en `null`. Si `low_confidence` es `true`, `traffic_light` es `null` y la recomendación
es "Revisión humana". La página web se puede construir con respuestas falsas en este formato.

## 5. Quién hace qué

Cada uno tiene tareas de las dos partes del proyecto. Las tareas se agruparon para que, dentro
de cada parte, las piezas que dependen mucho entre sí queden con la misma persona.

### Kevin

| # | Parte | Entregable | Detalle |
|---|---|---|---|
| 1 | Datos | `src/data/download_kaggle.py` | Contrato 4.1 |
| 2 | Datos | `src/common/preprocess.py` | Contrato 4.2. **Primero y subirlo pronto**: Eduardo lo usa en la API y en Grad-CAM |
| 3 | Datos | `notebooks/eda.ipynb` | Distribución de clases, duplicados (`imagehash`), recorte de retina y ejemplos de fotos borrosas y oscuras |
| 4 | Datos | `src/data/make_splits.py` | Quita duplicados (incluidos los de etiquetas distintas) y hace el split estratificado 70/15/15 con semilla fija. Sube `data/splits.csv` (`id_code,diagnosis,split`) al repo |
| 5 | Servicio | `src/common/quality_filter.py` | Contrato 4.8. Los umbrales salen de `notebooks/quality.ipynb` ✅ |
| 6 | Entrenamiento | `src/training/train.py` | Contrato 4.7. EfficientNet-B0 (`timm`), pesos por clase, checkpoints, `model.pth` + `metadata.json` a `models/<run_id>/` (contrato 4.3). Usa `metrics.py` de Eduardo. Probar en Studio con `--subset 64 --epochs 1` ✅ (falta la prueba en Studio) |
| 7 | Entrenamiento | `src/training/requirements.txt` | Solo lo que no trae el contenedor de PyTorch: `timm`, `opencv-python-headless`, `pandas`, `scikit-learn` ✅ |
| 8 | Servicio | Página web (`src/api/static/`) | Subir la foto y mostrar semáforo, recomendación, probabilidades y mapa de calor. Se construye contra el contrato 4.9 con respuestas falsas |
| 9 | Ambos | Pruebas | `tests/` para `preprocess.py`, `make_splits.py` y `quality_filter.py` |

### Eduardo

| # | Parte | Entregable | Detalle |
|---|---|---|---|
| 1 | Datos | `src/data/metrics.py` | Contrato 4.6. Funciones puras, no necesita datos reales para empezar |
| 2 | Servicio | `scripts/make_dummy_model.py` | `model.pth` + `metadata.json` de prueba que cumplen el contrato 4.3 (EfficientNet-B0 con pesos de ImageNet, umbrales inventados). Sirve también de `train.py` falso para probar el lanzamiento |
| 3 | Entrenamiento | `src/training/launch_sagemaker.py` | PyTorch con Python 3.11, `ml.g4dn.xlarge` Spot, `max_run`/`max_wait`, `checkpoint_s3_uri`, `metric_definitions` (contrato 4.7), variables `KAGGLE_TOKEN_PARAM` y `S3_BUCKET_MODELS`. Probarlo primero con el modelo de prueba (#2) |
| 4 | Servicio | `src/common/gradcam.py` | Grad-CAM sobre la última capa convolucional; funciona con cualquier modelo de `timm` |
| 5 | Servicio | `src/api/` | FastAPI con `POST /predict` (contrato 4.9) y `GET /health`: calidad → modelo → semáforo/baja confianza → Grad-CAM. Carga `models/current/` al arrancar y sirve la página web de Kevin |
| 6 | Servicio | Registro y métricas | Predicción a S3 (contrato 4.4) y métricas al namespace `BlindnessDetection`: predicciones, latencia, errores, rechazos por calidad, conteo por clase |
| 7 | Ambos | Pruebas y CI | `tests/` para `metrics.py` y la API (`TestClient` con el modelo de prueba); GitHub Actions corre `pytest` en cada push |
| 8 | Servicio | **Al final:** EC2 y CloudWatch | Lanzar la EC2 con el perfil `BlindnessDetection-EC2Role`, security group solo con el puerto de la API, alarma de distribución y dashboard |

### Juntos

| Entregable | Detalle |
|---|---|
| Acordar los contratos | Sección 4, antes de empezar |
| Entrenamientos | Corrida corta (`--subset 64 --epochs 1`), corrida completa, revisar métricas y umbrales, promover con `promote_model.py`. Kevin ajusta el modelo, Eduardo lanza y revisa en CloudWatch |
| Prueba de punta a punta | Con el modelo real en `models/current/`, primero en Studio y luego en EC2 |

En Studio, la API se puede probar con `TestClient` (sin levantar servidor) o con `uvicorn` en
el puerto 8000 y abrirla en `…/jupyterlab/default/proxy/8000/`.

## 6. Puntos de integración

| Momento | Qué se junta | Qué se prueba |
|---|---|---|
| Inicio | Contratos 4.1–4.9 | Que los dos estén de acuerdo |
| Kevin sube `preprocess.py` y `quality_filter.py` | Eduardo los usa en la API | La API funciona con el modelo de prueba |
| Eduardo sube `metrics.py` | Kevin lo usa en `train.py` | `train.py --subset 64 --epochs 1` escribe `metadata.json` con métricas y umbrales |
| `train.py` y `launch_sagemaker.py` listos | Primer entrenamiento en SageMaker | Corrida corta en Spot, artefactos en `models/<run_id>/` |
| Eduardo sube la API | Kevin conecta la página web | Subir una foto desde la página y ver el semáforo |
| Primer modelo real promovido | La API lee `models/current/` | Predicciones con fotos de validación; revisar que el semáforo tenga sentido |
| Antes de la entrega | Todo en EC2 | Prueba de punta a punta, alarma de CloudWatch y apagar lo que cobre |

## 7. Reglas para no pisarnos

- Cada quien trabaja en ramas propias (`kevin/...`, `eduardo/...`) y abre un pull request a `main`.
- Cada archivo tiene un dueño (sección 5). Si necesitas cambiar un archivo del otro, avísale
  o abre el pull request y que él lo revise.
- `src/api/static/` (página web) es de Kevin; el resto de `src/api/` es de Eduardo.
- Cambiar un contrato (sección 4) se avisa antes de hacer push.
- Nada de llaves, tokens ni `.env` en el repo ni en celdas de notebook.
- Detener el espacio de Studio al terminar, y apagar o borrar lo que cobre (EC2, entrenamientos)
  en cuanto no se use.
