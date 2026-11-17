# Arquitectura

![Diagrama de arquitectura](arquitectura.png)

## Flujo

1. **Kaggle → Amazon SageMaker AI:** el trabajo de entrenamiento descarga el dataset APTOS 2019 directo de Kaggle y entrena el modelo en una instancia con GPU.
2. **Amazon SageMaker AI → Amazon S3:** el modelo entrenado y su `metadata.json` (umbrales de decisión y de confianza, métricas de validación) se guardan en `models/<run_id>/`, el historial completo de entrenamientos.
3. **Promoción a producción:** después de revisar las métricas de un `run_id`, se promueve copiando sus archivos a `models/current/` — ver convención abajo. Es un paso manual/deliberado, no automático.
4. **Amazon S3 → Amazon EC2:** la página web y la API de inferencia (FastAPI) cargan el modelo **siempre desde `models/current/`**, nunca de un `run_id` directamente.
5. **Médico de primer contacto ↔ Amazon EC2:** el médico sube una foto de fondo de ojo. La API aplica el filtro de calidad, clasifica la imagen y regresa el semáforo con su recomendación y el mapa de calor (Grad-CAM). Si el modelo no está seguro, manda un aviso de baja confianza para revisión humana.
6. **Amazon EC2 → Amazon S3:** cada predicción se registra en el bucket de predicciones, sin datos que identifiquen al paciente.
7. **Amazon EC2 → Amazon CloudWatch:** se envían logs y métricas (predicciones, tiempo de respuesta, errores, fotos rechazadas por calidad y distribución por clase). Una alarma avisa si la distribución de predicciones cambia de forma brusca.

Todos los servicios están en la región `us-east-1`.

## Convención de modelos en S3 ("puntero" al modelo en producción)

El bucket de modelos guarda dos cosas distintas:

- `models/<run_id>/model.pth` + `metadata.json` — **historial**: uno por cada entrenamiento, nunca se sobrescribe.
- `models/current/model.pth` + `metadata.json` + `promoted_from.json` — **el modelo en producción ahora mismo**. Es la única ruta que lee EC2.

Promover un modelo es copiar (no entrenar de nuevo) los archivos de un `run_id` a `current/`, con
`src/training/promote_model.py`. El versionado activado en este bucket protege contra una
promoción accidental: si `current/` se sobrescribe por error, se puede recuperar la versión
anterior del objeto. `promoted_from.json` deja registro de qué `run_id` y cuándo se promovió, para
trazabilidad.

No hacen falta cambios de IAM para esto: los roles ya tienen permiso sobre `models/*` completo
(SageMaker lee/escribe, EC2 solo lee), lo cual ya cubre la subcarpeta `current/`.

## Versión en texto (Mermaid)

```mermaid
flowchart LR
    K[(Kaggle<br/>Dataset APTOS 2019)]
    U([Médico de primer contacto<br/>Sube foto, recibe semáforo y mapa de calor])

    subgraph AWS["AWS Cloud · Región us-east-1"]
        SM[Amazon SageMaker<br/>Entrenamiento con GPU]
        S3m[(Amazon S3<br/>models/run_id · historial<br/>models/current · producción)]
        EC2[Amazon EC2<br/>Página web + API FastAPI<br/>Filtro de calidad · Modelo · Baja confianza<br/>Semáforo + Grad-CAM]
        CW[Amazon CloudWatch<br/>Logs, métricas y alarma]
        S3r[(Amazon S3<br/>Registro de predicciones<br/>sin datos del paciente)]
    end

    K -->|Descarga el dataset| SM
    SM -->|Guarda el modelo| S3m
    S3m -->|Carga models/current/| EC2
    U <--> EC2
    EC2 -->|Métricas y logs| CW
    EC2 -->|Registro| S3r
```
