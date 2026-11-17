# Detección temprana de Retinopatía Diabética para evitar ceguera

Proyecto de la materia *Integración de Servicios de Aprendizaje Automático* (ITESO).
Equipo: Kevin Ayala y Eduardo Navarro.

## 1. Problema y justificación del uso de ML

### Problema

La retinopatía diabética es una complicación de la diabetes que daña los vasos sanguíneos de la
retina. Es la principal causa de ceguera prevenible en adultos en edad productiva, y como al
principio no da síntomas, muchos pacientes llegan con el especialista cuando el daño ya está avanzado.

En México el problema es grande. Según la Ensanut 2022, el 12.6% de los adultos tiene diabetes
diagnosticada y, sumando los casos sin diagnosticar, la cifra llega a 18.3% [1]. En estudios
mexicanos, entre el 38.9% y el 60.1% de las personas con diabetes tiene algún grado de
retinopatía [2]. Además, hay pocos especialistas: unos 42.5 oftalmólogos por millón de
habitantes [3], frente a más de 100 en países como España o Argentina [4]. Eso hace que la
detección temprana muchas veces no llegue a tiempo.

### ¿Por qué ML?

El diagnóstico se hace revisando fotos del fondo de ojo, y hacerlo a mano no escala cuando hay
millones de pacientes y pocos oftalmólogos. Un modelo de clasificación de imágenes puede analizar
cada foto en segundos y estimar el nivel de severidad, para priorizar a quién referir primero.
No reemplaza al médico: funciona como un filtro de tamizaje.

Este enfoque ya se probó en Jalisco, donde el Tec de Monterrey desarrolló un sistema de IA que
detecta la retinopatía sin un oftalmólogo presente [4]. Nosotros tomamos esa idea como caso de
estudio para construir el ciclo completo: entrenar, desplegar y monitorear el modelo en la nube.

### Nuestra propuesta

Lo planteamos como una herramienta de tamizaje para consultorios de primer contacto, donde no hay
oftalmólogo. El médico sube la foto del fondo de ojo desde una página web y recibe una respuesta
pensada para decidir qué hacer con el paciente, no solo un número. Además de clasificar, la
herramienta hace cuatro cosas:

- **Semáforo con recomendación:** verde (sin retinopatía o leve, revisión anual), amarillo
  (moderada, referir a oftalmología) y rojo (severa o proliferativa, referir pronto). Esta regla
  la vamos a revisar con un médico.
- **Filtro de calidad:** si la foto está borrosa u oscura, pide repetir la toma en lugar de dar un
  resultado poco confiable.
- **Aviso de baja confianza:** cuando el modelo no está seguro, no asigna color y manda el caso a
  revisión humana.
- **Mapa de calor:** con Grad-CAM [6] se marcan las zonas de la retina que más pesaron en la
  predicción, para que el médico vea en qué se fijó el modelo.

La herramienta no da un diagnóstico ni sustituye al oftalmólogo: sirve para ordenar a quién hay
que referir primero.

## 2. Dataset

[APTOS 2019 Blindness Detection | Kaggle](https://www.kaggle.com/competitions/aptos2019-blindness-detection)

Usamos el dataset APTOS 2019 Blindness Detection de Kaggle [5], creado por el Aravind Eye Hospital
de India con fotos de retina tomadas en zonas rurales.

Tiene 3,662 imágenes etiquetadas por médicos en 5 niveles:

| Clase | Nivel | Imágenes |
|---|---|---|
| 0 | Sin retinopatía | 1,805 |
| 1 | Leve | 370 |
| 2 | Moderada | 999 |
| 3 | Severa | 193 |
| 4 | Proliferativa | 295 |

Las dividiremos en entrenamiento, validación y prueba de forma estratificada, para que cada parte
conserve la proporción de clases, y antes revisaremos que no haya imágenes duplicadas.

El dataset está desbalanceado, porque casi la mitad son ojos sanos, y las fotos vienen de distintas
cámaras y calidades. Por eso habrá que preprocesarlas y usar métricas más allá de accuracy.

## 3. Métricas

Como es una herramienta de tamizaje, lo que más importa es no dejar pasar casos. Vamos a medir:

- **Sensibilidad y especificidad** para los casos referibles (moderada o más).
- **Kappa cuadrático ponderado**, que es la métrica oficial de la competencia APTOS.
- **Matriz de confusión** por clase.

El umbral de decisión se elige dando prioridad a la sensibilidad, porque un falso negativo cuesta
más que un falso positivo.

## 4. Servicios de AWS

- **Amazon S3:** almacenamiento del proyecto, en dos buckets. Uno de modelos: guarda cada
  entrenamiento en `models/<run_id>/` y el modelo que está en producción en `models/current/`
  (la API siempre carga de ahí). Otro de predicciones: el registro de cada predicción que hace la
  API, sin datos que identifiquen al paciente. El dataset no se guarda en S3: se descarga directo
  de Kaggle cuando se entrena.
- **Amazon SageMaker AI:** lo usamos para entrenar los modelos. Lanzamos trabajos de entrenamiento
  en instancias con GPU que descargan el dataset directo de Kaggle y, al terminar, guardan el
  modelo entrenado en S3. La instancia se apaga sola al finalizar, así que solo se paga el tiempo
  de entrenamiento.
- **Amazon EC2:** es el servidor donde corren la página web y nuestra API con FastAPI. Descarga el
  modelo desde S3, recibe una foto de fondo de ojo, revisa su calidad y regresa el semáforo, el
  nivel de severidad estimado (0 a 4) con su probabilidad y el mapa de calor.
- **Amazon CloudWatch:** recibe los logs y métricas del entrenamiento en SageMaker y monitorea la
  API: número de predicciones, tiempo de respuesta, errores, fotos rechazadas por calidad y
  distribución de predicciones por clase. Una alarma avisa si esa distribución cambia de forma
  brusca, señal de que las fotos que llegan ya no se parecen a las del entrenamiento.

## 5. Arquitectura

![Diagrama de arquitectura](docs/arquitectura.png)

1. **SageMaker AI** descarga el dataset APTOS 2019 directo de Kaggle y entrena el modelo con GPU.
2. El modelo entrenado se guarda en **Amazon S3**, en `models/<run_id>/`. Tras revisar sus
   métricas, se promueve a `models/current/` (el que usa producción).
3. **EC2** carga el modelo desde `models/current/` en S3 y sirve la página web y la API de
   inferencia con FastAPI.
4. El médico de primer contacto sube una foto de fondo de ojo; la API revisa su calidad, clasifica
   y regresa el **semáforo** y el **mapa de calor** (Grad-CAM), o un aviso de baja confianza.
5. Cada predicción se registra en **S3** sin datos del paciente.
6. **CloudWatch** recibe logs y métricas, con una alarma ante cambios en la distribución de predicciones.

Todos los servicios están en la región `us-east-1`. Más detalle en
[docs/arquitectura.md](docs/arquitectura.md).

## 6. Estructura del repositorio

```
├── README.md
├── requirements.txt
├── .gitignore
├── data/                 # vacío: el dataset se descarga de Kaggle al entrenar
├── notebooks/            # análisis y visualización de datos (EDA)
├── docs/                 # diagrama de arquitectura
├── src/
│   ├── data/             # scripts para descargar y preparar el dataset
│   ├── training/         # train.py (SageMaker AI) y promote_model.py (publica a producción)
│   └── api/              # página web y API con FastAPI (corre en EC2)
├── tests/                # pruebas con pytest
└── .github/workflows/    # CI con GitHub Actions
```

## Referencias

1. A. Basto-Abreu et al., "Prevalencia de prediabetes y diabetes en México: Ensanut 2022,"
   *Salud Pública de México*, vol. 65, supl. 1, pp. S163–S168, 2023, doi:
   [10.21149/14832](https://doi.org/10.21149/14832).
2. E. O. Graue-Hernandez et al., "Prevalence and associated risk factors of diabetic retinopathy
   and macular oedema in patients recently diagnosed with type 2 diabetes," *BMJ Open
   Ophthalmology*, vol. 5, no. 1, art. e000304, 2020, doi:
   [10.1136/bmjophth-2019-000304](https://doi.org/10.1136/bmjophth-2019-000304).
3. International Agency for the Prevention of Blindness, "Vision Atlas: Mexico." Accedido: 16 nov.
   2026. Disponible: <https://iapb.org/learn/vision-atlas/magnitude-and-projections/countries/mexico>
4. K. Pérez, "Tec es referencia mundial al detectar retinopatía diabética con IA," *CONECTA*,
   Tecnológico de Monterrey, 12 abr. 2021. Accedido: 16 nov. 2026. Disponible:
   <https://conecta.tec.mx/en/news/guadalajara/research/tec-world-leader-using-ai-detect-diabetic-retinopathy>
5. Asia Pacific Tele-Ophthalmology Society, "APTOS 2019 Blindness Detection," Kaggle, 2019.
   Accedido: 16 nov. 2026. Disponible: <https://www.kaggle.com/competitions/aptos2019-blindness-detection>
6. R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh y D. Batra, "Grad-CAM: Visual
   explanations from deep networks via gradient-based localization," en *Proc. IEEE Int. Conf.
   Computer Vision (ICCV)*, 2017, pp. 618–626.
