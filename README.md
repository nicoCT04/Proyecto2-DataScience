# Proyecto 2 — Clasificación de acciones en video de la Bundesliga

CC3084 — Ciencia de Datos · Semestre II 2026 · Universidad del Valle de Guatemala

Reto 21 del proyecto: **detección de pases en videos de jugadas de fútbol de la
Bundesliga** mediante visión por computadora. El proyecto tiene dos entregas:

1. **Análisis exploratorio** (`Informe_Proyecto2.pdf`): caracterización de 200 clips de
   20 partidos.
2. **Resultados** (`Informe_Proyecto2_Resultados.docx`): modelos que clasifican ventanas de
   video en **pase, disputa, saque de banda o fondo**, evaluados en un partido no visto, y
   una aplicación para explorar los datos y clasificar video nuevo.

## Integrantes

- Carlos Nicolás Concuá Trujillo — 23197
- Jorge Luis Felipe Aguilar Portillo — 23195
- Fernando José Rueda Rodas — 23748
- Fernando Andree Hernández Martínez — 23645

## Resultados principales

Evaluación en el partido de prueba `3c993bd2` (362 subclips, no usado en ningún ajuste):

| Modelo | F1 macro prueba | F1 macro CV (7 partidos) |
|---|---|---|
| Regresión logística | 0.289 | 0.286 |
| Random Forest | 0.304 | 0.324 |
| **XGBoost** | **0.347** | **0.351** |
| GRU sobre embeddings de ResNet18 | 0.114 | 0.316 |

XGBoost es el modelo elegido. Reconoce el pase (F1 0.45) y el saque de banda (F1 0.44)
mejor que el azar (0.25 con cuatro clases), pero confunde la disputa con el fondo. La GRU
no generaliza al partido nuevo. Los detalles están en el informe y en
`outputs/results/`.

## Datos

- **`data/clips/`** — 200 clips `.mp4` del dataset `dbal0503/Bundesliga` (Hugging Face),
  usados en el análisis exploratorio. No se versionan; `src/download_clips.sh`.
- **`data/subclips/`** — 762 subclips etiquetados de `ManuD/dfl_clips` (Hugging Face),
  usados en la fase de resultados: 400 de 7 partidos para entrenar y 362 del partido
  `3c993bd2` para probar. No se versionan (~4.6 GB); `src/download_subclips.sh`.
- **`data/train.csv`** — anotaciones originales de la competencia *DFL – Bundesliga Data
  Shootout*; sirven para validar las etiquetas de los subclips.
- **`data/subclip_features.csv`** — una fila por subclip con 48 variables de los modelos
  (más `duration_s` y `lead_s`, excluidas por filtrar la etiqueta).
- **`data/subclip_embeddings.npz`** — secuencias de 16 × 512 embeddings de ResNet18.
- Los demás CSV de `data/` son del análisis exploratorio; ver [`codebook.md`](codebook.md).

## Estructura

```
src/
  preprocessing.py · features.py        preprocesamiento y variables (compartidos con la app)
  build_subclip_index.py · build_features.py   índice de subclips y matriz de variables
  modeling.py · train_models.py         modelos tabulares (logística, RF, XGBoost)
  train_temporal.py                     GRU sobre embeddings de ResNet18
  evaluate.py                           evaluación en el partido de prueba + figuras 20-27
  inference.py                          inferencia sobre video nuevo (usada por la app)
  make_app_samples.py                   videos de ejemplo de la app
  (download_clips.sh, build_inventory.py, extract_features.py, detect_players.py,
   eda_figures.py: análisis exploratorio)
app/            aplicación Streamlit (app.py, theme.py, samples/)
models/         modelos entrenados (*.joblib, gru.pt); pretrained/ se descarga solo
outputs/        figuras 01-19 (EDA) y 20-27 (resultados); results/ con métricas en CSV/JSON
informe/        secciones en Markdown, capturas de la app y build_informe.py
notebooks/      00-04 del análisis exploratorio
```

## Ejecución

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

**Solo la aplicación** (los modelos y resultados ya están versionados):

```bash
streamlit run app/app.py
```

La primera clasificación descarga YOLOv8n y ResNet18 a `models/pretrained/`.

**Pipeline completo de la fase de resultados:**

```bash
bash src/download_subclips.sh            # 1. subclips etiquetados -> data/subclips/
python src/build_subclip_index.py        # 2. índice y validación de etiquetas
python src/build_features.py             # 3. variables + embeddings (YOLO y ResNet18 usan GPU/MPS si hay)
python src/train_models.py               # 4. modelos tabulares con ajuste por partido
python src/train_temporal.py             # 5. GRU
python src/evaluate.py                   # 6. métricas, tablas y figuras 20-27
python src/make_app_samples.py           # 7. videos de ejemplo de la app
python informe/build_informe.py          # 8. Informe_Proyecto2_Resultados.docx
```

Los pasos 4 a 8 no necesitan los videos: usan los CSV y embeddings versionados.

> En macOS, PyTorch y XGBoost cargan cada uno su propio OpenMP y el proceso puede cerrarse
> con *segfault*. Los scripts fijan `OMP_NUM_THREADS=1` antes de importarlos.

## Reparto del trabajo

| Integrante | Exploratorio | Resultados |
|---|---|---|
| Nicolás Concuá | Marco del problema, estructura del repo, descripción de los datos | Modelos tabulares; introducción, objetivos y metodología |
| Felipe Aguilar | Extracción de frames y propiedades visuales | Modelo temporal GRU; marco teórico y referencias |
| Fernando Rueda | Detección de jugadores y equipos, informe | Dependencias, preprocesamiento, subclips y extracción de variables |
| Fernando Hernández | Movimiento entre frames, densidad de pases, conclusiones | Evaluación, aplicación, resultados y conclusiones del informe |
