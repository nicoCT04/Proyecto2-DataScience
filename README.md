# Proyecto 2 · Análisis Exploratorio — Detección de pases en video (Bundesliga)

CC3084 — Ciencia de Datos · Semestre II 2026 · Universidad del Valle de Guatemala

Reto 21 del proyecto: **detección de pases en videos de jugadas de fútbol de la
Bundesliga** mediante visión por computadora. Este entregable es el **análisis
exploratorio** de los datos, previo al modelo de detección.

## Integrantes

- Carlos Nicolás Concuá Trujillo — 23197
- Jorge Luis Felipe Aguilar Portillo — 23195
- Fernando José Rueda Rodas — 23748
- Fernando Hernández

## Datos

Dataset re-alojado en Hugging Face: `dbal0503/Bundesliga`
(https://huggingface.co/datasets/dbal0503/Bundesliga), derivado de la
competencia *DFL - Bundesliga Data Shootout* de Kaggle.

- **`data/clips/`** — 200 clips `.mp4` (20 partidos × 10 clips), 1920×1080,
  25 fps, 30 s cada uno (~3.8 GB). **Fuente principal del análisis visual.**
  No se versionan en git; se regeneran con `src/download_clips.sh`.
- **`data/train.csv`** — anotaciones de eventos (11,218 filas) de 12 partidos
  completos. Se usa **solo como contexto**: da la taxonomía de pases
  (openplay, cross, freekick, corner) y la densidad temporal de los eventos.
  Sus `video_id` **no** corresponden a los 200 clips (son partidos distintos),
  por lo que no funciona como etiquetas de los clips.
- **`data/clips.csv`** — listado de los 200 clips (manifiesto original).
- **`data/clips_inventory.csv`** — inventario de metadatos generado localmente.

## Estructura

```
data/            train.csv, clips.csv, clips_inventory.csv (clips/ ignorado)
notebooks/       00_marco_del_problema.ipynb, 01_descripcion_datos.ipynb
src/             download_clips.sh, build_inventory.py, eda_figures.py
outputs/         figuras del análisis exploratorio
```

## Reproducibilidad

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

bash src/download_clips.sh      # descarga los 200 clips a data/clips/
python src/build_inventory.py   # genera data/clips_inventory.csv
python src/eda_figures.py       # genera las figuras en outputs/
```

## Reparto del trabajo (grupo de 4)

| Integrante | Parte |
|---|---|
| Nicolás | Marco del problema + estructura del repo + descripción de los datos |
| Compañero 2 | Extracción de frames + propiedades visuales (brillo, color, cancha) |
| Compañero 3 | Detección de jugadores + colores de camiseta / separación por equipo |
| Compañero 4 | Movimiento entre frames + densidad temporal de pases + conclusiones |
