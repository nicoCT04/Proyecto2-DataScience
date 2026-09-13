# Proyecto 2 · Análisis Exploratorio — Detección de pases en video (Bundesliga)

CC3084 — Ciencia de Datos · Semestre II 2026 · Universidad del Valle de Guatemala

Reto 21 del proyecto: **detección de pases en videos de jugadas de fútbol de la
Bundesliga** mediante visión por computadora. Este entregable es el **análisis
exploratorio** de los datos, previo al modelo de detección.

## Integrantes

- Carlos Nicolás Concuá Trujillo — 23197
- Jorge Luis Felipe Aguilar Portillo — 23195
- Fernando José Rueda Rodas — 23748
- Fernando Hernández — 23645

## Datos

Dataset re-alojado en Hugging Face: `dbal0503/Bundesliga`
(https://huggingface.co/datasets/dbal0503/Bundesliga), derivado de la
competencia *DFL - Bundesliga Data Shootout* de Kaggle.

- **`data/clips/`** — 200 clips `.mp4` (20 partidos × 10 clips), 1920×1080,
  25 fps, 30 s cada uno (~3.8 GB). **Fuente principal del análisis visual.**
  No se versionan en git; se regeneran con `src/download_clips.sh`.
- **`data/train.csv`** — anotaciones de eventos (11,218 filas) de 12 videos
  (tiempos de juego de ~47 min) de 8 partidos. Se usa **solo como contexto**: da la taxonomía de pases
  (openplay, cross, freekick, corner) y la densidad temporal de los eventos.
  Sus `video_id` **no** corresponden a los 200 clips (son partidos distintos),
  por lo que no funciona como etiquetas de los clips.
- **`data/clips.csv`** — listado de los 200 clips (manifiesto original).
- **`data/clips_inventory.csv`** — inventario de metadatos generado localmente.
- **`data/frame_features.csv`**, **`data/player_detections.csv`**,
  **`data/jersey_colors.csv`** — variables derivadas de los frames.

La descripción completa de cada variable (tipo, rango y transformaciones) está en
[`codebook.md`](codebook.md).

## Estructura

```
data/            train.csv, clips.csv, *_inventory/features/detections/jersey .csv (clips/ ignorado)
notebooks/       00_marco_del_problema · 01_descripcion_datos · 02_eda_visual_frames
                 03_deteccion_jugadores_equipos · 04_movimiento_temporal_conclusiones
src/             download_clips.sh · build_inventory.py · eda_figures.py
                 extract_features.py · detect_players.py
outputs/         19 figuras del análisis exploratorio
docs/            enunciado del proyecto
codebook.md      descripción de las variables
Informe_Proyecto2.pdf / .docx     informe de análisis exploratorio
```

## Reproducibilidad

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt

bash src/download_clips.sh       # 1. descarga los 200 clips a data/clips/
python src/build_inventory.py    # 2. inventario de clips -> data/clips_inventory.csv
python src/extract_features.py   # 3. features visuales -> data/frame_features.csv
python src/detect_players.py     # 4. jugadores YOLO + equipos -> data/player_detections.csv, data/jersey_colors.csv
python src/eda_figures.py        # 5. figuras de contexto -> outputs/
jupyter notebook                 # 6. ejecutar notebooks 00-04 en orden (leen los CSV y generan el resto de figuras)
```

Los pasos 1, 3 y 4 necesitan los videos; los CSV resultantes ya están versionados,
así que los notebooks se pueden ejecutar directamente sin descargar los clips.

## Reparto del trabajo (grupo de 4)

| Integrante | Parte |
|---|---|
| Nicolás Concuá | Marco del problema + estructura del repo + descripción de los datos |
| Felipe Aguilar | Extracción de frames + propiedades visuales (brillo, color, cancha) |
| Fernando Rueda | Detección de jugadores + colores de camiseta / separación por equipo + informe |
| Fernando Hernández | Movimiento entre frames + densidad temporal de pases + conclusiones |
