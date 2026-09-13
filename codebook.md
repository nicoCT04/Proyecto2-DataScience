# Codebook — Proyecto 2 · Detección de pases en video (Bundesliga)

Descripción de los datos usados en el análisis exploratorio: fuentes, variables,
tipos, rangos observados y transformaciones aplicadas.

## Fuentes

| Conjunto | Origen | Observaciones | Uso |
|---|---|---|---|
| `data/clips/*.mp4` | Hugging Face `dbal0503/Bundesliga` (DFL – Bundesliga Data Shootout) | 200 clips de 20 partidos (10 por partido) | Fuente principal del análisis visual. No se versiona (~3.8 GB); se descarga con `src/download_clips.sh`. |
| `data/train.csv` | Misma competencia | 11,218 eventos de 12 videos (tiempos de juego) de 8 partidos | Solo contexto: taxonomía y densidad temporal de los eventos. Sus `video_id` **no** coinciden con los clips. |
| `data/clips.csv` | Manifiesto del dataset | 200 filas | Listado original de clips. |

## `data/clips_inventory.csv` — 200 filas (una por clip) · `src/build_inventory.py`

| Variable | Tipo | Descripción | Valores / rango |
|---|---|---|---|
| `file` | categórica (id) | Nombre del archivo | `08fd33_0.mp4` … |
| `match_id` | categórica | Identificador del partido | 20 niveles, 10 clips cada uno |
| `clip_idx` | cuantitativa discreta (id) | Índice del clip dentro del partido | 0–9 |
| `width`, `height` | cuantitativa discreta | Resolución en píxeles | 1920 × 1080 (constante) |
| `fps` | cuantitativa | Cuadros por segundo | 25 (constante) |
| `duration_s` | cuantitativa continua | Duración en segundos | 30.00–30.04 |
| `size_mb` | cuantitativa continua | Tamaño del archivo | 14.87–26.27 MB (media 19.67) |

## `data/train.csv` — 11,218 filas (una por evento)

| Variable | Tipo | Descripción | Valores / rango |
|---|---|---|---|
| `video_id` | categórica | Video anotado: `<partido>_<tiempo>` (0 = primer tiempo, 1 = segundo) | 12 niveles de 8 partidos |
| `time` | cuantitativa continua | Momento del evento (s) | segundos desde el inicio del video |
| `event` | categórica | Tipo de evento | `play` 3,586 (32.0 %), `start` 3,418 (30.5 %), `end` 3,418 (30.5 %), `challenge` 624 (5.6 %), `throwin` 172 (1.5 %) |
| `event_attributes` | texto (lista) | Atributos del evento | `play`: acción (`pass`/`cross`) + contexto (`openplay`/`freekick`/`corner`); `challenge`: resultado; `throwin`: `pass`/`cross`. Vacío en los 6,836 `start`/`end` (faltante estructural, no se imputa). |

Derivadas en el análisis (notebook 01): `accion` y `contexto`, separando la lista de
atributos de los eventos `play`.

## `data/frame_features.csv` — 2,000 filas (10 frames por clip) · `src/extract_features.py`

Frames equiespaciados entre el 5 % y el 95 % de cada clip (uno cada ≈3 s), leídos al
vuelo con OpenCV.

| Variable | Tipo | Descripción | Rango observado |
|---|---|---|---|
| `file`, `match_id`, `clip_idx` | categóricas (id) | Clip y partido del frame | — |
| `frame_pos` | cuantitativa discreta (id) | Posición del frame en la muestra del clip | 0–9 |
| `time_s` | cuantitativa continua | Segundo del clip | 1.48–28.48 |
| `brightness` | cuantitativa continua | Media de gris (0–255) | 64.5–153.1 |
| `contrast` | cuantitativa continua | Desviación estándar de gris | 27.8–69.7 |
| `saturation` | cuantitativa continua | Media del canal S de HSV (0–255) | 59.8–147.5 |
| `green_ratio` | cuantitativa continua | Proporción de píxeles de césped (máscara HSV H 35–90, S≥40, V≥40) | 0.13–0.91 |
| `edge_density` | cuantitativa continua | Proporción de píxeles de borde Canny (100, 200) | 0.024–0.134 |
| `player_count` | cuantitativa discreta | Blobs no verdes de tamaño de jugador dentro de la cancha (proxy clásico) | 15–199 · **descartado**: no correlaciona con YOLO (r = −0.04) |
| `motion` | cuantitativa continua | Diferencia absoluta media de gris respecto al frame muestreado anterior | 4.1–57.6 · vale 0 en `frame_pos` 0 (sin frame previo): se trata como `NaN` |

## `data/player_detections.csv` — 600 filas (3 frames por clip) · `src/detect_players.py`

| Variable | Tipo | Descripción | Rango observado |
|---|---|---|---|
| `file`, `match_id`, `clip_idx`, `frame_pos` | categóricas (id) | Frame analizado | — |
| `player_count` | cuantitativa discreta | Personas detectadas por YOLOv8n (clase COCO *person*) | 0–23 (media 11.95) |

## `data/jersey_colors.csv` — 7,168 filas (una por jugador detectado)

| Variable | Tipo | Descripción | Rango observado |
|---|---|---|---|
| `file`, `match_id` | categóricas (id) | Clip y partido | — |
| `hue` | cuantitativa continua (circular) | Matiz medio del torso (OpenCV 0–180), sin píxeles con S ≤ 40 | 5.9–157.7 |
| `sat` | cuantitativa continua | Saturación media del torso | 46.7–193.7 |
| `val` | cuantitativa continua | Valor (brillo) medio del torso | 16.3–238.3 |

## Limpieza y transformaciones

- Clips: 200/200 legibles, sin faltantes ni duplicados; formato constante.
- `train.csv`: sin imputación (los vacíos son estructurales); `event_attributes` se
  convierte de texto a lista.
- `motion = 0` en el primer frame de cada clip → `NaN` (200 frames).
- Outliers (regla 1.5 × IQR) se conservan: corresponden a tomas cerradas y repeticiones.
- Para comparar distribuciones en un mismo eje, las variables visuales se estandarizan
  (z-score) solo en los diagramas de caja.
