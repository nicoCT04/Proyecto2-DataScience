# Proyecto 2 — Análisis Exploratorio
## Detección de pases en video de la Bundesliga (Reto 21)

**CC3084 – Ciencia de Datos · Semestre II 2026 · Universidad del Valle de Guatemala**

Integrantes: Nicolás Concuá (23197) · Felipe Aguilar (23195) · Fernando Rueda (23748) · Fernando Hernández

> Documento base del informe. Cada figura citada está en `outputs/`. El detalle
> ejecutable está en los cuadernos `notebooks/00`–`04`.

---

## 1. Situación problemática (10 pts)

El análisis del rendimiento en el fútbol profesional depende cada vez más de datos
extraídos del video. Anotar manualmente cada pase de un partido es lento, costoso y
propenso a errores: un partido tiene cientos de acciones con balón. Automatizar la
detección de pases a partir del video de transmisión permitiría generar estadísticas
a escala, pero el video presenta dificultades: cámara en movimiento, jugadores que se
ocultan entre sí, un balón pequeño y rápido, y gráficos superpuestos (marcas de agua,
marcador).

## 2. Problema científico (10 pts)

¿Es posible **detectar automáticamente los pases** en clips de video de partidos de la
Bundesliga a partir de sus características visuales (frames, jugadores, balón y
movimiento), y qué propiedades de los datos condicionan esa detección? Como paso
previo, este trabajo responde: **¿qué contienen los datos y qué señales visuales están
disponibles** para una futura detección de pases?

## 3. Objetivos (10 pts)

**General.** Realizar un análisis exploratorio de los clips de video de la Bundesliga y
de sus anotaciones, para caracterizar los datos y evaluar la viabilidad de detectar
pases mediante visión por computadora.

**Específicos.**
1. Describir el inventario de los 200 clips (resolución, fps, duración, tamaño, partidos)
   y verificar la consistencia del formato.
2. Caracterizar la taxonomía y la distribución temporal de los eventos en `train.csv`
   (pases openplay/cross/freekick/corner, challenges, saques de banda).
3. Identificar las señales visuales y los retos para la detección a partir de los frames
   (cancha, jugadores, equipos por color, movimiento).

## 4. Descripción de los datos (20 pts)

Fuente: dataset `dbal0503/Bundesliga` (Hugging Face), derivado de la competencia
*DFL – Bundesliga Data Shootout*.

- **200 clips de video** (`data/clips/`): 20 partidos × 10 clips. **Formato 100%
  homogéneo**: 1920×1080, 25 fps, 30 s (≈750 frames/clip), 3.8 GB en total. Sin
  corrupción ni faltantes (Fig. 01, Fig. 05).
- **`train.csv`** (contexto): 11,218 anotaciones de eventos de 12 partidos completos.
  **Sus `video_id` no coinciden con los clips descargables**, por lo que se usa como
  referencia de la estructura del problema, no como etiquetas. Los vacíos de
  `event_attributes` son estructurales (eventos start/end/throwin), no faltantes.
- **Variables visuales derivadas** (`data/frame_features.csv`, 2,000 frames): brillo,
  contraste, saturación, `green_ratio`, `edge_density`, `motion`, conteo de jugadores.

## 5. Análisis exploratorio (30 pts)

### 5.1 Variables cuantitativas visuales (Cuaderno 02)
Sobre 2,000 frames (10 por clip). Estadística descriptiva, histogramas (Fig. 08),
diagramas de caja (Fig. 09) y correlaciones (Fig. 10).
- La **cobertura de cancha** (`green_ratio`) es alta y estable (media 0.60) con una cola
  hacia valores bajos: 100 frames (percentil 5) corresponden a repeticiones y primeros
  planos, con **más densidad de bordes**. Correlación `green_ratio`–`edge_density` = **−0.47**.
- La cobertura es **consistente entre los 20 partidos** (Fig. 11) → estilo de transmisión
  homogéneo, favorable para generalizar un detector.

### 5.2 Detección de jugadores y equipos (Cuaderno 03)
Detector **YOLOv8n** preentrenado sobre 600 frames.
- **11.9 jugadores/frame** en promedio (máx 23), coherente con el plano táctico
  (Fig. 12, ejemplo anotado Fig. 06). Supera con creces al proxy clásico por blobs
  (que sobre-estimaba a ~63 por ruido de gradas).
- **Separación de equipos por color de camiseta**: el matiz de las 7,168 camisetas
  detectadas se agrupa en dos equipos con k-means (Fig. 07, Fig. 13) → base para
  contar pases por equipo.

### 5.3 Movimiento y contexto temporal (Cuaderno 04)
- La variable **`motion`** (diferencia entre frames) resume la dinámica; correlaciona
  con el detalle de escena (Fig. 15, Fig. 16) y marca los momentos de acción.
- Contexto `train.csv`: **3,468 pases**, dominados por juego abierto (openplay 3,417 »
  freekick 132, cross 118, corner 37) → **distribución desbalanceada** (Fig. 03). Los
  pases ocurren cada **4.1 s (mediana)** en fase de juego (Fig. 04, Fig. 17).

## 6. Hallazgos y conclusiones (20 pts)

**Hallazgos.**
1. Datos homogéneos y de calidad (200 clips, 1080p/25fps/30s, sin faltantes).
2. La cancha se aísla fácil con color; `green_ratio` separa juego abierto de repeticiones.
3. Los jugadores son detectables de forma confiable (YOLO), muy por encima del método clásico.
4. Los dos equipos son separables por color de camiseta.
5. El movimiento entre frames es la señal temporal clave para localizar acciones.
6. En el contexto anotado, los pases son frecuentes y desbalanceados por tipo.

**Retos.** Balón pequeño/rápido; marca de agua y marcador superpuestos; desbalance de
tipos de pase; ausencia de etiquetas en los clips descargables.

**Conclusiones y siguientes pasos.** El dataset es **viable** para un sistema de
detección de pases por visión. Pipeline propuesto: muestrear frames → detectar
jugadores y balón (YOLO) → asignar equipo por color → seguir el balón (tracking) →
clasificar la acción con un modelo temporal, usando los picos de `motion` como
candidatos. Para entrenar y evaluar se requieren los videos anotados o etiquetar
manualmente una muestra de los clips.
