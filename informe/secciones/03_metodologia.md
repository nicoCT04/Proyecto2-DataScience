# 3. Metodología

## 3.1 Datos y construcción de los subclips

A partir de las anotaciones de `train.csv` se recortaron **762 subclips** centrados en
los eventos del juego, etiquetados en cuatro clases: `background` (200), `challenge`
(200), `play` (200) y `throwin` (162). Cada subclip conserva su partido de origen
(`match_id`), lo que permite separar entrenamiento y prueba por partido.

Los datos se dividen en dos conjuntos **por partido**, nunca mezclando frames de un mismo
encuentro entre entrenamiento y prueba:

- **Entrenamiento:** 400 subclips de 7 partidos.
- **Prueba (partido no visto):** 362 subclips del partido `3c993bd2`.

## 3.2 Variables

De cada subclip se calcularon **50 variables agregadas** que resumen la ventana completa,
agrupadas en:

- **Color e iluminación:** brillo, contraste, saturación y cobertura de cancha
  (`green_ratio`), con su media, desviación y extremos.
- **Detalle y cortes:** densidad de bordes y detección de cortes de cámara (`shot_cut`).
- **Movimiento:** movimiento global entre frames, movimiento de cámara, flujo óptico de
  los jugadores, coherencia del flujo y posición del pico de movimiento.
- **Jugadores:** número medio de jugadores detectados (YOLOv8) y su dispersión espacial.
- **Balón:** visibilidad, velocidad, altura y distancia al jugador más cercano.

No se encontraron valores faltantes en la matriz de variables.

## 3.3 Modelos

Se entrenaron dos familias de modelos complementarias:

- **Modelos tabulares** sobre las 50 variables agregadas: **regresión logística**
  (con estandarización), **Random Forest** y **XGBoost**. Resuelven el problema con la
  información resumida de toda la ventana.
- **Modelo temporal (GRU)** sobre la secuencia de embeddings de ResNet18 de los frames,
  que aprovecha la dinámica cuadro a cuadro (ver sección de marco teórico).

## 3.4 Ajuste de hiperparámetros y validación

El ajuste de los modelos tabulares se hizo con **búsqueda en malla** y **validación
cruzada por partido** (`GroupKFold` sobre `match_id`, 5 pliegues), de modo que en cada
pliegue se validan partidos distintos a los de entrenamiento. La métrica de selección es
el **F1 macro**, que pondera por igual las cuatro clases pese al leve desbalance de
`throwin`. Con los mejores hiperparámetros, cada modelo se reentrena sobre todo el
conjunto de entrenamiento y se guarda para su evaluación e integración en la aplicación.

## 3.5 Protocolo de evaluación

El desempeño final se mide sobre el **partido de prueba retenido**, reportando F1 macro,
F1 por clase, matrices de confusión y curvas de precisión-recuerdo. Evaluar sobre un
partido completo no visto es deliberadamente exigente: refleja el uso real del sistema
—analizar un encuentro nuevo— y evita el optimismo de mezclar frames de un mismo partido
entre entrenamiento y prueba.
