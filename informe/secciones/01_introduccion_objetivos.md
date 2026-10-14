# 1. Introducción y objetivos

## 1.1 Introducción

En la fase de análisis exploratorio se caracterizaron los 200 clips de video de la
Bundesliga y se confirmó que las señales visuales necesarias para reconocer acciones
con balón están presentes: la cancha se aísla por color, los jugadores son detectables
y el movimiento entre frames marca los momentos de acción. Esta fase de **resultados**
da el siguiente paso: construir y evaluar modelos que clasifiquen automáticamente la
acción de cada jugada a partir de esas señales.

El problema se plantea como una **clasificación de ventanas de video (subclips)** en
cuatro categorías: pase (*play*), disputa (*challenge*), saque de banda (*throwin*) y
fondo sin acción (*background*). Para cada subclip se extrajeron variables agregadas de
movimiento, color, bordes, cámara, jugadores y balón, y a partir de ellas se entrenaron
dos familias de modelos: modelos **tabulares** sobre las variables resumidas (regresión
logística, Random Forest y XGBoost) y un modelo **temporal** (GRU) sobre la secuencia de
embeddings de los frames. La evaluación se hace sobre un **partido completo no visto**
durante el entrenamiento, para medir la generalización de forma realista.

## 1.2 Objetivos

**Objetivo general.** Desarrollar y comparar modelos de clasificación que identifiquen
la acción (pase, disputa, saque de banda o fondo) en subclips de video de la Bundesliga,
evaluando su capacidad de generalizar a un partido no visto.

**Objetivos específicos.**

1. Entrenar modelos tabulares (regresión logística, Random Forest y XGBoost) sobre las
   variables agregadas de cada subclip, ajustando sus hiperparámetros mediante validación
   cruzada por partido (GroupKFold sobre `match_id`) con métrica F1 macro.
2. Entrenar un modelo temporal (GRU) sobre la secuencia de embeddings de ResNet18 de los
   frames y compararlo con los modelos tabulares.
3. Evaluar todos los modelos en un partido de prueba retenido, reportando F1 macro,
   F1 por clase y matrices de confusión, e interpretar los errores.
4. Integrar el mejor modelo en una aplicación que permita explorar los datos y predecir
   la acción sobre video nuevo.
