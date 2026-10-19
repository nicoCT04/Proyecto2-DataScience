# 1. Introducción y objetivos

## 1.1 Introducción

Hoy cada pase, disputa y saque de banda de un partido lo anota una persona que revisa los
90 minutos de transmisión. Es un trabajo lento y caro que no escala a todos los partidos de
todas las ligas. Este proyecto busca responder una pregunta concreta: **¿se puede reconocer
automáticamente qué acción ocurre en una ventana de 1.6 s de transmisión?**

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

**Objetivo general.** Comparar modelos de clasificación de acciones (pase, disputa, saque
de banda o fondo) en video de la Bundesliga según su capacidad de generalizar a un partido
no visto.

**Objetivos específicos.**

1. Entrenar regresión logística, Random Forest y XGBoost sobre las variables agregadas de
   cada subclip, ajustando sus hiperparámetros con validación cruzada por partido
   (GroupKFold sobre `match_id`) y F1 macro como métrica.
2. Entrenar un modelo temporal (GRU) sobre la secuencia de cuadros de cada ventana,
   representados con embeddings de ResNet18.
3. Evaluar los cuatro modelos en un partido no visto mediante F1 macro, tomando el azar
   con cuatro clases (0.25) como referencia, e interpretar sus errores con el F1 por clase
   y las matrices de confusión.
4. Integrar los modelos en una aplicación que clasifique video nuevo con el mismo
   preprocesamiento que el entrenamiento.
