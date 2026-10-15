# 2. Marco teórico

## 2.1 Reconocimiento de acciones en fútbol

El análisis automático de fútbol comprende tareas distintas: detectar jugadores y
balón, seguir sus trayectorias, reconocer eventos e interpretar el juego. Akan y
Varlı (2023) revisan estas aplicaciones y muestran que el reconocimiento de eventos
forma parte de un problema más amplio de comprensión del video. En este proyecto
nos concentramos en clasificar ventanas cortas en `play`, `challenge`, `throwin` y
`background`. La categoría `play` se utiliza como aproximación al pase según las
anotaciones disponibles; no determina por sí sola si un pase fue exitoso ni quién
lo realizó.

Una imagen aislada puede contener jugadores y cancha sin mostrar la acción que
ocurre. La secuencia aporta cambios de posición y movimiento, pero también incluye
oclusiones, variaciones de escala y cambios de iluminación o punto de vista. Estos
son problemas habituales del reconocimiento de actividades descritos por Vrigkas
et al. (2015). En una transmisión deportiva, además, el movimiento de cámara puede
confundirse con el desplazamiento de los jugadores o del balón.

Clasificar subclips ya delimitados no equivale a localizar automáticamente todos
los pases en un partido continuo. Esta última tarea requiere proponer ventanas o
instantes candidatos y evaluar su localización temporal. Por ello, el alcance de
nuestros clasificadores es reconocer la clase de cada ventana suministrada.

## 2.2 Representación visual y preprocesamiento

Las representaciones de video pueden combinar apariencia y dinámica temporal
(Vrigkas et al., 2015). En el repositorio se siguen dos rutas. La primera resume
cada ventana mediante variables de color, bordes, flujo óptico, movimiento de
cámara y detecciones de jugadores y balón. El flujo óptico aproxima desplazamientos
entre imágenes consecutivas; no identifica por sí mismo una acción. Las detecciones
permiten resumir posiciones y cantidades, pero sus errores se trasladan a las
variables calculadas.

La segunda ruta utiliza una red convolucional para representar cada cuadro.
ResNet introduce conexiones residuales que suman la entrada de un bloque a su
transformación, facilitando el entrenamiento de redes profundas (He et al., 2016).
Aquí se reutiliza ResNet18 preentrenada en ImageNet, sin su capa clasificadora y con
pesos congelados. Cada cuadro queda representado por un vector de 512 componentes
(*embedding*); este vector describe apariencia, no una etiqueta de pase.

El preprocesamiento existente toma 16 cuadros equiespaciados por subclip, recorta
las franjas superior e inferior y prepara imágenes RGB de 224 × 224 con la
normalización de ImageNet. La entrada temporal conserva así una matriz de
16 × 512, en orden cronológico. Reutilizar los embeddings evita entrenar toda la
CNN con los 400 ejemplos de entrenamiento, aunque una representación aprendida en
imágenes generales puede no capturar bien detalles pequeños como el balón.

## 2.3 Modelo temporal GRU

Una red recurrente actualiza un estado oculto a partir del cuadro actual y del
estado anterior. La unidad recurrente con compuertas, GRU, utiliza una compuerta de
actualización para regular cuánto estado conserva y una de reinicio para modular
la información previa al calcular el estado candidato (Cho et al., 2014). Esto
permite modelar dependencias entre cuadros, en lugar de reducir toda la ventana a
un promedio.

Ullah et al. (2018) estudiaron una combinación de características CNN y una LSTM
bidireccional para reconocer acciones en video. Su trabajo respalda la estrategia
de separar representación espacial y modelado temporal, pero no demuestra que una
GRU sea superior ni garantiza resultados equivalentes en fútbol. Elegimos una GRU
unidireccional de una capa como alternativa temporal compacta para este conjunto
pequeño; su utilidad debe comprobarse experimentalmente.

La implementación recibe los embeddings, los estandariza con estadísticas del
entrenamiento y utiliza el último estado oculto para clasificar el subclip. Una
capa *dropout* de 0.3 precede a la salida lineal de cuatro clases. El entrenamiento
minimiza entropía cruzada con Adam y recorte del gradiente. Se comparan 32 y 64
unidades ocultas y 20 y 40 épocas; la selección depende del F1 macro de validación,
no del error sobre el partido de prueba. Los pesos y la normalización se guardan
juntos para reproducir la inferencia.

## 2.4 Modelos tabulares de comparación

La regresión logística multinomial funciona como referencia lineal: transforma
las variables estandarizadas en probabilidades de clase. Random Forest combina
árboles entrenados con aleatoriedad en los datos y las variables, lo que permite
representar relaciones no lineales (Breiman, 2001). El *gradient boosting* construye
un modelo aditivo de forma secuencial (Friedman, 2001); XGBoost desarrolla esta
familia con regularización y optimizaciones de cálculo (Chen y Guestrin, 2016).

Estas alternativas permiten contrastar una referencia lineal, ensambles de árboles
y una representación temporal. Sin embargo, los modelos tabulares reciben variables
agregadas y la GRU recibe embeddings CNN. La comparación mide sistemas completos,
no exclusivamente el efecto del algoritmo: no permite atribuir cualquier diferencia
de desempeño solo a la arquitectura recurrente.

## 2.5 Validación y límites de la comparación

Los subclips de un mismo partido comparten condiciones visuales. Separarlos al azar
podría favorecer resultados optimistas. Por eso usamos cinco pliegues agrupados por
`match_id` dentro de los siete partidos de entrenamiento. En la GRU, la media y la
desviación de cada componente se calculan únicamente con el entrenamiento de cada
pliegue; después de seleccionar parámetros, se reajusta sobre los 400 subclips.

El F1 de cada clase combina precisión y exhaustividad mediante su media armónica;
el F1 macro promedia las cuatro clases con igual peso. Así, la clase menos frecuente
no queda oculta por las demás. La matriz de confusión y el F1 por clase complementan
esta medida al mostrar qué acciones se confunden.

El conjunto denominado `valid` en los archivos corresponde al partido retenido
para la prueba final y no interviene en la selección de la GRU. El mejor F1 de la
búsqueda es una medida de selección, no una estimación independiente. La evaluación
final debe realizarse por separado y sus conclusiones quedan limitadas por disponer
de un único partido de prueba. La revisión se apoya en artículos de revista y se
complementa con los trabajos originales de congreso sobre GRU, ResNet y XGBoost.
