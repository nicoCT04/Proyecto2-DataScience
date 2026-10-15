# 8. Conclusiones

## 8.1 Cumplimiento de los objetivos

**Objetivo 1 — modelos tabulares.** Se cumplió. Se entrenaron regresión logística, Random
Forest y XGBoost sobre 48 variables por ventana, con búsqueda en malla y validación
cruzada agrupada por partido. En el proceso se detectó y corrigió una fuga de información:
la duración del subclip delataba la clase. Excluirla bajó el F1 de XGBoost de 0.381 a 0.347,
pero deja cifras que sí corresponden a reconocer la acción y que se pueden reproducir en un
video nuevo.

**Objetivo 2 — modelo temporal y comparación.** Se cumplió. La GRU sobre los *embeddings* de
ResNet18 alcanzó 0.316 de F1 macro en validación cruzada, comparable a los tabulares, pero
cayó a 0.114 en el partido de prueba porque predice saque de banda para casi todo. Con 400
secuencias, una representación de apariencia congelada no separa la acción del aspecto
visual del partido.

**Objetivo 3 — evaluación en un partido no visto.** Se cumplió. Los cuatro modelos se
evaluaron una sola vez en el partido `3c993bd2` con F1 macro, F1 por clase, matrices de
confusión, curvas de precisión-recall, importancia por permutación y tiempos. **XGBoost es
el mejor modelo** (F1 macro 0.347, exactitud 36 %, frente a 0.25 del azar) y el único cuya
validación cruzada anticipó bien su desempeño en prueba.

**Objetivo 4 — aplicación.** Se cumplió. La aplicación en Streamlit permite explorar las
variables tras la limpieza, clasificar un video subido con uno, varios o todos los modelos
(un momento o el video completo) y revisar el rendimiento en gráficas interactivas que se
pueden ocultar. Se verificó que su preprocesamiento reproduce el del entrenamiento.

## 8.2 Efectividad de los algoritmos

Ningún modelo alcanza un desempeño suficiente para detectar de forma automática y confiable
todas las acciones de un partido nuevo: el mejor F1 macro (0.347) está por encima del azar,
pero lejos de un sistema utilizable sin supervisión. Los resultados sí muestran qué funciona
y qué no:

- **Las variables de movimiento diseñadas a partir del análisis exploratorio son las que más
  aportan.** XGBoost usa sobre todo el flujo óptico propio de los jugadores (sin el paneo de
  cámara), la dispersión de jugadores detectados por YOLO y el movimiento de cámara. Con
  ellas reconoce el **pase** (F1 0.45, AP 0.48 frente a 0.28 del azar) y el **saque de
  banda** (F1 0.44). Para el pase, cuando está muy seguro acierta cerca del 90 % de las
  veces, así que es útil para encontrar los pases más claros.
- **La disputa no se distingue del juego sin acción** en una ventana de 1.6 s (F1 de 0.02 a
  0.23 según el modelo). Separarla probablemente requiere seguir a los jugadores y el balón
  cuadro a cuadro.
- **El cambio entre partidos es el principal límite.** Los árboles alcanzan F1 = 1 en
  entrenamiento y unos 0.33 en validación por partido. Con solo siete partidos para
  entrenar, el modelo no ve suficiente variedad de encuadres, iluminación y uniformes.
  Agregar complejidad no ayuda; agregar partidos sí debería.
- **Validar por partido fue clave.** Mezclar ventanas de un mismo encuentro entre
  entrenamiento y prueba habría producido cifras mucho más altas y engañosas. La
  validación cruzada agrupada anticipó bien el resultado de XGBoost en el partido nuevo
  (0.351 frente a 0.347).

## 8.3 Trabajo futuro

1. **Más partidos de entrenamiento** y normalización de color por partido para reducir el
   cambio de dominio, la limitación dominante.
2. **Ajuste fino de la CNN** o un modelo de video preentrenado en acciones, en lugar de
   *embeddings* de ImageNet congelados, con aumento de datos de color y encuadre para que la
   red temporal no dependa de la apariencia del partido.
3. **Seguimiento de jugadores y balón** entre cuadros (por ejemplo, YOLO reentrenado para el
   balón más un *tracker*) para describir la interacción que define una disputa o un pase.
4. **Localización temporal**: usar la línea de tiempo de la aplicación como detector de
   candidatos y evaluarla con tolerancia temporal sobre partidos completos.
