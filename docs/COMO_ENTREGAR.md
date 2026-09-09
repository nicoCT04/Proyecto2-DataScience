# Cómo entregar el Proyecto 2

Según el enunciado hay que entregar **tres cosas** (fecha: 11 de septiembre):

## 1. Informe en PDF
El contenido ya está redactado en [`docs/INFORME.md`](INFORME.md), mapeado a la
rúbrica, con los números reales y las figuras de `outputs/`.

Para producir el PDF, cualquiera de estas vías:
- **Desde los cuadernos** (recomendado, muestran el análisis con código y gráficos):
  ```bash
  # exporta cada cuaderno a HTML y luego "Imprimir → Guardar como PDF" en el navegador
  ./.venv/bin/jupyter nbconvert --to html notebooks/*.ipynb --output-dir docs/html
  ```
- **Desde el informe:** abrir `docs/INFORME.md` en el editor (o pandoc/Typora) y
  exportar a PDF, insertando las figuras de `outputs/` donde se citan.

Estructura del informe = las 6 secciones de la rúbrica (situación, problema,
objetivos, datos, análisis exploratorio, hallazgos y conclusiones).

## 2. Link del repositorio
`https://github.com/nicoCT04/Proyecto2-DataScience` (ya versionado). Los videos no
se suben; se regeneran con `bash src/download_clips.sh`.

## 3. Presentación de PowerPoint
Guion sugerido (10–12 diapositivas), reutilizando las figuras de `outputs/`:
1. Portada + integrantes
2. Situación problemática y problema científico
3. Objetivos
4. Datos: 200 clips (Fig. 05 frame) + inventario (Fig. 01)
5. Contexto de pases en `train.csv` (Fig. 02, 03)
6. EDA visual: histogramas y boxplots (Fig. 08, 09)
7. Correlaciones + cobertura de cancha (Fig. 10, 11)
8. Detección de jugadores YOLO (Fig. 06, 12)
9. Separación de equipos por color (Fig. 07)
10. Movimiento y densidad temporal (Fig. 15, 17)
11. Hallazgos
12. Conclusiones y siguientes pasos

## Checklist de rúbrica (100 pts)
- [x] Situación problemática (10)
- [x] Problema científico (10)
- [x] Objetivos: 1 general + 3 específicos medibles (10)
- [x] Descripción de los datos: variables, observaciones, limpieza (20)
- [x] Análisis exploratorio: cuantitativas, categóricas, gráficos, correlaciones,
      outliers, faltantes (30)
- [x] Hallazgos y conclusiones (20)
