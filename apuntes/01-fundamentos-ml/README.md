# Módulo 1 · Fundamentos de IA y Machine Learning

Andrés Farall (clases 1 a 5) y Débora Chan (clase 6) · 7 al 23 de abril de 2026

| Clase | Fecha | Tema | Apunte |
|---|---|---|---|
| 01 | mar 07-abr | Qué es la IA, IA discriminativa vs. generativa, supervisado vs. no supervisado | [Qué es la IA y tipos de aprendizaje](clase-01-que-es-ia-y-tipos-de-aprendizaje.md) |
| 02 | jue 09-abr | Razonar con datos, árboles de decisión, clasificador de Bayes y KNN | [Árboles de decisión y KNN](clase-02-arboles-clasificacion-knn.md) |
| 03 | mar 14-abr | Bootstrap, bagging, Random Forest, AdaBoost y XGBoost | [Ensambles: Random Forest y Boosting](clase-03-ensambles-random-forest-boosting.md) |
| 04 | jue 16-abr | Reducción de dimensión y componentes principales | [Reducción de dimensión y PCA](clase-04-reduccion-dimension-pca.md) |
| 05 | mar 21-abr | LDA, t-SNE, UMAP, autoencoders y k-medias | [Más allá de PCA y k-medias](clase-05-reduccion-no-lineal-y-k-medias.md) |
| 06 | jue 23-abr | Datos faltantes, imputación, distancias e ingeniería de atributos | [Imputación e ingeniería de atributos](clase-06-imputacion-y-feature-engineering.md) |

## El hilo del módulo

1. **Qué problema se resuelve.** La clase 1 separa aprender con etiqueta (supervisado: clasificar o predecir un número) de buscar estructura sin etiqueta (no supervisado), y predecir de explicar.
2. **Modelos supervisados clásicos.** Árboles y KNN (clase 2) muestran el compromiso entre complejidad y sobreajuste. Los ensambles (clase 3) lo atacan combinando muchos modelos, a costa de la explicabilidad.
3. **No supervisado.** PCA (clase 4) resume muchas variables en pocas componentes. La clase 5 suma métodos no lineales (t-SNE, UMAP, autoencoders) y segmentación con k-medias.
4. **Los datos antes del modelo.** La clase 6 cubre lo que pasa antes de entrenar: faltantes, imputación, distancias, codificación, escalado y selección de atributos.

## Notas sobre las fuentes

- Los nombres de carpeta del curso no siempre coinciden con el número de clase. El material de la clase 6 está en la carpeta `Clase 5` (láminas de extracción y selección de atributos y `clase_5_modulo1.ipynb`).
- En las clases 4 y 5 la transcripción invierte la cantidad de vinos tintos y blancos. Los apuntes usan las cifras de la tabla de k-medias (unos 4.900 blancos y 1.600 tintos).
- Las imágenes de `img/` vienen de la carpeta `_assets` del curso.
