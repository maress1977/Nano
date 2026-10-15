# Módulo 3 · Visión Artificial

Martín Masci · 26 de mayo al 18 de junio de 2026

| Clase | Fecha | Tema | Apunte |
|---|---|---|---|
| 15 | mar 26-may | Qué es la visión por computadora, imágenes como matrices y transformaciones geométricas | [Introducción a la visión por computadora](clase-15-introduccion-vision-por-computadora.md) |
| 16 | jue 28-may | Clasificación con redes en PyTorch y puntos clave con SIFT | [Clasificación con PyTorch y SIFT](clase-16-clasificacion-pytorch-y-sift.md) |
| 17 | mar 02-jun | Clasificación múltiple y detección de objetos con Detectron2 | [Detección con Detectron2](clase-17-clasificacion-multiple-detectron2.md) |
| 18 | jue 04-jun | Imágenes satelitales y límites de la detección | [Imágenes satelitales](clase-18-imagenes-satelitales-y-limites-de-la-deteccion.md) |
| 19 | mar 09-jun | Reconocimiento facial: embeddings, similitud coseno y MTCNN | [Reconocimiento facial](clase-19-reconocimiento-facial.md) |
| 20 | jue 11-jun | Explicabilidad (LIME), sesgo demográfico y definiciones de fairness | [Explicabilidad, sesgo y fairness](clase-20-explicabilidad-sesgo-fairness.md) |
| 21 | mar 16-jun | OCR: reconocimiento de texto en imágenes | [OCR](clase-21-ocr-reconocimiento-de-texto.md) |
| 22 | jue 18-jun | Segmentación con U-Net, GAN y Vision Transformers | [U-Net, GAN y ViT](clase-22-unet-gan-vision-transformers.md) |

## El hilo del módulo

1. **Las bases.** La clase 15 traduce la imagen a una matriz de números y presenta las transformaciones geométricas y las redes convolucionales. La clase 16 arma el pipeline de visión, clasifica con redes en PyTorch y usa SIFT para encontrar puntos clave.
2. **Detectar objetos.** Las clases 17 y 18 pasan de una etiqueta por imagen a cajas por objeto con Detectron2, y prueban sus límites en imágenes satelitales.
3. **Comparar con embeddings.** La clase 19 reconoce rostros comparando vectores con similitud coseno y un umbral, la misma lógica de búsqueda que usa el retrieval semántico del TFI.
4. **Usarlo con responsabilidad.** La clase 20 explica por qué un modelo decide lo que decide y cómo medir y mitigar el sesgo entre grupos.
5. **Más tareas y arquitecturas.** La clase 21 lee texto en imágenes (OCR) y la clase 22 cierra con U-Net, GAN y Vision Transformers.

## Notas sobre las fuentes

- **Clase 18:** solo quedó grabado el tramo final y la carpeta no trae láminas.
- **Clase 20:** no hay transcripción. El apunte se armó con las láminas de Domingo Mery sobre explicabilidad y sesgo, y no se sabe con certeza quién dio la clase.
- **Clase 21:** el trabajo en equipos posterior al corte no quedó grabado.
- **Láminas:** las de las clases 15 a 21 se numeran como "Clase 1" a "Clase 7" del módulo.
- **Imágenes:** las de `img/` vienen de la carpeta `_assets` del curso.
