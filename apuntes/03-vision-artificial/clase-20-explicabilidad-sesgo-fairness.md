# Clase 20 · Explicabilidad, sesgo y fairness

> **Módulo 3 · Visión Artificial** · Domingo Mery (según las láminas) · jueves 11-jun-2026
> **Fuentes:** sin transcripción; apunte armado solo con las láminas `PPT Explicabilidad_Domingo-Mery.pdf` y `PPT Bias y Fairness_Domingo-Mery.pptx`

## En pocas palabras

- Una red puede acertar por motivos equivocados: el clasificador de husky contra lobo "miraba" la nieve del fondo, no el animal. La **explicabilidad** (XAI) busca saber qué partes de la entrada pesaron en la decisión, para mejorar el modelo, justificar decisiones con impacto legal o ético y generar confianza.
- Hay métodos de explicación que tratan al modelo como **caja negra**: perturban la entrada y miran cómo cambia la salida. **LIME** apaga superpixeles al azar y ajusta una regresión lineal ponderada; en verificación facial, los métodos de **removal** y **aggregation** sacan o agregan regiones de la cara y miden cómo cambia el score.
- Un sistema de reconocimiento facial no rinde igual para todos los grupos demográficos. Las distribuciones de scores de pares genuinos e impostores se corren según el grupo, así que un **umbral único** produce tasas de error distintas para cada grupo.
- El sesgo no viene solo de datos etiquetados con prejuicios: también surge de muestras que se retroalimentan, features menos informativas para minorías, desbalance en el tamaño de las muestras y **proxies** de atributos sensibles. Los casos COMPAS, el reclutador de Amazon, los word embeddings y PULSE lo muestran fuera de la visión.
- La mitigación puede hacerse antes (pre-processing), durante (in-processing) o después del entrenamiento (post-processing). Para modelos de caja negra conviene el post-processing, por ejemplo **umbrales variables**, que el trabajo de Villalobos, Mery y Bowyer elige a partir de atributos blandos **no sensibles** en lugar de raza o género.

## Conceptos clave

### Por qué explicar un modelo

Según la definición del programa XAI de DARPA citada en las láminas, la IA explicable busca técnicas de machine learning que:

- produzcan modelos más explicables **sin perder** rendimiento predictivo, y
- permitan a las personas comprender, confiar adecuadamente y gestionar estos sistemas.

Las láminas dan tres razones para explicar:

1. **Mejorar y controlar:** si se entiende cómo decide un algoritmo, se lo puede corregir e incluso descubrir conocimiento nuevo.
2. **Justificar:** cuando hay temas legales o éticos, la persona afectada o beneficiada tiene derecho a saber por qué se tomó una decisión.
3. **Generar confianza.**

La pregunta concreta que responden los métodos de esta clase es: **¿qué lugares de la imagen fueron importantes para la clasificación?** La respuesta suele ser un **mapa de saliencia** (*saliency map*) superpuesto a la imagen.

### Explicaciones de caja negra

Un método de caja negra no mira los pesos ni las activaciones internas de la red: solo usa entradas y salidas. Por eso sirve para cualquier modelo, incluso uno comercial cerrado. Las láminas presentan dos para clasificación de imágenes:

- **RISE** (*Randomized Input Sampling for Explanation*, Petsiuk, Das y Saenko, 2018): explica modelos de caja negra a partir de muestreo aleatorio de la entrada. Las láminas solo lo nombran y muestran figuras del paper.
- **LIME** (*Local Interpretable Model-agnostic Explanations*, Ribeiro, Singh y Guestrin, 2016), que se desarrolla paso a paso.

### LIME paso a paso

La idea es aproximar el comportamiento del modelo, **alrededor de una imagen concreta**, con un modelo lineal simple cuyos coeficientes sí se pueden leer. El ejemplo de las láminas es una imagen clasificada como labrador:

| Paso | Qué se hace | En el ejemplo |
|---|---|---|
| 1. Imagen de entrada | La imagen a explicar | Foto de un perro |
| 2. Superpixeles | Se agrupan píxeles de color parecido con el algoritmo de segmentación *quickshift* (Vedaldi y Soatto, 2008) | |
| 3. Etiquetado | Se numera cada superpixel | $m = 68$ superpixeles |
| 4. Perturbaciones | Se prende o apaga cada superpixel al azar con probabilidad 0,5; cada perturbación es un vector binario de $m$ elementos | $n = 150$ perturbaciones |
| 5. Distancias | Se mide la similitud de cada perturbación con la imagen original, representada por el vector $z = [1,1,\dots,1]$ | |
| 6. Pesos | Una función kernel transforma cada distancia en un peso entre 0 y 1: las perturbaciones más parecidas al original pesan más | |
| 7. Predicción | Se pasa cada imagen perturbada por el modelo $f$ y se anota la probabilidad de la clase | Probabilidad de "labrador" |
| 8. Regresión lineal ponderada | Se ajusta una regresión con la matriz de perturbaciones ($n \times m$) como entrada, las predicciones ($n \times 1$) como salida y los pesos del paso 6 | Da $m \times 1$ coeficientes |
| 9. Top features | Se eligen los superpixeles con coeficientes más grandes | Los 4 mayores: superpixeles 9, 14, 25 y 30 |
| 10. Explicación | Se muestra la imagen solo con esos superpixeles | |

La clave está en el paso 8: **la magnitud de cada coeficiente indica qué tan fuerte influye ese superpixel en la predicción**. El modelo lineal no reemplaza a la red; solo es una explicación local válida cerca de esa imagen.

Algunos puntos a tener en cuenta:

- El resultado depende de cómo se segmenta la imagen (cantidad y forma de los superpixeles), de cuántas perturbaciones se generan y de cuántas features se muestran al final.
- "Apagar" un superpixel crea imágenes que el modelo quizás nunca vio en el entrenamiento.

### Explicabilidad en verificación facial

En **verificación facial** un *facial matcher* recibe una imagen enrolada (*enrolled*) y una imagen de consulta (*probe*), y devuelve un score. Si el score supera un umbral $t$, se declara "misma persona"; si no, "personas distintas". El matcher es una caja negra.

Las láminas muestran cómo el score cae cuando la *probe* se modifica: con la misma imagen da 1,0000, y con distintas versiones alteradas baja a valores como 0,5699, 0,5643 o 0,4226. Esa sensibilidad es la base de la explicación: se procesa la *probe*, se observa el score y se arma un mapa de saliencia.

Dos familias de algoritmos (trabajo presentado en WACV 2022, en el repositorio del curso del docente):

- **Removal:** se quita una región de la cara y se mide el score. Variante: quitar **una región tras otra**, acumulando.
- **Aggregation:** se parte de poca información y se **agregan** regiones, una sola o una tras otra, midiendo cómo sube el score.

Con eso salen cuatro mapas de saliencia (quitar una región, quitar una tras otra, agregar una región, agregar una tras otra), que se muestran también como curvas de nivel y se **combinan** en un mapa final.

Según las conclusiones del trabajo:

- Se proponen seis enfoques para construir mapas de saliencia **sin manipular el interior** del modelo de reconocimiento.
- Se mide la contribución de cada parte de la cara **por separado y en conjunto**.
- Se ven las zonas de la cara más relevantes para cada algoritmo.
- Sirve para evaluar sistemas comerciales cerrados (*off-the-shelf*).

Los experimentos usan cinco algoritmos de verificación (ArcFace, Dlib, VGGFace2, CasiaWebFace y LBP) en tres escenarios: imágenes sintéticas con distintas calidades y oclusiones, caras reales con distintas expresiones, poses y oclusiones, y caras de distintos grupos demográficos.

### Cómo se mide un sistema de verificación facial

Este es el punto de partida de la parte de sesgo:

1. Cada imagen pasa por un extractor de features y se convierte en un vector de $n$ elementos.
2. El score entre dos caras es la **similitud coseno** entre sus vectores: alto si son la misma persona, bajo si son personas distintas.
3. Para evaluar, se toman dos conjuntos A y B de $m$ caras, donde la cara $i$ de A y la cara $i$ de B son la misma persona. Se calcula la matriz de similitudes de $m \times m$:
   - la **diagonal** son los pares positivos (**genuinos**);
   - **fuera de la diagonal** están los pares negativos (**impostores**).
4. Se grafican las dos distribuciones de scores (densidad de frecuencia) y se fija un **umbral**. El ejemplo de las láminas muestra un par genuino con score 0,75 y un par de personas distintas con 0,10.

Para comparar algoritmos se usa **d′** (*d-prime*), que mide cuánto se separan las distribuciones de genuinos e impostores: cuanto mayor, mejor. Las láminas comparan cuatro algoritmos con d′ = 0,20; 1,59; 2,22 y 3,23, y marcan el de 3,23 como el mejor.

### Sesgo demográfico: un umbral, distintos errores

La pregunta de las láminas es si un algoritmo rinde igual para todos los grupos demográficos. La respuesta es **no**.

Siguiendo a Vangara, King, Albiero y Bowyer (2019), si se separan las distribuciones por grupo (por ejemplo, caucásicos y afroamericanos), las curvas de genuinos e impostores **no coinciden con la global ni entre sí**. Un umbral elegido para un grupo deja, en el otro grupo, una proporción distinta de falsos positivos y falsos negativos.

Otras evidencias que muestran las láminas:

- **Peinado y "píxeles de cara"** (arXiv 2008.06989): los mapas de calor muestran que, en promedio, las imágenes de rostros femeninos tienen menos píxeles de cara que las masculinas, y el peinado según el género es una de las causas principales.
- **Tono de piel** (Krishnapriya et al., 2020): la variación de precisión según raza y tono de piel, con resultados para ArcFace y VGGFace2.
- **Sistemas comerciales:** una lámina muestra precisiones muy dispares entre grupos (de 30,4% a 100%) con referencias a Amazon Rekognition y un estudio del MIT. El texto extraído no permite asignar cada cifra a su grupo.

### Opciones para un sistema más justo

| Enfoque | Idea | Problema |
|---|---|---|
| Umbral global | Un único umbral para todos | Errores distintos según el grupo |
| Umbrales demográficos | Un umbral por grupo (raza, género, edad) | Hay que definir los grupos y usar información sensible |
| Atributos blandos no sensibles | Elegir el umbral según atributos que no son sensibles | Es la propuesta de Villalobos, Mery y Bowyer (IEEE Access, 2022) |

Los dos problemas de usar datos demográficos en la operación real:

1. **¿Cómo definir el grupo?** ¿Solo género, o género + raza + edad? ¿Cuántas "razas" se definen? Haría falta una lista exhaustiva.
2. **Información sensible:** tanto pedir como predecir el grupo demográfico puede considerarse controvertido en reconocimiento facial.

Para elegir umbrales sin datos sensibles, las láminas muestran **estrategias basadas en clustering** y una **selección de umbral basada en un árbol de decisión**. Solo se ven como figuras, sin detalle en el texto.

### Causas del sesgo

| Causa | Qué pasa |
|---|---|
| **Muestras sesgadas** | Un sesgo inicial se agrava con el tiempo: las observaciones nuevas confirman la predicción y hay menos chances de contradecirla. Ejemplo: si la policía patrulla más donde ya registró delitos, registra más delitos ahí, y un sistema entrenado con esos datos hereda el sesgo. |
| **Datos contaminados** | El modelo reproduce el sesgo humano de los datos históricos. Ejemplo: usar como etiqueta las decisiones de contratación de un directivo, y no la capacidad real del candidato. |
| **Features limitadas** | Las features o etiquetas pueden ser menos fiables para grupos minoritarios, y ese ruido baja la precisión para ellos. |
| **Disparidad en el tamaño de la muestra** | Con muchos menos datos del grupo minoritario, es menos probable modelarlo bien. |
| **Proxies** | Aunque no se use el atributo sensible (raza, género), otras variables lo reflejan, como el lugar de residencia. Es difícil decidir si una variable relevante está demasiado correlacionada con el atributo protegido. |

Las láminas resumen: los datos que aporta el ser humano pueden estar sesgados, **pero esa no es la única fuente de sesgo**.

### Estrategias de mitigación

| Momento | Ejemplos en las láminas | Comentario |
|---|---|---|
| **Pre-processing** (antes de entrenar) | Construir y entrenar con datasets balanceados (Kärkkäinen y Joo, 2019; Wang et al., 2019; y otros) | Datos balanceados **no garantizan** precisión balanceada (Albiero et al., 2020) |
| **In-processing** (durante el entrenamiento) | Redes adversarias, aprendizaje por refuerzo, términos de penalización en la pérdida | Puede ser muy costoso y **no sirve para modelos de caja negra** |
| **Post-processing** (después de entrenar) | Calibración de scores, umbrales variables (Terhörst et al., 2020; Robinson et al., 2020) | Usa menos cómputo y se puede aplicar a cajas negras |

### Definiciones de fairness

Según las láminas, *fairness* (justicia, equidad) tiene muchas definiciones. Hay dos planos:

- Las **diferencias de rendimiento** entre grupos se pueden medir y cuantificar.
- El fairness en sí es una **construcción social**, definida por acuerdo general o por leyes, sobre las consecuencias de esas diferencias en una aplicación concreta.

| Criterio | Qué exige |
|---|---|
| **Statistical parity** | En clasificación binaria, todos los grupos tienen la misma probabilidad de recibir una predicción positiva. |
| **Disparate / adverse impact** | Las tasas de clasificación positiva entre dos grupos no deben diferir más allá de un límite del 80% (la conocida "regla del 80%"). |
| **Equalized odds / disparate mistreatment** | Ningún grupo recibe una cantidad desproporcionada de falsos positivos o falsos negativos. |
| **Demographic parity** (según la lámina) | No se centra en el grupo: individuos similares deben recibir predicciones similares. |

En el caso de verificación facial con umbral único, el criterio que se rompe es **equalized odds**: los falsos positivos y falsos negativos se reparten distinto entre grupos.

## Ejemplos y casos de la clase

**Explicabilidad**

- **Husky contra lobo** (paper de LIME): un clasificador entrenado con fotos de lobos en la nieve clasifica un husky como lobo. La explicación muestra que la red se fijaba en la **nieve**. El modelo parecía bueno, pero había aprendido un atajo.
- **Kinship detection** (detección de parentesco entre dos caras): López, Boutellaa y Hadid (2016) mostraron que comparar solo la **distancia de crominancia** (color) entre imágenes, sin ningún análisis facial, superaba a varios algoritmos publicados. El dataset tenía un atajo que los métodos explotaban.
- **Cómo engañar a una CNN:** video sobre la fragilidad de las redes frente a entradas manipuladas.
- **Summit** (Fred Hohman): visualización interactiva que resume qué features aprendió una red y cómo interactúan para predecir.
- **Labrador:** el ejemplo que recorre los 10 pasos de LIME.

**Sesgo y fairness**

- **COMPAS:** software que estima la probabilidad de reincidencia de un acusado con factores como pobreza o desempleo (la raza no se usa directamente). Los jueces lo usan para calibrar penas. El análisis de ProPublica muestra diferencias de falsos positivos y falsos negativos entre grupos, un ejemplo de *proxies*.
- **Reclutador de Amazon:** la herramienta de selección de personal de Amazon mostró sesgo contra las mujeres y fue descartada, un ejemplo de datos contaminados.
- **Word embeddings:** los vecinos más cercanos (KNN) de *man* y *woman* muestran sesgo de género en las nubes de palabras.
- **PULSE / depixelizer:** un modelo que genera caras de alta resolución a partir de una imagen pixelada devolvió una cara blanca a partir de una foto de Obama.
- **Reconocimiento de objetos y desigualdad global** (Facebook, "Does Object Recognition Work for Everyone?"): los algoritmos reconocen peor objetos del hogar de países de menores ingresos.
- **Biometría:** afecta a miles de millones de personas (vigilancia, control de acceso, gestión de identidad). Rathgeb (2021) resume el debate sobre si los sistemas son "parciales", "racistas" o "sexistas".
- Charla **"Ethics in AI: A Challenging Task"** de Ricardo Baeza-Yates, citada en varias láminas.

## Preguntas de repaso

<details><summary>1. El clasificador de husky contra lobo tenía buena precisión en test. ¿Por qué igual es un mal modelo, y cómo lo revela la explicabilidad?</summary>

Porque acertaba por una razón espuria: la nieve, que en el entrenamiento aparecía con los lobos. Con un husky en la nieve falla. Un mapa de saliencia como el de LIME muestra que las regiones decisivas eran el fondo y no el animal. La métrica sola no lo detecta si el test comparte el mismo atajo.

</details>

<details><summary>2. En LIME, ¿por qué se ponderan las perturbaciones según su distancia a la imagen original?</summary>

Porque LIME busca una explicación local: un modelo lineal que imite a la red cerca de esa imagen. Las perturbaciones que conservan casi todos los superpixeles describen mejor ese entorno, así que pesan más. Las que apagan casi todo están lejos y aportan poco.

</details>

<details><summary>3. ¿Qué representa cada coeficiente de la regresión lineal ponderada de LIME?</summary>

Cuánto influye ese superpixel en la probabilidad de la clase predicha. Los superpixeles con los coeficientes más grandes (en el ejemplo, los 4 mayores de 68) son la explicación que se muestra.

</details>

<details><summary>4. ¿Qué ventaja tienen los métodos de removal y aggregation para explicar verificación facial frente a métodos que miran dentro de la red?</summary>

No necesitan acceso al modelo: solo modifican la imagen *probe* y miran el score. Por eso sirven para sistemas comerciales cerrados. Además, al quitar o agregar regiones de a una o de forma acumulada, miden la contribución de cada zona por separado y en combinación con otras.

</details>

<details><summary>5. ¿Por qué un único umbral de verificación puede ser injusto aunque el rendimiento global sea bueno?</summary>

Porque las distribuciones de scores de genuinos e impostores son distintas para cada grupo demográfico. Un umbral fijo corta cada par de distribuciones en un lugar distinto, así que un grupo puede tener muchos más falsos positivos o falsos negativos que otro. La métrica global promedia y esconde esa diferencia.

</details>

<details><summary>6. ¿Por qué quitar la variable "raza" de un modelo no garantiza que deje de discriminar?</summary>

Por los proxies: otras variables, como el lugar de residencia o la pobreza en COMPAS, están correlacionadas con el atributo sensible y lo reintroducen. Además, el sesgo puede venir de datos contaminados, de muestras desbalanceadas o de features menos fiables para minorías.

</details>

<details><summary>7. Si se tiene un sistema de reconocimiento facial comercial cerrado, ¿qué tipo de mitigación es viable y por qué?</summary>

Post-processing, por ejemplo calibrar scores o usar umbrales variables. No requiere reentrenar ni acceder al modelo, y consume poco cómputo. El in-processing necesita modificar el entrenamiento, y el pre-processing con datos balanceados no asegura resultados balanceados.

</details>

<details><summary>8. ¿Qué problema resuelve usar atributos blandos no sensibles para elegir umbrales, frente a umbrales por grupo demográfico?</summary>

Evita tener que definir grupos demográficos (cuán específicos, cuántas categorías) y tener que pedir o inferir información sensible como raza o género, algo controvertido en reconocimiento facial. Igual permite ajustar el umbral según características de la cara.

</details>

## Conexiones

- **Hacia atrás:** retoma la verificación facial de la Clase 19 (embeddings, similitud coseno, umbral de matching): aquí se usa el mismo esquema para medir sesgo con distribuciones de genuinos e impostores y d′. La tensión entre rendimiento y explicabilidad ya apareció con los ensambles en la Clase 03. Los errores de detección comentados en la Clase 17 son el tipo de caso que un mapa de saliencia ayuda a diagnosticar.
- **TFI:** la lección de que una métrica global esconde diferencias entre grupos aplica al normalizador de nombres comerciales de Benefix. Conviene desagregar Hit@k y MRR por subgrupos de casos y no quedarse solo con el promedio. Lo mismo vale para un umbral único de similitud en el retrieval semántico, que puede funcionar distinto según el tipo de nombre.
