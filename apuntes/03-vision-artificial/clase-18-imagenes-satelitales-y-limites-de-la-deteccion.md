# Clase 18 · Imágenes satelitales y límites de la detección

> **Módulo 3 · Visión Artificial** · Martín Masci · jueves 04-jun-2026
> **Fuentes:** transcripción `DIA_Clase_18.vtt` (grabación parcial: solo quedó el tramo final de la clase, después del trabajo en grupos; corresponde a la "Clase 4" del docente, que no tiene láminas propias en la carpeta del curso) · demos en vivo de Kaggle/Medium (detección de barcos), Roboflow (detección de construcciones) y un repositorio con YOLO para imágenes aéreas · recapitulación que hace el docente al inicio de la Clase 19

Este apunte es más corto que los demás porque la grabación conserva solo el cierre de la clase. El tramo anterior (trabajo en grupos sobre el pipeline de detección de la Clase 17) no quedó registrado.

## En pocas palabras

- Hasta este punto del módulo, la visión artificial todavía se puede **reproducir de punta a punta**: combinando dos o tres herramientas (redes preentrenadas, detección, clasificación) se arma un pipeline para un problema propio. Pero cada problema necesita **su propio entrenamiento y su propia calibración**: un detector entrenado para la tortuga del patio no va a funcionar para el gato del living.
- Lo que no debería pasar es que el modelo "funcione y no se entienda por qué". En visión artificial todavía es posible ir línea a línea y entender qué hace cada parte, a diferencia de los grandes modelos generativos.
- Las **imágenes satelitales** son una rama particular: las clases son bastante estables (todo se ve desde arriba), lo que facilita el entrenamiento, pero las imágenes son muy densas en información y traen problemas propios: **nubes**, **coloración desigual** entre mosaicos, **escala o zoom** y **recortes (cropping)** que duplican o cortan objetos en los bordes.
- Un detector puede equivocarse aunque se haya entrenado con datasets enormes, si el dataset **no cubre la variedad real de la clase**. El caso de las "*zombie pools*" (piletas sucias, vacías o de colores no celestes) lo muestra.
- **Detección/clasificación** (encerrar el objeto en un rectángulo) alcanza para la mayoría de los casos; la **segmentación** (pintar los píxeles exactos) se justifica cuando se necesita el borde preciso, como en medicina. Lo que viene es pasar de *detectar* a **reconocer** (caras, Clase 19).

## Conceptos clave

### Hasta dónde llega lo visto: reproducible, pero específico

El docente usa la analogía de un curso de cocina: con lo aprendido se puede reproducir una receta y que salga bien, pero montar un restaurante requiere mucho más entrenamiento. En visión artificial:

- Con las arquitecturas vistas (redes convolucionales, Detectron2, YOLO) se puede armar, entrenar y depurar un detector para un problema propio.
- La estructura de fondo es siempre la misma: **procesamiento, análisis de la imagen y detección de patrones**.
- Lo que no se transfiere gratis es el entrenamiento: cada problema requiere sus datos, su etiquetado y su ajuste de hiperparámetros.
- La frontera del campo (arquitecturas grandes, problemas a gran escala) no se discute en detalle en el curso.

El docente contrasta esto con los modelos generativos actuales: funcionan muy bien, pero reproducirlos exige arquitecturas mucho más complejas y máquinas muy potentes. Con visión artificial, en cambio, "todavía se puede ir código a código".

### Imágenes satelitales: qué las hace distintas

Una foto común (tomada con un celular, bajada de internet o generada) es un conjunto de píxeles con objetos; su calidad no cambia la naturaleza del problema. Las imágenes satelitales son otro mundo:

| Aspecto | Ventaja o problema |
|---|---|
| Punto de vista | **Ventaja:** siempre desde arriba y con un estilo de imagen parecido; las clases son más o menos siempre las mismas, lo que simplifica el entrenamiento. |
| Densidad | **Problema:** cada pedacito de imagen trae mucha información. La pregunta clave es qué tan cerca o qué tan lejos hay que estar para detectar el objeto buscado. |
| Mosaico | **Problema:** la imagen es un pegado de muchas tomas, con zonas más claras ("quemadas") u oscuras y coloraciones distintas. |
| Nubes | **Problema típico:** la nubosidad tapa partes de la imagen, como una foto revelada con una parte quemada. |
| Sin bordes naturales | **Problema:** la imagen es "infinita"; los bordes los pone el recorte. |
| Zoom | **Problema:** los detectores reconocen bien los objetos cuando están muy cerca, pero el objetivo suele ser detectarlos con una vista más alejada. |

### El problema del recorte (cropping)

Para procesar una imagen satelital hay que cortarla en pedazos. Si los recortes no coinciden exactamente en los bordes, un objeto (por ejemplo, una casa) puede quedar en dos recortes a la vez y **contarse dos veces**, o quedar partido. El docente lo mostró moviendo levemente el recorte sobre Google Maps. Además, el problema cambia según la zona: en una ciudad densa hay objetos por todos lados; en el campo, encontrar una casita aislada es otro tipo de búsqueda.

En la demo de Roboflow (abajo) se vio lo mismo: el detector falla sobre todo **en los bordes** de la imagen.

### Modelos que se usan con imágenes satelitales

- Según el docente, los modelos tipo **ResNet** y en general los de PyTorch, y sobre todo **YOLO** (v8 en adelante), funcionan bien con imágenes satelitales.
- Por cómo codifican la imagen, rinden mejor cuando el objeto aparece grande, con mucho zoom; llevarlos a reconocer objetos con menos zoom requiere trabajo adicional sobre la imagen.
- Con Google Maps no se llega a nivel de píxel: incluso con el máximo zoom que permite la herramienta, la imagen sigue siendo densa.
- Google Maps es un dataset enorme y público, pero (hasta donde sabe el docente) no ofrece una API para bajar imágenes; se puede escribir código que las obtenga en un tamaño determinado.

### Por qué falla un detector bien entrenado: las *zombie pools*

En la demo con YOLO, el modelo marcó como "pileta" algo que claramente no lo era (un estadio) y no detectó otras. El modelo estaba entrenado con datasets grandes, así que la explicación no es "pocos datos". En la discusión con los alumnos se descartó que la causa fuera la forma rectangular o el agua alrededor; la respuesta fue la **variedad de apariencia de la clase**:

- Si se entrena solo con piletas de agua celeste y cristalina, no se detectan las ***zombie pools***: piletas sucias (agua verde u otra tonalidad) o vacías.
- Una pileta vacía, vista desde arriba, se parece al techo de un galpón.
- Hay piletas con revestimiento color arena, o tapadas.

Conclusión: el dataset tiene que incluir todas esas variantes. Para el ojo humano es obvio qué es una pileta; para la red hay que resolver **en qué mira** para decidir que lo es. Cada clase tiene su "doble clic" de particularidades.

### Clasificación/detección vs. segmentación

- **Detección y clasificación**: encontrar el objeto, encerrarlo en un rectángulo (*bounding box*) y etiquetarlo. Es lo que se usa en la mayoría de los trabajos.
- **Segmentación**: pintar los píxeles que pertenecen al objeto. No es mucho más complicada técnicamente, pero se usa cuando se necesita el **borde exacto**: tumores, fallas, temas médicos. Es otro tipo de entrenamiento.
- En patología digital, según comentó un alumno, el patólogo marca una **ROI** (*region of interest*) y el algoritmo trabaja sobre ese recorte y no sobre la imagen digitalizada completa.

| Uso | Enfoque | Rol del modelo |
|---|---|---|
| Contar objetos (vacas en una provincia, construcciones, piletas) | Detección | Automatizable |
| Seguridad (cámaras, rayos X de equipajes) | Detección, junto con otros dispositivos físicos como detectores de metales | Automatizable; también sirve para algunas cuestiones de fraude |
| Medicina (rayos X, tomografías, lunares) | Segmentación, más precisión | **Complementa** un diagnóstico que tiene que ser humano, como el informe del radiólogo que lee el médico |

### Detectar no es detectar anomalías

Todo lo visto hasta ahora busca cosas que **se sabe que se quieren encontrar** (una clase definida). La detección de anomalías es otra discusión, que el docente dejó para más adelante en el módulo si da el tiempo.

### Lo que viene: de detectar a reconocer

- Hasta ahora: cómo mira una computadora, qué mira, y cómo **detectar y clasificar**. Sigue habiendo una lógica binaria: es una pieza de ajedrez o no lo es.
- Próximo paso: **reconocer**. Con caras es más difícil, porque entre muchas personas hay que identificar a la que se busca. Lo mismo se puede hacer con animales: no detectar "perros", sino **tu** perro.
- El extremo es el ***face analysis***: una vez reconocida la persona, inferir rasgos, características o el sentimiento detrás de la cara.

## Ejemplos y casos de la clase

- **Detección de barcos (Kaggle + Medium):** un ejemplo del mismo estilo que el detector de ajedrez, pero sobre imágenes satelitales. El docente lo usó para abrir la "segunda capa" del tema: qué cambia cuando las imágenes no son fotos comunes. Lo recomendó por la cantidad de trabajo profesional que usa información satelital.
- **Google Maps en vivo:** sirvió para mostrar el problema del zoom y del recorte: un mismo objeto puede quedar en dos recortes contiguos.
- **Roboflow, detector de construcciones (*buildings*):** un modelo ya armado. Al probarlo en vivo sobre un recorte de Google Maps, detectó algunas construcciones pero no todas, y falló sobre todo en los bordes. El docente lo presentó como una herramienta para explorar, aunque es más un producto cerrado ("caja negra") que código limpio para modificar en Colab. Roboflow tiene modelos prearmados para muchas otras clases.
- **Repositorio YOLO para imágenes aéreas:** combina YOLO (usa YOLOv9; el docente trabajó con v8) con lo que el docente llamó "DOTA", una propuesta más moderna para objetos *(dudoso: DOTA suele ser el nombre de un dataset de imágenes aéreas, no de una arquitectura; conviene verificarlo en el repositorio)*. Es bastante *low code*, pero hay que entrenarlo (es supervisado). Devuelve recortes con las clases detectadas: piletas, helicópteros, puentes, aviones, barcos, aeropuertos. Acá aparecieron los errores con piletas que llevaron a las *zombie pools*.
- **Imágenes comerciales de alta resolución:** un alumno contó que, consultado por un proyecto de conteo de animales en un campo, encontró servicios que venden imágenes satelitales de entre 30 y 50 cm por píxel (la mayor definición que se comercializa, según su búsqueda), a un costo alto. El docente agregó que el problema es la escala: contar animales en una provincia entera o todas las piletas del país implica un dataset enorme y carísimo. Para algo provincial se puede trabajar "a mano" con muchas imágenes lo más chicas posible, y las arquitecturas vistas alcanzan.
- **Detector de ajedrez de la Clase 17:** el docente cerró la discusión de ese caso (dejó un video) y lo usó como referencia de detección sobre imágenes comunes.

## Práctica

No hubo notebook nuevo en el tramo grabado. Quedaron comentarios prácticos sobre el entorno de Colab, a partir de lo que les pasó a los alumnos al entrenar el detector de la Clase 17:

- Un alumno tuvo que bajar el backbone del modelo a **R50-FPN** porque el entorno gratuito colapsaba con uno más pesado.
- Para clasificación múltiple simple, con imágenes no muy pesadas, se puede usar **CPU sin GPU**. Con imágenes satelitales, no.
- No cambiar el tipo de entorno ni reiniciarlo mientras el modelo está entrenando: se pierde todo y hay que correr de nuevo desde cero.
- Si se usa mucho el entorno gratuito, Colab puede dejar al usuario sin backend. Hacer una copia del notebook con las modificaciones ayuda, aunque pueden quedar sesiones colgadas que hay que cerrar.

## Para recordar

- "Funciona y no entiendo por qué" es justamente lo que no debería pasar.
- Cada problema requiere su entrenamiento específico y su calibración específica.
- En imágenes satelitales, los problemas típicos son las nubes, la escala (zoom) y los bordes de los recortes.
- Un error del detector no siempre es falta de datos: puede ser falta de **variedad** en la clase (las *zombie pools*).
- Para el ojo humano es obvio; la pregunta es en qué mira la red para darse cuenta.
- Segmentar solo cuando se necesita el borde exacto (medicina, fallas); en medicina el modelo complementa al profesional, no lo reemplaza.
- Hasta acá se detecta lo que se quiere encontrar; detectar anomalías es otro problema.

## Preguntas de repaso

<details><summary>1. ¿Por qué las imágenes satelitales facilitan en parte el entrenamiento de un detector y a la vez lo complican?</summary>

Lo facilitan porque el punto de vista es siempre cenital y el estilo de imagen es parecido, así que las clases son bastante estables. Lo complican porque son muy densas en información, vienen de un mosaico de tomas con distinta coloración, pueden tener nubes que tapan la imagen y obligan a decidir el nivel de zoom y cómo recortar.

</details>

<details><summary>2. ¿Qué problema genera recortar una imagen satelital en pedazos para procesarla?</summary>

La imagen no tiene bordes naturales; los pone el recorte. Si los recortes no coinciden exactamente, un mismo objeto puede quedar en dos recortes y contarse dos veces, o quedar partido. Por eso los detectores suelen fallar en los bordes.

</details>

<details><summary>3. Un detector entrenado con un dataset enorme confunde un estadio con una pileta y no detecta piletas sucias. ¿Cuál es la explicación más probable?</summary>

Que el dataset no cubre la variedad real de la clase. Si se entrenó sobre todo con piletas de agua celeste, no reconoce las *zombie pools* (sucias, vacías, de color arena o tapadas), y una pileta vacía puede parecerse al techo de un galpón. El problema no es la cantidad de datos, sino su representatividad.

</details>

<details><summary>4. ¿Cuándo conviene segmentar en lugar de detectar con un rectángulo?</summary>

Cuando se necesita el borde exacto del objeto, como en medicina (tumores, lunares) o en la detección de fallas. Para contar objetos o para seguridad alcanza con la detección.

</details>

<details><summary>5. ¿Por qué el docente distingue entre seguridad y medicina en cuanto al rol del modelo?</summary>

En seguridad, o para contar objetos, la detección se puede automatizar y complementar con otros dispositivos físicos. En medicina el modelo acompaña un diagnóstico que tiene que ser humano: aporta una segunda mirada, como el informe del radiólogo que lee el médico.

</details>

<details><summary>6. ¿Por qué un detector entrenado para una tortuga en un patio no sirve para un gato en un living?</summary>

Porque cada problema necesita su propio entrenamiento y su propia calibración. El detector aprendió los patrones de un objeto en un contexto determinado, y otro objeto en otro contexto requiere otros datos etiquetados y otro ajuste.

</details>

<details><summary>7. ¿En qué se diferencia detectar de reconocer, según el adelanto de la próxima clase?</summary>

Detectar sigue siendo binario: el objeto de la clase está o no está (es una pieza de ajedrez o no). Reconocer exige identificar a un individuo concreto entre muchos parecidos: una persona entre muchas caras, o *tu* perro y no cualquier perro.

</details>

<details><summary>8. ¿Por qué lo visto hasta ahora no es detección de anomalías?</summary>

Porque los modelos buscan una clase definida de antemano: se sabe qué se quiere encontrar y se entrena con ejemplos etiquetados de eso. Detectar anomalías es buscar lo que se aparta de lo normal sin tener una clase definida, y es otro problema.

</details>

## Conexiones

- **Hacia atrás:** continúa la detección multiclase y el detector de ajedrez con Detectron2 de la Clase 17 (dataset etiquetado, ajuste de hiperparámetros, errores del modelo). Lo de "en qué mira la red" retoma cómo procesa la imagen una computadora, de las Clases 15 y 16.
- **Hacia adelante:** la Clase 19 empieza recapitulando este tema (imágenes satelitales) y pasa a reconocimiento facial: detección de rostros y comparación de caras. La segmentación y la detección de anomalías quedaron anunciadas para más adelante en el módulo.
- **TFI:** las *zombie pools* son un caso de un problema general que también aparece en el normalizador de nombres comerciales de Benefix: un sistema entrenado o calibrado solo con las variantes "limpias" de cada clase falla con las variantes reales (abreviaturas, errores de tipeo, formatos inesperados). El conjunto con el que se miden Hit@k y MRR tiene que incluir esa variedad.
