# Clase 13 · Introducción a la IA generativa: herramientas, riesgos y prompting

> **Módulo 2 · Intro a IA Generativa (caja de herramientas)** · Horacio Arrigo · martes 19-may-2026
> **Fuentes:** transcripción `DIA_Clase_13.vtt` (no hay láminas ni notebook en la carpeta; la presentación, armada originalmente por Diego Adrián Castro y actualizada por el profesor, solo se conoce por lo que se dijo en clase)

## En pocas palabras

- La **IA generativa** es una rama del aprendizaje profundo: redes neuronales mucho más grandes, entrenadas con mucha más información. No reemplaza al machine learning clásico ni al deep learning; cada uno tiene sus pros y sus contras. Los **agentes** no son una tecnología nueva, sino una aplicación: sistemas autónomos con un objetivo que usan estas tecnologías.
- Un LLM es un **modelo probabilístico de lenguaje**: predice la palabra más probable, no reflexiona sobre lo que dice. De ahí salen sus límites: **alucinaciones** convincentes, complacencia, debilidad en matemática y sesgos heredados de los datos (el reloj que siempre marca las 10:10).
- Las herramientas actuales suman cosas alrededor del modelo: **multimodalidad**, búsqueda en Internet para compensar que el entrenamiento es estático, y herramientas externas para imágenes o cuentas ("una caja de herramientas"). El **razonamiento paso a paso** mejora respuestas complejas a cambio de más procesamiento.
- En el trabajo, usar herramientas **públicas o gratuitas** con información de la empresa es una mala práctica, equivalente a mandar documentación interna desde el mail personal. Las versiones **corporativas** ofrecen confidencialidad, no entrenan con los datos y están gestionadas por la organización.
- El **prompting** es cómo nos comunicamos con estos modelos: rol, instrucción clara, contexto, audiencia y formato de salida, dados desde el principio. Técnicas: zero-shot, one-shot, few-shot (*in-context learning*), cadena de pensamientos y pedirle al propio modelo que escriba el prompt.

## Conceptos clave

### Dónde se ubica la IA generativa

El profesor la ubicó con el diagrama de conjuntos anidados:

| Nivel | Qué es según la clase |
|---|---|
| Inteligencia artificial | La disciplina completa. Nace en los años 60-70 a partir de estadística, matemática, ciencias de la computación e ingeniería, con la pregunta de si las máquinas pueden pensar o emular el comportamiento humano. |
| Machine learning | Modelos algorítmicos que aprenden patrones de los datos para clasificar, predecir o estimar. |
| Deep learning | Redes neuronales; necesitan muchos más datos, pero en general mejoran los resultados. |
| IA generativa | Redes neuronales mucho más grandes, que generan contenido a partir de instrucciones. |
| Agentes | No es una rama nueva: es una aplicación. Un sistema autónomo con un objetivo, que decide y ejecuta según su conocimiento y su contexto (la analogía es un robot con una tarea). |

Una idea que el profesor remarcó: que hoy se hable de "IA" pensando en ChatGPT no significa que lo generativo reemplace a lo anterior.

### Breve historia

- **2014, GAN** (redes generativas antagónicas): dos redes, una **genera** (por ejemplo, una imagen de un gato) y otra **evalúa** qué tan realista es. El generador aprende del feedback del evaluador e itera hasta que lo generado es indistinguible de lo real. Para el profesor, es la primera arquitectura generativa. (El nombre del autor quedó mal reconocido en la transcripción.)
- **2017, Transformer** (Google): nace para procesamiento de lenguaje natural, cuya tarea fundacional es la **traducción**, y produce un salto de calidad muy grande en ella. Es la arquitectura base de los **LLM** (*large language models*).
- **GPT-3 y ChatGPT** (OpenAI): las versiones anteriores eran científicas y no se usaban masivamente. La masificación llega con ChatGPT como aplicación web. (En la clase se mencionan 2020 y 2022 para el lanzamiento; la fecha quedó confusa.)
- Después aparecen generadores de imágenes como **DALL-E** y **Stable Diffusion**, y desde 2023 hasta hoy hay adopción masiva y modelos **multimodales**.

### Características de la IA generativa

- **Multimodalidad**: entiende y genera texto, imagen, audio y video, y puede producir resultados compuestos (por ejemplo, modificar fragmentos de películas).
- **Responde a instrucciones** en lenguaje natural: el **prompt**.
- **Consumo de recursos muy alto** (electricidad, agua). Entrenar un modelo como ChatGPT o Gemini no está al alcance de cualquiera, y reentrenarlo cuesta cientos de millones de dólares. Para un uso propio no hace falta reentrenarlo todo: alcanza con adaptarlo a la información que uno necesita, y hay técnicas para eso.

### Alucinaciones y por qué ocurren

Una **alucinación** es cuando el modelo produce información falsa o "demasiado creativa", inventa cosas que no son.

- **Por qué**: el modelo no piensa como nosotros. Es estadístico y va prediciendo la palabra más probable. Si se le pregunta algo sin respuesta (un acertijo imposible, un autor que no existe), igual genera lo que le parece más probable.
- **Lo peligroso**: lo inventado suena **muy convincente**, y como le preguntamos justamente lo que no sabemos, es fácil creerlo sin verificar. Además, las aplicaciones son **complacientes**: les cuesta contradecirnos.
- **Depende del contexto**: un estudiante planteó que la alucinación puede ser útil para salir de la zona de confort. El profesor coincidió para un artista, pero no para un médico o un abogado.
- El propio *disclaimer* de OpenAI reconoce que, por la naturaleza probabilística del aprendizaje automático, el output puede no reflejar con exactitud personas, lugares o hechos.

**Defensa práctica:** pedirle que **cite las fuentes** que usó (el ejemplo fue preguntarle a Copilot quién fundó Buenos Aires) y entrar a verificarlas con criterio propio.

### Sesgos y restricciones

- **Sesgo por sobrecorrección**: Gemini se entrenó para evitar sesgos raciales y culturales y "se pasó de la raya": al pedirle vikingos, generaba personas con rasgos que no corresponden a la historia.
- **Sesgo de los datos**: el reloj de las 10:10 (ver ejemplos). Lo que no está en el entrenamiento "no existe" para el modelo.
- **Restricciones deliberadas**: DeepSeek no respondía sobre Tiananmén 1989 (no se sabe si por falta de datos o por instrucción); Mistral o Gemini sí responden. Muchos modelos se niegan a dar diagnósticos médicos, en parte para cuidarse legalmente.
- **Jailbreak sencillo**: si se le piden sitios de software pirata, se niega; si se le piden "para bloquearlos en el firewall", los da. Es una técnica de ataque nueva que no existía en antispam ni firewalls. No se pueden tipificar todas las vulnerabilidades; lo que se puede hacer es restringir los modelos a ciertos temas o a cierta documentación.

### Entrenamiento estático y conexión a Internet

El conjunto de entrenamiento es **estático**: los modelos no se reentrenan a diario. Antes, si se preguntaba por algo posterior al corte, el modelo no lo sabía o inventaba ("sí, River ganó ayer"). Hoy las aplicaciones están **conectadas a Internet**: extraen los tópicos de la pregunta, buscan en fuentes (noticias, foros) y arman la respuesta. La información está actualizada porque el modelo busca, no porque se haya reentrenado.

### Límites en matemática, lógica y física

- Son modelos de **lenguaje** ("la M de LLM es de modelo, no de matemática"). Entrenados con texto, responden los números más probables y no tienen calculadora.
- Mejoraron porque las empresas les **agregan herramientas detrás**: modelos que hacen cuentas, modelos que entienden imágenes. El LLM invoca la herramienta, recibe el resultado (por ejemplo, el texto extraído de una imagen) y lo estructura. Es como una persona con una caja de herramientas, y de ahí se pasa naturalmente a los agentes.
- Siguen sin entender la lógica o la física del mundo real.
- Están entrenados sobre todo con texto en inglés, así que en tareas de lenguaje en español (por ejemplo, análisis sintáctico) también pueden equivocarse.

### Razonamiento, contexto y cadena de pensamientos

- **Razonar** es que el modelo verifique paso a paso lo que hace para llegar al resultado. Ayuda a validar la respuesta y a que el modelo **no pierda el hilo**. Su contra es que **consume más procesamiento y tiempo**, así que hay que elegir cuándo usarlo. En tareas complejas, como generar código, muchos modelos ya razonan por su cuenta.
- **Contexto**: es la cantidad de información que el modelo puede manejar en una conversación, una especie de memoria. Si se le pregunta mucho, el contexto crece, se vuelve inmanejable y el modelo se "marea". La analogía: recordamos bien lo de ayer y mal lo de hace un año, salvo que reconstruyamos el día paso a paso.
- **Cadena de pensamientos** (*chain of thought*): el razonamiento hecho "a mano". Uno divide el problema y guía al modelo pidiéndole un paso por vez. Con los modelos que razonan solos es casi equivalente, pero sigue sirviendo cuando necesitamos entender cómo resuelve o forzar un paso concreto.
- Corregirle errores ("no es así") le da información útil, y las empresas la usan para mejorar los modelos.

### Uso corporativo: herramientas públicas vs. corporativas

Las preguntas del profesor: ¿mandarías documentación interna desde tu mail personal? ¿La guardarías en tu Drive personal? ¿La publicarías en una red social? La respuesta es no, y usar una IA pública con esa información es lo mismo.

| | IA pública / gratuita | IA corporativa |
|---|---|---|
| Acceso | Libre por Internet o cuentas personales | Integrada al entorno de trabajo |
| Control | Entornos abiertos, no controlados por la empresa | Gestionada por una política interna (ciberseguridad, operaciones o una persona en una PYME) |
| Uso de los datos | Pueden usarse para entrenar y alimentar respuestas futuras | Confidencialidad: no se usan para entrenar |
| Revisión | Puede haber revisores humanos | Sin revisión externa explícita |
| Terceros | Riesgo de compartir (por ejemplo, para publicidad) | No se comparte con terceros |
| Infraestructura | Menos inversión en seguridad | Mayor seguridad y cumplimiento normativo |

- La información es uno de los activos más importantes de cualquier empresa, también de una PYME o una startup.
- La salida correcta es **contratar** la herramienta, asegurarla y dejar por contrato que los datos no se usan para entrenar ni se comparten.
- En el debate de clase, un estudiante señaló que las empresas grandes ya usan licencias corporativas o modelos privados (se mencionó AWS Bedrock), y que el riesgo de las cuentas free es justamente que no se puede controlar ese vector. Otro planteó más preocupación por las apps hechas con herramientas de *vibe coding* sin mínimos de seguridad.
- También aparece la **publicidad** basada en las conversaciones en versiones gratuitas, algo más delicado que en el mail porque los chats se usan de forma mucho más personal.

### Prompting

El prompt es cómo los humanos instruimos a la IA en lenguaje natural. Puede ser texto, pero también imagen o audio (si se dicta, también va nuestra voz).

**Partes de un buen prompt** (el framework de la presentación):

1. **Rol**: que conteste como un profesional determinado (arquitecto, diseñador de interiores, asesor de turismo). Los roles médicos suelen estar bloqueados.
2. **Instrucción** directa, clara y explícita.
3. **Detalles relevantes**: contexto, objetivo, de dónde viene el pedido.
4. **Audiencia**: para uno mismo, para un ejecutivo, para chicos de qué edad.
5. **Formato de salida**: mail, presentación, imagen para un diseño.

Conviene darlo todo **desde el principio** y de forma estructurada, para que el modelo planifique mejor y no se maree con correcciones posteriores ("si arranca mal, el error se incrementa"). Igual, para preguntas simples (la temperatura de mañana en Buenos Aires) un prompt corto alcanza.

**¿Hace falta el rol con los modelos nuevos?** Se discutió en clase. Los modelos entienden cada vez mejor la intención y adoptan solos un rol parecido. La conclusión del profesor: el rol y el contexto **nunca están de más**, y explicitarlos da respuestas más concretas desde el arranque. Otra técnica es contarle quiénes somos, qué hacemos y cuáles son nuestros objetivos, pero **solo en una herramienta corporativa**.

**Técnicas de *in-context learning* (ICL)**: darle ejemplos dentro del prompt.

| Técnica | Ejemplos | Uso típico |
|---|---|---|
| Zero-shot | Ninguno | Ideas generales, categorización |
| One-shot | Uno | Mostrar el pedido y el resultado esperado (por ejemplo, un número capicúa para que genere otros) |
| Few-shot | Varios | Generar nuevos casos con el mismo patrón (por ejemplo, speeches de venta de call center para otros productos) |

**Meta-prompting** (aporte de estudiantes, avalado por el profesor):
- Pedirle al modelo, con rol de "ingeniero de prompts", que escriba el prompt, y usarlo en una **ventana nueva** para no arrastrar el contexto del primer mensaje.
- Después de iterar hasta llegar al resultado buscado, preguntarle qué prompt habría producido esa versión final, y guardarlo para tareas repetitivas.

**Privacidad en el prompt**: en herramientas públicas, omitir los datos que no cambian el resultado. En vez de "mi mamá de 65 años vive sola en Córdoba…", preguntar por "una persona mayor que vive lejos". Un estudiante señaló que la edad o la zona pueden ser relevantes, y el profesor aceptó dar un rango. Ojo: el modelo puede **inferir** datos de lo que se le fue contando en otras conversaciones.

**Costo**: cada token tiene costo. Un prompt mal armado o muchas iteraciones encarecen el uso, y hay empresas que concluyeron que un programador junior puede salir más barato que generar código con IA.

## Ejemplos y casos de la clase

- **Comprobante de transferencia** (multimodalidad): con solo "¿me podrías describir la imagen?", el modelo deduce que un número es un monto, reconoce fecha y hora con su formato y el banco. La diferencia con el OCR clásico, que un estudiante explicó, es que el OCR extrae el texto tal cual (con errores ante tipografías raras, documentos torcidos o mala calidad), mientras que el LLM **estructura** la información para volcarla en una planilla o una base de datos. Hasta interpreta escritura a mano, con límites.
- **Diseño de cocina con Copilot**: un estudiante subió su plano y fue indicando materiales (porcelanato, maderas, mesada) hasta lograr un render hiperrealista en un día, peleándose con el modelo que insistía en poner el horno donde no iba. Para el profesor, los profesionales siguen haciendo falta, pero usando estas herramientas para acelerar.
- **El capicúa con Claude** (razonamiento): el modelo verifica posición por posición. Al agregar un cero adelante, concluye que no es capicúa, pero aclara que, interpretado como entero, el cero no cuenta y sí lo es. Con un decimal (0,…) plantea si incluir o no el punto y analiza cada opción. Muestra cómo el razonamiento detecta ambigüedades del pedido.
- **El reloj de las 10:10**: al pedir un reloj analógico, casi todos los modelos marcan las 10:10, porque la publicidad de relojes usa esa hora como estándar. Es un sesgo de los datos: lo que el modelo no vio, no lo puede generar.
- **Vikingos de Gemini**: sobrecorrección de sesgos.
- **Comparativas de imágenes** (un camello con lentes azules, un hombre en una reposera con binoculares) y la evolución del video de **Will Smith comiendo fideos**: de grotesco a indistinguible de lo real. También el video del Pensador de Rodin con un televisor y el Obelisco. Contracara: el riesgo de las *fake news*, que se difunden más rápido que su desmentida.
- **Caso Samsung** (2023): un empleado usó ChatGPT gratuito con código propietario, el código se filtró y Samsung prohibió su uso. También se mencionaron una filtración de datos personales y medios de pago de usuarios de ChatGPT (visibles nueve horas) y casos de mal uso de chatbots en Toyota y McDonald's.
- **Hijo estudiando con chats separados por materia**: una estudiante contó que al marcarle los errores (por ejemplo, en análisis sintáctico) el modelo mejoraba, y que buscar en qué se equivocaba era en sí un ejercicio de aprendizaje.
- **Proyecto Glasswing y Claude Mythos Preview** (Anthropic): según se contó en clase, el modelo resultó tan capaz para encontrar vulnerabilidades (una de 27 años en OpenBSD, otra de 16 años en FFmpeg, varias en el motor JavaScript de Firefox) que Anthropic no lo liberó al público y lo compartió con un grupo inicial de unas 12 organizaciones (entre ellas Amazon, Google, Microsoft, Apple, Cisco, Nvidia y Broadcom) para revisar su propio software. Sirvió para explicar las **vulnerabilidades zero-day** (existen desde que se libera el software y nadie las detectó) y el **system card**, el documento con las pruebas y los benchmarks de un modelo (en este caso, de 244 páginas). La pregunta abierta: ¿y si otra empresa logró algo parecido y no lo informó?

## Práctica

No hubo ejercicios formales. Hubo dos actividades en vivo:

- Pedirle a cualquier IA la imagen de un **reloj analógico** y ver qué hora marca.
- **Demo de NotebookLM** (Google): un chatbot al que se le cargan fuentes (webs, PDF, videos de YouTube, CSV, entre otros formatos) para consultarlas y generar contenido a partir de ellas: resúmenes en audio (podcast), presentaciones, video, mapas mentales, reportes, flashcards y guías de estudio. El profesor cargó la web de Mythos y su system card y le pidió un listado de las vulnerabilidades encontradas por software. Tiene versión gratuita, y se mencionó que tiene API y que se le pueden conectar agentes. La recomendó para el trabajo final (informes, reportes, estudio de documentos), con la advertencia de no cargar información personal en la versión gratuita.

También se mencionaron herramientas para código: Claude Code, Codex, Cursor, Visual Studio Code con asistentes y **Granite Code**, de descarga libre e instalable localmente. Sin acceso a Internet, los prompts no salen de la máquina.

## Para recordar

- "La IA es un aliado, una herramienta. No decide por nosotros": siempre hace falta supervisión humana.
- El modelo no piensa: predice el texto más probable. Por eso puede inventar con total convicción.
- Pedir **fuentes** y verificar, sobre todo si la respuesta genera dudas.
- Que la información esté actualizada no significa que el modelo se reentrenó: busca en Internet.
- Usar una IA pública con datos de la empresa es como mandar documentación interna desde el mail personal.
- El razonamiento mejora las respuestas complejas, pero cuesta más tiempo y procesamiento.
- "Mejor arrancar bien y refinar que arrancar mal y corregir": dar rol, contexto, audiencia y formato desde el primer prompt.
- No compartir en un prompt datos personales que no cambian el resultado.
- Las nuevas tecnologías traen nuevas técnicas de ataque que todavía no están todas descubiertas.

## Preguntas de repaso

<details><summary>1. ¿Por qué un LLM puede alucinar aunque haya sido entrenado con muchísima información?</summary>

Porque no razona sobre la verdad de lo que dice: es un modelo probabilístico que genera la continuación más probable. Si la respuesta no está en sus datos (o no existe), igual produce algo verosímil. El riesgo es que suena convincente y que la aplicación tiende a ser complaciente.

</details>

<details><summary>2. ¿Qué enseña el ejemplo del reloj que marca las 10:10?</summary>

Que el modelo reproduce la distribución de sus datos de entrenamiento: como la publicidad de relojes usa casi siempre esa hora, otras posiciones de las agujas son improbables para él. Lo que no vio con suficiente frecuencia, el modelo no lo puede generar sin una indicación explícita.

</details>

<details><summary>3. Si un chatbot responde correctamente sobre el partido de ayer, ¿significa que el modelo se reentrenó?</summary>

No. El entrenamiento es estático y reentrenar cuesta cientos de millones de dólares. La aplicación busca en Internet según los tópicos de la pregunta y arma la respuesta con esas fuentes.

</details>

<details><summary>4. ¿Por qué los LLM fallaban en matemática y cómo mejoraron?</summary>

Porque son modelos de lenguaje entrenados con texto, no con las reglas del álgebra: devuelven los números más probables y no tienen calculadora. Mejoraron porque las aplicaciones invocan herramientas externas (modelos que calculan, que leen imágenes) y porque se les pide razonar paso a paso.

</details>

<details><summary>5. ¿Cuál es la diferencia entre pedirle al modelo que razone y aplicar cadena de pensamientos, y qué costo tienen?</summary>

En el primer caso, el modelo decide sus propios pasos. En la cadena de pensamientos, uno divide el problema y le pide un paso por vez, lo que sirve cuando hay que entender o forzar cómo resuelve. Las dos mejoran la calidad y evitan que el modelo pierda el hilo, pero consumen más procesamiento y tiempo.

</details>

<details><summary>6. ¿Qué riesgos concretos tiene usar una IA gratuita con información de la empresa y qué ofrece la versión corporativa?</summary>

Que los datos se usen para entrenar y reaparezcan en otras respuestas, que los vean revisores humanos, que queden en servidores menos seguros o se compartan con terceros (caso Samsung). La corporativa garantiza por contrato que no se entrena con los datos, no hay revisión externa ni se comparte con terceros, y está gestionada por una política interna.

</details>

<details><summary>7. ¿Qué elementos debería tener un prompt para una tarea compleja y por qué conviene darlos al principio?</summary>

Rol, instrucción clara, detalles de contexto y objetivo, audiencia y formato de salida. Dados desde el principio, el modelo planifica mejor la respuesta. Corregir después agranda el contexto y tiende a arrastrar el error inicial.

</details>

<details><summary>8. ¿En qué se diferencian zero-shot, one-shot y few-shot?</summary>

En la cantidad de ejemplos que se le dan al modelo dentro del prompt (ninguno, uno o varios) para que infiera el patrón de la respuesta. Es aprendizaje en contexto: el modelo no se reentrena, solo usa los ejemplos de esa conversación.

</details>

## Conexiones

- **Clase 11**: la base técnica de esta clase son los embeddings, el Transformer con atención y la generación token a token con temperatura. Acá se ven sus consecuencias de uso: predicción probabilística, alucinaciones y límites de contexto.
- **Clase 12**: las redes convolucionales para imágenes son el antecedente de lo que hoy hacen los modelos multimodales al "leer" una imagen.
- **Clases 7 a 10**: las redes neuronales y su entrenamiento, que la IA generativa lleva a una escala mucho mayor de parámetros y datos.
- **Clase 14 y TFI**: el profesor recomendó NotebookLM para armar informes y estudiar documentos del trabajo final. Para el enfoque **generativo** de normalización de nombres comerciales en Benefix aplican directamente varias ideas de esta clase: el few-shot para mostrar el formato del nombre normalizado, el riesgo de que el modelo **alucine** un nombre canónico que no existe en el catálogo (de ahí la idea de restringirlo a un contexto o una documentación) y el costo por token de cada consulta. Esos errores son los que después se miden con Hit@k y MRR frente a los enfoques léxico, semántico e híbrido.
- **Módulos de NLP e IA generativa**: el profesor anticipó que se profundizará en procesamiento de lenguaje natural y en agentes.
