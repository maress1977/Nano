# Módulo 4 · Procesamiento de Lenguaje Natural

Horacio Arrigo · 23 de junio al 23 de julio de 2026

| Clase | Fecha | Tema | Apunte |
|---|---|---|---|
| 23 | mar 23-jun | Qué es el NLP, preprocesamiento y análisis de sentimiento básico | [Introducción y preprocesamiento](clase-23-introduccion-nlp-preprocesamiento.md) |
| 24 | jue 25-jun | Bag of Words, TF-IDF, clasificación de textos y topic modeling | [Espacios vectoriales](clase-24-espacios-vectoriales-bow-tfidf-topic-modeling.md) |
| 25 | mar 30-jun | Word2Vec, GloVe y FastText | [Word embeddings](clase-25-word-embeddings-word2vec-glove-fasttext.md) |
| 26 | jue 02-jul | RNN, LSTM, GRU y la llegada de los Transformers | [Redes recurrentes y Transformers](clase-26-rnn-lstm-gru-transformers.md) |
| 27 | mar 07-jul | Hugging Face, model cards y traducción automática | [Hugging Face y traducción](clase-27-hugging-face-traduccion-automatica.md) |
| 28 | mar 14-jul | BERT, embeddings contextuales, fine-tuning y BERTopic | [Encoders tipo BERT](clase-28-bert-encoders-fine-tuning-topic-modeling.md) |
| 29 | jue 16-jul | Decoders tipo GPT, sampling, prompting y LLMs locales | [Decoders tipo GPT](clase-29-decoders-gpt-prompting-llm-locales.md) |
| 30 | mar 21-jul | Bake-off de modelos de sentimiento y tareas NLP con LLM y Pydantic | [Bake-off y salida estructurada](clase-30-bakeoff-sentimiento-llm-pydantic.md) |
| 31 | jue 23-jul | Ciclo de vida, inferencia batch o en tiempo real, containers y APIs | [Modelos en producción](clase-31-modelos-en-produccion-repaso.md) |

## El hilo del módulo

1. **Del texto a números.** La clase 23 limpia y tokeniza el texto. La clase 24 lo representa con conteos (BoW, TF-IDF) y la clase 25 con embeddings densos que capturan significado.
2. **Modelos que leen en orden.** La clase 26 pasa de redes recurrentes a Transformers, la arquitectura que domina el resto del módulo.
3. **Las familias de Transformers.** La clase 27 usa modelos preentrenados del Hub (encoder-decoder para traducir), la clase 28 los encoders tipo BERT y la clase 29 los decoders tipo GPT.
4. **Comparar y llevar a producción.** La clase 30 compara modelos clásicos, embeddings, BERT y LLMs sobre la misma tarea: un LLM no siempre gana. La clase 31 muestra cómo se sirve un modelo y repasa el módulo.

Para el TFI, las clases 24 (TF-IDF, base del retrieval léxico), 28 (sentence embeddings, base del semántico) y 30 (comparar variantes con las mismas métricas) son las más directas.

## Notas sobre las fuentes

- **Numeración:** las láminas y notebooks usan la numeración del curso de NLP. Las clases 23 a 31 son las "Clase 01" a "Clase 09" del curso.
- **Clase 27:** la carpeta trae también notebooks de las clases 07 y 08 del curso; el apunte usa solo el material de la clase 05.
- **Clase 30:** fue casi toda trabajo en grupos. La teoría de Pydantic y log-probs sale del notebook, y las cifras del bake-off salen de la corrida guardada por los docentes.
- **Imágenes:** las de `img/` vienen de la carpeta `_assets` del curso.
