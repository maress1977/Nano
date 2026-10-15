# Módulo 5 · IA Generativa

Juan Pablo Sokil · 4 al 20 de agosto de 2026 (clases 32 a 37; el cronograma sigue hasta la 41)

| Clase | Fecha | Tema | Apunte |
|---|---|---|---|
| 32 | mar 04-ago | Cómo funciona un LLM: tokens, sampling, fases de entrenamiento, APIs y modelos locales | [Introducción a la IA generativa](clase-32-introduccion-ia-generativa-llms.md) |
| 33 | jue 06-ago | Prompt engineering, RAG y fine-tuning (LoRA, QLoRA) | [Tres formas de ajustar un LLM](clase-33-prompt-engineering-rag-fine-tuning.md) |
| 34 | mar 11-ago | LangChain, chunking, bases vectoriales y RAG básico | [LangChain y RAG básico](clase-34-langchain-rag-basico.md) |
| 35 | jue 13-ago | Búsqueda híbrida con RRF, reranking, evaluación de retrievers e introducción a agentes | [RAG avanzado y evaluación](clase-35-rag-avanzado-evaluacion-intro-agentes.md) |
| 36 | mar 18-ago | Agentes: tools, memoria, planificación y ReAct | [Agentes y ReAct](clase-36-agentes-tools-react.md) |
| 37 | jue 20-ago | LangGraph, agente que enruta y skills | [LangGraph y skills](clase-37-langgraph-y-skills.md) |

## El hilo del módulo

1. **Qué es un LLM.** La clase 32 explica cómo genera texto (siguiente token, temperatura) y cómo se usa por API o en local.
2. **Cómo adaptarlo.** La clase 33 compara las tres palancas: mejorar el prompt, darle contexto con RAG o reentrenarlo con fine-tuning.
3. **RAG en serio.** La clase 34 arma un RAG básico con LangChain. La clase 35 lo mejora (búsqueda híbrida, reranking) y, sobre todo, enseña a **medirlo** con Hit@k y MRR cambiando una cosa por vez.
4. **Agentes.** La clase 36 le da herramientas al modelo con el loop ReAct. La clase 37 orquesta agentes con LangGraph y empaqueta conocimiento en skills.

**Para el TFI**, la clase 35 es la base metodológica del experimento: comparar retrievers léxico, semántico e híbrido con RRF (k=60) sobre el mismo conjunto de evaluación y las mismas métricas.

## Notas sobre las fuentes

- **Numeración:** las láminas usan la del curso de IA Generativa. Las clases 32 a 35 son las "Clase 01" a "Clase 04". La clase 36 usa las láminas "Clase 06" y la 37 las "Clase 07": el docente adelantó agentes y pasó la clase teórica que correspondía a la "Clase 05" para el final del módulo.
- **Clase 35:** usa también las láminas de RAG avanzado de la carpeta de la clase 34 y las primeras láminas de agentes de la carpeta de la clase 36. La parte sobre cuándo conviene armar un RAG propio no tiene láminas y sale solo de la transcripción.
- **Partes no dadas en clase:** n8n y producción (FastAPI, Docker) quedaron como lectura y figuran como material no visto.
- **Imágenes:** las de `img/` vienen de la carpeta `_assets` del curso.
