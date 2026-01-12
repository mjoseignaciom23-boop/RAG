# Sistema RAG - Retrieval Augmented Generation

Sistema de preguntas y respuestas basado en documentos propios. Permite indexar documentos y realizar consultas inteligentes con citación de fuentes.

## Tecnologías

- **Python 3.8+**
- **LangChain** - Orquestación del pipeline RAG
- **ChromaDB** - Base de datos vectorial
- **Sentence Transformers** - Embeddings locales (all-MiniLM-L6-v2)
- **OpenAI GPT-3.5-turbo** - Generación de respuestas

## Formatos Soportados

PDF, TXT, DOCX y Markdown

## Instalación

```bash
# Crear entorno virtual
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt
```

Crear archivo `.env` con tu API key de OpenAI:

```env
OPENAI_API_KEY=tu_api_key_aqui
```

## Uso

```bash
python main.py
```

El menú interactivo permite:

1. **Indexar documentos** - Procesa documentos de `data/documentos/`
2. **Cargar índice existente** - Carga un índice previamente creado
3. **Hacer una consulta** - Realiza preguntas sobre tus documentos
4. **Eliminar índice** - Limpia la base de datos vectorial

## Costos Estimados

- **Embeddings**: Gratuitos (modelo local)
- **ChromaDB**: Gratuito (local)
- **OpenAI API**:
  - GPT-3.5-turbo: ~$0.001 por 1K tokens
  - GPT-4: ~$0.03 por 1K tokens
  - Consulta típica: ~2-3K tokens → $0.002-$0.003 por pregunta
