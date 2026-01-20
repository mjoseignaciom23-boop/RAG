# RAG Local con Ollama

Sistema de RAG (Retrieval Augmented Generation) 100% local y privado. Usa **Ollama** con **Qwen 2.5** para razonamiento y `multilingual-e5` para embeddings.

## Tecnologías

- **LangChain**: Orquestación.
- **ChromaDB**: Base de datos vectorial persistente.
- **Ollama**: Ejecución local de LLMs.
- **Sentence Transformers**: Embeddings en CPU/GPU.

## Requisitos Previs

1. **Instalar Ollama**: [ollama.com](https://ollama.com)
2. **Descargar modelo**:
   ```bash
   ollama pull qwen2.5:7b
   ```

## Instalación

```bash
# Entorno virtual
python -m venv venv
venv\Scripts\activate  # Windows

# Dependencias
pip install -r requirements.txt

# Configuración
cp .env.example .env
# (No hace falta API Key, todo es local)
```

## Uso

```bash
python main.py
```

1. **Indexar**: Lee PDFs/TXT de `data/documentos`.
2. **Consultar**: Pregunta sobre tu información.
