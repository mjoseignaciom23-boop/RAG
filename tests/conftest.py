"""
Configuración y fixtures compartidas para pytest.
"""
import os
import sys
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def mock_settings():
    """Settings mockeado para tests."""
    from src.config import Settings
    from unittest.mock import patch

    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key-12345"}):
        settings = Settings(
            openai_api_key="sk-test-key-12345",
            llm_model_name="gpt-3.5-turbo",
            llm_temperature=0.7,
            vectorstore_path="./test_vectorstore",
            chunk_size=500,
            chunk_overlap=100,
            retrieval_k=3,
        )
        yield settings


@pytest.fixture
def sample_documents():
    """Documentos de ejemplo para tests."""
    return [
        Document(
            page_content="Este es el contenido del primer documento sobre Python.",
            metadata={"source": "/path/to/doc1.pdf", "page": 1},
        ),
        Document(
            page_content="Segundo documento que habla sobre machine learning.",
            metadata={"source": "/path/to/doc2.pdf", "page": 1},
        ),
        Document(
            page_content="Tercer documento con información sobre RAG systems.",
            metadata={"source": "/path/to/doc3.txt", "page": "N/A"},
        ),
    ]


@pytest.fixture
def mock_vector_store():
    """VectorStore mockeado."""
    mock = MagicMock()
    mock.is_initialized.return_value = True
    mock.create_vectorstore.return_value = None
    mock.load_vectorstore.return_value = True
    mock.delete_vectorstore.return_value = None
    return mock


@pytest.fixture
def mock_document_loader(sample_documents):
    """DocumentLoader mockeado."""
    mock = MagicMock()
    mock.process_documents.return_value = sample_documents
    mock.load_document.return_value = sample_documents[:1]
    mock.load_directory.return_value = sample_documents
    mock.split_documents.return_value = sample_documents
    return mock


@pytest.fixture
def mock_llm():
    """LLM mockeado."""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content="Esta es una respuesta de prueba.")
    return mock


@pytest.fixture
def mock_embeddings():
    """Embeddings mockeados."""
    mock = MagicMock()
    mock.embed_documents.return_value = [[0.1] * 384 for _ in range(3)]
    mock.embed_query.return_value = [0.1] * 384
    return mock
