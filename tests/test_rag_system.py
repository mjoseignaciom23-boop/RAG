"""
Tests para el sistema RAG principal.
"""
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.rag_system import RAGSystem
from src.models import QueryResult, SourceDocument
from src.exceptions import (
    VectorStoreNotInitializedError,
    QueryError,
    IndexError as RAGIndexError,
)


class TestRAGSystemInit:
    """Tests para la inicialización del RAGSystem."""

    def test_init_with_all_dependencies(
        self, mock_settings, mock_vector_store, mock_document_loader, mock_llm
    ):
        """Test que se inicializa correctamente con todas las dependencias."""
        rag = RAGSystem(
            settings=mock_settings,
            vector_store=mock_vector_store,
            document_loader=mock_document_loader,
            llm=mock_llm,
        )

        assert rag._settings == mock_settings
        assert rag._vector_store == mock_vector_store
        assert rag._document_loader == mock_document_loader
        assert rag._llm == mock_llm

    def test_init_creates_default_llm(
        self, mock_settings, mock_vector_store, mock_document_loader
    ):
        """Test que crea un LLM por defecto si no se provee."""
        with patch("src.rag_system.ChatOpenAI") as MockChatOpenAI:
            mock_llm_instance = MagicMock()
            MockChatOpenAI.return_value = mock_llm_instance

            rag = RAGSystem(
                settings=mock_settings,
                vector_store=mock_vector_store,
                document_loader=mock_document_loader,
            )

            MockChatOpenAI.assert_called_once_with(
                model_name=mock_settings.llm_model_name,
                temperature=mock_settings.llm_temperature,
                openai_api_key=mock_settings.openai_api_key,
            )

    def test_vector_store_property(
        self, mock_settings, mock_vector_store, mock_document_loader, mock_llm
    ):
        """Test que la propiedad vector_store funciona."""
        rag = RAGSystem(
            settings=mock_settings,
            vector_store=mock_vector_store,
            document_loader=mock_document_loader,
            llm=mock_llm,
        )

        assert rag.vector_store == mock_vector_store


class TestIndexDocuments:
    """Tests para index_documents."""

    def test_index_documents_success(
        self,
        mock_settings,
        mock_vector_store,
        mock_document_loader,
        mock_llm,
        sample_documents,
    ):
        """Test que indexa documentos correctamente."""
        mock_document_loader.process_documents.return_value = sample_documents

        rag = RAGSystem(
            settings=mock_settings,
            vector_store=mock_vector_store,
            document_loader=mock_document_loader,
            llm=mock_llm,
        )

        result = rag.index_documents("/path/to/docs")

        assert result is True
        mock_document_loader.process_documents.assert_called_once_with("/path/to/docs")
        mock_vector_store.create_vectorstore.assert_called_once_with(sample_documents)

    def test_index_documents_no_documents(
        self, mock_settings, mock_vector_store, mock_document_loader, mock_llm
    ):
        """Test que retorna False si no hay documentos."""
        mock_document_loader.process_documents.return_value = []

        rag = RAGSystem(
            settings=mock_settings,
            vector_store=mock_vector_store,
            document_loader=mock_document_loader,
            llm=mock_llm,
        )

        result = rag.index_documents("/empty/path")

        assert result is False
        mock_vector_store.create_vectorstore.assert_not_called()

    def test_index_documents_raises_on_error(
        self, mock_settings, mock_vector_store, mock_document_loader, mock_llm
    ):
        """Test que lanza RAGIndexError en caso de error."""
        mock_document_loader.process_documents.side_effect = Exception("Load error")

        rag = RAGSystem(
            settings=mock_settings,
            vector_store=mock_vector_store,
            document_loader=mock_document_loader,
            llm=mock_llm,
        )

        with pytest.raises(RAGIndexError):
            rag.index_documents("/path/to/docs")


class TestLoadExistingIndex:
    """Tests para load_existing_index."""

    def test_load_existing_index_success(
        self, mock_settings, mock_vector_store, mock_document_loader, mock_llm
    ):
        """Test que carga un índice existente."""
        mock_vector_store.load_vectorstore.return_value = True

        rag = RAGSystem(
            settings=mock_settings,
            vector_store=mock_vector_store,
            document_loader=mock_document_loader,
            llm=mock_llm,
        )

        result = rag.load_existing_index()

        assert result is True
        mock_vector_store.load_vectorstore.assert_called_once()

    def test_load_existing_index_not_found(
        self, mock_settings, mock_vector_store, mock_document_loader, mock_llm
    ):
        """Test que retorna False si no hay índice."""
        mock_vector_store.load_vectorstore.return_value = False

        rag = RAGSystem(
            settings=mock_settings,
            vector_store=mock_vector_store,
            document_loader=mock_document_loader,
            llm=mock_llm,
        )

        result = rag.load_existing_index()

        assert result is False


class TestQuery:
    """Tests para query."""

    def test_query_raises_if_not_initialized(
        self, mock_settings, mock_vector_store, mock_document_loader, mock_llm
    ):
        """Test que lanza error si el vector store no está inicializado."""
        mock_vector_store.is_initialized.return_value = False

        rag = RAGSystem(
            settings=mock_settings,
            vector_store=mock_vector_store,
            document_loader=mock_document_loader,
            llm=mock_llm,
        )

        with pytest.raises(VectorStoreNotInitializedError):
            rag.query("Test question")

    def test_query_success(
        self,
        mock_settings,
        mock_vector_store,
        mock_document_loader,
        mock_llm,
        sample_documents,
    ):
        """Test que realiza una consulta correctamente."""
        mock_vector_store.is_initialized.return_value = True
        mock_vector_store.similarity_search.return_value = [
            (sample_documents[0], 0.2),
            (sample_documents[1], 0.3),
        ]

        rag = RAGSystem(
            settings=mock_settings,
            vector_store=mock_vector_store,
            document_loader=mock_document_loader,
            llm=mock_llm,
        )

        # Crear un chain mock que devuelve string al invocar
        class MockChain:
            def invoke(self, *args, **kwargs):
                return "Respuesta de prueba"

            def __or__(self, other):
                return self

            def __ror__(self, other):
                return self

        mock_chain = MockChain()

        with patch("src.rag_system.StrOutputParser", return_value=mock_chain):
            with patch("src.rag_system.RunnablePassthrough", return_value=mock_chain):
                with patch("src.rag_system.ChatPromptTemplate") as mock_prompt_cls:
                    mock_prompt_cls.from_template.return_value = mock_chain
                    rag._prompt_template = mock_chain
                    rag._llm = mock_chain

                    result = rag.query("Pregunta de test")

        assert isinstance(result, QueryResult)
        assert result.query == "Pregunta de test"
        assert result.answer == "Respuesta de prueba"
        mock_vector_store.similarity_search.assert_called_once()

    def test_query_uses_default_k(
        self, mock_settings, mock_vector_store, mock_document_loader, mock_llm
    ):
        """Test que usa el k por defecto de settings."""
        mock_vector_store.is_initialized.return_value = True
        mock_vector_store.similarity_search.return_value = []

        rag = RAGSystem(
            settings=mock_settings,
            vector_store=mock_vector_store,
            document_loader=mock_document_loader,
            llm=mock_llm,
        )

        with pytest.raises(Exception):
            rag.query("Test")

        mock_vector_store.similarity_search.assert_called_with(
            "Test", k=mock_settings.retrieval_k
        )

    def test_query_uses_custom_k(
        self, mock_settings, mock_vector_store, mock_document_loader, mock_llm
    ):
        """Test que acepta k personalizado."""
        mock_vector_store.is_initialized.return_value = True
        mock_vector_store.similarity_search.return_value = []

        rag = RAGSystem(
            settings=mock_settings,
            vector_store=mock_vector_store,
            document_loader=mock_document_loader,
            llm=mock_llm,
        )

        with pytest.raises(Exception):
            rag.query("Test", k=10)

        mock_vector_store.similarity_search.assert_called_with("Test", k=10)


class TestDeleteIndex:
    """Tests para delete_index."""

    def test_delete_index(
        self, mock_settings, mock_vector_store, mock_document_loader, mock_llm
    ):
        """Test que elimina el índice."""
        rag = RAGSystem(
            settings=mock_settings,
            vector_store=mock_vector_store,
            document_loader=mock_document_loader,
            llm=mock_llm,
        )

        rag.delete_index()

        mock_vector_store.delete_vectorstore.assert_called_once()


class TestFormatContext:
    """Tests para _format_context."""

    def test_format_context(
        self,
        mock_settings,
        mock_vector_store,
        mock_document_loader,
        mock_llm,
        sample_documents,
    ):
        """Test que formatea el contexto correctamente."""
        rag = RAGSystem(
            settings=mock_settings,
            vector_store=mock_vector_store,
            document_loader=mock_document_loader,
            llm=mock_llm,
        )

        context = rag._format_context(sample_documents)

        assert "[Documento 1]" in context
        assert "[Documento 2]" in context
        assert "doc1.pdf" in context
        assert "Python" in context


class TestExtractSources:
    """Tests para _extract_sources."""

    def test_extract_sources(
        self,
        mock_settings,
        mock_vector_store,
        mock_document_loader,
        mock_llm,
        sample_documents,
    ):
        """Test que extrae las fuentes correctamente."""
        rag = RAGSystem(
            settings=mock_settings,
            vector_store=mock_vector_store,
            document_loader=mock_document_loader,
            llm=mock_llm,
        )

        scores = [0.2, 0.3, 0.4]
        sources = rag._extract_sources(sample_documents, scores)

        assert len(sources) == 3
        assert all(isinstance(s, SourceDocument) for s in sources)
        assert sources[0].file_name == "doc1.pdf"
        assert sources[0].similarity_score == pytest.approx(0.8, rel=0.01)

    def test_extract_sources_deduplicates(
        self,
        mock_settings,
        mock_vector_store,
        mock_document_loader,
        mock_llm,
    ):
        """Test que elimina fuentes duplicadas."""
        rag = RAGSystem(
            settings=mock_settings,
            vector_store=mock_vector_store,
            document_loader=mock_document_loader,
            llm=mock_llm,
        )

        duplicate_docs = [
            Document(
                page_content="Content 1",
                metadata={"source": "/path/doc.pdf", "page": 1},
            ),
            Document(
                page_content="Content 2",
                metadata={"source": "/path/doc.pdf", "page": 1},
            ),
        ]

        sources = rag._extract_sources(duplicate_docs, [0.2, 0.3])

        assert len(sources) == 1
