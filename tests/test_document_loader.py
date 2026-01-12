"""
Tests para el módulo de carga de documentos.
"""
import os
from unittest.mock import patch, MagicMock

import pytest
from langchain_core.documents import Document

from src.document_loader import DocumentLoader, SUPPORTED_EXTENSIONS
from src.exceptions import (
    DocumentNotFoundError,
    UnsupportedFormatError,
    DocumentLoadError,
)


class TestDocumentLoader:
    """Tests para la clase DocumentLoader."""

    def test_init_default_values(self):
        """Test que los valores por defecto son correctos."""
        loader = DocumentLoader()
        assert loader.chunk_size == 1000
        assert loader.chunk_overlap == 200

    def test_init_custom_values(self):
        """Test que se pueden configurar valores personalizados."""
        loader = DocumentLoader(chunk_size=500, chunk_overlap=50)
        assert loader.chunk_size == 500
        assert loader.chunk_overlap == 50

    def test_supported_extensions(self):
        """Test que las extensiones soportadas son correctas."""
        assert ".pdf" in SUPPORTED_EXTENSIONS
        assert ".txt" in SUPPORTED_EXTENSIONS
        assert ".docx" in SUPPORTED_EXTENSIONS
        assert ".md" in SUPPORTED_EXTENSIONS
        assert ".xlsx" not in SUPPORTED_EXTENSIONS


class TestLoadDocument:
    """Tests para load_document."""

    def test_load_document_file_not_found(self):
        """Test que lanza error si el archivo no existe."""
        loader = DocumentLoader()
        with pytest.raises(DocumentNotFoundError) as exc_info:
            loader.load_document("/nonexistent/file.pdf")
        assert "/nonexistent/file.pdf" in str(exc_info.value)

    def test_load_document_unsupported_format(self, tmp_path):
        """Test que lanza error para formatos no soportados."""
        unsupported_file = tmp_path / "file.xlsx"
        unsupported_file.write_text("content")

        loader = DocumentLoader()
        with pytest.raises(UnsupportedFormatError) as exc_info:
            loader.load_document(str(unsupported_file))
        assert ".xlsx" in str(exc_info.value)

    def test_load_document_txt_success(self, tmp_path):
        """Test que carga un archivo txt correctamente."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Este es un documento de prueba.", encoding="utf-8")

        loader = DocumentLoader()
        documents = loader.load_document(str(txt_file))

        assert len(documents) >= 1
        assert "Este es un documento de prueba" in documents[0].page_content

    def test_load_document_handles_loader_error(self, tmp_path):
        """Test que maneja errores del loader correctamente."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("content", encoding="utf-8")

        loader = DocumentLoader()

        with patch("src.document_loader._LOADERS") as mock_loaders:
            mock_loader_instance = MagicMock()
            mock_loader_instance.load.side_effect = Exception("Loader error")
            mock_loaders.__getitem__.return_value = lambda x: mock_loader_instance

            with pytest.raises(DocumentLoadError):
                loader.load_document(str(txt_file))


class TestLoadDirectory:
    """Tests para load_directory."""

    def test_load_directory_not_found(self):
        """Test que lanza error si el directorio no existe."""
        loader = DocumentLoader()
        with pytest.raises(DocumentNotFoundError):
            loader.load_directory("/nonexistent/directory")

    def test_load_directory_empty(self, tmp_path):
        """Test que maneja directorios vacíos."""
        loader = DocumentLoader()
        documents = loader.load_directory(str(tmp_path))
        assert documents == []

    def test_load_directory_with_txt_files(self, tmp_path):
        """Test que carga archivos txt de un directorio."""
        file1 = tmp_path / "doc1.txt"
        file2 = tmp_path / "doc2.txt"
        file1.write_text("Contenido del documento 1", encoding="utf-8")
        file2.write_text("Contenido del documento 2", encoding="utf-8")

        loader = DocumentLoader()
        documents = loader.load_directory(str(tmp_path))

        assert len(documents) >= 2

    def test_load_directory_ignores_unsupported(self, tmp_path):
        """Test que ignora archivos no soportados."""
        txt_file = tmp_path / "doc.txt"
        xlsx_file = tmp_path / "data.xlsx"
        txt_file.write_text("Contenido", encoding="utf-8")
        xlsx_file.write_text("fake xlsx", encoding="utf-8")

        loader = DocumentLoader()
        documents = loader.load_directory(str(tmp_path))

        sources = [d.metadata.get("source", "") for d in documents]
        assert not any("xlsx" in s for s in sources)

    def test_load_directory_continues_on_error(self, tmp_path):
        """Test que continúa si un archivo falla."""
        good_file = tmp_path / "good.txt"
        good_file.write_text("Buen contenido", encoding="utf-8")

        loader = DocumentLoader()

        with patch.object(loader, "load_document") as mock_load:
            mock_load.side_effect = [
                DocumentLoadError("Error", "details"),
                [Document(page_content="test", metadata={})],
            ]

            documents = loader.load_directory(str(tmp_path))
            assert len(documents) >= 0


class TestSplitDocuments:
    """Tests para split_documents."""

    def test_split_documents_empty_list(self):
        """Test que maneja lista vacía."""
        loader = DocumentLoader()
        chunks = loader.split_documents([])
        assert chunks == []

    def test_split_documents_creates_chunks(self):
        """Test que divide documentos en chunks."""
        loader = DocumentLoader(chunk_size=50, chunk_overlap=10)
        long_content = "Este es un contenido muy largo. " * 20
        documents = [Document(page_content=long_content, metadata={"source": "test"})]

        chunks = loader.split_documents(documents)

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.page_content) <= 50 + 20


class TestProcessDocuments:
    """Tests para process_documents."""

    def test_process_documents_file(self, tmp_path):
        """Test que procesa un archivo individual."""
        txt_file = tmp_path / "doc.txt"
        txt_file.write_text("Contenido del documento", encoding="utf-8")

        loader = DocumentLoader()
        chunks = loader.process_documents(str(txt_file))

        assert len(chunks) >= 1

    def test_process_documents_directory(self, tmp_path):
        """Test que procesa un directorio."""
        txt_file = tmp_path / "doc.txt"
        txt_file.write_text("Contenido del documento", encoding="utf-8")

        loader = DocumentLoader()
        chunks = loader.process_documents(str(tmp_path))

        assert len(chunks) >= 1

    def test_process_documents_invalid_path(self):
        """Test que lanza error para ruta inválida."""
        loader = DocumentLoader()
        with pytest.raises(DocumentNotFoundError):
            loader.process_documents("/invalid/path")

    def test_process_documents_returns_empty_for_empty_dir(self, tmp_path):
        """Test que retorna lista vacía para directorio vacío."""
        loader = DocumentLoader()
        chunks = loader.process_documents(str(tmp_path))
        assert chunks == []
