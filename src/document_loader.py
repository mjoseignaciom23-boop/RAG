"""
Módulo para cargar documentos de diferentes formatos.
Soporta: PDF, TXT, DOCX, MD
"""
import os
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.exceptions import (
    DocumentLoadError,
    DocumentNotFoundError,
    UnsupportedFormatError,
)
from src.logger import get_logger

logger = get_logger("rag.document_loader")

SUPPORTED_EXTENSIONS = frozenset({".pdf", ".txt", ".docx", ".md"})

_LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": lambda path: TextLoader(path, encoding="utf-8"),
    ".docx": Docx2txtLoader,
    ".md": UnstructuredMarkdownLoader,
}


class DocumentLoader:
    """Clase para cargar y procesar documentos de múltiples formatos."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Inicializa el cargador de documentos.

        Args:
            chunk_size: Tamaño de cada chunk de texto
            chunk_overlap: Solapamiento entre chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def load_document(self, file_path: str) -> list[Document]:
        """
        Carga un documento según su extensión.

        Args:
            file_path: Ruta al archivo

        Returns:
            Lista de documentos cargados

        Raises:
            DocumentNotFoundError: Si el archivo no existe
            UnsupportedFormatError: Si el formato no es soportado
            DocumentLoadError: Si hay error al cargar el documento
        """
        if not os.path.exists(file_path):
            raise DocumentNotFoundError(file_path)

        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFormatError(file_extension)

        try:
            loader_factory = _LOADERS[file_extension]
            loader = loader_factory(file_path)
            documents = loader.load()
            logger.info(
                f"Cargado: {os.path.basename(file_path)} "
                f"({len(documents)} paginas/secciones)"
            )
            return documents

        except (DocumentNotFoundError, UnsupportedFormatError):
            raise
        except Exception as e:
            raise DocumentLoadError(
                message=f"Error cargando documento",
                details=f"{file_path}: {str(e)}",
            ) from e

    def load_directory(self, directory_path: str) -> list[Document]:
        """
        Carga todos los documentos soportados de un directorio.

        Args:
            directory_path: Ruta al directorio

        Returns:
            Lista de todos los documentos cargados

        Raises:
            DocumentNotFoundError: Si el directorio no existe
        """
        if not os.path.exists(directory_path):
            raise DocumentNotFoundError(directory_path)

        all_documents: list[Document] = []
        errors: list[str] = []

        logger.info(f"Cargando documentos desde: {directory_path}")

        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)

            if not os.path.isfile(file_path):
                continue

            file_extension = os.path.splitext(filename)[1].lower()

            if file_extension not in SUPPORTED_EXTENSIONS:
                continue

            try:
                documents = self.load_document(file_path)
                all_documents.extend(documents)
            except DocumentLoadError as e:
                errors.append(str(e))
                logger.warning(f"Error cargando {filename}: {e}")

        if errors:
            logger.warning(f"Se encontraron {len(errors)} errores durante la carga")

        if not all_documents:
            logger.warning("No se encontraron documentos soportados")
        else:
            logger.info(f"Total de documentos cargados: {len(all_documents)}")

        return all_documents

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """
        Divide los documentos en chunks más pequeños.

        Args:
            documents: Lista de documentos a dividir

        Returns:
            Lista de documentos divididos en chunks
        """
        logger.info("Dividiendo documentos en chunks...")
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Documentos divididos en {len(chunks)} chunks")
        return chunks

    def process_documents(self, source_path: str) -> list[Document]:
        """
        Procesa documentos desde un archivo o directorio.

        Args:
            source_path: Ruta al archivo o directorio

        Returns:
            Lista de chunks procesados

        Raises:
            DocumentNotFoundError: Si la ruta no existe
        """
        if os.path.isfile(source_path):
            documents = self.load_document(source_path)
        elif os.path.isdir(source_path):
            documents = self.load_directory(source_path)
        else:
            raise DocumentNotFoundError(source_path)

        if not documents:
            return []

        return self.split_documents(documents)
