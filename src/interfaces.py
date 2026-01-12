"""
Interfaces (Protocols) para inyección de dependencias.
Permiten desacoplar las implementaciones concretas y facilitar el testing.
"""
from typing import Protocol, runtime_checkable
from langchain_core.documents import Document


@runtime_checkable
class DocumentLoaderInterface(Protocol):
    """Interface para cargadores de documentos."""

    def load_document(self, file_path: str) -> list[Document]:
        """Carga un documento desde una ruta."""
        ...

    def load_directory(self, directory_path: str) -> list[Document]:
        """Carga todos los documentos de un directorio."""
        ...

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Divide documentos en chunks."""
        ...

    def process_documents(self, source_path: str) -> list[Document]:
        """Procesa documentos desde archivo o directorio."""
        ...


@runtime_checkable
class VectorStoreInterface(Protocol):
    """Interface para vector stores."""

    def create_vectorstore(self, documents: list[Document]) -> None:
        """Crea un vector store desde documentos."""
        ...

    def load_vectorstore(self) -> bool:
        """Carga un vector store existente."""
        ...

    def similarity_search(
        self,
        query: str,
        k: int = 4,
    ) -> list[tuple[Document, float]]:
        """Realiza búsqueda por similitud."""
        ...

    def delete_vectorstore(self) -> None:
        """Elimina el vector store."""
        ...

    def is_initialized(self) -> bool:
        """Verifica si el vector store está inicializado."""
        ...


@runtime_checkable
class EmbeddingsInterface(Protocol):
    """Interface para modelos de embeddings."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Genera embeddings para una lista de textos."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Genera embedding para una consulta."""
        ...


@runtime_checkable
class LLMInterface(Protocol):
    """Interface para modelos de lenguaje."""

    def invoke(self, input: str) -> str:
        """Invoca el LLM con un input."""
        ...
