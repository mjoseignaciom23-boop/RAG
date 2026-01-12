"""
Módulo para gestionar la base de datos vectorial usando ChromaDB.
"""
import os
import shutil
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from src.embeddings import LocalEmbeddings
from src.exceptions import (
    VectorStoreError,
    VectorStoreNotInitializedError,
    VectorStoreEmptyError,
)
from src.logger import get_logger
from src.interfaces import EmbeddingsInterface

logger = get_logger("rag.vector_store")


class VectorStore:
    """Clase para gestionar el almacenamiento vectorial con ChromaDB."""

    def __init__(
        self,
        persist_directory: str = "./vectorstore",
        embeddings: EmbeddingsInterface | None = None,
    ):
        """
        Inicializa el vector store.

        Args:
            persist_directory: Directorio donde se persistirá la base de datos
            embeddings: Modelo de embeddings (inyectable para testing)
        """
        self.persist_directory = persist_directory
        self._embeddings = embeddings
        self._vectorstore: Chroma | None = None

    @property
    def embeddings(self) -> EmbeddingsInterface:
        """Lazy loading de embeddings."""
        if self._embeddings is None:
            self._embeddings = LocalEmbeddings()
        return self._embeddings

    def is_initialized(self) -> bool:
        """Verifica si el vector store está inicializado."""
        return self._vectorstore is not None

    def create_vectorstore(self, documents: list[Document]) -> None:
        """
        Crea una nueva base de datos vectorial desde documentos.

        Args:
            documents: Lista de documentos a indexar

        Raises:
            VectorStoreEmptyError: Si no hay documentos para indexar
            VectorStoreError: Si hay error al crear el vector store
        """
        if not documents:
            raise VectorStoreEmptyError()

        try:
            logger.info("Creando base de datos vectorial...")
            logger.info(f"Procesando {len(documents)} chunks...")

            self._vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
            )

            logger.info(
                f"Base de datos vectorial creada y guardada en: {self.persist_directory}"
            )

        except VectorStoreEmptyError:
            raise
        except Exception as e:
            raise VectorStoreError(
                message="Error creando vector store",
                details=str(e),
            ) from e

    def load_vectorstore(self) -> bool:
        """
        Carga una base de datos vectorial existente.

        Returns:
            True si se cargó correctamente, False si no existe

        Raises:
            VectorStoreError: Si hay error al cargar
        """
        if not os.path.exists(self.persist_directory):
            logger.warning(
                f"No existe una base de datos vectorial en: {self.persist_directory}"
            )
            return False

        try:
            logger.info(
                f"Cargando base de datos vectorial desde: {self.persist_directory}"
            )
            self._vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
            )
            logger.info("Base de datos vectorial cargada correctamente")
            return True

        except Exception as e:
            raise VectorStoreError(
                message="Error cargando vector store",
                details=str(e),
            ) from e

    def similarity_search(
        self,
        query: str,
        k: int = 4,
    ) -> list[tuple[Document, float]]:
        """
        Realiza una búsqueda por similitud en el vector store.

        Args:
            query: Consulta a buscar
            k: Número de resultados a retornar

        Returns:
            Lista de tuplas (documento, score de similitud)

        Raises:
            VectorStoreNotInitializedError: Si el vector store no está inicializado
        """
        if not self.is_initialized():
            raise VectorStoreNotInitializedError()

        results = self._vectorstore.similarity_search_with_score(query, k=k)
        return results

    def get_retriever(self, k: int = 4):
        """
        Obtiene un retriever para usar con LangChain.

        Args:
            k: Número de documentos a recuperar

        Returns:
            Retriever de LangChain

        Raises:
            VectorStoreNotInitializedError: Si el vector store no está inicializado
        """
        if not self.is_initialized():
            raise VectorStoreNotInitializedError()

        return self._vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    def delete_vectorstore(self) -> None:
        """Elimina la base de datos vectorial."""
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
            logger.info(f"Base de datos eliminada: {self.persist_directory}")
        else:
            logger.warning(f"No existe base de datos en: {self.persist_directory}")

        self._vectorstore = None
