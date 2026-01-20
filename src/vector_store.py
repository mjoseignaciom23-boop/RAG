"""Gestión de ChromaDB."""
import os
import shutil
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from src.embeddings import LocalEmbeddings
from src.exceptions import VectorStoreError, VectorStoreNotInitializedError, VectorStoreEmptyError
from src.logger import get_logger
from src.interfaces import EmbeddingsInterface

logger = get_logger("rag.vector_store")

class VectorStore:
    """Manejo de base de datos vectorial ChromaDB."""

    def __init__(self, persist_directory: str = "./vectorstore", embeddings: EmbeddingsInterface | None = None):
        self.persist_directory = persist_directory
        self._embeddings = embeddings
        self._vectorstore: Chroma | None = None

    @property
    def embeddings(self) -> EmbeddingsInterface:
        if self._embeddings is None:
            self._embeddings = LocalEmbeddings()
        return self._embeddings

    def is_initialized(self) -> bool:
        return self._vectorstore is not None

    def create_vectorstore(self, documents: list[Document]) -> None:
        """Crea y persiste la base de datos vectorial desde documentos."""
        if not documents:
            raise VectorStoreEmptyError()

        try:
            logger.info(f"Creando DB vectorial con {len(documents)} chunks...")
            self._vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
            )
            logger.info(f"Guardado en: {self.persist_directory}")
        except Exception as e:
            raise VectorStoreError("Error creando vector store", str(e)) from e

    def load_vectorstore(self) -> bool:
        """Carga la base de datos existente."""
        if not os.path.exists(self.persist_directory):
            logger.warning(f"No existe DB en: {self.persist_directory}")
            return False

        try:
            logger.info("Cargando DB vectorial...")
            self._vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
            )
            return True
        except Exception as e:
            raise VectorStoreError("Error cargando vector store", str(e)) from e

    def similarity_search(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        """Busca documentos similares a la consulta."""
        if not self.is_initialized():
            raise VectorStoreNotInitializedError()
        return self._vectorstore.similarity_search_with_score(query, k=k)

    def get_retriever(self, k: int = 4):
        """Retorna un objeto retriever de LangChain."""
        if not self.is_initialized():
            raise VectorStoreNotInitializedError()
        return self._vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})

    def delete_vectorstore(self) -> None:
        """Elimina físicamente la base de datos."""
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
            logger.info("Base de datos eliminada")
        self._vectorstore = None
