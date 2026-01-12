"""
Sistema RAG principal que integra el vector store y el modelo LLM.
"""
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.config import Settings
from src.models import QueryResult, SourceDocument
from src.exceptions import (
    VectorStoreNotInitializedError,
    QueryError,
    IndexError as RAGIndexError,
)
from src.logger import get_logger
from src.interfaces import (
    VectorStoreInterface,
    DocumentLoaderInterface,
)

logger = get_logger("rag.system")

DEFAULT_PROMPT_TEMPLATE = """
Eres un asistente útil que responde preguntas basándose en el contexto proporcionado.

Contexto:
{context}

Pregunta: {question}

Instrucciones:
1. Responde la pregunta usando SOLO la información del contexto proporcionado.
2. Si la información no está en el contexto, indica claramente que no tienes esa información.
3. Proporciona una respuesta clara y concisa.
4. Cita las fuentes mencionando de qué documento proviene la información.

Respuesta:
"""


class RAGSystem:
    """Sistema RAG completo con consultas y citación de fuentes."""

    def __init__(
        self,
        settings: Settings,
        vector_store: VectorStoreInterface,
        document_loader: DocumentLoaderInterface,
        llm: ChatOpenAI | None = None,
    ):
        """
        Inicializa el sistema RAG con inyección de dependencias.

        Args:
            settings: Configuración de la aplicación
            vector_store: Instancia del vector store
            document_loader: Instancia del cargador de documentos
            llm: Modelo LLM (opcional, se crea uno por defecto si no se provee)
        """
        self._settings = settings
        self._vector_store = vector_store
        self._document_loader = document_loader

        if llm is not None:
            self._llm = llm
        else:
            self._llm = ChatOpenAI(
                model_name=settings.llm_model_name,
                temperature=settings.llm_temperature,
                openai_api_key=settings.openai_api_key,
            )

        self._prompt_template = ChatPromptTemplate.from_template(DEFAULT_PROMPT_TEMPLATE)

    @property
    def vector_store(self) -> VectorStoreInterface:
        """Acceso al vector store para compatibilidad."""
        return self._vector_store

    def index_documents(self, source_path: str) -> bool:
        """
        Indexa documentos en el vector store.

        Args:
            source_path: Ruta al archivo o directorio con documentos

        Returns:
            True si se indexó correctamente

        Raises:
            RAGIndexError: Si hay error durante la indexación
        """
        try:
            logger.info("=" * 60)
            logger.info("INDEXANDO DOCUMENTOS")
            logger.info("=" * 60)

            chunks = self._document_loader.process_documents(source_path)

            if not chunks:
                logger.warning("No se pudieron procesar documentos")
                return False

            self._vector_store.create_vectorstore(chunks)

            logger.info("=" * 60)
            logger.info("INDEXACION COMPLETADA")
            logger.info("=" * 60)
            return True

        except Exception as e:
            raise RAGIndexError(
                message="Error durante la indexacion",
                details=str(e),
            ) from e

    def load_existing_index(self) -> bool:
        """
        Carga un índice existente.

        Returns:
            True si se cargó correctamente
        """
        return self._vector_store.load_vectorstore()

    def query(self, question: str, k: int | None = None) -> QueryResult:
        """
        Realiza una consulta al sistema RAG.

        Args:
            question: Pregunta a consultar
            k: Número de documentos relevantes a recuperar (default: settings.retrieval_k)

        Returns:
            QueryResult con la respuesta y las fuentes

        Raises:
            VectorStoreNotInitializedError: Si no hay índice cargado
            QueryError: Si hay error durante la consulta
        """
        if not self._vector_store.is_initialized():
            raise VectorStoreNotInitializedError()

        k = k or self._settings.retrieval_k

        try:
            logger.info("Buscando informacion relevante...")

            results = self._vector_store.similarity_search(question, k=k)

            documents = [doc for doc, score in results]
            scores = [score for doc, score in results]

            context = self._format_context(documents)

            chain = (
                {"context": lambda x: context, "question": RunnablePassthrough()}
                | self._prompt_template
                | self._llm
                | StrOutputParser()
            )

            logger.info("Generando respuesta...")
            answer = chain.invoke(question)

            sources = self._extract_sources(documents, scores)

            return QueryResult(
                answer=answer,
                sources=sources,
                query=question,
            )

        except VectorStoreNotInitializedError:
            raise
        except Exception as e:
            raise QueryError(
                message="Error durante la consulta",
                details=str(e),
            ) from e

    def _format_context(self, documents: list) -> str:
        """
        Formatea los documentos como contexto para el prompt.

        Args:
            documents: Lista de documentos

        Returns:
            Contexto formateado
        """
        context_parts = []

        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")

            context_parts.append(
                f"[Documento {i}]\n"
                f"Fuente: {os.path.basename(source)}\n"
                f"Pagina: {page}\n"
                f"Contenido: {doc.page_content}\n"
            )

        return "\n".join(context_parts)

    def _extract_sources(
        self,
        documents: list,
        scores: list[float],
    ) -> list[SourceDocument]:
        """
        Extrae información de las fuentes como modelos Pydantic.

        Args:
            documents: Lista de documentos
            scores: Lista de scores de similitud

        Returns:
            Lista de SourceDocument
        """
        sources: list[SourceDocument] = []
        seen_sources: set[str] = set()

        for doc, score in zip(documents, scores):
            source_path = doc.metadata.get("source", "Unknown")
            source_name = os.path.basename(source_path)
            page = doc.metadata.get("page", "N/A")

            source_id = f"{source_name}_{page}"

            if source_id not in seen_sources:
                seen_sources.add(source_id)
                sources.append(
                    SourceDocument.from_langchain_doc(doc, score)
                )

        return sources

    def delete_index(self) -> None:
        """Elimina el índice actual."""
        self._vector_store.delete_vectorstore()
