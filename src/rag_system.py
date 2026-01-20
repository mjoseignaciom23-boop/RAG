"""Sistema RAG principal."""
import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.config import Settings
from src.models import QueryResult, SourceDocument
from src.exceptions import VectorStoreNotInitializedError, QueryError, IndexError as RAGIndexError
from src.logger import get_logger
from src.interfaces import VectorStoreInterface, DocumentLoaderInterface

logger = get_logger("rag.system")

PROMPT_TEMPLATE = """Responde usando solo el contexto.
Contexto: {context}
Pregunta: {question}
Si no sabes, dilo. Sé conciso y cita fuentes."""

class RAGSystem:
    """Orquestador del flujo RAG."""

    def __init__(self, settings: Settings, vector_store: VectorStoreInterface, document_loader: DocumentLoaderInterface, llm=None):
        self._settings = settings
        self._vector_store = vector_store
        self._document_loader = document_loader
        self._llm = llm or ChatOllama(
            model=settings.llm_model_name,
            temperature=settings.llm_temperature,
            base_url=settings.ollama_base_url,
        )
        self._prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    @property
    def vector_store(self):
        return self._vector_store

    def index_documents(self, source_path: str) -> bool:
        """Procesa, divide e indexa documentos."""
        try:
            logger.info("Iniciando indexación...")
            chunks = self._document_loader.process_documents(source_path)
            if not chunks: return False
            self._vector_store.create_vectorstore(chunks)
            logger.info("Indexación completada")
            return True
        except Exception as e:
            raise RAGIndexError("Falló la indexación", str(e)) from e

    def load_existing_index(self) -> bool:
        """Carga el índice del disco."""
        return self._vector_store.load_vectorstore()

    def query(self, question: str, k: int | None = None) -> QueryResult:
        """Consulta el sistema RAG."""
        if not self._vector_store.is_initialized():
            raise VectorStoreNotInitializedError()

        try:
            logger.info("Buscando contexto...")
            docs_scores = self._vector_store.similarity_search(question, k=k or self._settings.retrieval_k)
            docs = [d for d, _ in docs_scores]
            
            chain = (
                {"context": lambda _: self._format(docs), "question": RunnablePassthrough()}
                | self._prompt | self._llm | StrOutputParser()
            )
            
            logger.info("Generando respuesta...")
            return QueryResult(
                answer=chain.invoke(question),
                sources=self._extract_sources(docs, [s for _, s in docs_scores]),
                query=question
            )
        except VectorStoreNotInitializedError: raise
        except Exception as e:
            raise QueryError("Falló la consulta", str(e)) from e

    def _format(self, docs) -> str:
        return "\n".join([f"[Doc {i+1}]: {d.page_content}" for i, d in enumerate(docs)])

    def _extract_sources(self, docs, scores) -> list[SourceDocument]:
        """Extrae metadatos únicos de las fuentes."""
        seen, sources = set(), []
        for doc, score in zip(docs, scores):
            sid = f"{doc.metadata.get('source')}_{doc.metadata.get('page')}"
            if sid not in seen:
                seen.add(sid)
                sources.append(SourceDocument.from_langchain_doc(doc, score))
        return sources

    def delete_index(self) -> None:
        self._vector_store.delete_vectorstore()
