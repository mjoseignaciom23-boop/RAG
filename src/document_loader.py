"""Carga y procesamiento de documentos (PDF, TXT, DOCX, MD)."""
import os
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.exceptions import DocumentLoadError, DocumentNotFoundError, UnsupportedFormatError
from src.logger import get_logger

logger = get_logger("rag.document_loader")

SUPPORTED = {
    ".pdf": PyPDFLoader,
    ".txt": lambda p: TextLoader(p, encoding="utf-8"),
    ".docx": Docx2txtLoader,
    ".md": UnstructuredMarkdownLoader,
}

class DocumentLoader:
    """Procesa documentos de diversos formatos."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
        )

    def load_document(self, file_path: str) -> list[Document]:
        """Carga un único archivo si el formato es soportado."""
        if not os.path.exists(file_path):
            raise DocumentNotFoundError(file_path)

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in SUPPORTED:
            raise UnsupportedFormatError(ext)

        try:
            loader = SUPPORTED[ext](file_path)
            docs = loader.load()
            logger.info(f"Cargado: {os.path.basename(file_path)} ({len(docs)} págs)")
            return docs
        except (DocumentNotFoundError, UnsupportedFormatError):
            raise
        except Exception as e:
            raise DocumentLoadError(f"Error cargando {file_path}", str(e)) from e

    def load_directory(self, directory_path: str) -> list[Document]:
        """Carga todos los archivos soportados de un directorio."""
        if not os.path.exists(directory_path):
            raise DocumentNotFoundError(directory_path)

        docs = []
        logger.info(f"Escaneando: {directory_path}")

        for f in os.listdir(directory_path):
            full_path = os.path.join(directory_path, f)
            if not os.path.isfile(full_path): continue
            
            try:
                if os.path.splitext(f)[1].lower() in SUPPORTED:
                    docs.extend(self.load_document(full_path))
            except DocumentLoadError as e:
                logger.warning(str(e))

        logger.info(f"Total procesado: {len(docs)} documentos")
        return docs

    def process_documents(self, source_path: str) -> list[Document]:
        """Carga y divide documentos desde archivo o carpeta."""
        if os.path.isfile(source_path):
            raw_docs = self.load_document(source_path)
        elif os.path.isdir(source_path):
            raw_docs = self.load_directory(source_path)
        else:
            raise DocumentNotFoundError(source_path)

        if not raw_docs: return []
        
        chunks = self.splitter.split_documents(raw_docs)
        logger.info(f"Generados {len(chunks)} chunks")
        return chunks
