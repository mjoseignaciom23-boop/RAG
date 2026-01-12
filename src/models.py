"""
Modelos Pydantic (DTOs) para el sistema RAG.
"""
from pydantic import BaseModel, Field


class SourceDocument(BaseModel):
    """Representa un documento fuente o chunk recuperado."""

    file_name: str = Field(description="Nombre del archivo fuente")
    page: int | str = Field(description="Número de página o 'N/A'")
    content: str = Field(description="Contenido del chunk")
    similarity_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Score de similitud (0-1)",
    )
    preview: str = Field(description="Vista previa del contenido")

    @classmethod
    def from_langchain_doc(
        cls,
        doc,
        score: float,
        preview_length: int = 150,
    ) -> "SourceDocument":
        """
        Crea un SourceDocument desde un documento de LangChain.

        Args:
            doc: Documento de LangChain
            score: Score de distancia (menor = más similar)
            preview_length: Longitud de la vista previa

        Returns:
            SourceDocument
        """
        import os

        source_path = doc.metadata.get("source", "Unknown")
        file_name = os.path.basename(source_path)
        page = doc.metadata.get("page", "N/A")
        content = doc.page_content
        similarity = max(0.0, min(1.0, 1 - score))

        return cls(
            file_name=file_name,
            page=page,
            content=content,
            similarity_score=similarity,
            preview=content[:preview_length] + "..." if len(content) > preview_length else content,
        )


class QueryResult(BaseModel):
    """Resultado de una consulta RAG."""

    answer: str = Field(description="Respuesta generada por el LLM")
    sources: list[SourceDocument] = Field(
        default_factory=list,
        description="Fuentes utilizadas para generar la respuesta",
    )
    query: str = Field(description="Consulta original")

    @property
    def num_sources(self) -> int:
        """Número de fuentes utilizadas."""
        return len(self.sources)

    def format_sources(self) -> str:
        """Formatea las fuentes para mostrar al usuario."""
        if not self.sources:
            return ""

        lines = [
            "",
            "=" * 60,
            "FUENTES DE INFORMACION",
            "=" * 60,
        ]

        seen = set()
        for i, source in enumerate(self.sources, 1):
            source_id = f"{source.file_name}_{source.page}"
            if source_id in seen:
                continue
            seen.add(source_id)

            lines.extend([
                f"\n[{i}] Archivo: {source.file_name}",
                f"    Pagina: {source.page}",
                f"    Similitud: {source.similarity_score:.2%}",
                f"    Vista previa: {source.preview}",
            ])

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


class IndexStats(BaseModel):
    """Estadísticas del índice de documentos."""

    total_documents: int = Field(
        ge=0,
        description="Número total de documentos procesados",
    )
    total_chunks: int = Field(
        ge=0,
        description="Número total de chunks generados",
    )
    vectorstore_path: str = Field(description="Ruta del vector store")
    is_loaded: bool = Field(description="Si el índice está cargado en memoria")

    def format_stats(self) -> str:
        """Formatea las estadísticas para mostrar al usuario."""
        status = "Cargado" if self.is_loaded else "No cargado"
        return (
            f"Documentos: {self.total_documents}\n"
            f"Chunks: {self.total_chunks}\n"
            f"Ruta: {self.vectorstore_path}\n"
            f"Estado: {status}"
        )


class DocumentChunk(BaseModel):
    """Representa un chunk de documento para indexar."""

    content: str = Field(description="Contenido del chunk")
    source: str = Field(description="Ruta del archivo fuente")
    page: int | str = Field(default="N/A", description="Número de página")
    metadata: dict = Field(default_factory=dict, description="Metadata adicional")
