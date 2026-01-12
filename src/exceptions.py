"""
Excepciones personalizadas para el sistema RAG.
"""


class RAGError(Exception):
    """Excepción base para todos los errores del sistema RAG."""

    def __init__(self, message: str, details: str | None = None):
        self.message = message
        self.details = details
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message}: {self.details}"
        return self.message


class ConfigurationError(RAGError):
    """Error de configuración (API keys faltantes, variables de entorno, etc.)."""

    pass


class DocumentLoadError(RAGError):
    """Error al cargar documentos."""

    pass


class UnsupportedFormatError(DocumentLoadError):
    """Formato de documento no soportado."""

    def __init__(self, file_extension: str):
        super().__init__(
            message="Formato de documento no soportado",
            details=f"Extensión '{file_extension}' no está soportada. "
            "Formatos válidos: .pdf, .txt, .docx, .md",
        )
        self.file_extension = file_extension


class DocumentNotFoundError(DocumentLoadError):
    """Documento o directorio no encontrado."""

    def __init__(self, path: str):
        super().__init__(
            message="Archivo o directorio no encontrado",
            details=f"La ruta '{path}' no existe",
        )
        self.path = path


class VectorStoreError(RAGError):
    """Error relacionado con el vector store."""

    pass


class VectorStoreNotInitializedError(VectorStoreError):
    """El vector store no está inicializado."""

    def __init__(self):
        super().__init__(
            message="Vector store no inicializado",
            details="Debes indexar documentos o cargar un índice existente primero",
        )


class VectorStoreEmptyError(VectorStoreError):
    """No hay documentos para indexar."""

    def __init__(self):
        super().__init__(
            message="No hay documentos para indexar",
            details="Proporciona al menos un documento válido",
        )


class QueryError(RAGError):
    """Error durante una consulta."""

    pass


class IndexError(RAGError):
    """Error durante la indexación."""

    pass
