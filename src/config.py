"""
Configuración centralizada del sistema RAG usando pydantic-settings.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator

from src.exceptions import ConfigurationError


class Settings(BaseSettings):
    """Configuración de la aplicación cargada desde variables de entorno."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Keys
    openai_api_key: str = Field(
        ...,
        description="API Key de OpenAI",
    )

    # Modelos
    llm_model_name: str = Field(
        default="gpt-3.5-turbo",
        description="Nombre del modelo LLM a usar",
    )
    llm_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperatura del LLM",
    )
    embedding_model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="Nombre del modelo de embeddings local",
    )

    # Vector Store
    vectorstore_path: str = Field(
        default="./vectorstore",
        description="Ruta al directorio del vector store",
    )

    # Document Processing
    chunk_size: int = Field(
        default=1000,
        gt=0,
        description="Tamaño de cada chunk de texto",
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        description="Solapamiento entre chunks",
    )

    # Retrieval
    retrieval_k: int = Field(
        default=4,
        gt=0,
        description="Número de documentos a recuperar por consulta",
    )

    @field_validator("openai_api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Valida que la API key no sea un placeholder."""
        if not v or v == "tu_api_key_aqui" or v.startswith("sk-xxx"):
            raise ValueError(
                "OPENAI_API_KEY no está configurada correctamente. "
                "Por favor, configura tu API key en el archivo .env"
            )
        return v


@lru_cache
def get_settings() -> Settings:
    """
    Obtiene la configuración de la aplicación (singleton cacheado).

    Returns:
        Instancia de Settings

    Raises:
        ConfigurationError: Si la configuración es inválida
    """
    try:
        return Settings()
    except Exception as e:
        raise ConfigurationError(
            message="Error de configuración",
            details=str(e),
        ) from e
