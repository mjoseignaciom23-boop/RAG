"""Configuración del sistema."""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from src.exceptions import ConfigurationError

class Settings(BaseSettings):
    """Carga configuración desde variables de entorno (.env)."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Ollama
    ollama_base_url: str = Field(default="http://localhost:11434")

    # Modelos
    llm_model_name: str = Field(default="qwen2.5:7b")
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    embedding_model_name: str = Field(default="intfloat/multilingual-e5-small")

    # Vector Store
    vectorstore_path: str = Field(default="./vectorstore")

    # Procesamiento
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    retrieval_k: int = Field(default=4, gt=0)

@lru_cache
def get_settings() -> Settings:
    """Singleton de configuración."""
    try:
        return Settings()
    except Exception as e:
        raise ConfigurationError(f"Error de configuración: {e}") from e
