"""
Tests para el módulo de configuración.
"""
import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.config import Settings, get_settings
from src.exceptions import ConfigurationError


class TestSettings:
    """Tests para la clase Settings."""

    def test_settings_with_valid_api_key(self):
        """Test que Settings se crea correctamente con API key válida."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-valid-test-key"}):
            settings = Settings()
            assert settings.openai_api_key == "sk-valid-test-key"

    def test_settings_default_values(self):
        """Test que los valores por defecto son correctos."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-valid-test-key"}):
            settings = Settings()
            assert settings.llm_model_name == "gpt-3.5-turbo"
            assert settings.llm_temperature == 0.7
            assert settings.embedding_model_name == "all-MiniLM-L6-v2"
            assert settings.vectorstore_path == "./vectorstore"
            assert settings.chunk_size == 1000
            assert settings.chunk_overlap == 200
            assert settings.retrieval_k == 4

    def test_settings_custom_values(self):
        """Test que se pueden configurar valores personalizados."""
        env_vars = {
            "OPENAI_API_KEY": "sk-custom-key",
            "LLM_MODEL_NAME": "gpt-4",
            "LLM_TEMPERATURE": "0.5",
            "CHUNK_SIZE": "500",
            "RETRIEVAL_K": "6",
        }
        with patch.dict(os.environ, env_vars):
            settings = Settings()
            assert settings.llm_model_name == "gpt-4"
            assert settings.llm_temperature == 0.5
            assert settings.chunk_size == 500
            assert settings.retrieval_k == 6

    def test_settings_fails_without_api_key(self, monkeypatch):
        """Test que falla sin API key."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValidationError):
            Settings(_env_file=None)

    def test_settings_fails_with_placeholder_api_key(self):
        """Test que falla con API key placeholder."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "tu_api_key_aqui"}):
            with pytest.raises(ValidationError):
                Settings()

    def test_settings_fails_with_invalid_api_key(self):
        """Test que falla con API key inválida."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-xxx-invalid"}):
            with pytest.raises(ValidationError):
                Settings()

    def test_temperature_validation_min(self):
        """Test que temperature debe ser >= 0."""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-valid-key",
            "LLM_TEMPERATURE": "-0.5",
        }):
            with pytest.raises(ValidationError):
                Settings()

    def test_temperature_validation_max(self):
        """Test que temperature debe ser <= 2."""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-valid-key",
            "LLM_TEMPERATURE": "2.5",
        }):
            with pytest.raises(ValidationError):
                Settings()

    def test_chunk_size_must_be_positive(self):
        """Test que chunk_size debe ser positivo."""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-valid-key",
            "CHUNK_SIZE": "0",
        }):
            with pytest.raises(ValidationError):
                Settings()


class TestGetSettings:
    """Tests para la función get_settings."""

    def test_get_settings_returns_settings(self):
        """Test que get_settings retorna un Settings válido."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key-12345"}):
            get_settings.cache_clear()
            settings = get_settings()
            assert isinstance(settings, Settings)

    def test_get_settings_raises_configuration_error(self):
        """Test que get_settings lanza ConfigurationError cuando falla."""
        get_settings.cache_clear()
        with patch("src.config.Settings", side_effect=ValidationError.from_exception_data("test", [])):
            with pytest.raises(ConfigurationError):
                get_settings()

    def test_get_settings_is_cached(self):
        """Test que get_settings retorna la misma instancia (cache)."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key-12345"}):
            get_settings.cache_clear()
            settings1 = get_settings()
            settings2 = get_settings()
            assert settings1 is settings2
