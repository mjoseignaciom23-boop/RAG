"""
Logger centralizado para el sistema RAG.
"""
import logging
import sys
from typing import Literal

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

_SENSITIVE_KEYS = frozenset({
    "api_key",
    "openai_api_key",
    "password",
    "secret",
    "token",
    "authorization",
})


class SensitiveDataFilter(logging.Filter):
    """Filtro para evitar loggear información sensible."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Filtra y sanitiza mensajes con datos sensibles."""
        if hasattr(record, "msg") and isinstance(record.msg, str):
            msg_lower = record.msg.lower()
            for key in _SENSITIVE_KEYS:
                if key in msg_lower:
                    record.msg = self._sanitize_message(record.msg)
                    break
        return True

    def _sanitize_message(self, message: str) -> str:
        """Reemplaza valores sensibles con asteriscos."""
        return "[CONTENIDO SENSIBLE REDACTADO]"


def setup_logger(
    name: str = "rag",
    level: LogLevel = "INFO",
    log_to_file: bool = False,
    log_file_path: str = "rag.log",
) -> logging.Logger:
    """
    Configura y retorna un logger.

    Args:
        name: Nombre del logger
        level: Nivel de logging
        log_to_file: Si debe escribir a archivo
        log_file_path: Ruta del archivo de log

    Returns:
        Logger configurado
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level))

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(SensitiveDataFilter())
    logger.addHandler(console_handler)

    if log_to_file:
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.addFilter(SensitiveDataFilter())
        logger.addHandler(file_handler)

    logger.propagate = False

    return logger


def get_logger(name: str = "rag") -> logging.Logger:
    """
    Obtiene un logger existente o crea uno nuevo.

    Args:
        name: Nombre del logger

    Returns:
        Logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger
