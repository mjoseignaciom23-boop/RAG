"""
Módulo para generar embeddings usando sentence-transformers.
Utiliza el modelo local all-MiniLM-L6-v2 (gratuito).
"""
from typing import List
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings


class LocalEmbeddings(Embeddings):
    """Clase para generar embeddings usando sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Inicializa el modelo de embeddings.

        Args:
            model_name: Nombre del modelo de sentence-transformers
        """
        print(f"[*] Cargando modelo de embeddings: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print(f"[OK] Modelo cargado correctamente")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Genera embeddings para una lista de textos.

        Args:
            texts: Lista de textos a procesar

        Returns:
            Lista de vectores de embeddings
        """
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Genera embedding para una consulta individual.

        Args:
            text: Texto de la consulta

        Returns:
            Vector de embedding
        """
        embedding = self.model.encode(
            text,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embedding.tolist()


def get_embeddings() -> LocalEmbeddings:
    """
    Obtiene una instancia del modelo de embeddings.

    Returns:
        Instancia de LocalEmbeddings
    """
    return LocalEmbeddings()
