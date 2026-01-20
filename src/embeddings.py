"""Embeddings locales usando sentence-transformers."""
from typing import List
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings

class LocalEmbeddings(Embeddings):
    """Wrapper para usar modelos de embeddings locales."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small"):
        print(f"[*] Cargando embeddings: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print(f"[OK] Modelo cargado")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Genera embeddings para una lista de textos."""
        return self.model.encode(
            texts, 
            show_progress_bar=False, 
            convert_to_numpy=True
        ).tolist()

    def embed_query(self, text: str) -> List[float]:
        """Genera embedding para una consulta."""
        return self.model.encode(
            text, 
            show_progress_bar=False, 
            convert_to_numpy=True
        ).tolist()

def get_embeddings() -> LocalEmbeddings:
    return LocalEmbeddings()
