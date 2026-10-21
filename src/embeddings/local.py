import numpy as np
from sentence_transformers import SentenceTransformer

from src.embeddings.base import EmbeddingBackend

# Instruction that BGE models recommend adding in front of search queries
BEG_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

class LocalEmbeddingBackend(EmbeddingBackend):
    def __init__(
            self, 
            model_name: str = "BAAI/bge-small-en-v1.5",
            query_prefix: str = BEG_QUERY_PREFIX
    ):
        self.model = SentenceTransformer(model_name)
        self.query_prefix = query_prefix

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    def embed_query(self, text: str) -> np.ndarray:
        return self.model.encode(self.query_prefix + text, normalize_embeddings=True)