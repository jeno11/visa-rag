"""Same bge-small model, run with ONNX runtime via fastembed: no PyTorch, much less memory."""
import os
import numpy as np
from fastembed import TextEmbedding

from src.embeddings.base import BGE_QUERY_PREFIX, EmbeddingBackend

def _normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale every row to length 1, like normalize_embeddings=True in PyTorch backend."""
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

class FastEmbedBackend(EmbeddingBackend):
    def __init__(
            self, 
            model_name: str = "BAAI/bge-small-en-v1.5",
            query_prefix: str = BGE_QUERY_PREFIX,
        ):
        self.model = TextEmbedding(model_name, cache_dir=os.getenv("FASTEMBED_CACHE_PATH"))
        self.query_prefix = query_prefix

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.array(list(self.model.embed(texts))))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(np.array(list(self.model.embed([self.query_prefix + text]))))[0]