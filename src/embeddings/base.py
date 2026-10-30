from abc import ABC, abstractmethod

import numpy as np

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

class EmbeddingBackend(ABC):
    """Turns text into vectors. Any embedding model must provide these two methods."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed chunks for the index. Returns shape (number of texts, dimensions)."""

    @abstractmethod
    def embed_query(self, text:str) -> np.ndarray:
        """Embed one user question. Returns shape (dimensions,)."""