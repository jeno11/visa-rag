import os

def get_backend(name: str | None = None):
    """Create the embedding backend named by EMBEDDING_BACKEND (default: fastembed)."""
    name = name or os.getenv("EMBEDDING_BACKEND", "fastembed")
    if name == "fastembed":
        from src.embeddings.onnx_backend import FastEmbedBackend
        return FastEmbedBackend()
    if name == "torch":
        from src.embeddings.local import LocalEmbeddingBackend
        return LocalEmbeddingBackend()
    raise ValueError(f"Unknown EMBEDDING_BACKEND: {name!r}")
