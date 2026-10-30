"""Embed every chunk once and score it in a Chroma index on disk."""
import json
from pathlib import Path

import chromadb

from src.embeddings import get_backend

CHUNKS_PATH = Path("data/chunks.jsonl")
INDEX_DIR = "data/chroma"
COLLECTION = "visa_chunks"

def main() -> None:
    chunks = [json.loads(line) for line in CHUNKS_PATH.open()]

    texts = [f"{c['title_path']}\n\n{c['body']}" for c in chunks]

    backend = get_backend()
    vectors = backend.embed_documents(texts)

    client = chromadb.PersistentClient(path=INDEX_DIR)
    try:
        client.delete_collection(COLLECTION)    # rebuild from scratch every time
    except Exception:
        pass    # first run: nothing to delete yet
    collection = client.create_collection(COLLECTION)

    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=vectors.tolist(),
        documents=[c["body"] for c in chunks],
        metadatas=[{"title_path": c["title_path"], "url": c["url"], "guide": c["guide"]} for c in chunks],
    )

    print(f"Indexed {collection.count()} chunks into {INDEX_DIR}/{COLLECTION}")

if __name__ == "__main__":
    main()