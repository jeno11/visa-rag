"""Check the ONNX backend reproduces the PyTorch one: embed the same texts with both, compare."""
import json
from pathlib import Path

from src.embeddings.local import LocalEmbeddingBackend
from src.embeddings.onnx_backend import FastEmbedBackend

chunks = [json.loads(line) for line in Path("data/chunks.jsonl").open()][:20]
texts = [f"{c['title_path']}\n\n{c['body']}" for c in chunks]

torch_b, onnx_b = LocalEmbeddingBackend(), FastEmbedBackend()
docs_sim = (torch_b.embed_documents(texts) * onnx_b.embed_documents(texts)).sum(axis=1)

question = "How much does a Graduate visa cost?"
query_sim = float(torch_b.embed_query(question) @ onnx_b.embed_query(question))

print(f"20 chunks, torch vs ONNX similarity: min={docs_sim.min():.4f}, mean={docs_sim.mean():.4f}")
print(f"1 query,   torch vs ONNX similarity: {query_sim:.4f}")