"""Rough peak memory of the retrieval stack: model + index + BM25 loaded, one search run."""
import resource
import sys

from src.bm25 import BM25Retriever
from src.hybrid import HybridRetriever
from src.search import Retriever

retriever = HybridRetriever([BM25Retriever(), Retriever()])
retriever.search("How much does a Student visa cost?", k=5)

peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
mb = peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024
print(f"Peak memory: {mb:.0f} MB (Render free tier limit: 512MB)")
