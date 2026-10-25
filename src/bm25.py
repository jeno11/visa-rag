"""Keyword baseline: BM25 over the same chunks, with the same search() interface as Retriever."""
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

def tokenize(text: str) -> list[str]:
    """Lowecase and split into words: 'How much money?' -> ['how', 'much', 'money']."""
    return re.findall(r"[a-z0-9£]+", text.lower())

class BM25Retriever:
    def __init__(self, chunks_path: str = "data/chunks.jsonl"):
        self.chunks = [json.loads(line) for line in Path(chunks_path).open()]
        corpus = [tokenize(f"{c['title_path']}\n\n{c['body']}") for c in self.chunks]
        self.bm25 = BM25Okapi(corpus)

    def search(self, question: str, k: int = 3) -> list[dict]:
        scores = self.bm25.get_scores(tokenize(question))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            {
                "chunk_id": self.chunks[i]["chunk_id"],
                "title_path":self.chunks[i]["title_path"],
                "url": self.chunks[i]["url"],
                "body": self.chunks[i]["body"],
                "score": float(scores[i]),
            }
            for i in top
        ]      
