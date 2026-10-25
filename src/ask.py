"""Ask a question: hybrid retrieval, then a cited answer from Gemini."""
import json
import sys
from pathlib import Path

from src.bm25 import BM25Retriever
from src.generate import AnswerGenerator
from src.hybrid import HybridRetriever
from src.search import Retriever

def snapshot_date() -> str:
    """Date the gov.uk pages were fetched (the earliest, if they differ)."""
    return min(json.loads(p.read_text())["_fetched_at"][:10] for p in Path("data/raw").glob("*.json"))

def main() -> None:
    retriever = HybridRetriever([BM25Retriever(), Retriever()])
    generator = AnswerGenerator()
    date = snapshot_date()

    for question in sys.argv[1:]:
        chunks = retriever.search(question, k=5)
        result = generator.answer(question, chunks)
        print(f"\nQ: {question}\n\n{result['answers']}\n")
        for c in result["citations"]:
            print(f"    [{c['n']}] {c['title_path']}\n  {c['url']}")
        print(f"\n  Based on gov.uk guidance as of {date}. Not legal advice.")

if __name__ == "__main__":
    main()