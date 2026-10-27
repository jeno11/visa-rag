"""Run every eval question through RAG and closed-book generation; save answers and automatic metrics."""
import json
import sys
from pathlib import Path

EVAL_PATH = Path("data/eval/questions.jsonl")
OUT_PATH = Path("results/answers_v2.jsonl")

def run() -> list[dict]:
    from src.bm25 import BM25Retriever
    from src.generate import AnswerGenerator
    from src.hybrid import HybridRetriever
    from src.search import Retriever

    items = [json.loads(l) for l in EVAL_PATH.open() if l.strip()]
    retriever = HybridRetriever([BM25Retriever(), Retriever()])
    gen = AnswerGenerator()

    OUT_PATH.parent.mkdir(exist_ok=True)
    rows = []
    with OUT_PATH.open("w") as f:
        for i, item in enumerate(items, start=1):
            chunks = retriever.search(item["question"], k=5)
            row = {
                "id": item["id"],
                "category": item["category"],
                "question": item["question"],
                "expected": item["answer"],
                "relevant_chunk_ids": item["relevant_chunk_ids"],
                "retrieved_ids": [c["chunk_id"] for c in chunks],
                "rag": gen.answer(item["question"], chunks),
                "closed_book": gen.answer_without_context(item["question"]),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            rows.append(row)
            print(f"{i}/{len(items)} {item['id']} done")
    return rows

def report(rows: list[dict]) -> None:
    in_scope = [r for r in rows if r["category"] != "out_of_scope"]
    out_scope = [r for r in rows if r["category"] == "out_of_scope"]

    for mode in ("rag", "closed_book"):
        refused_out = sum(r[mode]["not_found"] for r in out_scope)
        refused_in = sum(r[mode]["not_found"] for r in in_scope)
        print(f"\n=== {mode} ===")
        print(f"out-of-scope correctly refused: {refused_out}/{len(out_scope)}")
        print(f"in-scope wrongly refused:       {refused_in}/{len(in_scope)}")

    answered = [r for r in in_scope if not r["rag"]["not_found"]]
    good = sum(any(c["chunk_id"] in r["relevant_chunk_ids"] for c in r["rag"]["citations"]) for r in answered)
    uncited = sum(not r["rag"]["citations"] for r in answered)
    print(f"\nRAG citations: {good}/{len(answered)} answers cite at least one labelled-correct chunk;" f" {uncited} cite nothing")

if __name__ == "__main__":
    if "--report" in sys.argv:
        report([json.loads(l) for l in OUT_PATH.open()])
    else:
        report(run())