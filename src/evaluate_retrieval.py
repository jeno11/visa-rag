"""Score a retriever on the eval set: hit rate@k and MRR."""
import json
from pathlib import Path

from src.search import Retriever

EVAL_PATH = Path("data/eval/questions.jsonl")
KS = [1, 3, 5]

def first_hit_rank(retrieved_ids: list[str], relevant: set[str]) -> int | None:
    """Rank (1-based) of the first correct chunk, or None if none was retrieved."""
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant:
            return rank
    return None

def evaluate(retriever, label: str) -> None:
    items = [json.loads(l) for l in EVAL_PATH.open() if l.strip()]
    scored = [item for item in items if item["category"] != "out_of_scope"]

    ranks = []
    for item in scored:
        hits = retriever.search(item["question"], k=max(KS))
        ids = [h["chunk_id"] for h in hits]
        rank = first_hit_rank(ids, set(item["relevant_chunk_ids"]))
        ranks.append((item, rank, ids))

    n = len(ranks)
    print(f"\n=== {label}: {n} questions ===")
    for k in KS:
        hit = sum(1 for _, r, _ in ranks if r is not None and r <= k)
        print(f"hit@{k}: {hit / n:.1%}  ({hit}/{n})")
    mrr = sum(1 / r for _, r, _ in ranks if r is not None) / n
    print(f"MRR:    {mrr:.3f}")

    for cat in ("standard", "hard"):
        sub = [r for item, r, _ in ranks if item["category"] == cat]
        hit5 = sum(1 for r in sub if r is not None)
        print(f"    {cat:<8} hit@5: {hit5}/{len(sub)}")

    print("\nMisses / not at #1:")
    for item, r, ids in ranks:
        if r != 1:
            where = f"rank {r}" if r else "not in top 5"
            print(f"    {item['id']} ({where}): {item['question']}")
            print(f"    got #1: {ids[0]}")

if __name__ == "__main__":
    from src.bm25 import BM25Retriever
    from src.hybrid import HybridRetriever

    bm25, dense = BM25Retriever(), Retriever()
    evaluate(bm25, "keyword (BM25)")
    evaluate(dense, "dense (bge-small)")
    evaluate(HybridRetriever([bm25, dense]), "hybrid (RRF)")
