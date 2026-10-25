"""Hybrid retrieval: combine several retrievers' rankings with Reciprocal Rank Fusion."""

class HybridRetriever:
    def __init__(self, retrievers: list, depth: int = 20, c: int = 60):
        self.retrievers = retrievers
        self.depth = depth # how many results to take from each retriever
        self.c = c # standard RRF constant; NOT tuned on our eval set

    def search(self, question: str, k: int = 3) -> list[dict]:
        scores: dict[str, float] = {}
        info: dict[str, dict] = {}
        for retriever in self.retrievers:
            for rank, hit in enumerate(retriever.search(question, k=self.depth), start=1):
                cid = hit["chunk_id"]
                scores[cid] = scores.get(cid, 0.0) + 1 / (self.c + rank)
                info[cid] = hit
        top = sorted(scores, key=scores.get, reverse=True)[:k]
        return [{**info[cid], "score": scores[cid]} for cid in top]
            