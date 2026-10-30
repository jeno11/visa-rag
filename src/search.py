"""Search the Chroma index: a question in, the closes chunks out."""
import sys
import chromadb

from src.build_index import COLLECTION, INDEX_DIR
from src.embeddings import get_backend

class Retriever:
    def __init__(self, backend=None):
        self.backend = backend or get_backend()
        client = chromadb.PersistentClient(path=INDEX_DIR)
        self.collection = client.get_collection(COLLECTION)

    def search(self, question: str, k: int = 3) -> list[dict]:
        q = self.backend.embed_query(question)
        res = self.collection.query(query_embeddings=[q.tolist()], n_results=k)

        results = []
        for cid, body, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            results.append({
                "chunk_id": cid,
                "title_path": meta["title_path"],
                "url": meta["url"],
                "body": body,
                "score": 1 - dist / 2 # converst Chroma's distance back to cosine similarity
            })
        return results

TEST_QUESTIONS = [
    "How long does it take to get a decision if I extend my Student visa?",
    "How much does a Graduate visa cost?",
    "Can I bring my family on a Student visa?",
    "How long can I stay on a Graduate visa?",
    "What is the minimum salary for a Skilled Worker visa?",
    "Do I need to prove my English for a Student visa if I'm from Canada?",
]

def main() -> None:
    retriever = Retriever()
    questions = sys.argv[1:] or TEST_QUESTIONS
    for q in questions:
        print(f"\nQ: {q}")
        for rank, hit in enumerate(retriever.search(q), start=1):
            print(f"    {rank}. {hit['score']:.3f}  {hit['title_path']}")

if __name__ == "__main__":
    main()