"""Tiny demo: embed a few sentences and compare them by cosine similarity."""
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

sentences = {
    "Q: money": "How much money do I need for a Student visa?",
    "A: money": "You need £1,171 per month for courses outside London.",
    "A: decision time": "A decision will usually be made within weeks.",
    "Q: funds": "What are the maintenance funds requirements for sudying in the UK?"
}

names = list(sentences)
vectors = model.encode(list(sentences.values()), normalize_embeddings=True)

print("shape:", vectors.shape)
print("first 5 numbers of 'Q: money':", vectors[0][:5].round(3))

print("\nCosine similarity:")
for i, a in enumerate(names):
    for j, b in enumerate(names):
        if j > i:
            score = float(vectors[i] @ vectors[j])
            print(f"    {score:.3f} {a} <-> {b}")