"""One-off: fill in missing closed-book answers in the results file, keeping the RAG answer."""
import json
from pathlib import Path

from src.generate import AnswerGenerator

path = Path("results/answers_v1.jsonl")
rows = [json.loads(line) for line in path.open()]
gen = AnswerGenerator()

for i, r in enumerate(rows, start=1):
    if r["closed_book"] is None:
        r["closed_book"] = gen.answer_without_context(r["question"])
        print(f"{i}/{len(rows)} {r['id']} done")

path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))