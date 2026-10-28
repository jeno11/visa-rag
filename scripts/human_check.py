"""Blind human check of the LLM judge: grade a sample by myself, then compare with the judge."""
import json
import random
import re
from pathlib import Path

JUDGED = Path("results/judged_v2.jsonl")
OUT = Path("results/human_labels_v2.jsonl")
KEYS = {"c": "correct", "p": "partial", "i": "incorrect"}

rows = [json.loads(line) for line in JUDGED.open()]
graded = [r for r in rows if r["verdict"] in KEYS.values()]
not_correct = [r for r in graded if r["verdict"] != "correct"]
correct = [r for r in graded if r["verdict"] == "correct"]

rng = random.Random(42)
sample = not_correct + rng.sample(correct, 10)
rng.shuffle(sample)

labels = []
for n, r in enumerate(sample, start=1):
    answer = re.sub(r"\s*\[[\d,\s]+\]", "", r["answer"]) # remove [1, 2] so you can't tell RAG from closed-book
    print(f"\n--- {n}/{len(sample)} ---")
    print(f"QUESTION:  {r['question']}")
    print(f"REFERENCE: {r['expected']}")
    print(f"ANSWER:    {answer}")
    key = ""
    while key not in KEYS:
        key = input("Your grade (c = correct, p = partial, i = incorrect): ").strip().lower()
        labels.append({
            "id": r["id"],
            "mode": r["mode"],
            "human": KEYS[key],
            "judge": r["verdict"],
        })
OUT.write_text("".join(json.dumps(label) + "\n" for label in labels))

agree = sum(label["human"] == label["judge"] for label in labels)
print(f"\nAgreement: {agree}/{len(labels)} ({agree / len(labels):.0%})")
for label in labels:
    if label["human"] != label["judge"]:
        print(f"  {label['id']} {label['mode']:<11} you: {label['human']:<9} judge: {label['judge']}")