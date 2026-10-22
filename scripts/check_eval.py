"""Validate the eval set: every labelled chunk id must exist, and question ids must be unique."""
import json
from collections import Counter
from pathlib import Path

chunk_ids = {json.loads(line)['chunk_id'] for line in Path("data/chunks.jsonl").open()}
items = [json.loads(line) for line in Path("data/eval/questions.jsonl").open() if line.strip()]

problems = 0
for qid, n in Counter(item["id"] for item in items).items():
    if n > 1:
        print(f"duplicate queston id: {qid}")
        problems += 1

for item in items:
    for cid in item["relevant_chunk_ids"]:
        if cid not in chunk_ids:
            print(f"{item['id']}: unknown chunk id {cid}")
            problems += 1

print(f"\n{len(items)} questions, {problems} problems")
print("by category:", dict(Counter(item["category"] for item in items)))