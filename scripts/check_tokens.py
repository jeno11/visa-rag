"""Measure chunk lengths in real model tokens, to check nothing gets truncated."""
import json
from pathlib import Path

from transformers import AutoTokenizer

MODEL = "BAAI/bge-small-en-v1.5"

chunks = [json.loads(line) for line in Path("data/chunks.jsonl").open()]
tok = AutoTokenizer.from_pretrained(MODEL)

lengths = sorted(
    (len(tok(f"{c['title_path']}\n\n{c['body']}")["input_ids"]), c["chunk_id"])
    for c in chunks
)
limit = tok.model_max_length

print(f"model max tokens: {limit}")
print(f"min={lengths[0][0]}, median={lengths[len(lengths) // 2][0]}, max={lengths[-1][0]}")
over = [x for x in lengths if x[0] > limit]
print(f"chunks over the limit: {len(over)}")
for n, cid in over:
    print(f"    {n} {cid}") 