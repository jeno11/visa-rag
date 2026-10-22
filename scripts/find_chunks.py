"""Find chunk ids by keyword, for labelling eval questions.
Deliberately does NOT use the retriever."""
import json
import sys
from pathlib import Path

terms = [t.lower() for t in sys.argv[1:]]

for line in Path("data/chunks.jsonl").open():
    c = json.loads(line)
    text = f"{c['title_path']}\n{c['body']}".lower()
    if all(t in text for t in terms):
        preview = c["body"][:120].replace(chr(10), " ")
        print(f"{c['chunk_id']}\n   {preview}...\n")