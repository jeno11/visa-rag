"""Split raw gov.uk guide JSON into section-level chunks for retrieval."""
import re
import json
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path

from bs4 import BeautifulSoup, Tag

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/chunks.jsonl")
GOVUK = "https://www.gov.uk"
_SPACE_BEFORE_PUNCT = re.compile(r" +([.,;:!?)])")
_SPACE_AFTER_PAREN = re.compile(r"\( +")

@dataclass
class Chunk:
    chunk_id: str
    guide: str
    title_path: str
    url: str
    body: str

def clean(text: str) -> str:
    """Undo the stray spaces get_text(' ') leaves around links and <abbr> tags."""
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    return _SPACE_AFTER_PAREN.sub("(", text)

def element_to_text(el: Tag) -> str:
    """Turn one top-level HTML element into readable plain text."""
    if el.name in ("ul", "ol"):
        items = el.find_all("li", recursive=False)
        return "\n".join(f"- {li.get_text(' ', strip=True)}" for li in items)
    if el.name == "table":
        rows = []
        for tr in el.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
            rows.append(" | ".join(cells))
            return "\n".join(rows)
    return el.get_text(" ", strip=True)

def split_part(guide_slug: str, guide_title: str, part: dict) -> list[Chunk]:
    soup = BeautifulSoup(part["body"], "html.parser")
    for button in soup.select("a.govuk-button"): # drop "Start now" buttons
        button.decompose()

    part_title = part["title"].strip()
    part_url = f"{GOVUK}/{guide_slug}/{part['slug']}"

    # Each section = text under one <h2>; text before the first <h2> is th eintro.
    sections = []
    current = {"heading": None, "anchor": None, "lines": []}
    for el in soup.children:
        if not isinstance(el, Tag):
            continue # skip bare whitespaces between tags
        if el.name == "h2":
            if current["lines"]:
                sections.append(current)
            current = {
                "heading": el.get_text(" ", strip=True), 
                "anchor": el.get("id"),
                "lines": []
            }

            continue
        text = element_to_text(el)
        if text:
            current["lines"].append(text)
    if current["lines"]:
        sections.append(current)

    chunks = []
    for s in sections:
        path = [guide_title, part_title] + ([s["heading"]] if s["heading"] else [])
        anchor = s["anchor"]
        chunks.append(Chunk(
            chunk_id=f"{guide_slug}/{part['slug']}#{anchor or 'intro'}",
            guide=guide_slug,
            title_path=" > ".join(path),
            url=f"{part_url}#{anchor}" if anchor else part_url,
            body=clean("\n".join(s["lines"])),
        ))
    return chunks

def main() -> None:
    all_chunks = []
    for raw_file in sorted(RAW_DIR.glob("*.json")):
        data = json.loads(raw_file.read_text())
        guide_slug = data["base_path"].strip("/")
        for part in data["details"]["parts"]:
            all_chunks.extend(split_part(guide_slug, data["title"], part))

    with OUT_PATH.open("w") as f:
        for c in all_chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    sizes = [len(c.body) for c in all_chunks]
    print(f"{len(all_chunks)} chunks written to {OUT_PATH}")
    print(f"body chars: min={min(sizes)}, median={statistics.median(sizes):.0f}, max={max(sizes)}")
    by_size = sorted(all_chunks, key=lambda c: len(c.body))
    print("\nSmallest:")
    for c in by_size[:3]:
        print(f"    {len(c.body):>5}    {c.chunk_id}")
    print("Largest:")
    for c in by_size[-3:]:
        print(f"    {len(c.body):>5}    {c.chunk_id}")
    print("\nSample chunk:\n", json.dumps(asdict(all_chunks[5]), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()