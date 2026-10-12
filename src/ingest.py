"""Fetch gov.uk visa guides via the Content API and save raw JSON sanpshots."""
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

GUIDES = ["student-visa", "graduate-visa", "skilled-worker-visa"]
API_BASE = "https://www.gov.uk/api/content"
RAW_DIR = Path("data/raw")

def fetch_guide(slug: str) -> dict:
    resp = requests.get(f"{API_BASE}/{slug}", timeout=30)
    resp.raise_for_status()
    return resp.json()

def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    fetched_at = datetime.now(timezone.utc).isoformat()

    for slug in GUIDES:
        data = fetch_guide(slug)
        data["_fetched_at"] = fetched_at # record the snapshot date with data
        (RAW_DIR / f"{slug}.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False)
        )

        parts = data.get("details", {}).get("parts", [])
        print(f"{slug}: type={data.get('document_type')}, "
              f"parts={len(parts)}, updated={data.get('public_updated_at')}")
        for p in parts:
            print(f"    - {p['slug']}: {p['title']} ({len(p['body'])} chars)")

if __name__ == "__main__":
    main()