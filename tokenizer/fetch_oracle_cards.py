"""
fetch_oracle_cards.py

Downloads a fresh copy of Scryfall's "Oracle Cards" bulk data file
(oracle-cards.jsonl) directly from Scryfall's API, in place of downloading
it manually. Cards released after the last download are otherwise invisible
to mtg_tokenizer.py's vocab-building pass - re-run this periodically to
stay current.

Scryfall's API rejects the default `requests` User-Agent with a 400, so a
custom User-Agent/Accept pair is required (see _HEADERS below).

Usage:
    python3 fetch_oracle_cards.py   # overwrites ./oracle-cards.jsonl
"""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path

import requests

BULK_DATA_CATALOG_URL = "https://api.scryfall.com/bulk-data"
_HEADERS = {
    "User-Agent": "mtg-card-embeddings/0.1 (fetch_oracle_cards.py)",
    "Accept": "application/json",
}


def _oracle_cards_jsonl_uri() -> str:
    resp = requests.get(BULK_DATA_CATALOG_URL, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    for entry in resp.json()["data"]:
        if entry["type"] == "oracle_cards":
            return entry["jsonl_download_uri"]
    raise RuntimeError("No 'oracle_cards' entry found in Scryfall's bulk-data catalog")


def fetch_oracle_cards(out_path: str = "./oracle-cards.jsonl") -> int:
    """Download and decompress the current Oracle Cards bulk file to
    out_path, streaming straight to disk (the ~25MB compressed download is
    never fully buffered in memory). Returns the number of cards written.
    """
    jsonl_uri = _oracle_cards_jsonl_uri()
    print(f"Fetching Oracle Cards from {jsonl_uri} ...")

    out = Path(out_path)
    with requests.get(jsonl_uri, headers=_HEADERS, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        resp.raw.decode_content = True  # transparently gunzip the response body
        with gzip.GzipFile(fileobj=resp.raw) as gz, open(out, "wb") as f:
            shutil.copyfileobj(gz, f)

    with out.open("rb") as f:
        line_count = sum(1 for _ in f)
    size_mb = out.stat().st_size / 1_000_000
    print(f"Saved {line_count} cards to {out_path} ({size_mb:.1f} MB)")
    return line_count


if __name__ == "__main__":
    fetch_oracle_cards()
