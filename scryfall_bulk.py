"""
scryfall_bulk.py

Shared helper for loading a Scryfall bulk data export (a JSON array or true
JSON Lines file - see load_bulk_json). Used by tokenizer/mtg_tokenizer.py to
build the card vocabulary from Scryfall's "Oracle Cards" bulk file - kept as
a standalone top-level module since it's a generic Scryfall-bulk-file
parser, not tokenizer-specific.
"""

from __future__ import annotations

import json


def load_bulk_json(path: str) -> list[dict]:
    """Load a Scryfall bulk data file, auto-detecting JSON array vs JSON Lines."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    text = text.strip()
    if not text:
        raise ValueError(f"{path} is empty")

    # Standard bulk export: whole file is one JSON array.
    if text[0] == "[":
        return json.loads(text)

    # JSON Lines: one JSON object per line.
    records = []
    for i, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse line {i} of {path} as JSON. "
                f"File does not look like a valid Scryfall bulk export "
                f"(JSON array) or JSON Lines file."
            ) from e
    return records
