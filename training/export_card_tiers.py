"""
export_card_tiers.py

One-off export of training.metrics.card_tiers() to a small, oracle_id-keyed
JSON file -- so anything that needs a card's popularity tier at inference
time (training/ensemble.py) only has to load this small file, not the full
data/dataset.jsonl.gz + a deck-loading pass -- useful for any deployment
that doesn't want the whole training dataset resident just to know a
card's tier.

Keyed by oracle_id, not token id -- survives a tokenizer rebuild, since
oracle_id is Scryfall's stable per-card identity while token ids are only
stable across tokenizer rebuilds that were given the previous tokenizer to
reuse ids from (see CardTokenizer.from_scryfall_bulk's
previous_tokenizer_path).

Usage:
    python3 -m training.export_card_tiers
"""

from __future__ import annotations

import json

from tokenizer.mtg_tokenizer import CardTokenizer
from training.deck_dataset import load_decks_auto, split_decks
from training.metrics import card_tiers
from training.train import TOKENIZER_PATH

OUT_PATH = "./data/card_tiers.json"


def run(tokenizer_path: str = TOKENIZER_PATH, out_path: str = OUT_PATH) -> dict[str, str]:
    tok = CardTokenizer.load_pretrained(tokenizer_path)
    decks = load_decks_auto(tok)
    splits = split_decks(decks)
    tiers = card_tiers(splits["train"])

    by_oracle_id = {
        tok.card_by_token[tok.id_to_token[cid]].oracle_id: tier
        for cid, tier in tiers.items()
    }
    with open(out_path, "w") as f:
        json.dump(by_oracle_id, f)
    print(f"Wrote {len(by_oracle_id)} card tiers to {out_path}")
    return by_oracle_id


if __name__ == "__main__":
    run()
