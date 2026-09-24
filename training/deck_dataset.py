"""
deck_dataset.py

Loads data/dataset.jsonl.gz into masked-set training examples for the
deck-completion model (see README.md's "Architecture" section). A deck is
a *set* of oracle_ids, not a sequence: each __getitem__
call holds out a random fraction of a deck's cards as prediction targets and
returns the rest as context, alongside the deck's commander(s).

Deliberately does NOT use CardTokenizer.encode_deck() / the [MASK]/[CMDR]
special tokens: those exist for encode_deck's flat-sequence-with-markers
design, but this pipeline represents the commander(s) and the context cards
as two separately-pooled *sets* (see training/model.py), not one marked
sequence -- there's no "slot" to place a [MASK] token into when the encoder
has no notion of position in the first place. oracle_id -> token id is
looked up directly against the tokenizer's vocab instead.

Card quantity is deliberately ignored (each deck is treated as a set of
distinct oracle_ids, presence/absence only) -- real Commander decks are
singleton other than a handful of named exceptions (Seven Dwarves, etc.),
so this is a minor simplification, not a meaningful information loss.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import Dataset

from tokenizer.mtg_tokenizer import CardTokenizer, PAD

DATASET_PATH = "./data/dataset.jsonl.gz"
HUB_DATASET_REPO = "nsaroiu/moxfield-dump"

# Real decks include a small number of incomplete/work-in-progress
# decklists (confirmed live: 106 of 122,571 decks have under 30 mainboard
# cards, some as few as 1) -- too degenerate to be a useful
# masked-prediction example, so they're dropped.
MIN_DECK_SIZE = 30

# Mask ratio range: randomized per example rather than a fixed BERT-style
# 15%, since the real use case spans "commander + a few cards" all the way
# to "one card left". Widened from an original (0.1, 0.7) once evaluation
# showed the narrower range never trained on below ~30% context, so
# "commander only, suggest everything" -- a real, advertised use case
# (training/complete_deck.py with no --card args) -- was untested
# extrapolation rather than a measured capability.
DEFAULT_MASK_RATIO_RANGE = (0.05, 0.95)

# Held-out-commander eval: sampled only from commanders with a deck count in
# this band. Below the low end there's too little data per commander for a
# stable eval number; above the high end, holding them out would remove a
# disproportionate amount of training data for one eval slice.
HELD_OUT_COMMANDER_DECK_RANGE = (20, 200)
N_HELD_OUT_COMMANDERS = 25


@dataclass
class ParsedDeck:
    public_id: str
    commander_ids: list[int]  # token ids, usually 1 (2 for partner commanders)
    card_ids: list[int]  # token ids, deduped, quantity ignored


def load_decks(tokenizer: CardTokenizer, dataset_path: str = DATASET_PATH) -> list[ParsedDeck]:
    """Parse data/dataset.jsonl.gz into token-id decks, resolved against
    `tokenizer`'s vocab. oracle_ids absent from the vocab are dropped
    (confirmed live at ~0.0135% of references, all genuinely not_legal
    cards a real deck can still reference despite not being paper-legal)
    rather than failing the whole deck.
    """
    known = tokenizer.oracle_id_to_token
    decks = []
    with gzip.open(dataset_path, "rt", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)

            commander_ids = [
                tokenizer.token_to_id[known[oid]]
                for oid in entry.get("commanders", [])
                if oid in known
            ]
            card_ids = list({
                tokenizer.token_to_id[known[oid]]
                for oid, _qty in entry.get("cards", [])
                if oid in known
            })

            if len(card_ids) < MIN_DECK_SIZE or not commander_ids:
                continue

            decks.append(ParsedDeck(
                public_id=entry["public_id"],
                commander_ids=commander_ids,
                card_ids=card_ids,
            ))
    return decks


def load_decks_from_hub(tokenizer: CardTokenizer, repo_id: str = HUB_DATASET_REPO) -> list[ParsedDeck]:
    """Sibling of load_decks that reads straight from the published
    Hugging Face Hub dataset instead of a local data/dataset.jsonl.gz --
    lazily imports `datasets` (only needed for this path, same pattern as
    CardTokenizer.text_embeddings()'s lazy sentence-transformers import) so
    every other function in this module still needs nothing beyond torch.

    The Hub dataset's schema is a flat, Arrow-friendly shape
    (`commander_oracle_ids`/`card_oracle_ids`/`card_quantities` as parallel
    lists) rather than load_decks' local `{"commanders": [...], "cards":
    [[oracle_id, qty], ...]}` -- this adapts each row into the exact same
    ParsedDeck shape either loader produces, so every downstream caller
    (split_decks, DeckCompletionDataset, ...) is identical either way."""
    from datasets import load_dataset

    known = tokenizer.oracle_id_to_token
    ds = load_dataset(repo_id, split="train")

    decks = []
    for row in ds:
        commander_ids = [
            tokenizer.token_to_id[known[oid]]
            for oid in row["commander_oracle_ids"]
            if oid in known
        ]
        card_ids = list({
            tokenizer.token_to_id[known[oid]]
            for oid in row["card_oracle_ids"]
            if oid in known
        })

        if len(card_ids) < MIN_DECK_SIZE or not commander_ids:
            continue

        decks.append(ParsedDeck(
            public_id=row["public_id"],
            commander_ids=commander_ids,
            card_ids=card_ids,
        ))
    return decks


def load_decks_auto(
    tokenizer: CardTokenizer,
    dataset_path: str = DATASET_PATH,
    hub_repo_id: str = HUB_DATASET_REPO,
) -> list[ParsedDeck]:
    """load_decks if `dataset_path` exists locally, else load_decks_from_hub
    -- the one place train.py/evaluate.py go so neither has to duplicate
    this fallback check."""
    import os
    if os.path.exists(dataset_path):
        return load_decks(tokenizer, dataset_path)
    return load_decks_from_hub(tokenizer, hub_repo_id)


def _public_id_bucket(public_id: str) -> int:
    """Deterministic 0-99 bucket from public_id, for a reproducible
    train/val/test split with no split-assignment file to keep in sync."""
    digest = hashlib.md5(public_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


def split_decks(
    decks: list[ParsedDeck],
    n_held_out_commanders: int = N_HELD_OUT_COMMANDERS,
    held_out_commander_deck_range: tuple[int, int] = HELD_OUT_COMMANDER_DECK_RANGE,
    seed: int = 42,
) -> dict[str, list[ParsedDeck]]:
    """Splits decks by public_id (90/5/5 train/val/test), plus a separate
    held_out_commander slice: decks led by a small set of commanders never
    seen anywhere in train/val/test, to test whether the hybrid content
    embeddings let the model generalize to an unfamiliar commander via its
    color identity/type/keywords rather than only memorizing per-commander
    patterns.
    """
    single_commander_counts = Counter()
    for d in decks:
        if len(d.commander_ids) == 1:
            single_commander_counts[d.commander_ids[0]] += 1

    lo, hi = held_out_commander_deck_range
    eligible = sorted(
        cid for cid, n in single_commander_counts.items() if lo <= n <= hi
    )
    rng = random.Random(seed)
    held_out_commanders = set(rng.sample(eligible, min(n_held_out_commanders, len(eligible))))

    splits: dict[str, list[ParsedDeck]] = {"train": [], "val": [], "test": [], "held_out_commander": []}
    for d in decks:
        if any(cid in held_out_commanders for cid in d.commander_ids):
            splits["held_out_commander"].append(d)
            continue
        bucket = _public_id_bucket(d.public_id)
        if bucket < 90:
            splits["train"].append(d)
        elif bucket < 95:
            splits["val"].append(d)
        else:
            splits["test"].append(d)

    return splits


class DeckCompletionDataset(Dataset):
    """Yields (context_card_ids, target_card_ids, commander_ids) per deck,
    with a freshly sampled random mask on every __getitem__ call (so the
    same deck contributes a different context/target split each epoch)."""

    def __init__(
        self,
        decks: list[ParsedDeck],
        mask_ratio_range: tuple[float, float] = DEFAULT_MASK_RATIO_RANGE,
        seed: Optional[int] = None,
    ):
        self.decks = decks
        self.mask_ratio_range = mask_ratio_range
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.decks)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int], list[int]]:
        deck = self.decks[idx]
        cards = list(deck.card_ids)
        self._rng.shuffle(cards)

        ratio = self._rng.uniform(*self.mask_ratio_range)
        k = min(len(cards), max(1, round(len(cards) * ratio)))

        target = cards[:k]
        context = cards[k:]
        return context, target, deck.commander_ids


def make_collate_fn(vocab_size: int, pad_id: int):
    """Pads context/commander id lists to the batch max and builds a
    multi-hot target vector per example (cheap at this vocab size, and
    exactly what BCE-over-the-whole-vocab needs -- see training/model.py)."""

    def collate_fn(batch):
        contexts, targets, commanders = zip(*batch)
        batch_size = len(batch)
        max_ctx = max((len(c) for c in contexts), default=1) or 1
        max_cmdr = max((len(c) for c in commanders), default=1) or 1

        ctx_ids = torch.full((batch_size, max_ctx), pad_id, dtype=torch.long)
        ctx_mask = torch.zeros((batch_size, max_ctx), dtype=torch.bool)
        cmdr_ids = torch.full((batch_size, max_cmdr), pad_id, dtype=torch.long)
        cmdr_mask = torch.zeros((batch_size, max_cmdr), dtype=torch.bool)
        target_multihot = torch.zeros((batch_size, vocab_size), dtype=torch.float32)

        for i, (ctx, tgt, cmdr) in enumerate(zip(contexts, targets, commanders)):
            if ctx:
                ctx_ids[i, :len(ctx)] = torch.tensor(ctx, dtype=torch.long)
                ctx_mask[i, :len(ctx)] = True
            if cmdr:
                cmdr_ids[i, :len(cmdr)] = torch.tensor(cmdr, dtype=torch.long)
                cmdr_mask[i, :len(cmdr)] = True
            for tid in tgt:
                target_multihot[i, tid] = 1.0

        return ctx_ids, ctx_mask, cmdr_ids, cmdr_mask, target_multihot

    return collate_fn


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from-hub", action="store_true",
        help=f"Force loading from the {HUB_DATASET_REPO} Hub dataset even if a "
             f"local {DATASET_PATH} exists (requires the `datasets` package). "
             f"Without this flag, the local file is used if present, else the "
             f"Hub dataset is used automatically -- see load_decks_auto.",
    )
    args = parser.parse_args()

    tok = CardTokenizer.load_pretrained("./tokenizer.json")
    decks = load_decks_from_hub(tok) if args.from_hub else load_decks_auto(tok)
    print(f"Loaded {len(decks)} decks (min size {MIN_DECK_SIZE}).")

    splits = split_decks(decks)
    for name, ds in splits.items():
        print(f"  {name}: {len(ds)} decks")

    ds = DeckCompletionDataset(splits["train"], seed=0)
    ctx, tgt, cmdr = ds[0]
    print(f"Sample: {len(ctx)} context cards, {len(tgt)} target cards, {len(cmdr)} commander(s)")

    collate = make_collate_fn(tok.vocab_size, tok.token_to_id[PAD])
    batch = collate([ds[i] for i in range(4)])
    ctx_ids, ctx_mask, cmdr_ids, cmdr_mask, target_multihot = batch
    print("Batch shapes:", ctx_ids.shape, ctx_mask.shape, cmdr_ids.shape, cmdr_mask.shape, target_multihot.shape)
