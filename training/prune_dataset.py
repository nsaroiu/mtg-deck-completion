"""
prune_dataset.py

Builds real-vs-corrupted training examples for the deck-pruning model
(training/model.py's PruneModel, training/train_prune.py). Reuses
training/deck_dataset.py's ParsedDeck/load_decks_auto/split_decks
unchanged -- this only changes how a ParsedDeck becomes a training
example, not how decks are parsed or split.

Corruption mechanism: for each deck, replace a few of its real cards with
real cards pulled from OTHER decks that happen to be legal for this deck's
commander(s) -- not a uniform-random legal card. A random legal negative is
too easy (the model would just re-learn "is this card popular/on-color",
signal we already have for free from the tier data); a real card that was
actually built into someone else's deck is a genuinely harder, more honest
negative, forcing the model to learn contextual fit to THIS deck rather
than surface stats.
"""

from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset

from tokenizer.mtg_tokenizer import CardTokenizer
from training.complete_deck import legal_mask
from training.deck_dataset import ParsedDeck

# Fraction of a deck's cards replaced with cross-deck negatives, per
# example. Kept small (a real deck has a FEW weak cards, not mostly-bad
# ones) so the task stays "spot the weak link(s)" -- a majority-corrupted
# example would teach bulk-noise detection instead of individual fit.
CORRUPT_RATIO_RANGE = (0.03, 0.08)

# Bounded retries per negative before giving up on that slot -- keeps a
# narrow-color-identity deck (e.g. mono-color, where most other decks'
# cards are illegal here) from stalling __getitem__; a slot that runs out
# of tries is simply left real, yielding fewer corruptions than requested
# for that example rather than failing it.
MAX_SAMPLE_TRIES = 40

# "inject" mode's corruption density -- see PruneDataset's mode docstring.
# Deliberately higher than CORRUPT_RATIO_RANGE (0.03-0.08): the point is to
# simulate a genuinely oversized candidate pool (~15-20 extra cards for a
# ~100-card deck -- a real "I drafted/collected too many good cards" case,
# and a direct rehearsal for a possible future ">100 cards, cut to 100"
# feature), forcing the model to rank several competing candidates against
# each other rather than mostly spotting one obvious misfit among a sea of
# confident-real cards. Still well short of the "majority corrupted teaches
# bulk-noise detection, not individual fit" danger zone noted above.
INJECT_RATIO_RANGE = (0.15, 0.20)


class PruneDataset(Dataset):
    """Yields (deck_card_ids, labels, commander_ids): labels[i] == 1 if
    deck_card_ids[i] is really from this deck, 0 if it's a cross-deck real
    card added as a negative. A fresh corruption is sampled on every
    __getitem__ call (same spirit as DeckCompletionDataset's fresh mask per
    call), so the same deck contributes a different corrupted example
    across epochs.

    Two modes, both using the same cross-deck negative-sampling mechanism
    (_sample_negative) -- this isn't a new negative SOURCE, just a
    different way of applying it:
      - "swap" (default, unchanged from this class's original behavior):
        REPLACES corrupt_ratio_range's fraction of the real deck's own
        cards in place -- the deck stays its normal (~90-99 card) size
        throughout.
      - "inject": ADDS inject_ratio_range's fraction of extra cards on top
        of the (fully intact) real deck instead of replacing any of it --
        the resulting deck is genuinely LARGER than a normal deck, mirroring
        an oversized candidate pool that needs cutting DOWN to size, not a
        normal-sized deck with a few cards secretly swapped. See
        INJECT_RATIO_RANGE's comment for why this is worth training on."""

    def __init__(
        self,
        decks: list[ParsedDeck],
        tok: CardTokenizer,
        corrupt_ratio_range: tuple[float, float] = CORRUPT_RATIO_RANGE,
        seed: int | None = None,
        mode: str = "swap",
        inject_ratio_range: tuple[float, float] = INJECT_RATIO_RANGE,
    ):
        if mode not in ("swap", "inject"):
            raise ValueError(f"Unknown mode: {mode!r}")
        self.decks = decks
        self.tok = tok
        self.corrupt_ratio_range = corrupt_ratio_range
        self.mode = mode
        self.inject_ratio_range = inject_ratio_range
        self._rng = random.Random(seed)
        self._legal_mask_cache: dict[frozenset, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.decks)

    def _commander_colors(self, deck: ParsedDeck) -> frozenset:
        colors: set[str] = set()
        for cid in deck.commander_ids:
            card = self.tok.card_by_token[self.tok.id_to_token[cid]]
            colors |= set(card.color_identity)
        return frozenset(colors)

    def _legal_mask_for(self, colors: frozenset) -> torch.Tensor:
        cached = self._legal_mask_cache.get(colors)
        if cached is None:
            cached = legal_mask(self.tok, set(colors))
            self._legal_mask_cache[colors] = cached
        return cached

    def _sample_negative(self, mask: torch.Tensor, excluded: set[int]) -> int | None:
        for _ in range(MAX_SAMPLE_TRIES):
            other = self._rng.choice(self.decks)
            if not other.card_ids:
                continue
            cand = self._rng.choice(other.card_ids)
            if cand in excluded or not mask[cand]:
                continue
            return cand
        return None

    def __getitem__(self, idx: int) -> tuple[list[int], list[int], list[int]]:
        deck = self.decks[idx]
        cards = list(deck.card_ids)

        colors = self._commander_colors(deck)
        mask = self._legal_mask_for(colors)
        excluded = set(cards) | set(deck.commander_ids)

        if self.mode == "inject":
            ratio = self._rng.uniform(*self.inject_ratio_range)
            n_target = max(1, round(len(cards) * ratio))
            negatives: list[int] = []
            for _ in range(n_target):
                neg = self._sample_negative(mask, excluded)
                if neg is None:
                    continue
                excluded.add(neg)
                negatives.append(neg)

            all_cards = cards + negatives
            all_labels = [1] * len(cards) + [0] * len(negatives)
            # Shuffle so "corrupted" isn't trivially "at the end" -- PruneEncoder
            # has no positional encoding so this can't actually leak a
            # positional shortcut, but keeping position uninformative by
            # construction removes any doubt.
            order = list(range(len(all_cards)))
            self._rng.shuffle(order)
            return [all_cards[i] for i in order], [all_labels[i] for i in order], deck.commander_ids

        ratio = self._rng.uniform(*self.corrupt_ratio_range)
        n_target = max(1, round(len(cards) * ratio))

        positions = list(range(len(cards)))
        self._rng.shuffle(positions)
        labels = [1] * len(cards)
        n_done = 0
        for pos in positions:
            if n_done >= n_target:
                break
            neg = self._sample_negative(mask, excluded)
            if neg is None:
                continue
            excluded.add(neg)
            cards[pos] = neg
            labels[pos] = 0
            n_done += 1

        return cards, labels, deck.commander_ids


def make_prune_collate_fn(pad_id: int):
    """Pads deck/commander id lists to the batch max -- same shape
    convention as deck_dataset.make_collate_fn, but with a per-position
    label vector instead of a multi-hot whole-vocab target: there's no
    fixed vocab to score here, just "is this specific slot real"."""

    def collate_fn(batch):
        card_lists, label_lists, commanders = zip(*batch)
        batch_size = len(batch)
        max_deck = max((len(c) for c in card_lists), default=1) or 1
        max_cmdr = max((len(c) for c in commanders), default=1) or 1

        deck_ids = torch.full((batch_size, max_deck), pad_id, dtype=torch.long)
        deck_mask = torch.zeros((batch_size, max_deck), dtype=torch.bool)
        labels = torch.zeros((batch_size, max_deck), dtype=torch.float32)
        cmdr_ids = torch.full((batch_size, max_cmdr), pad_id, dtype=torch.long)
        cmdr_mask = torch.zeros((batch_size, max_cmdr), dtype=torch.bool)

        for i, (cards, lbls, cmdr) in enumerate(zip(card_lists, label_lists, commanders)):
            if cards:
                deck_ids[i, : len(cards)] = torch.tensor(cards, dtype=torch.long)
                deck_mask[i, : len(cards)] = True
                labels[i, : len(lbls)] = torch.tensor(lbls, dtype=torch.float32)
            if cmdr:
                cmdr_ids[i, : len(cmdr)] = torch.tensor(cmdr, dtype=torch.long)
                cmdr_mask[i, : len(cmdr)] = True

        return deck_ids, deck_mask, cmdr_ids, cmdr_mask, labels

    return collate_fn


class HardNegativeDeckDataset(Dataset):
    """Yields (real_card_ids, held_out_card_ids, commander_ids) for
    hard-negative pruner training (training/train_prune_hardneg.py) --
    unlike PruneDataset above, this does NOT pick the negative substitute
    itself. It only decides which cards to hold out; the substitute comes
    at TRAIN TIME from the (frozen) synergy generator's own top
    mispredictions when asked to complete a deck missing those cards (see
    train_prune_hardneg.build_hard_negative_batch) -- closer to the
    mistakes the pruner actually needs to catch in production, since a
    deck it prunes there was built by that same generator, rather than the
    cross-deck-real-card negatives above.

    held_out_card_ids are returned (not just their count) so the batch
    builder can exclude them from the generator's predictions -- otherwise
    a lucky correct guess could get labeled a "negative" by mistake."""

    def __init__(
        self,
        decks: list[ParsedDeck],
        corrupt_ratio_range: tuple[float, float] = CORRUPT_RATIO_RANGE,
        seed: int | None = None,
    ):
        self.decks = decks
        self.corrupt_ratio_range = corrupt_ratio_range
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.decks)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int], list[int]]:
        deck = self.decks[idx]
        cards = list(deck.card_ids)
        self._rng.shuffle(cards)

        ratio = self._rng.uniform(*self.corrupt_ratio_range)
        k = max(1, min(len(cards) - 1, round(len(cards) * ratio)))

        held_out = cards[:k]
        real = cards[k:]
        return real, held_out, deck.commander_ids


def make_hard_negative_collate_fn(pad_id: int):
    """Pads real/held-out/commander id lists to the batch max, plus an
    n_corrupt count per example -- the batch builder needs the count before
    it knows how many generator predictions to mine per example."""

    def collate_fn(batch):
        real_lists, held_out_lists, commanders = zip(*batch)
        batch_size = len(batch)
        max_real = max((len(c) for c in real_lists), default=1) or 1
        max_held = max((len(c) for c in held_out_lists), default=1) or 1
        max_cmdr = max((len(c) for c in commanders), default=1) or 1

        real_ids = torch.full((batch_size, max_real), pad_id, dtype=torch.long)
        real_mask = torch.zeros((batch_size, max_real), dtype=torch.bool)
        held_out_ids = torch.full((batch_size, max_held), pad_id, dtype=torch.long)
        held_out_mask = torch.zeros((batch_size, max_held), dtype=torch.bool)
        n_corrupt = torch.zeros((batch_size,), dtype=torch.long)
        cmdr_ids = torch.full((batch_size, max_cmdr), pad_id, dtype=torch.long)
        cmdr_mask = torch.zeros((batch_size, max_cmdr), dtype=torch.bool)

        for i, (real, held, cmdr) in enumerate(zip(real_lists, held_out_lists, commanders)):
            if real:
                real_ids[i, : len(real)] = torch.tensor(real, dtype=torch.long)
                real_mask[i, : len(real)] = True
            if held:
                held_out_ids[i, : len(held)] = torch.tensor(held, dtype=torch.long)
                held_out_mask[i, : len(held)] = True
            n_corrupt[i] = len(held)
            if cmdr:
                cmdr_ids[i, : len(cmdr)] = torch.tensor(cmdr, dtype=torch.long)
                cmdr_mask[i, : len(cmdr)] = True

        return real_ids, real_mask, held_out_ids, held_out_mask, n_corrupt, cmdr_ids, cmdr_mask

    return collate_fn


if __name__ == "__main__":
    from tokenizer.mtg_tokenizer import PAD
    from training.deck_dataset import load_decks_auto, split_decks

    tok = CardTokenizer.load_pretrained("./tokenizer.json")
    decks = load_decks_auto(tok)
    splits = split_decks(decks)

    ds = PruneDataset(splits["train"], tok, seed=0)
    cards, labels, cmdr = ds[0]
    n_corrupt = sum(1 for l in labels if l == 0)
    print(f"Sample: {len(cards)} deck cards, {n_corrupt} corrupted, {len(cmdr)} commander(s)")

    collate = make_prune_collate_fn(tok.token_to_id[PAD])
    batch = collate([ds[i] for i in range(4)])
    deck_ids, deck_mask, cmdr_ids, cmdr_mask, labels = batch
    print("Batch shapes:", deck_ids.shape, deck_mask.shape, cmdr_ids.shape, cmdr_mask.shape, labels.shape)

    hn_ds = HardNegativeDeckDataset(splits["train"], seed=0)
    real, held_out, cmdr = hn_ds[0]
    print(f"\nHard-negative sample: {len(real)} real cards, {len(held_out)} held out, {len(cmdr)} commander(s)")

    hn_collate = make_hard_negative_collate_fn(tok.token_to_id[PAD])
    hn_batch = hn_collate([hn_ds[i] for i in range(4)])
    real_ids, real_mask, held_out_ids, held_out_mask, n_corrupt, cmdr_ids, cmdr_mask = hn_batch
    print(
        "Hard-negative batch shapes:", real_ids.shape, real_mask.shape,
        held_out_ids.shape, held_out_mask.shape, n_corrupt.shape, cmdr_ids.shape, cmdr_mask.shape,
    )
