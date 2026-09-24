"""
complete_deck.py

CLI: given a commander (or two, for partners) and an optional partial
decklist by name, print the model's top-K suggested completions -- the
actual deliverable of the project, not just an evaluation number.

Applies the deterministic color-identity legality filter described in the
training plan (a candidate's color identity must be a subset of the
commander(s)' color identity) as a post-filter on the model's raw scores,
rather than relying on the model to have learned the rule from data.

Generation strategy: single-shot (default) vs. iterative (--chunk-size /
--temperature) -- see complete_deck()'s docstring below for what these do
and why they exist. Single-shot (chunk_size=None) is unchanged from the
original implementation, so every existing example in this file's
docstring and README.md still reproduces exactly.

Usage:
    python3 -m training.complete_deck --commander "Atraxa, Praetors' Voice"
    python3 -m training.complete_deck --commander "Atraxa, Praetors' Voice" \\
        --card "Sol Ring" --card "Arcane Signet" --top 15
    python3 -m training.complete_deck \\
        --commander "Silas Renn, Seeker Adept" --commander "Rograkh, Son of Rohgahh"
    python3 -m training.complete_deck --commander "Meren of Clan Nel Toth" \\
        --chunk-size 5 --temperature 0.5
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from tokenizer.mtg_tokenizer import COLORS, SPECIAL_TOKENS, CardTokenizer
from training.evaluate import DEFAULT_CHECKPOINT, load_checkpoint
from training.model import DeckCompletionModel
from training.train import pick_device


class UnknownCardError(ValueError):
    """Raised when a given commander/card name doesn't resolve against the
    tokenizer's vocab. A plain ValueError subclass (not SystemExit) so
    non-CLI callers can catch it and turn it into a proper error response
    instead of killing the process; __main__ below is the one place that
    still turns it into a CLI exit."""


def legal_mask(tok: CardTokenizer, commander_colors: set[str]) -> torch.Tensor:
    """(vocab_size,) bool tensor: True where a card's color identity is a
    subset of commander_colors. Special tokens (all-zero feature rows) are
    excluded explicitly, since an empty color identity would otherwise
    trivially pass the subset check."""
    feats = tok.feature_matrix()
    color_cols = feats[:, : len(COLORS)]  # (vocab_size, 5), 0/1
    allowed = np.array([1.0 if c in commander_colors else 0.0 for c in COLORS], dtype=np.float32)
    illegal = (color_cols * (1.0 - allowed)).sum(axis=1) > 0
    mask = torch.tensor(~illegal)
    for tok_str in SPECIAL_TOKENS:
        mask[tok.token_to_id[tok_str]] = False
    return mask


def resolve_or_die(tok: CardTokenizer, name: str) -> int:
    token = tok.resolve_name(name)
    if token is None:
        raise UnknownCardError(f"Could not resolve card name: {name!r}")
    return tok.token_to_id[token]


def _run_model(model, tok, device, cmdr_tensor, cmdr_mask, picked: list[int], mask: torch.Tensor) -> torch.Tensor:
    """One forward pass, masked to -inf for already-picked cards and
    illegal colors. Shared by both generation modes below."""
    if picked:
        ctx_tensor = torch.tensor([picked], dtype=torch.long, device=device)
        ctx_mask = torch.ones_like(ctx_tensor, dtype=torch.bool)
    else:
        # Empty context is a valid case (commander-only completion) -- the
        # DeepSets encoder's mask-aware pool handles an all-False mask by
        # clamping the count to 1, giving a zero-vector context contribution
        # rather than dividing by zero. See training/model.py.
        ctx_tensor = torch.zeros((1, 1), dtype=torch.long, device=device)
        ctx_mask = torch.zeros((1, 1), dtype=torch.bool, device=device)

    logits = model(ctx_tensor, ctx_mask, cmdr_tensor, cmdr_mask)[0].cpu()
    already_visible = set(picked) | set(cmdr_tensor[0].tolist())
    logits[list(already_visible)] = float("-inf")
    logits[~mask] = float("-inf")
    return logits


def _select_chunk(logits: torch.Tensor, k: int, temperature: float | None, rng: random.Random) -> list[int]:
    """Selects up to k token ids from one set of (already-masked) logits.
    temperature=None is deterministic top-k (what the CLI's default and
    every single-shot call use). Otherwise samples k ids without
    replacement from softmax(logits/temperature) via sequential
    renormalization -- k is always small (<=~10 in practice) so the O(k^2)
    cost here is negligible."""
    pool_ids = torch.nonzero(logits > float("-inf"), as_tuple=True)[0]
    k_eff = min(k, len(pool_ids))
    if k_eff == 0:
        return []
    if temperature is None:
        top_idx = torch.topk(logits[pool_ids], k_eff).indices
        return pool_ids[top_idx].tolist()

    remaining_ids = pool_ids.tolist()
    remaining_scores = logits[pool_ids].clone()
    picked = []
    for _ in range(k_eff):
        probs = torch.softmax(remaining_scores / temperature, dim=0)
        pos = rng.choices(range(len(remaining_ids)), weights=probs.tolist(), k=1)[0]
        picked.append(remaining_ids[pos])
        del remaining_ids[pos]
        remaining_scores = torch.cat([remaining_scores[:pos], remaining_scores[pos + 1:]])
    return picked


@torch.no_grad()
def complete_deck(
    commanders: list[str],
    cards: list[str] | None = None,
    top: int = 20,
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    loaded: tuple[DeckCompletionModel, CardTokenizer, torch.device] | None = None,
    chunk_size: int | None = None,
    temperature: float | None = None,
    seed: int = 0,
) -> list[tuple[str, float]]:
    """`loaded`, if given, skips the (slow: disk + model init) checkpoint
    load and reuses an already-resident (model, tokenizer, device) -- for a
    caller that loads once at startup and must not re-load per request. The
    CLI path below leaves `loaded` unset and keeps loading fresh, as before.

    `chunk_size` / `temperature` select the generation strategy -- added in
    response to a real report that completions "feel like a card
    recommender, not a deck builder": individually reasonable-looking cards
    that don't add up to one coherent gameplan. Root cause: the default
    behavior below (chunk_size=None) runs ONE forward pass against the
    original `cards` and takes the top-`top` scores all at once, so pick
    #47 has no idea picks #1-46 exist -- nothing before this ever
    conditioned a new pick on the model's OWN prior picks in the same call,
    only on the caller's original input.

    chunk_size, when set, re-runs the forward pass every `chunk_size` picks
    instead of once for everything, feeding each new batch's picks back
    into the context first -- so pick #47 (if chunk_size divides evenly)
    really can react to #1-46. Small chunk sizes are maximally reactive but
    risk a self-reinforcing collapse: each pick nudges the deck-so-far
    representation further toward whatever it's already leaning on,
    making the next pick even more likely to be "more of the same" (real
    example found while testing this: a commander whose completion
    degenerated into a run of near-identical cheap filler spells under
    chunk_size=1). temperature counteracts that by sampling each chunk's
    picks from softmax(logits/temperature) instead of always taking the
    single top-scoring card, so a non-top pick occasionally wins and the
    run doesn't narrow into one repetitive lane. temperature is ignored
    when chunk_size is None (single-shot has no "chunk" to sample within).

    Recommended starting point if you want to try iterative generation:
    chunk_size=5, temperature=0.5 -- the best compromise found across an
    11-commander comparison (redundancy pulled back down from chunk_size=1's
    self-reinforcing collapse, while still reproducing chunk_size=1's clear
    win on at least one commander where single-shot stayed generic). It's a
    real, reportable improvement on one axis, not a settled best config, so
    don't take chunk_size=5/temperature=0.5 as more final than that.
    """
    if loaded is not None:
        model, tok, device = loaded
    else:
        device = pick_device()
        model, tok, _ckpt = load_checkpoint(checkpoint_path, device)

    cmdr_ids = [resolve_or_die(tok, name) for name in commanders]
    card_ids = [resolve_or_die(tok, name) for name in (cards or [])]

    cmdr_tensor = torch.tensor([cmdr_ids], dtype=torch.long, device=device)
    cmdr_mask = torch.ones_like(cmdr_tensor, dtype=torch.bool)

    commander_colors: set[str] = set()
    for cid in cmdr_ids:
        card = tok.card_by_token[tok.id_to_token[cid]]
        commander_colors |= set(card.color_identity)
    mask = legal_mask(tok, commander_colors)

    if chunk_size is None:
        # Single-shot: exactly the original implementation, so every
        # existing documented example reproduces byte-identically.
        logits = _run_model(model, tok, device, cmdr_tensor, cmdr_mask, card_ids, mask)
        scores = torch.sigmoid(logits)
        top_ids = torch.topk(scores, top).indices.tolist()
        return [(tok.card_by_token[tok.id_to_token[i]].name, scores[i].item()) for i in top_ids]

    rng = random.Random(seed)
    picked = list(card_ids)
    results: list[tuple[int, float]] = []
    while len(results) < top:
        k = min(chunk_size, top - len(results))
        logits = _run_model(model, tok, device, cmdr_tensor, cmdr_mask, picked, mask)
        chosen = _select_chunk(logits, k, temperature, rng)
        if not chosen:
            break  # ran out of legal, unpicked candidates
        for cid in chosen:
            picked.append(cid)
            results.append((cid, torch.sigmoid(logits[cid]).item()))

    return [(tok.card_by_token[tok.id_to_token[cid]].name, score) for cid, score in results]


def _print_completion(
    commanders: list[str], cards: list[str], results: list[tuple[str, float]],
    chunk_size: int | None, temperature: float | None,
) -> None:
    cmdr_str = " + ".join(commanders)
    print(f"\nCommander(s): {cmdr_str}")
    if cards:
        print(f"Known cards ({len(cards)}): {', '.join(cards)}")
    else:
        print("Known cards: (none -- completing from commander alone)")
    if chunk_size is None:
        print("Generation strategy: single-shot")
    else:
        print(f"Generation strategy: iterative (chunk_size={chunk_size}, temperature={temperature})")
    print(f"\nTop {len(results)} suggestions:")
    for name, score in results:
        print(f"  {score:.3f}  {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--commander", action="append", required=True, dest="commanders",
                         help="Commander name; repeat for partner commanders.")
    parser.add_argument("--card", action="append", default=[], dest="cards",
                         help="A card already in the deck; repeat for each known card.")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--chunk-size", type=int, default=None,
        help=(
            "Re-run the model every N picks instead of once for the whole completion "
            "(the default, single-shot). Each new batch of picks is conditioned on "
            "everything picked so far in this run -- including the model's own prior "
            "picks, not just --card -- fixing single-shot's tendency to produce a pile "
            "of independently-good-looking cards instead of one coherent gameplan. "
            "Small values react fastest but risk a self-reinforcing narrow collapse "
            "(see --temperature). Recommended starting point: 5. See complete_deck()'s "
            "docstring for the full writeup."
        ),
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help=(
            "Only used with --chunk-size. Samples each chunk's picks from "
            "softmax(logit/temperature) instead of always taking the single top-scoring "
            "card, so a non-top pick occasionally wins instead of --chunk-size narrowing "
            "into one repetitive lane. Omit for deterministic greedy picks within each "
            "chunk. Recommended starting point: 0.5."
        ),
    )
    args = parser.parse_args()

    try:
        results = complete_deck(
            args.commanders, args.cards, top=args.top, checkpoint_path=args.checkpoint,
            chunk_size=args.chunk_size, temperature=args.temperature,
        )
    except UnknownCardError as e:
        raise SystemExit(str(e))
    _print_completion(args.commanders, args.cards, results, args.chunk_size, args.temperature)
