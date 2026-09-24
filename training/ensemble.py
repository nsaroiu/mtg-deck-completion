"""
ensemble.py

Deficit-based staple/synergy ensemble: instead of asking one model to hold
both "ranks staples well" and "ranks synergy/mid-tier cards well" under one
loss (a trade-off no loss function tried here escapes -- see
train_two_head.py and losses.py for the attempts and why each was
rejected), combine two already-trained SPECIALIZED checkpoints' decisions
at inference time -- generate normally with the synergy specialist, then
deterministically correct the staple count toward a target measured from
real decks. "Generate, then deterministically correct toward a
real-data-derived target" is a general pattern worth reusing wherever a
hard constraint (like a target card count) needs to be met exactly rather
than hoped for statistically.

Checkpoint choice, revised from this module's first version after a direct
test settled a real question -- does a BCE-trained model actually deserve
its "staple specialist" role, or was that assumed rather than measured?
Checked directly: given the ~50 legal, not-yet-picked staple-tier
candidates (exactly what the backfill phase below ranks), which one is the
true held-out target -- across both the jointly-trained two-head staple
head AND the standalone bce_baseline checkpoint, BCE LOSES to softmax at
this exact task (bce_baseline: 69.6% top-1 / 0.794 MRR; softmax: 84.7% /
0.908; set_transformer_softmax: 82.7% / 0.899). BCE's real strength (90%+ whole-vocab
precision) comes from confidently, independently recognizing a handful of
always-safe staples -- it was never trained to compare two competing
staples against EACH OTHER, which is exactly the job backfill needs and
exactly what softmax's competitive/ranking objective is built for. So:
  - staple specialist (backfill phase): deepsets_checkpoint.pt's SOFTMAX
    sibling, deepsets_softmax_checkpoint.pt -- not BCE. The actual measured
    best at this specific restricted-pool ranking task, marginally ahead
    of even set_transformer_softmax there (84.7% vs 82.7%).
  - synergy specialist (main phase): set_transformer_softmax_checkpoint.pt,
    unchanged -- mid-tier MRR 0.0579, the highest of all checkpoints
    evaluated this session, and its job (open-ended iterative generation)
    is a different task than the backfill phase's narrow restricted-pool
    ranking.
No BCE-trained checkpoint plays any role in this module anymore -- both
phases are softmax-family models, differing only in encoder, each chosen
because it's the measured best at its specific job. "staple_model"/
"staple_view" throughout describes the ROLE (does staple-tier backfill),
not the loss it happens to be trained with.

TARGET_STAPLE_COUNT_BY_COLORS (see below) gives the target staple-tier card
count for a full completion, looked up by the commander(s)' own
color-identity size -- calibrated from real per-color-identity-size
averages across all 122,464 real decks (training.metrics.card_tiers over
every real deck's card_ids). Not a single flat constant: real average staple
count varies close to 5x by color-identity size (4.0 for colorless up to
18.5 for 4-color), so a flat target would badly overshoot narrow commanders
and undershoot wide ones -- see the constant's own comment for the numbers
and the legality check that ruled out the 50-card staple-tier pool itself
ever being the binding constraint instead.

Order matters: the synergy phase runs FIRST, against the caller's own
clean context, using the iterative chunk_size=5/temperature=0.5 strategy
an 11-commander generation-strategy comparison found gives the most
coherent completions (see complete_deck()'s docstring for the full
writeup) -- not single-shot, since coherence is the whole
reason to have a synergy specialist. Staple sufficiency is corrected
AFTERWARD, by trimming the synergy phase's own lowest-confidence non-staple
picks and backfilling with the staple specialist -- never the reverse:
reserving staple slots up front would hand the synergy model a context
pre-diluted with several context-independent picks before it ever runs,
undermining the coherence it exists to provide.

An optional THIRD phase -- pass prune_model to enable it -- closes the
loop the other direction: every phase above only ever ADDS cards, nothing
ever reconsiders one already in the list. training/train_prune.py trains a
dedicated PruneModel (see training/model.py) as a real-vs-corrupted
discriminator, and this phase uses it to find the current list's weakest-
fitting picks, trim them, and refill via the synergy specialist -- the same
"generate, then correct" shape as the staple phase, not a new mechanism.
Scope is deliberately restricted to NONLAND, NON-STAPLE-TIER picks: a
direct qualitative check (score real and model-generated decks, read the
bottom-8 by predicted fit) found the prune model's cross-deck-negative
training signal is noisy specifically for lands and generic staples --
both are near-interchangeable across same-color decks BY DESIGN (a dual
land or Sol Ring genuinely fits almost any deck in its colors), so a
cross-deck "swap" there is often still a perfectly fine card, not a real
corruption the way a synergy card pulled from an unrelated deck usually
is. Confirmed live: Dark Ritual and Blood Crypt (both staple-tier) were
the MOST confidently "flagged" cards in a Ghen, Arcanum Weaver test deck
(fit scores 0.019/0.078) despite being perfectly reasonable includes,
while genuinely narrow misfits among synergy-tier picks (Aftershock for
Krark, the Thumbless: 0.001; Beetleback Chief: 0.071) were caught
correctly. Restricting the prune model's authority to the exact card
population it was actually discriminative about -- not retraining --
fixes this the same way legality/staple-deficit corrections elsewhere in
this module are deterministic post-filters rather than something asked of
a model to get right on its own.

Usage:
    python3 -m training.ensemble --commander "Ghen, Arcanum Weaver"
    python3 -m training.ensemble --commander "Ghen, Arcanum Weaver" \\
        --card "Sol Ring" --card "Command Tower" --top 30
    python3 -m training.ensemble --commander "Ghen, Arcanum Weaver" --single-shot
"""

from __future__ import annotations

import argparse
import json
import random

import torch

from tokenizer.mtg_tokenizer import CardTokenizer
from training.complete_deck import _run_model, _select_chunk, complete_deck, legal_mask, resolve_or_die
from training.evaluate import load_checkpoint
from training.train import pick_device

STAPLE_CHECKPOINT = "./data/deepsets_softmax_checkpoint.pt"
SYNERGY_CHECKPOINT = "./data/set_transformer_softmax_checkpoint.pt"

# training/train_prune.py's PruneModel, trained in "inject" mode (extra
# cross-deck negatives ADDED on top of an intact real deck, rather than
# swapped in place) at the ORIGINAL accepted density (~3-8%, prune_dataset.
# CORRUPT_RATIO_RANGE) -- not prune_checkpoint.pt (the original "swap" mode
# checkpoint), and deliberately not either of the higher-density variants
# tried along the way (prune_checkpoint_inject.pt / _swap_highdensity.pt),
# both of which were rejected after a live ablation found corruption
# density -- not deck size -- is what degrades calibration: a model trained
# at ~15-20% negatives learns to shift its whole output distribution toward
# that base rate (the same class-imbalance-calibration lesson as this
# project's own pos_weight/MIN_POS_WEIGHT story for the completion model),
# and in practice that meant real cuts to cards like Wheel of Fortune,
# Smothering Tithe, and Solitude -- not sharper judgment. Density held at
# the original safe level while deck size was allowed to genuinely exceed
# 100 cards showed NONE of that degradation (val_loss and real-deck
# false-positive rate both matched the original), so this checkpoint keeps
# the original's proven quality while also being the one actually trained
# on oversized candidate pools -- a direct step toward a possible future
# "trim my >100-card decklist down to size" feature, at no measured cost.
PRUNE_CHECKPOINT = "./data/prune_checkpoint_inject_lowdensity.pt"
TIERS_PATH = "./data/card_tiers.json"

# Real staple-tier card count, computed live from all 122,464 real decks,
# grouped by commander color-identity SIZE (not exact WUBRG combination --
# a cleaner 6-bucket split with plenty of decks per bucket). Replaces a
# flat TARGET_STAPLE_COUNT=12 (this module's original single global
# average) after checking directly: the true average varies nearly 5x
# across bucket, from 4.0 (colorless) to 18.5 (4-color) -- a flat 12 would
# have overshot colorless commanders by ~3x and undershot 4-color ones by
# ~35%. The other half of that check -- whether the 50-card staple-tier
# pool's LEGAL subset is ever the binding constraint instead -- came back
# negative: even colorless commanders have 26 of the 50 legally available,
# comfortably above the real ~4-card target, so legality never actually
# caps this in practice; only the real-count lookup below is needed.
TARGET_STAPLE_COUNT_BY_COLORS = {0: 4, 1: 6, 2: 10, 3: 14, 4: 19, 5: 15}


def target_staple_count_for(commander_colors: set[str]) -> int:
    return TARGET_STAPLE_COUNT_BY_COLORS[len(commander_colors)]


def _is_land(tok: CardTokenizer, cid: int) -> bool:
    return "Land" in tok.card_by_token[tok.id_to_token[cid]].type_line


def load_tiers(tok: CardTokenizer, path: str = TIERS_PATH) -> dict[int, str]:
    """oracle_id-keyed JSON (training/export_card_tiers.py) -> token-id-keyed
    dict, resolved against the currently-loaded tokenizer's vocab."""
    with open(path) as f:
        by_oracle_id: dict[str, str] = json.load(f)
    tiers: dict[int, str] = {}
    for oracle_id, tier in by_oracle_id.items():
        token = tok.oracle_id_to_token.get(oracle_id)
        if token is not None:
            tiers[tok.token_to_id[token]] = tier
    return tiers


def _prune_round(
    tagged: list[tuple[str, float, str]],
    known_ids: list[int],
    commander_colors: set[str],
    tok: CardTokenizer,
    device: torch.device,
    prune_model,
    synergy_model,
    tiers: dict[int, str],
    max_cuts: int,
    prune_threshold: float,
    chunk_size: int | None,
    temperature: float | None,
    cmdr_tensor: torch.Tensor,
    cmdr_mask: torch.Tensor,
) -> tuple[list[tuple[str, float, str]], int]:
    """One cut-then-refill round -- see _prune_and_refill below for why this
    runs in an outer loop instead of once. Iteratively cuts the single
    worst-scoring eligible (nonland, non-staple-tier, model-suggested)
    pick, RE-SCORING the remaining deck after each cut, until nothing left
    scores below prune_threshold or max_cuts cuts have been made this round
    -- then refills every cut slot in one pass via the synergy specialist,
    restricted to legal + unpicked candidates same as every other
    generation call in this module. Returns (new_tagged, n_cut).

    prune_threshold matters as much as the land/staple scope restriction
    (see module docstring): an early version always cut a fixed count
    regardless of confidence, which meant an already-good completion (every
    eligible card scoring >0.88 "real" -- confirmed live on a Meren, Clan
    Nel Toth completion) still had its relatively-lowest-of-a-high-scoring-
    bunch cards cut, including Jarad, Golgari Lich Lord (0.977) -- a
    genuinely strong, on-theme pick, not a weak link. 0.5 is PruneModel's
    own calibrated decision boundary (see training/train_prune.py): only
    cut a card the model actually doubts.

    Scoring is ITERATIVE within a round, not one static batch pass, for the
    same reason the synergy phase above moved deck completion off
    single-shot: when a few candidate cards are mutually
    redundant, each one's score is computed WITH the others still present
    in context, so a single batch pass can't tell "these 4 are all
    individually a bit weak" apart from "these 4 are collectively redundant
    and 1-2 of them leaving would fix the rest" -- only re-scoring after
    each cut lets a later candidate's fit be judged against the deck as it
    actually stands once an earlier cut has already happened."""
    cut_names: list[str] = []
    for _ in range(max_cuts):
        tagged_ids = [resolve_or_die(tok, name) for name, _, _ in tagged]
        eligible_idx = [
            i for i, cid in enumerate(tagged_ids)
            if not _is_land(tok, cid) and tiers.get(cid) != "staple"
        ]
        if not eligible_idx:
            break

        full_deck_ids = known_ids + tagged_ids
        deck_tensor = torch.tensor([full_deck_ids], dtype=torch.long, device=device)
        deck_mask = torch.ones_like(deck_tensor, dtype=torch.bool)

        logits = prune_model(deck_tensor, deck_mask, cmdr_tensor, cmdr_mask)[0].cpu()
        tagged_probs = torch.sigmoid(logits[len(known_ids):])  # aligns with `tagged`'s order

        worst_idx = min(eligible_idx, key=lambda i: tagged_probs[i].item())
        if tagged_probs[worst_idx].item() >= prune_threshold:
            break  # nothing left the model actually doubts

        cut_names.append(tagged[worst_idx][0])
        tagged = [t for i, t in enumerate(tagged) if i != worst_idx]

    if not cut_names:
        return tagged, 0

    kept_ids = [resolve_or_die(tok, name) for name, _, _ in tagged]
    mask = legal_mask(tok, commander_colors)
    picked = known_ids + kept_ids
    rng = random.Random(0)
    n_needed = len(cut_names)
    refill: list[tuple[str, float, str]] = []
    while len(refill) < n_needed:
        k = min(chunk_size or n_needed, n_needed - len(refill))
        chunk_logits = _run_model(synergy_model, tok, device, cmdr_tensor, cmdr_mask, picked, mask)
        chosen = _select_chunk(chunk_logits, k, temperature, rng)
        if not chosen:
            break
        for cid in chosen:
            picked.append(cid)
            refill.append((
                tok.card_by_token[tok.id_to_token[cid]].name,
                torch.sigmoid(chunk_logits[cid]).item(),
                "synergy",
            ))

    return tagged + refill, len(cut_names)


def _prune_and_refill(
    tagged: list[tuple[str, float, str]],
    cmdr_ids: list[int],
    known_ids: list[int],
    commander_colors: set[str],
    tok: CardTokenizer,
    device: torch.device,
    prune_model,
    synergy_model,
    tiers: dict[int, str],
    max_prune: int,
    max_rounds: int,
    prune_threshold: float,
    chunk_size: int | None,
    temperature: float | None,
) -> list[tuple[str, float, str]]:
    """Phase 3 -- see module docstring for scope/motivation. Runs up to
    max_rounds full cut-then-refill cycles (_prune_round above), stopping
    the moment a round makes zero cuts. One round alone was found to have a
    real gap: it cuts everything it's going to cut, then refills ONCE at
    the end, so a freshly refilled card is never itself scrutinized -- if
    it's a weak fit, or if removing one card changes the picture enough
    that a second, previously-fine-looking card now genuinely crosses
    prune_threshold too, one round can't catch it. Repeating the whole
    cycle closes that gap.

    This is deliberately NOT the same as just raising max_prune and letting
    one round cut more: a fixed-count "cut the worst N, repeat a few times,
    no cutoff" design was considered and rejected, because "worst-scoring"
    and "bad" aren't the same thing once a completion is already solid (see
    _prune_round's docstring's Jarad example) -- a cutoff-free version would
    keep finding SOMETHING to cut every round forever, eroding an already-
    good deck instead of refining it. prune_threshold stays the actual
    stopping condition every round; max_rounds/max_prune are only safety
    caps on top of it, and most calls stop in 0-1 rounds because nothing
    crosses threshold at all."""
    cmdr_tensor = torch.tensor([cmdr_ids], dtype=torch.long, device=device)
    cmdr_mask = torch.ones_like(cmdr_tensor, dtype=torch.bool)

    total_cuts = 0
    for _ in range(max_rounds):
        budget = max_prune - total_cuts
        if budget <= 0:
            break
        tagged, n_cut = _prune_round(
            tagged, known_ids, commander_colors, tok, device,
            prune_model, synergy_model, tiers, budget, prune_threshold,
            chunk_size, temperature, cmdr_tensor, cmdr_mask,
        )
        total_cuts += n_cut
        if n_cut == 0:
            break

    return tagged


@torch.no_grad()
def ensemble_complete_deck(
    commanders: list[str],
    cards: list[str] | None,
    top: int,
    tok: CardTokenizer,
    device: torch.device,
    staple_model,
    synergy_model,
    tiers: dict[int, str],
    chunk_size: int | None = 5,
    temperature: float | None = 0.5,
    target_staple_count: int | None = None,
    prune_model=None,
    max_prune: int = 10,
    max_rounds: int = 3,
    prune_threshold: float = 0.5,
) -> list[tuple[str, float, str]]:
    """Returns `top` (name, score, source) triples, source in
    {"synergy", "staple"} -- see module docstring for the two/three-phase
    flow.

    target_staple_count: None (the default) looks up a real-data-derived
    target from the commander(s)' own color-identity size via
    target_staple_count_for/TARGET_STAPLE_COUNT_BY_COLORS -- pass an
    explicit int to override with a fixed target instead.

    prune_model: None (the default) disables the third phase entirely --
    every existing caller reproduces exactly. Pass a loaded PruneModel to
    enable it. Pruning is iterative and threshold-gated, not a fixed count,
    across up to max_rounds full cut-then-refill cycles (see
    _prune_and_refill): it keeps cutting the single worst-scoring eligible
    (nonland, non-staple-tier) pick and re-scoring the rest, refilling and
    re-examining the result, for as long as something still scores below
    prune_threshold (default 0.5, the model's own calibrated decision
    boundary) -- so a completion with several genuinely weak picks (even
    ones only revealed after an earlier round's cuts) gets all of them
    addressed, while one with nothing weak is returned unchanged after
    round 1 finds nothing to do. max_prune/max_rounds are safety caps
    against runaway over-pruning, not targets."""
    cards = cards or []

    # -- phase 1: synergy specialist, iterative by default (see module
    # docstring for why this goes first and why iterative is the default) --
    synergy_picks = complete_deck(
        commanders, cards, top=top,
        loaded=(synergy_model, tok, device),
        chunk_size=chunk_size, temperature=temperature,
    )

    cmdr_ids = [resolve_or_die(tok, name) for name in commanders]
    known_ids = [resolve_or_die(tok, name) for name in cards]
    synergy_ids = [resolve_or_die(tok, name) for name, _ in synergy_picks]

    commander_colors: set[str] = set()
    for cid in cmdr_ids:
        card = tok.card_by_token[tok.id_to_token[cid]]
        commander_colors |= set(card.color_identity)

    def is_staple(cid: int) -> bool:
        return tiers.get(cid) == "staple"

    if target_staple_count is None:
        target_staple_count = target_staple_count_for(commander_colors)
    staples_present = sum(is_staple(c) for c in known_ids) + sum(is_staple(c) for c in synergy_ids)
    staple_deficit = min(max(0, target_staple_count - staples_present), top)

    tagged: list[tuple[str, float, str]] = [(name, score, "synergy") for name, score in synergy_picks]

    cmdr_tensor = torch.tensor([cmdr_ids], dtype=torch.long, device=device)
    cmdr_mask = torch.ones_like(cmdr_tensor, dtype=torch.bool)

    if staple_deficit == 0:
        result = tagged
    else:
        # -- trim: lowest-scoring non-staple synergy picks make room. A
        # staple the synergy model picked on its own is never trimmed -- it
        # earned that slot, same as the land-backfill logic never trims a
        # nonbasic land the model itself suggested. --
        non_staple_idx = [i for i, cid in enumerate(synergy_ids) if not is_staple(cid)]
        non_staple_idx.sort(key=lambda i: synergy_picks[i][1])  # ascending score
        trim_idx = set(non_staple_idx[:staple_deficit])
        kept = [t for i, t in enumerate(tagged) if i not in trim_idx]
        kept_ids = [cid for i, cid in enumerate(synergy_ids) if i not in trim_idx]

        # -- backfill: one single-shot pass through the staple specialist,
        # restricted to legal + unpicked + staple-tier candidates --
        mask = legal_mask(tok, commander_colors)

        already_present = known_ids + kept_ids
        logits = _run_model(staple_model, tok, device, cmdr_tensor, cmdr_mask, already_present, mask)
        scores = torch.sigmoid(logits)
        staple_pool = torch.tensor(
            [cid for cid in range(logits.shape[0]) if tiers.get(cid) == "staple"], dtype=torch.long,
        )
        pool_logits = logits[staple_pool]
        k = min(staple_deficit, int((pool_logits > float("-inf")).sum().item()))
        backfill = []
        if k > 0:
            top_idx = torch.topk(pool_logits, k).indices
            backfill = [
                (tok.card_by_token[tok.id_to_token[int(staple_pool[i])]].name, scores[staple_pool[i]].item(), "staple")
                for i in top_idx
            ]
        result = kept + backfill

    if prune_model is None:
        return result

    return _prune_and_refill(
        result, cmdr_ids, known_ids, commander_colors, tok, device,
        prune_model, synergy_model, tiers, max_prune, max_rounds, prune_threshold, chunk_size, temperature,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commander", action="append", required=True, dest="commanders")
    ap.add_argument("--card", action="append", default=[], dest="cards")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--staple-checkpoint", default=STAPLE_CHECKPOINT)
    ap.add_argument("--synergy-checkpoint", default=SYNERGY_CHECKPOINT)
    ap.add_argument("--chunk-size", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument(
        "--single-shot", action="store_true",
        help="Use single-shot generation for the synergy phase instead of the default iterative strategy (for comparison).",
    )
    ap.add_argument(
        "--prune-checkpoint", default=None,
        help="Path to a PruneModel checkpoint (training/train_prune.py) to enable the third "
             "prune-and-refill phase. Omit to disable it (the default).",
    )
    ap.add_argument(
        "--max-prune", type=int, default=10,
        help="prune-checkpoint only. Safety cap on total cuts across all rounds, not a "
             "target -- pruning stops as soon as a round finds nothing below --prune-threshold.",
    )
    ap.add_argument(
        "--max-rounds", type=int, default=3,
        help="prune-checkpoint only. Safety cap on cut-then-refill rounds -- stops early "
             "the moment a round makes zero cuts, so most completions use far fewer.",
    )
    ap.add_argument(
        "--prune-threshold", type=float, default=0.5,
        help="prune-checkpoint only. Only cut a pick scoring below this real-probability; "
             "default 0.5 is the model's own calibrated decision boundary.",
    )
    args = ap.parse_args()

    device = pick_device()
    staple_model, tok, _ = load_checkpoint(args.staple_checkpoint, device)
    synergy_model, _, _ = load_checkpoint(args.synergy_checkpoint, device)
    tiers = load_tiers(tok)

    prune_model = None
    if args.prune_checkpoint:
        from training.train_prune import load_prune_checkpoint
        prune_model, _, _ = load_prune_checkpoint(args.prune_checkpoint, device)

    chunk_size = None if args.single_shot else args.chunk_size
    temperature = None if args.single_shot else args.temperature

    results = ensemble_complete_deck(
        args.commanders, args.cards, args.top, tok, device,
        staple_model, synergy_model, tiers,
        chunk_size=chunk_size, temperature=temperature,
        prune_model=prune_model, max_prune=args.max_prune, max_rounds=args.max_rounds,
        prune_threshold=args.prune_threshold,
    )

    cmdr_str = " + ".join(args.commanders)
    print(f"\nCommander(s): {cmdr_str}")
    print(f"Known cards ({len(args.cards)}): {', '.join(args.cards) if args.cards else '(none)'}")
    strategy = "single-shot" if args.single_shot else f"iterative (chunk_size={args.chunk_size}, temperature={args.temperature})"
    print(f"Synergy phase strategy: {strategy}")
    n_staple = sum(1 for _, _, src in results if src == "staple")
    print(f"\nTop {len(results)} suggestions ({n_staple} staple-backfilled, {len(results) - n_staple} synergy):")
    for name, score, source in results:
        print(f"  {score:.3f}  [{source:7}]  {name}")


if __name__ == "__main__":
    main()
