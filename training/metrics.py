"""
metrics.py

Shared evaluation logic for the deck-completion model: Recall@K, Precision@K,
and Mean Reciprocal Rank on held-out (masked) cards, broken out by card
popularity tier, plus embedding nearest-neighbor lookups. Factored out of
evaluate.py so train.py can also track a lightweight Recall@K during
training (for the loss/recall-vs-epoch plots) without a circular import
between the two.

Recall@K alone doesn't penalize a model for padding its top-K with
plausible-looking noise alongside the real answer -- a model that just
dumped every card into its top 100 would score well on Recall@100 while
being useless. Precision@K (what fraction of the top-K guesses were
actually right) and MRR (how highly, on average, are the real missing cards
actually ranked -- not just whether they cleared some fixed K) are tracked
alongside it for that reason.
"""

from __future__ import annotations

import random
from collections import Counter

import torch

from tokenizer.mtg_tokenizer import CardTokenizer
from training.model import DeckCompletionModel

# The tier_names to pass to recall_at_k alongside staple_color_tiers() below.
STAPLE_COLOR_TIER_NAMES = ("universal", "color_popular", "mid", "long_tail")

EVAL_MASK_RATIOS = (0.1, 0.3, 0.5, 0.7, 0.9)
EVAL_KS = (20, 50, 100)

# Popularity tiers for the breakdown below: "staple" = among the most common
# cards in training decks, "long_tail" = rare, "mid" = everything else.
STAPLE_TOP_N = 50
LONG_TAIL_MAX_DECKS = 20


def card_tiers(train_decks, min_count_for_mid: int = LONG_TAIL_MAX_DECKS) -> dict[int, str]:
    """token id -> 'staple' | 'mid' | 'long_tail', based on training-deck
    frequency (see module docstring)."""
    freq = Counter()
    for d in train_decks:
        for cid in d.card_ids:
            freq[cid] += 1

    ranked = [cid for cid, _ in freq.most_common()]
    tiers = {}
    for i, cid in enumerate(ranked):
        if i < STAPLE_TOP_N:
            tiers[cid] = "staple"
        elif freq[cid] < min_count_for_mid:
            tiers[cid] = "long_tail"
        else:
            tiers[cid] = "mid"
    return tiers


def staple_color_tiers(
    train_decks, tok: CardTokenizer, min_count_for_mid: int = LONG_TAIL_MAX_DECKS,
) -> dict[int, str]:
    """Like card_tiers, but splits the 'staple' bucket into 'universal'
    (colorless -- fits any commander regardless of color identity, e.g. Sol
    Ring) vs 'color_popular' (has a color identity -- popular within its
    colors, e.g. Swords to Plowshares, but not universally playable). "mid"
    and "long_tail" are unchanged.

    One-off diagnostic for a real training-run finding: staple Recall@50
    barely moved after fixing the
    pos_weight floor, while staple Precision@50 was already high (89.5%) and
    staple MRR stayed low (0.006). This tests whether that gap is explained
    by color-restricted "staples" needing real contextual reasoning to rank
    well (not a flaw), or whether it holds even for the universal,
    context-independent cards (a genuine narrow-confidence problem)."""
    freq = Counter()
    for d in train_decks:
        for cid in d.card_ids:
            freq[cid] += 1

    ranked = [cid for cid, _ in freq.most_common()]
    tiers = {}
    for i, cid in enumerate(ranked):
        if i < STAPLE_TOP_N:
            card = tok.card_by_token[tok.id_to_token[cid]]
            tiers[cid] = "universal" if not card.color_identity else "color_popular"
        elif freq[cid] < min_count_for_mid:
            tiers[cid] = "long_tail"
        else:
            tiers[cid] = "mid"
    return tiers


@torch.no_grad()
def recall_at_k(
    model: DeckCompletionModel,
    decks,
    tiers: dict[int, str],
    ks=EVAL_KS,
    mask_ratios=EVAL_MASK_RATIOS,
    device: torch.device = torch.device("cpu"),
    max_decks: int | None = None,
    seed: int = 123,
    tier_names: tuple[str, ...] = ("staple", "mid", "long_tail"),
) -> dict:
    """tier_names must cover every label that can appear in `tiers` (plus
    "mid", used as the fallback for any token id `tiers` doesn't cover) --
    defaults to card_tiers()'s three buckets, but pass a custom tuple
    alongside a custom `tiers` dict (see staple_color_tiers below) for a
    different breakdown without touching this function."""
    rng = random.Random(seed)

    hits = {(k, ratio): 0 for k in ks for ratio in mask_ratios}
    totals = {(k, ratio): 0 for k in ks for ratio in mask_ratios}
    # tier_* is recall-by-tier: of the true staple/mid/long_tail *targets*,
    # how many did we find. pred_tier_* is precision-by-tier: of the cards
    # the model actually *predicted* that happen to be staple/mid/long_tail,
    # how many were correct -- a different question (e.g. "when the model
    # recommends a staple, is it usually right?").
    tier_hits = {(k, tier): 0 for k in ks for tier in tier_names}
    tier_totals = {(k, tier): 0 for k in ks for tier in tier_names}
    pred_tier_hits = {(k, tier): 0 for k in ks for tier in tier_names}
    pred_tier_totals = {(k, tier): 0 for k in ks for tier in tier_names}
    n_examples_by_ratio = {ratio: 0 for ratio in mask_ratios}

    # MRR is K-independent (it's about how highly a target is ranked, not
    # whether it cleared a cutoff), so it's tracked once per ratio/tier
    # rather than per-K like recall/precision.
    rr_sum_by_ratio = {ratio: 0.0 for ratio in mask_ratios}
    rr_count_by_ratio = {ratio: 0 for ratio in mask_ratios}
    rr_sum_by_tier = {tier: 0.0 for tier in tier_names}
    rr_count_by_tier = {tier: 0 for tier in tier_names}

    eval_decks = decks if max_decks is None else decks[:max_decks]
    was_training = model.training
    model.eval()

    for deck in eval_decks:
        cards = list(deck.card_ids)
        rng.shuffle(cards)

        for ratio in mask_ratios:
            k_mask = min(len(cards), max(1, round(len(cards) * ratio)))
            target = cards[:k_mask]
            context = cards[k_mask:]

            ctx_ids = torch.tensor([context], dtype=torch.long, device=device)
            ctx_mask = torch.ones_like(ctx_ids, dtype=torch.bool)
            cmdr_ids = torch.tensor([deck.commander_ids], dtype=torch.long, device=device)
            cmdr_mask = torch.ones_like(cmdr_ids, dtype=torch.bool)

            logits = model(ctx_ids, ctx_mask, cmdr_ids, cmdr_mask)[0]
            # Exclude cards already visible (context + commander) from the
            # candidate ranking -- recommending what's already in the deck
            # is trivially "correct" and would just push real targets out
            # of the top-K.
            already_visible = set(context) | set(deck.commander_ids)
            logits[list(already_visible)] = float("-inf")

            # Full ranking (not just top max_k): needed for MRR, since a
            # target's exact rank matters even when it falls outside every K
            # we report Recall/Precision@K for.
            order = torch.argsort(logits, descending=True)
            ranks = torch.empty_like(order)
            ranks[order] = torch.arange(order.numel(), device=order.device)

            n_examples_by_ratio[ratio] += 1
            target_set = set(target)

            for k in ks:
                predicted_list = order[:k].tolist()
                predicted = set(predicted_list)
                found = predicted & target_set
                hits[(k, ratio)] += len(found)
                totals[(k, ratio)] += len(target)

                for tid in target:
                    tier = tiers.get(tid, "mid")
                    tier_totals[(k, tier)] += 1
                    if tid in predicted:
                        tier_hits[(k, tier)] += 1

                for pid in predicted_list:
                    tier = tiers.get(pid, "mid")
                    pred_tier_totals[(k, tier)] += 1
                    if pid in target_set:
                        pred_tier_hits[(k, tier)] += 1

            for tid in target:
                rr = 1.0 / (ranks[tid].item() + 1)  # +1: ranks are 0-indexed
                rr_sum_by_ratio[ratio] += rr
                rr_count_by_ratio[ratio] += 1
                tier = tiers.get(tid, "mid")
                rr_sum_by_tier[tier] += rr
                rr_count_by_tier[tier] += 1

    if was_training:
        model.train()

    by_ratio = {
        f"k={k}_ratio={ratio}": hits[(k, ratio)] / max(totals[(k, ratio)], 1)
        for k in ks for ratio in mask_ratios
    }
    by_tier = {
        f"k={k}_tier={tier}": tier_hits[(k, tier)] / max(tier_totals[(k, tier)], 1)
        for k in ks for tier in tier_names
    }
    # Precision@K: fraction of the top-K guesses that were actually correct.
    # Note the ceiling this is naturally bounded by: if a deck only has 8
    # held-out targets, Precision@50 can never exceed 8/50 = 0.16 even for a
    # perfect model -- it's capped by target-set size, not just model skill.
    precision_by_ratio = {
        f"k={k}_ratio={ratio}": hits[(k, ratio)] / max(k * n_examples_by_ratio[ratio], 1)
        for k in ks for ratio in mask_ratios
    }
    precision_by_tier = {
        f"k={k}_tier={tier}": pred_tier_hits[(k, tier)] / max(pred_tier_totals[(k, tier)], 1)
        for k in ks for tier in tier_names
    }
    mrr_by_ratio = {
        ratio: rr_sum_by_ratio[ratio] / max(rr_count_by_ratio[ratio], 1)
        for ratio in mask_ratios
    }
    mrr_by_tier = {
        tier: rr_sum_by_tier[tier] / max(rr_count_by_tier[tier], 1)
        for tier in tier_names
    }
    return {
        "n_decks": len(eval_decks),
        "by_ratio": by_ratio,
        "by_tier": by_tier,
        "precision_by_ratio": precision_by_ratio,
        "precision_by_tier": precision_by_tier,
        "mrr_by_ratio": mrr_by_ratio,
        "mrr_by_tier": mrr_by_tier,
    }


@torch.no_grad()
def nearest_neighbors(model: DeckCompletionModel, tok: CardTokenizer, name: str, k: int = 6, device=torch.device("cpu")):
    was_training = model.training
    model.eval()
    table = model.card_embedding.full_table().cpu()
    tid = tok.token_to_id[tok.resolve_name(name)]
    v = table[tid]
    sims = torch.nn.functional.cosine_similarity(table, v.unsqueeze(0), dim=1)
    top = torch.argsort(sims, descending=True)[1 : k + 1].tolist()
    if was_training:
        model.train()
    return [(tok.card_by_token[tok.id_to_token[i]].name, sims[i].item()) for i in top]
