"""
losses.py

Alternative training losses for the deck-completion model, built to address
a specific, measured problem: staple-tier cards (Sol Ring, Command Tower,
...) get decent precision (~90% -- when the model predicts one, it's usually
right) but terrible MRR (~0.007 -- even when it IS the true held-out target,
it's typically ranked far below the top). The same signature showed up
identically across every encoder architecture tried (DeepSets and Set
Transformer both), which is why the fix being tried here is a loss change,
not another architecture change.

training/train.py's default loss (BCEWithLogitsLoss with a per-card
pos_weight) is a POINTWISE loss: every one of the ~30K vocab logits is
scored independently against its own 0/1 label, with no term that ever
compares two cards' scores to each other. That's a real mismatch with the
task -- Recall@K/MRR are about relative ORDER, and nothing in plain BCE
penalizes "target card scored below a wrong card" specifically, only "this
card's own score is far from its own label." Both losses below attack that
mismatch, from two different angles -- see each one's docstring.

Both loss functions share BCEWithLogitsLoss's calling convention,
`loss_fn(logits, target) -> scalar`, so train.py's training loop and
evaluate_loss() need no changes to use either one.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def multi_positive_softmax_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    invalid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Softmax cross-entropy generalized to multiple simultaneous positives
    per example (a training example can have up to ~95 held-out target
    cards at once, at the wide end of DEFAULT_MASK_RATIO_RANGE) -- the
    direct fix for "high precision, low MRR": under plain BCE, a card only
    needs its own logit close to 1 regardless of what else is in the
    vocabulary; under softmax, every card's probability is normalized
    against EVERY other card's score, so a card can only score highly by
    actually outscoring its competitors, which is exactly what MRR
    measures.

    logits, target: (B, V). invalid_mask: (V,) bool, True = never a valid
    prediction (special tokens like [PAD]/[MASK]/[CMDR] -- see
    tokenizer.mtg_tokenizer.SPECIAL_TOKENS) -- excluded from the softmax
    denominator so their scores can't dilute probability mass that should
    go to real cards.

    Per-example loss is -log P(target) averaged over that example's own
    positives (not summed), so examples with many simultaneous targets and
    examples with few both contribute a comparably-scaled loss -- without
    this per-example averaging, an example with 90 targets would dominate
    the batch gradient over one with 5 purely by having more terms to sum,
    which has nothing to do with how well-ranked either one's targets are.

    Real tradeoff, not hidden: positives within the SAME example now also
    compete against each other for softmax probability mass (unlike BCE,
    where every positive can independently approach 1) -- more simultaneous
    targets means less probability mass available per target, a form of
    imbalance BCE's pos_weight scheme doesn't have and this loss doesn't
    correct for. Also drops pos_weight's explicit frequency reweighting
    entirely; if a trained checkpoint still shows frequency-correlated
    ranking problems, that's the first place to add it back, not assumed
    away here.
    """
    masked_logits = logits.masked_fill(invalid_mask, float("-inf")) if invalid_mask is not None else logits
    log_probs = F.log_softmax(masked_logits, dim=-1)
    # log_probs is -inf exactly at invalid_mask's positions (correctly zero
    # probability there). target is always 0 at those same positions (a
    # special token is never a real target) -- but 0 * (-inf) is NaN, not 0,
    # in IEEE float arithmetic, so left alone this poisons the whole sum.
    # nan_to_num after the fact is safe: it only replaces the already-
    # -inf VALUE at positions target is guaranteed to be 0 at, it doesn't
    # change what the softmax denominator excluded during log_softmax above.
    log_probs = torch.nan_to_num(log_probs, neginf=0.0)
    n_targets = target.sum(dim=-1).clamp(min=1.0)
    per_example = -(target * log_probs).sum(dim=-1) / n_targets
    return per_example.mean()


class HybridStapleRankingLoss:
    """BCEWithLogitsLoss (unchanged, including its pos_weight correction)
    plus a small auxiliary pairwise ranking term scoped ONLY to staple-tier
    cards (see training/metrics.py's card_tiers, STAPLE_TOP_N=50) -- the
    most surgical of the two options tried here: it targets exactly the
    diagnosed problem (staples specifically) without touching the loss's
    behavior on mid/long-tail cards, which is already the model's real
    strength (mid-tier MRR ~0.05, the source of the genuine synergy-pick
    wins in real completions) -- a general loss change risks regressing
    that to fix something narrower.

    For every (example, staple-tier target) pair in a batch, samples a
    shared pool of `n_neg` random vocab columns (shared across the WHOLE
    batch for one training step, not resampled per example -- cheap, and
    every example still gets its own logits at those columns) and applies a
    hinge margin loss: max(0, margin - (staple_score - neg_score)), pushing
    the staple's score above the sampled negatives' by at least `margin`.
    Only ~50 staple-tier columns are ever gathered (not the full ~30K
    vocab), so this stays cheap: (B, 50, n_neg) tensor, not (B, V, n_neg).

    Known, accepted approximation (same tradition as word2vec/NCE-style
    negative sampling): `n_neg` columns are sampled uniformly at random from
    the full vocab with no explicit exclusion of true positives -- at this
    vocab size (~30K) an accidental true-positive-as-negative collision is
    rare and, per NCE's own justification, a small amount of that noise is
    an accepted cost of avoiding the bookkeeping of an exact per-example
    exclusion set.
    """

    def __init__(
        self,
        base_loss_fn,
        staple_indices: torch.Tensor,
        lam: float = 0.5,
        margin: float = 1.0,
        n_neg: int = 64,
    ):
        self.base_loss_fn = base_loss_fn
        self.staple_indices = staple_indices  # (n_staples,) long
        self.lam = lam
        self.margin = margin
        self.n_neg = n_neg

    def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        base = self.base_loss_fn(logits, target)

        staple_logits = logits[:, self.staple_indices]  # (B, S)
        staple_targets = target[:, self.staple_indices]  # (B, S)
        n_positive = staple_targets.sum()
        if n_positive.item() == 0:
            # No staple-tier card was held out in this batch at all -- rare
            # (staples appear in nearly every deck) but possible at small
            # batch sizes/high mask ratios; nothing to rank, skip the term
            # rather than dividing by zero.
            return base

        neg_idx = torch.randint(0, logits.size(1), (self.n_neg,), device=logits.device)
        neg_logits = logits[:, neg_idx]  # (B, n_neg)

        diff = staple_logits.unsqueeze(2) - neg_logits.unsqueeze(1)  # (B, S, n_neg)
        hinge = (self.margin - diff).clamp(min=0.0)
        pos_mask = staple_targets.unsqueeze(2)  # (B, S, 1), broadcasts over n_neg

        aux = (hinge * pos_mask).sum() / (n_positive * self.n_neg)
        return base + self.lam * aux


class BlendedSoftmaxBCELoss:
    """alpha * multi_positive_softmax_loss + (1 - alpha) * BCEWithLogitsLoss
    (pos_weight-corrected). alpha=0 reproduces the plain BCE baseline
    exactly; alpha=1 reproduces the pure softmax loss exactly.

    Motivation: the pure softmax loss (above) fixed staple MRR dramatically
    (0.007 -> 0.13, val split) but overcorrected -- staple precision fell
    from ~90% to ~52%, and qualitative completions got noticeably more
    generic (see the "Fixing the Staple-MRR Problem" report). Independent
    BCE keeps precision high specifically because each card's score only
    has to be close to its own 0/1 label, never forced to compete against
    the entire vocabulary the way softmax's normalization does -- blending
    the two lets BCE's calibration pressure pull precision back up while
    still keeping most of softmax's ranking pressure. Since alpha=0 and
    alpha=1 are both already trained and evaluated checkpoints, a sweep of
    intermediate alpha values interpolates between two known references
    rather than exploring blind.
    """

    def __init__(self, bce_loss_fn, invalid_mask: torch.Tensor, alpha: float = 0.5):
        self.bce_loss_fn = bce_loss_fn
        self.invalid_mask = invalid_mask
        self.alpha = alpha

    def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = self.bce_loss_fn(logits, target)
        sm = multi_positive_softmax_loss(logits, target, invalid_mask=self.invalid_mask)
        return self.alpha * sm + (1 - self.alpha) * bce


class NormalizedBlendedLoss:
    """Like BlendedSoftmaxBCELoss, but divides each term by an EMA of its
    own recent magnitude before blending -- so alpha controls each term's
    RELATIVE gradient contribution, not raw magnitude.

    Real finding this fixes: a 3-point sweep of BlendedSoftmaxBCELoss
    (alpha 0.25/0.5/0.75) found a phase transition, not a smooth trade-off
    curve -- staple precision had already collapsed to essentially pure
    softmax's level by alpha=0.25. Checking the two losses' own recorded
    val_loss explained why: BCE's scale (~0.06) is ~100x smaller than
    softmax's (~6.1, close to ln(vocab_size)~10.4), so a naive weighted sum
    is dominated by softmax's raw magnitude the moment alpha leaves zero --
    the nominal 1:3 ratio at alpha=0.25 was actually more like 50:1 in real
    gradient contribution.

    An EMA (not a fixed pre-measured constant) is used because the two
    losses' relative scale drifts over training -- roughly 53x early,
    102x by epoch 40 in the already-trained checkpoints -- so a snapshot
    normalization would only be correct at the point it was measured.
    """

    def __init__(
        self,
        bce_loss_fn,
        invalid_mask: torch.Tensor,
        alpha: float = 0.5,
        ema_decay: float = 0.99,
        eps: float = 1e-8,
    ):
        self.bce_loss_fn = bce_loss_fn
        self.invalid_mask = invalid_mask
        self.alpha = alpha
        self.ema_decay = ema_decay
        self.eps = eps
        self.ema_bce: torch.Tensor | None = None
        self.ema_softmax: torch.Tensor | None = None

    def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = self.bce_loss_fn(logits, target)
        sm = multi_positive_softmax_loss(logits, target, invalid_mask=self.invalid_mask)

        # Only update the running estimate during actual training steps --
        # train.py's evaluate_loss() calls this same object on validation
        # batches under torch.no_grad(), and those shouldn't perturb the EMA.
        if torch.is_grad_enabled():
            bce_val, sm_val = bce.detach(), sm.detach()
            if self.ema_bce is None:
                self.ema_bce, self.ema_softmax = bce_val, sm_val
            else:
                self.ema_bce = self.ema_decay * self.ema_bce + (1 - self.ema_decay) * bce_val
                self.ema_softmax = self.ema_decay * self.ema_softmax + (1 - self.ema_decay) * sm_val

        bce_norm = bce / self.ema_bce.clamp(min=self.eps)
        sm_norm = sm / self.ema_softmax.clamp(min=self.eps)
        return self.alpha * sm_norm + (1 - self.alpha) * bce_norm


def cooccurrence_contrastive_loss(
    full_table: torch.Tensor,
    pos_a: torch.Tensor,
    pos_b: torch.Tensor,
    neg_c: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """Margin-based contrastive loss on the shared card-embedding table
    directly (not either head's deck-scoring logits): for a real
    within-deck pair (a, b), pushes cos_sim(a, b) above cos_sim(a, c) for a
    random card c by at least `margin`.

    Motivation: a nearest-neighbor validity study on the learned embeddings
    found real structure overall, but one clear, honest miss -- Sol Ring's
    nearest neighbors are other cards sharing its exact oracle text
    ("{T}: Add {C}{C}.") rather than cards it's actually played alongside,
    because nothing in the deck-completion objective ever directly rewards
    "these two cards co-occur in real decks" -- that's an emergent, not
    targeted, property of the completion task. This term targets it
    directly. pos_a/pos_b/neg_c: (P,) long index tensors, sampled by the
    training loop from a precomputed real-co-occurrence pair pool (see
    train_two_head.py) -- not filtered for accidental true positives among
    the negatives, same word2vec/NCE-style tradeoff HybridStapleRankingLoss
    above already accepts.
    """
    def cos(i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
        vi, vj = full_table[i], full_table[j]
        return (vi * vj).sum(-1) / (vi.norm(dim=-1) * vj.norm(dim=-1)).clamp(min=1e-8)

    return torch.relu(margin - cos(pos_a, pos_b) + cos(pos_a, neg_c)).mean()


class MultiTaskNormalizedLoss:
    """Generalizes NormalizedBlendedLoss from a 2-term alpha blend to N
    named terms, each with its own weight, each EMA-normalized to a
    comparable scale before being combined: sum(weight_i * loss_i / ema_i).

    Same motivation as NormalizedBlendedLoss, one level up: in
    TwoHeadDeckCompletionModel, the staple/synergy heads' losses only
    directly own their own final projection layer, but both backprop into
    the SHARED trunk (the embedding table + encoder) alongside the
    co-occurrence contrastive term -- so an unnormalized sum would let
    whichever term has the largest raw magnitude dominate how the shared
    trunk gets updated, the exact failure mode that broke the first,
    unnormalized BlendedSoftmaxBCELoss attempt, recurring here across three
    terms instead of two.

    `weights`: {name: weight}, fixed at construction (this object persists
    for the whole training run, so its EMA state accumulates across steps).
    Each call passes the CURRENT step's already-computed raw loss tensors
    (`raw: {name: scalar tensor}`) -- the training loop computes each
    term's value itself first (the two heads' (logits, target) losses and
    the contrastive term's (full_table, pos_a, pos_b, neg_c) call have
    nothing in common signature-wise), and this just combines the results,
    rather than this object needing to know how to compute any of them.
    """

    def __init__(self, weights: dict[str, float], ema_decay: float = 0.99, eps: float = 1e-8):
        self.weights = weights
        self.ema_decay = ema_decay
        self.eps = eps
        self.ema: dict[str, torch.Tensor] = {}

    def __call__(self, raw: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        if torch.is_grad_enabled():
            for name, val in raw.items():
                val_detached = val.detach()
                if name not in self.ema:
                    self.ema[name] = val_detached
                else:
                    self.ema[name] = self.ema_decay * self.ema[name] + (1 - self.ema_decay) * val_detached

        total = 0.0
        for name, weight in self.weights.items():
            ema_val = self.ema.get(name, raw[name].detach()).clamp(min=self.eps)
            total = total + weight * raw[name] / ema_val

        return total, {name: val.item() for name, val in raw.items()}


def pairwise_margin_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    n_neg: int = 64,
    margin: float = 1.0,
) -> torch.Tensor:
    """A primary loss untried in this exact form: pairwise margin ranking
    applied to EVERY true target, not just the staple tier the way
    HybridStapleRankingLoss scopes its own pairwise term. Comparative like
    multi_positive_softmax_loss (a card can only reduce this loss by
    actually outscoring negatives, not just moving its own score toward a
    fixed label the way BCE does) -- but WITHOUT softmax's documented cost:
    softmax normalizes every card's probability against the whole
    vocabulary at once, so an example's own simultaneous true targets end
    up competing against EACH OTHER too for shared probability mass ("more
    simultaneous targets means less probability mass available per
    target" -- see multi_positive_softmax_loss's docstring). Here, each
    true target is only ever compared against sampled NEGATIVES, never
    against its own example's other true targets.

    logits, target: (B, V). For every real (example, positive-target) pair
    in the batch, pushes that target's score above `n_neg` shared random
    negative columns' scores by at least `margin`, hinge-style --
    HybridStapleRankingLoss's existing mechanism, generalized from "only
    the ~50 staple columns" to "every true target, made the primary loss."

    Scaling: naively doing this by densely sweeping every vocab column as a
    potential "positive" (HybridStapleRankingLoss's existing (B, 50, n_neg)
    trick, extended to all V columns) would be (B, V, n_neg) -- ~491M
    elements at this vocab size, not viable. Fixed by gathering only the
    SPARSE actual positive entries via torch.nonzero instead: a real,
    non-approximated mean over n_neg individual hinges per positive (not a
    cheaper-but-weaker "hinge of the mean negative" shortcut), materializing
    only (n_pairs, n_neg) -- tens of thousands of elements, not hundreds of
    millions.

    Known, accepted approximation, same tradition as HybridStapleRankingLoss
    and multi_positive_softmax_loss: `n_neg` negative columns are shared
    across the whole batch for one training step (not resampled per
    example) and sampled uniformly at random with no explicit exclusion of
    accidental true positives -- cheap, and per NCE's own justification, a
    small amount of that noise is an accepted cost at this vocab size.
    """
    B, V = logits.shape
    neg_idx = torch.randint(0, V, (n_neg,), device=logits.device)
    neg_logits = logits[:, neg_idx]  # (B, n_neg)

    pos_batch_idx, pos_card_idx = torch.nonzero(target, as_tuple=True)
    if pos_batch_idx.numel() == 0:
        # No positive target anywhere in this batch -- vanishingly rare
        # (every deck example has at least one held-out card) but possible
        # at extreme mask ratios on a tiny batch; nothing to rank, return a
        # zero that still participates in the autograd graph rather than a
        # bare Python 0 (which MultiTaskNormalizedLoss-style combiners
        # calling .detach() on would choke on).
        return logits.sum() * 0.0

    pos_logits = logits[pos_batch_idx, pos_card_idx]  # (n_pairs,)
    pos_neg_logits = neg_logits[pos_batch_idx]  # (n_pairs, n_neg) -- each positive vs. its OWN example's negative pool

    hinge = torch.relu(margin - (pos_logits.unsqueeze(1) - pos_neg_logits))  # (n_pairs, n_neg)
    per_pair = hinge.mean(dim=1)  # (n_pairs,)

    # Same per-example averaging convention as multi_positive_softmax_loss:
    # an example with many simultaneous targets shouldn't dominate the
    # batch gradient purely by having more (example, positive) pairs to
    # sum, which has nothing to do with how well-ranked any of them are.
    n_targets = target.sum(dim=1).clamp(min=1.0)  # (B,)
    weight = 1.0 / n_targets[pos_batch_idx]
    return (per_pair * weight).sum() / B
