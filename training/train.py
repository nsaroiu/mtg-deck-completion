"""
train.py

Training loop for the DeepSets deck-completion model (see model.py and
README.md's "Architecture" section).

Usage:
    python3 -m training.train
    python3 -m training.train --epochs 5 --batch-size 256 --embed-dim 128
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tokenizer.mtg_tokenizer import SPECIAL_TOKENS, CardTokenizer, PAD
from training.deck_dataset import (
    DeckCompletionDataset,
    load_decks_auto,
    make_collate_fn,
    split_decks,
)
from training.losses import (
    BlendedSoftmaxBCELoss,
    HybridStapleRankingLoss,
    NormalizedBlendedLoss,
    multi_positive_softmax_loss,
    pairwise_margin_loss,
)
from training.metrics import card_tiers, recall_at_k
from training.model import DeckCompletionModel, HybridCardEmbedding, build_encoder

TOKENIZER_PATH = "./tokenizer.json"
TEXT_EMBEDDING_CACHE = "./tokenizer_text_embeddings.npy"
CHECKPOINT_PATH = "./data/deepsets_checkpoint.pt"
HISTORY_PATH = "./data/deepsets_train_history.json"

# In addition to the always-latest checkpoint (saved every epoch, at
# checkpoint_path), also keep a numbered snapshot every N epochs -- see the
# save site in run() for the naming scheme that keeps different runs' (e.g.
# base vs. large Set Transformer) snapshots from colliding.
CHECKPOINT_SNAPSHOT_EVERY = 10

# Per-epoch tracked recall: cheap enough to run every epoch (unlike the full
# multi-ratio/multi-tier sweep in evaluate.py) so train_loss/val_loss/recall
# can all be plotted against epoch together.
EPOCH_RECALL_K = (50,)
EPOCH_RECALL_RATIO = (0.3,)
EPOCH_RECALL_MAX_DECKS = 300

# Cards like Sol Ring/Command Tower appear in nearly every deck -- naive BCE
# would be dominated by "always predict staples" being trivially correct
# most of the time. pos_weight upweights each card's positive-label loss
# inversely to how common it is in training decks (clipped to avoid huge,
# unstable weights for singleton-appearance cards), so rare/synergy cards
# actually contribute gradient instead of being drowned out.
#
# MIN_POS_WEIGHT is the fix for a real finding from the first run: with no
# floor, a card in ~99% of decks got weight (1-0.99)/0.99 ~= 0.01 -- its
# gradient was almost entirely suppressed, not just "de-emphasized". Recall@K
# broken out by popularity tier came back with staples as the *worst*
# tier (worse than long-tail cards), the opposite of the intended effect.
# Real Commander decks do run their staples; a 0.2 floor keeps the loss from
# treating "missed a staple" as nearly free while still discounting it well
# below a genuinely rare card's weight (up to MAX_POS_WEIGHT).
MAX_POS_WEIGHT = 100.0
MIN_POS_WEIGHT = 0.2


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_pos_weight(train_decks, vocab_size: int) -> torch.Tensor:
    freq = Counter()
    for d in train_decks:
        for cid in d.card_ids:
            freq[cid] += 1
    n = len(train_decks)

    weight = torch.ones(vocab_size, dtype=torch.float32)
    for cid, count in freq.items():
        p = count / n
        weight[cid] = min(MAX_POS_WEIGHT, max(MIN_POS_WEIGHT, (1 - p) / max(p, 1e-6)))
    return weight


def build_loss_fn(
    loss_type: str,
    tok: CardTokenizer,
    splits: dict,
    tiers: dict,
    device: torch.device,
    loss_kwargs: dict,
):
    """Constructs a train.py-compatible loss_fn (`(logits, target) ->
    scalar`) by name -- the loss analogue of training/model.py's
    build_encoder, so train.py and any future re-loading code have one
    place to go. See training/losses.py's module docstring for why these
    exist and training/train.py's own pos_weight comment for the original
    "bce" baseline this compares against.
    """
    def _special_token_invalid_mask() -> torch.Tensor:
        special_ids = [tok.token_to_id[t] for t in SPECIAL_TOKENS if t in tok.token_to_id]
        invalid_mask = torch.zeros(tok.vocab_size, dtype=torch.bool, device=device)
        invalid_mask[special_ids] = True
        return invalid_mask

    if loss_type == "bce":
        pos_weight = compute_pos_weight(splits["train"], tok.vocab_size).to(device)
        return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    if loss_type == "softmax":
        invalid_mask = _special_token_invalid_mask()

        def loss_fn(logits, target):
            return multi_positive_softmax_loss(logits, target, invalid_mask=invalid_mask)

        return loss_fn

    if loss_type == "blend":
        pos_weight = compute_pos_weight(splits["train"], tok.vocab_size).to(device)
        base_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        invalid_mask = _special_token_invalid_mask()
        return BlendedSoftmaxBCELoss(base_loss_fn, invalid_mask, alpha=loss_kwargs.get("alpha", 0.5))

    if loss_type == "norm_blend":
        pos_weight = compute_pos_weight(splits["train"], tok.vocab_size).to(device)
        base_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        invalid_mask = _special_token_invalid_mask()
        return NormalizedBlendedLoss(
            base_loss_fn,
            invalid_mask,
            alpha=loss_kwargs.get("alpha", 0.5),
            ema_decay=loss_kwargs.get("ema_decay", 0.99),
        )

    if loss_type == "pairwise":
        margin = loss_kwargs.get("margin", 1.0)
        n_neg = loss_kwargs.get("n_neg", 64)

        def loss_fn(logits, target):
            return pairwise_margin_loss(logits, target, n_neg=n_neg, margin=margin)

        return loss_fn

    if loss_type == "hybrid_staple":
        pos_weight = compute_pos_weight(splits["train"], tok.vocab_size).to(device)
        base_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        staple_ids = sorted(tid for tid, t in tiers.items() if t == "staple")
        staple_indices = torch.tensor(staple_ids, dtype=torch.long, device=device)
        return HybridStapleRankingLoss(
            base_loss_fn,
            staple_indices,
            lam=loss_kwargs.get("lam", 0.5),
            margin=loss_kwargs.get("margin", 1.0),
            n_neg=loss_kwargs.get("n_neg", 64),
        )

    raise ValueError(f"Unknown loss_type: {loss_type!r}")


@torch.no_grad()
def evaluate_loss(model, loader, loss_fn, device) -> float:
    model.eval()
    total, n_batches = 0.0, 0
    for ctx_ids, ctx_mask, cmdr_ids, cmdr_mask, target in loader:
        ctx_ids, ctx_mask = ctx_ids.to(device), ctx_mask.to(device)
        cmdr_ids, cmdr_mask = cmdr_ids.to(device), cmdr_mask.to(device)
        target = target.to(device)
        logits = model(ctx_ids, ctx_mask, cmdr_ids, cmdr_mask)
        total += loss_fn(logits, target).item()
        n_batches += 1
    model.train()
    return total / max(n_batches, 1)


def scheduled_sampling_prob(epoch: int, warmup_epochs: int, p_end: float) -> float:
    """Linear ramp from 0 at epoch 1 to p_end by `warmup_epochs`, then holds
    at p_end -- standard scheduled-sampling curriculum, avoiding
    destabilizing early training when the model's own predictions are
    still close to noise. p_end=0.0 (the default everywhere this is
    called) makes this always return 0 regardless of epoch/warmup_epochs --
    the inert no-op path every existing experiment relies on."""
    if warmup_epochs <= 0:
        return p_end
    return min(p_end, p_end * epoch / warmup_epochs)


def build_scheduled_sampling_batch(
    model, ctx_ids, ctx_mask, cmdr_ids, cmdr_mask, target, pad_id: int, invalid_mask: torch.Tensor, k: int,
):
    """Simulates a few steps into iterative generation for one training
    step: one no_grad forward pass on the REAL (context, commander) pair
    gets the model's own current top-k predictions (masking out
    already-visible cards and special tokens), which are appended to
    context and removed from the target set -- the model is then trained
    to still find the true remaining targets correctly with those extra,
    possibly-imperfect self-picks already sitting in its context, instead
    of only ever seeing a clean ground-truth-only context the way every
    training example otherwise does. See train.py's --scheduled-sampling-*
    flags for why this exists: iterative generation's context is exactly
    this kind of self-built pile, never a clean random real-deck subset, and closing
    that training/inference gap is the best-supported explanation for
    iterative generation's self-reinforcing collapse at aggressive
    (chunk_size=1) settings.

    Returns (new_ctx_ids, new_ctx_mask, new_target), each with k extra
    columns/updated entries -- ctx_ids/target are otherwise the same shape
    convention make_collate_fn already produces.
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        logits = model(ctx_ids, ctx_mask, cmdr_ids, cmdr_mask)  # (B, V)
        already_visible = torch.zeros_like(logits, dtype=torch.bool)
        already_visible.scatter_(1, ctx_ids, ctx_mask)
        already_visible.scatter_(1, cmdr_ids, cmdr_mask)
        logits = logits.masked_fill(already_visible, float("-inf"))
        logits = logits.masked_fill(invalid_mask.unsqueeze(0), float("-inf"))
        self_picks = torch.topk(logits, k, dim=1).indices  # (B, k)
    if was_training:
        model.train()

    B, L = ctx_ids.shape
    new_ctx_ids = torch.full((B, L + k), pad_id, dtype=ctx_ids.dtype, device=ctx_ids.device)
    new_ctx_mask = torch.zeros((B, L + k), dtype=torch.bool, device=ctx_ids.device)
    new_ctx_ids[:, :L] = ctx_ids
    new_ctx_mask[:, :L] = ctx_mask
    new_ctx_ids[:, L:] = self_picks
    new_ctx_mask[:, L:] = True

    new_target = target.clone()
    new_target.scatter_(1, self_picks, 0.0)

    return new_ctx_ids, new_ctx_mask, new_target


def run(
    epochs: int = 3,
    batch_size: int = 256,
    embed_dim: int = 128,
    hidden_dim: int = 256,
    lr: float = 1e-3,
    log_every: int = 100,
    limit_train_batches: int | None = None,
    checkpoint_path: str = CHECKPOINT_PATH,
    history_path: str = HISTORY_PATH,
    encoder_type: str = "deepsets",
    encoder_kwargs: dict | None = None,
    loss_type: str = "bce",
    loss_kwargs: dict | None = None,
    use_completeness_feature: bool = False,
    scheduled_sampling_p_end: float = 0.0,
    scheduled_sampling_warmup_epochs: int | None = None,
    scheduled_sampling_k: int = 3,
    device: torch.device | None = None,
) -> dict:
    device = device or pick_device()
    encoder_kwargs = encoder_kwargs or {}
    loss_kwargs = loss_kwargs or {}
    if scheduled_sampling_warmup_epochs is None:
        scheduled_sampling_warmup_epochs = epochs // 2
    print(f"Device: {device}", flush=True)
    print(f"Encoder: {encoder_type} {encoder_kwargs}", flush=True)
    print(f"Loss: {loss_type} {loss_kwargs}", flush=True)
    if scheduled_sampling_p_end > 0:
        print(
            f"Scheduled sampling: p_end={scheduled_sampling_p_end} "
            f"warmup_epochs={scheduled_sampling_warmup_epochs} k={scheduled_sampling_k}",
            flush=True,
        )

    tok = CardTokenizer.load_pretrained(TOKENIZER_PATH)
    decks = load_decks_auto(tok)
    splits = split_decks(decks)
    print(f"Decks -- train: {len(splits['train'])}, val: {len(splits['val'])}, "
          f"test: {len(splits['test'])}, held_out_commander: {len(splits['held_out_commander'])}", flush=True)

    pad_id = tok.token_to_id[PAD]
    collate_fn = make_collate_fn(tok.vocab_size, pad_id)

    tiers = card_tiers(splits["train"])

    train_ds = DeckCompletionDataset(splits["train"])
    val_ds = DeckCompletionDataset(splits["val"], seed=1)  # fixed seed: stable val loss across epochs

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    struct_feats = tok.feature_matrix()
    text_feats = tok.text_embeddings(cache_path=TEXT_EMBEDDING_CACHE)

    card_embedding = HybridCardEmbedding(
        tok.vocab_size, embed_dim, struct_feats, text_feats, pad_id
    ).to(device)
    encoder = build_encoder(encoder_type, embed_dim, hidden_dim, **encoder_kwargs)
    model = DeckCompletionModel(
        card_embedding, encoder, hidden_dim, use_completeness_feature=use_completeness_feature
    ).to(device)

    loss_fn = build_loss_fn(loss_type, tok, splits, tiers, device, loss_kwargs)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}", flush=True)

    # Only built/used when scheduled sampling is actually enabled below --
    # cheap either way (a (vocab_size,) bool tensor), but there's no reason
    # to construct it when scheduled_sampling_p_end=0.0 will never consult it.
    ss_special_ids = [tok.token_to_id[t] for t in SPECIAL_TOKENS if t in tok.token_to_id]
    ss_invalid_mask = torch.zeros(tok.vocab_size, dtype=torch.bool, device=device)
    ss_invalid_mask[ss_special_ids] = True

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running_loss, n_seen = 0.0, 0
        epoch_loss_sum, epoch_n = 0.0, 0
        p_ss = scheduled_sampling_prob(epoch, scheduled_sampling_warmup_epochs, scheduled_sampling_p_end)

        for step, (ctx_ids, ctx_mask, cmdr_ids, cmdr_mask, target) in enumerate(train_loader, start=1):
            if limit_train_batches is not None and step > limit_train_batches:
                break

            ctx_ids, ctx_mask = ctx_ids.to(device), ctx_mask.to(device)
            cmdr_ids, cmdr_mask = cmdr_ids.to(device), cmdr_mask.to(device)
            target = target.to(device)

            # Whole-batch (not per-example) probabilistic augmentation --
            # simpler than mixing real/augmented examples within one batch,
            # and matches how this was scoped ("a second forward pass on
            # part of every batch"). p_ss=0 (scheduled_sampling_p_end=0.0,
            # the default) makes this condition always False -- a true
            # no-op, not just a cheap one, so every existing experiment's
            # training behavior is unaffected unless explicitly opted in.
            if p_ss > 0 and random.random() < p_ss:
                ctx_ids, ctx_mask, target = build_scheduled_sampling_batch(
                    model, ctx_ids, ctx_mask, cmdr_ids, cmdr_mask, target,
                    pad_id, ss_invalid_mask, scheduled_sampling_k,
                )

            optimizer.zero_grad()
            logits = model(ctx_ids, ctx_mask, cmdr_ids, cmdr_mask)
            loss = loss_fn(logits, target)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            running_loss += loss_val
            n_seen += 1
            epoch_loss_sum += loss_val
            epoch_n += 1
            if step % log_every == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss={running_loss / n_seen:.4f} ({time.time() - t0:.1f}s elapsed)", flush=True)
                running_loss, n_seen = 0.0, 0

        train_loss = epoch_loss_sum / max(epoch_n, 1)
        val_loss = evaluate_loss(model, val_loader, loss_fn, device)
        recall = recall_at_k(
            model, splits["val"], tiers,
            ks=EPOCH_RECALL_K, mask_ratios=EPOCH_RECALL_RATIO,
            device=device, max_decks=EPOCH_RECALL_MAX_DECKS,
        )
        recall_at_50 = recall["by_ratio"][f"k={EPOCH_RECALL_K[0]}_ratio={EPOCH_RECALL_RATIO[0]}"]

        elapsed = time.time() - t0
        ss_suffix = f" p_ss={p_ss:.3f}" if scheduled_sampling_p_end > 0 else ""
        print(f"Epoch {epoch} done in {elapsed:.1f}s -- train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} recall@{EPOCH_RECALL_K[0]}={recall_at_50:.4f}{ss_suffix}", flush=True)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "recall_at_50": recall_at_50,
            "elapsed_s": elapsed,
            "scheduled_sampling_p": p_ss,
        })

        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "vocab_size": tok.vocab_size,
            "epoch": epoch,
            "val_loss": val_loss,
            "encoder_type": encoder_type,
            "encoder_kwargs": encoder_kwargs,
            "loss_type": loss_type,
            "loss_kwargs": loss_kwargs,
            "use_completeness_feature": use_completeness_feature,
            "scheduled_sampling": {
                "p_end": scheduled_sampling_p_end,
                "warmup_epochs": scheduled_sampling_warmup_epochs,
                "k": scheduled_sampling_k,
            },
        }
        torch.save(checkpoint, checkpoint_path)

        # Periodic snapshot, kept alongside (not instead of) the
        # always-latest checkpoint_path above -- lets a later comparison
        # look at an earlier epoch (e.g. to check for overfitting) instead
        # of only ever having the final epoch's weights. Named off
        # checkpoint_path itself (e.g. foo.pt -> foo_epoch10.pt), so two
        # runs with different --checkpoint-path values (as the base and
        # large Set Transformer runs use) can never collide on snapshots.
        if epoch % CHECKPOINT_SNAPSHOT_EVERY == 0:
            ckpt_path = Path(checkpoint_path)
            snapshot_path = ckpt_path.with_name(f"{ckpt_path.stem}_epoch{epoch}{ckpt_path.suffix}")
            torch.save(checkpoint, snapshot_path)

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    return {"model": model, "tokenizer": tok, "splits": splits, "history": history, "device": device}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--checkpoint-path", default=CHECKPOINT_PATH)
    parser.add_argument("--history-path", default=HISTORY_PATH)
    parser.add_argument("--encoder-type", default="deepsets", choices=["deepsets", "set_transformer"])
    parser.add_argument("--num-heads", type=int, default=4, help="set_transformer only")
    parser.add_argument("--num-layers", type=int, default=3, help="set_transformer only")
    parser.add_argument("--dim-feedforward", type=int, default=512, help="set_transformer only")
    parser.add_argument("--dropout", type=float, default=0.1, help="set_transformer only")
    parser.add_argument("--loss-type", default="bce", choices=["bce", "softmax", "pairwise", "hybrid_staple", "blend", "norm_blend"])
    parser.add_argument(
        "--use-completeness-feature", action="store_true",
        help=(
            "Append known-card-count/DECK_SIZE as an extra scalar into deck_proj's "
            "input (training/model.py's DeckCompletionModel docstring has the full "
            "writeup). Off by default; existing checkpoints are unaffected. A real "
            "training run found this regressed staple-tier MRR and held-out-commander "
            "recall with no compensating gain -- kept for the record, not recommended."
        ),
    )
    parser.add_argument("--hybrid-lambda", type=float, default=0.5, help="hybrid_staple only")
    parser.add_argument("--hybrid-margin", type=float, default=1.0, help="hybrid_staple only")
    parser.add_argument("--hybrid-neg-samples", type=int, default=64, help="hybrid_staple only")
    parser.add_argument("--blend-alpha", type=float, default=0.5, help="blend/norm_blend only")
    parser.add_argument("--blend-ema-decay", type=float, default=0.99, help="norm_blend only")
    parser.add_argument("--pairwise-margin", type=float, default=1.0, help="pairwise only")
    parser.add_argument("--pairwise-neg-samples", type=int, default=64, help="pairwise only")
    parser.add_argument(
        "--scheduled-sampling-p-end", type=float, default=0.0,
        help=(
            "Probability (by the end of the warmup ramp) that a training step uses an "
            "augmented context built from the model's own current top-k predictions "
            "instead of only the real ground-truth context -- targets the training/"
            "inference mismatch behind iterative generation's self-reinforcing collapse. "
            "0.0 (the default) disables this entirely; every added code path is a true "
            "no-op in that case."
        ),
    )
    parser.add_argument(
        "--scheduled-sampling-warmup-epochs", type=int, default=None,
        help="Epochs to linearly ramp p from 0 to --scheduled-sampling-p-end. Defaults to epochs // 2.",
    )
    parser.add_argument("--scheduled-sampling-k", type=int, default=3, help="Self-picks added to context per augmented step.")
    args = parser.parse_args()

    encoder_kwargs = (
        {
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "dim_feedforward": args.dim_feedforward,
            "dropout": args.dropout,
        }
        if args.encoder_type == "set_transformer"
        else {}
    )
    if args.loss_type == "hybrid_staple":
        loss_kwargs = {
            "lam": args.hybrid_lambda,
            "margin": args.hybrid_margin,
            "n_neg": args.hybrid_neg_samples,
        }
    elif args.loss_type == "blend":
        loss_kwargs = {"alpha": args.blend_alpha}
    elif args.loss_type == "norm_blend":
        loss_kwargs = {"alpha": args.blend_alpha, "ema_decay": args.blend_ema_decay}
    elif args.loss_type == "pairwise":
        loss_kwargs = {"margin": args.pairwise_margin, "n_neg": args.pairwise_neg_samples}
    else:
        loss_kwargs = {}

    run(
        epochs=args.epochs,
        batch_size=args.batch_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        limit_train_batches=args.limit_train_batches,
        checkpoint_path=args.checkpoint_path,
        history_path=args.history_path,
        encoder_type=args.encoder_type,
        encoder_kwargs=encoder_kwargs,
        loss_type=args.loss_type,
        loss_kwargs=loss_kwargs,
        use_completeness_feature=args.use_completeness_feature,
        scheduled_sampling_p_end=args.scheduled_sampling_p_end,
        scheduled_sampling_warmup_epochs=args.scheduled_sampling_warmup_epochs,
        scheduled_sampling_k=args.scheduled_sampling_k,
    )
