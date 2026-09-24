"""
train_two_head.py

Training loop for TwoHeadDeckCompletionModel (training/model.py) -- a
shared trunk (HybridCardEmbedding + a set encoder) with two specialized
scoring heads (staple/synergy) plus a co-occurrence-aware contrastive term
on the shared embedding table. See training/model.py's
TwoHeadDeckCompletionModel docstring for why this was tried as a
replacement for training/ensemble.py's two-full-model fusion (and
ultimately rejected in favor of it -- kept here for the record), and
training/losses.py's cooccurrence_contrastive_loss/MultiTaskNormalizedLoss
docstrings for the loss design.

A new file, not a rewrite of train.py -- DeckCompletionModel's single-head
training path is untouched, every existing checkpoint/complete_deck.py
caller keeps working exactly as before.

Usage:
    python3 -m training.train_two_head --epochs 2   # smoke test first
    python3 -m training.train_two_head --epochs 40 --batch-size 256
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tokenizer.mtg_tokenizer import PAD, SPECIAL_TOKENS, CardTokenizer
from training.deck_dataset import (
    DeckCompletionDataset,
    load_decks_auto,
    make_collate_fn,
    split_decks,
)
from training.losses import (
    MultiTaskNormalizedLoss,
    cooccurrence_contrastive_loss,
    multi_positive_softmax_loss,
)
from training.metrics import card_tiers, recall_at_k
from training.model import HeadView, HybridCardEmbedding, TwoHeadDeckCompletionModel, build_encoder
from training.train import (
    TEXT_EMBEDDING_CACHE,
    TOKENIZER_PATH,
    compute_pos_weight,
    pick_device,
)

CHECKPOINT_PATH = "./data/two_head_checkpoint.pt"
HISTORY_PATH = "./data/two_head_train_history.json"
CHECKPOINT_SNAPSHOT_EVERY = 10

EPOCH_RECALL_K = (50,)
EPOCH_RECALL_RATIO = (0.3,)
EPOCH_RECALL_MAX_DECKS = 300

PAIRS_PER_DECK = 3
COOC_BATCH_SIZE = 256
COOC_MARGIN = 0.2


def build_positive_pair_pool(train_decks, pairs_per_deck: int = PAIRS_PER_DECK, seed: int = 0) -> torch.Tensor:
    """(N, 2) long tensor of real within-deck card pairs, sampled ONCE
    before training -- each training step just indexes into this pool
    (cheap tensor gather) rather than re-touching ParsedDeck objects live
    every batch. See training/losses.py's cooccurrence_contrastive_loss."""
    rng = random.Random(seed)
    pairs = []
    for deck in train_decks:
        cards = deck.card_ids
        if len(cards) < 2:
            continue
        for _ in range(pairs_per_deck):
            a, b = rng.sample(cards, 2)
            pairs.append((a, b))
    return torch.tensor(pairs, dtype=torch.long)


def sample_cooc_batch(pos_pool: torch.Tensor, vocab_size: int, device, n: int = COOC_BATCH_SIZE):
    idx = torch.randint(0, pos_pool.size(0), (n,))
    pos_a = pos_pool[idx, 0].to(device)
    pos_b = pos_pool[idx, 1].to(device)
    neg_c = torch.randint(0, vocab_size, (n,), device=device)
    return pos_a, pos_b, neg_c


def compute_losses(
    model: TwoHeadDeckCompletionModel,
    ctx_ids, ctx_mask, cmdr_ids, cmdr_mask, target,
    bce_loss_fn, invalid_mask: torch.Tensor,
    pos_pool: torch.Tensor, device,
) -> dict[str, torch.Tensor]:
    """One forward pass through the shared trunk, scored by both heads,
    plus one contrastive-loss batch on the shared embedding table -- computes
    full_table() ONCE and reuses it for both heads' scoring AND the
    co-occurrence term, rather than calling model(...) (which would
    recompute full_table() internally) and then again for the contrastive
    term."""
    full_table = model.card_embedding.full_table()  # (V, D)
    pooled = model._pool(ctx_ids, ctx_mask, cmdr_ids, cmdr_mask)
    staple_logits = model.staple_proj(pooled) @ full_table.T
    synergy_logits = model.synergy_proj(pooled) @ full_table.T

    staple_loss = bce_loss_fn(staple_logits, target)
    synergy_loss = multi_positive_softmax_loss(synergy_logits, target, invalid_mask=invalid_mask)

    pos_a, pos_b, neg_c = sample_cooc_batch(pos_pool, full_table.size(0), device)
    cooc_loss = cooccurrence_contrastive_loss(full_table, pos_a, pos_b, neg_c, margin=COOC_MARGIN)

    return {"staple": staple_loss, "synergy": synergy_loss, "cooc": cooc_loss}


@torch.no_grad()
def evaluate_loss(model, loader, bce_loss_fn, invalid_mask, pos_pool, combiner, device) -> dict[str, float]:
    model.eval()
    totals = {"staple": 0.0, "synergy": 0.0, "cooc": 0.0, "total": 0.0}
    n_batches = 0
    for ctx_ids, ctx_mask, cmdr_ids, cmdr_mask, target in loader:
        ctx_ids, ctx_mask = ctx_ids.to(device), ctx_mask.to(device)
        cmdr_ids, cmdr_mask = cmdr_ids.to(device), cmdr_mask.to(device)
        target = target.to(device)
        raw = compute_losses(model, ctx_ids, ctx_mask, cmdr_ids, cmdr_mask, target, bce_loss_fn, invalid_mask, pos_pool, device)
        total, raw_floats = combiner(raw)
        for name, val in raw_floats.items():
            totals[name] += val
        totals["total"] += total.item()
        n_batches += 1
    model.train()
    return {k: v / max(n_batches, 1) for k, v in totals.items()}


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
    term_weights: dict[str, float] | None = None,
    device: torch.device | None = None,
) -> dict:
    device = device or pick_device()
    encoder_kwargs = encoder_kwargs or {}
    term_weights = term_weights or {"staple": 1 / 3, "synergy": 1 / 3, "cooc": 1 / 3}
    print(f"Device: {device}", flush=True)
    print(f"Encoder: {encoder_type} {encoder_kwargs}", flush=True)
    print(f"Term weights: {term_weights}", flush=True)

    tok = CardTokenizer.load_pretrained(TOKENIZER_PATH)
    decks = load_decks_auto(tok)
    splits = split_decks(decks)
    print(f"Decks -- train: {len(splits['train'])}, val: {len(splits['val'])}, "
          f"test: {len(splits['test'])}, held_out_commander: {len(splits['held_out_commander'])}", flush=True)

    pad_id = tok.token_to_id[PAD]
    collate_fn = make_collate_fn(tok.vocab_size, pad_id)
    tiers = card_tiers(splits["train"])

    train_ds = DeckCompletionDataset(splits["train"])
    val_ds = DeckCompletionDataset(splits["val"], seed=1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    struct_feats = tok.feature_matrix()
    text_feats = tok.text_embeddings(cache_path=TEXT_EMBEDDING_CACHE)
    card_embedding = HybridCardEmbedding(tok.vocab_size, embed_dim, struct_feats, text_feats, pad_id).to(device)
    encoder = build_encoder(encoder_type, embed_dim, hidden_dim, **encoder_kwargs)
    model = TwoHeadDeckCompletionModel(card_embedding, encoder, hidden_dim).to(device)

    pos_weight = compute_pos_weight(splits["train"], tok.vocab_size).to(device)
    bce_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    special_ids = [tok.token_to_id[t] for t in SPECIAL_TOKENS if t in tok.token_to_id]
    invalid_mask = torch.zeros(tok.vocab_size, dtype=torch.bool, device=device)
    invalid_mask[special_ids] = True

    pos_pool = build_positive_pair_pool(splits["train"])
    print(f"Co-occurrence positive-pair pool: {pos_pool.size(0)} pairs", flush=True)

    combiner = MultiTaskNormalizedLoss(term_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}", flush=True)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running = {"total": 0.0, "staple": 0.0, "synergy": 0.0, "cooc": 0.0}
        n_seen = 0
        epoch_totals = {"total": 0.0, "staple": 0.0, "synergy": 0.0, "cooc": 0.0}
        epoch_n = 0

        for step, (ctx_ids, ctx_mask, cmdr_ids, cmdr_mask, target) in enumerate(train_loader, start=1):
            if limit_train_batches is not None and step > limit_train_batches:
                break

            ctx_ids, ctx_mask = ctx_ids.to(device), ctx_mask.to(device)
            cmdr_ids, cmdr_mask = cmdr_ids.to(device), cmdr_mask.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            raw = compute_losses(model, ctx_ids, ctx_mask, cmdr_ids, cmdr_mask, target, bce_loss_fn, invalid_mask, pos_pool, device)
            total, raw_floats = combiner(raw)
            total.backward()
            optimizer.step()

            running["total"] += total.item()
            for name, val in raw_floats.items():
                running[name] += val
                epoch_totals[name] += val
            epoch_totals["total"] += total.item()
            n_seen += 1
            epoch_n += 1
            if step % log_every == 0:
                print(
                    f"  epoch {epoch} step {step}/{len(train_loader)} "
                    f"total={running['total']/n_seen:.4f} staple={running['staple']/n_seen:.4f} "
                    f"synergy={running['synergy']/n_seen:.4f} cooc={running['cooc']/n_seen:.4f} "
                    f"({time.time()-t0:.1f}s elapsed)", flush=True,
                )
                running = {k: 0.0 for k in running}
                n_seen = 0

        train_losses = {k: v / max(epoch_n, 1) for k, v in epoch_totals.items()}
        val_losses = evaluate_loss(model, val_loader, bce_loss_fn, invalid_mask, pos_pool, combiner, device)

        staple_view = HeadView(model, "staple")
        synergy_view = HeadView(model, "synergy")
        staple_recall = recall_at_k(
            staple_view, splits["val"], tiers,
            ks=EPOCH_RECALL_K, mask_ratios=EPOCH_RECALL_RATIO,
            device=device, max_decks=EPOCH_RECALL_MAX_DECKS,
        )["by_ratio"][f"k={EPOCH_RECALL_K[0]}_ratio={EPOCH_RECALL_RATIO[0]}"]
        synergy_recall = recall_at_k(
            synergy_view, splits["val"], tiers,
            ks=EPOCH_RECALL_K, mask_ratios=EPOCH_RECALL_RATIO,
            device=device, max_decks=EPOCH_RECALL_MAX_DECKS,
        )["by_ratio"][f"k={EPOCH_RECALL_K[0]}_ratio={EPOCH_RECALL_RATIO[0]}"]

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch} done in {elapsed:.1f}s -- "
            f"train_total={train_losses['total']:.4f} val_total={val_losses['total']:.4f} "
            f"val_staple={val_losses['staple']:.4f} val_synergy={val_losses['synergy']:.4f} "
            f"val_cooc={val_losses['cooc']:.4f} "
            f"staple_recall@{EPOCH_RECALL_K[0]}={staple_recall:.4f} "
            f"synergy_recall@{EPOCH_RECALL_K[0]}={synergy_recall:.4f}", flush=True,
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_losses,
            "val_loss": val_losses,
            "staple_recall_at_50": staple_recall,
            "synergy_recall_at_50": synergy_recall,
            "elapsed_s": elapsed,
        })

        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "vocab_size": tok.vocab_size,
            "epoch": epoch,
            "val_loss": val_losses["total"],
            "val_loss_by_term": val_losses,
            "encoder_type": encoder_type,
            "encoder_kwargs": encoder_kwargs,
            "architecture": "two_head",
            "term_weights": term_weights,
        }
        torch.save(checkpoint, checkpoint_path)

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
    parser.add_argument("--staple-weight", type=float, default=1 / 3)
    parser.add_argument("--synergy-weight", type=float, default=1 / 3)
    parser.add_argument("--cooc-weight", type=float, default=1 / 3)
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
        term_weights={"staple": args.staple_weight, "synergy": args.synergy_weight, "cooc": args.cooc_weight},
    )
