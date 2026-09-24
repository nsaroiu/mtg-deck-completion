"""
train_prune.py

Training loop for the deck-pruning model (training/model.py's PruneModel,
training/prune_dataset.py) -- a separate script from training/train.py
since this trains a structurally different model (permutation-equivariant,
not invariant), a different dataset (real-vs-corrupted, not masked
completion), and a different eval (corruption-recovery precision, not
Recall@K against the vocab). See training/model.py's PruneModel docstring
for the full design and motivation.

Usage:
    python3 -m training.train_prune
    python3 -m training.train_prune --epochs 20 --batch-size 128
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tokenizer.mtg_tokenizer import CardTokenizer, PAD
from training.deck_dataset import load_decks_auto, split_decks
from training.model import HybridCardEmbedding, PruneEncoder, PruneModel
from training.prune_dataset import CORRUPT_RATIO_RANGE, INJECT_RATIO_RANGE, PruneDataset, make_prune_collate_fn
from training.train import pick_device

TOKENIZER_PATH = "./tokenizer.json"
TEXT_EMBEDDING_CACHE = "./tokenizer_text_embeddings.npy"
CHECKPOINT_PATH = "./data/prune_checkpoint.pt"
HISTORY_PATH = "./data/prune_train_history.json"


def prune_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-position BCE, averaged over only the real (non-padding) deck-card
    positions -- unlike train.py's whole-vocab BCE, there's no class
    imbalance to correct for here: the real:corrupted ratio per example is
    already controlled directly by prune_dataset.CORRUPT_RATIO_RANGE, so no
    pos_weight is needed."""
    per_pos = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    mask_f = mask.to(per_pos.dtype)
    return (per_pos * mask_f).sum() / mask_f.sum().clamp(min=1.0)


@torch.no_grad()
def evaluate_corruption_recovery(model, loader, device) -> dict:
    """For each example, ranks deck-card positions by predicted
    real-probability (ascending) and checks, among the bottom-n_corrupt
    predictions (n_corrupt = the TRUE number corrupted in that example),
    what fraction are actually the corrupted positions -- precision@true_k,
    the exact narrow task the pruning phase of the generate/prune/backfill
    loop needs to be good at: given a completed deck, point at the actual
    weak links, not just "is this deck corrupted at all"."""
    model.eval()
    total_recall, n_examples = 0.0, 0
    for deck_ids, deck_mask, cmdr_ids, cmdr_mask, labels in loader:
        deck_ids, deck_mask = deck_ids.to(device), deck_mask.to(device)
        cmdr_ids, cmdr_mask = cmdr_ids.to(device), cmdr_mask.to(device)
        labels = labels.to(device)

        logits = model(deck_ids, deck_mask, cmdr_ids, cmdr_mask)
        logits = logits.masked_fill(~deck_mask, float("inf"))  # padding never looks "suspicious"

        for i in range(deck_ids.shape[0]):
            true_corrupt = (labels[i] == 0) & deck_mask[i]
            n_corrupt = int(true_corrupt.sum().item())
            if n_corrupt == 0:
                continue
            bottom_idx = torch.topk(logits[i], n_corrupt, largest=False).indices
            predicted_corrupt = torch.zeros_like(true_corrupt)
            predicted_corrupt[bottom_idx] = True
            hit = (predicted_corrupt & true_corrupt).sum().item()
            total_recall += hit / n_corrupt
            n_examples += 1

    model.train()
    return {"corruption_recovery_at_true_k": total_recall / max(n_examples, 1), "n_examples": n_examples}


def run(
    epochs: int = 20,
    batch_size: int = 128,
    embed_dim: int = 128,
    hidden_dim: int = 256,
    lr: float = 1e-3,
    num_heads: int = 4,
    num_layers: int = 3,
    dim_feedforward: int = 512,
    dropout: float = 0.1,
    train_mode: str = "swap",
    corrupt_ratio_range: tuple[float, float] | None = None,
    inject_ratio_range: tuple[float, float] | None = None,
    log_every: int = 100,
    limit_train_batches: int | None = None,
    checkpoint_path: str = CHECKPOINT_PATH,
    history_path: str = HISTORY_PATH,
    device: torch.device | None = None,
) -> dict:
    device = device or pick_device()
    # corrupt_ratio_range only affects train_mode="swap" (PruneDataset's own
    # corrupt_ratio_range attribute); inject_ratio_range only affects
    # train_mode="inject" (its separate inject_ratio_range attribute) --
    # PruneDataset consults whichever one matches its mode and ignores the
    # other, so both can be passed unconditionally. None (the default for
    # each) reproduces that mode's own original density unchanged. Together
    # these are what let the density-vs-size question be asked as a clean
    # 2x2 (swap/inject x low/high density) instead of only ever changing
    # both variables at once.
    ratio_range = corrupt_ratio_range if corrupt_ratio_range is not None else CORRUPT_RATIO_RANGE
    resolved_inject_range = inject_ratio_range if inject_ratio_range is not None else INJECT_RATIO_RANGE
    print(f"Device: {device}", flush=True)
    print(
        f"Train corruption mode: {train_mode} "
        f"(swap ratio_range={ratio_range}, inject ratio_range={resolved_inject_range})",
        flush=True,
    )

    tok = CardTokenizer.load_pretrained(TOKENIZER_PATH)
    decks = load_decks_auto(tok)
    splits = split_decks(decks)
    print(f"Decks -- train: {len(splits['train'])}, val: {len(splits['val'])}, "
          f"test: {len(splits['test'])}, held_out_commander: {len(splits['held_out_commander'])}", flush=True)

    pad_id = tok.token_to_id[PAD]
    collate_fn = make_prune_collate_fn(pad_id)

    train_ds = PruneDataset(
        splits["train"], tok, mode=train_mode,
        corrupt_ratio_range=ratio_range, inject_ratio_range=resolved_inject_range,
    )
    # ALWAYS the original swap-mode/default-density benchmark, regardless of
    # train_mode -- so corruption_recovery stays directly comparable across
    # every prune checkpoint trained so far (prune_checkpoint.pt included),
    # same "train differently, eval on the same yardstick" methodology
    # train_prune_hardneg.py already uses for its own comparison.
    val_ds = PruneDataset(splits["val"], tok, seed=1)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    struct_feats = tok.feature_matrix()
    text_feats = tok.text_embeddings(cache_path=TEXT_EMBEDDING_CACHE)

    card_embedding = HybridCardEmbedding(tok.vocab_size, embed_dim, struct_feats, text_feats, pad_id).to(device)
    encoder = PruneEncoder(
        embed_dim, num_heads=num_heads, num_layers=num_layers,
        dim_feedforward=dim_feedforward, dropout=dropout,
    ).to(device)
    model = PruneModel(card_embedding, encoder, hidden_dim).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}", flush=True)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running_loss, n_seen = 0.0, 0
        epoch_loss_sum, epoch_n = 0.0, 0

        for step, (deck_ids, deck_mask, cmdr_ids, cmdr_mask, labels) in enumerate(train_loader, start=1):
            if limit_train_batches is not None and step > limit_train_batches:
                break

            deck_ids, deck_mask = deck_ids.to(device), deck_mask.to(device)
            cmdr_ids, cmdr_mask = cmdr_ids.to(device), cmdr_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(deck_ids, deck_mask, cmdr_ids, cmdr_mask)
            loss = prune_loss(logits, labels, deck_mask)
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

        model.eval()
        val_loss_sum, val_n = 0.0, 0
        with torch.no_grad():
            for deck_ids, deck_mask, cmdr_ids, cmdr_mask, labels in val_loader:
                deck_ids, deck_mask = deck_ids.to(device), deck_mask.to(device)
                cmdr_ids, cmdr_mask = cmdr_ids.to(device), cmdr_mask.to(device)
                labels = labels.to(device)
                logits = model(deck_ids, deck_mask, cmdr_ids, cmdr_mask)
                val_loss_sum += prune_loss(logits, labels, deck_mask).item()
                val_n += 1
        val_loss = val_loss_sum / max(val_n, 1)

        recovery = evaluate_corruption_recovery(model, val_loader, device)

        elapsed = time.time() - t0
        print(f"Epoch {epoch} done in {elapsed:.1f}s -- train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} corruption_recovery={recovery['corruption_recovery_at_true_k']:.4f}", flush=True)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "corruption_recovery_at_true_k": recovery["corruption_recovery_at_true_k"],
            "elapsed_s": elapsed,
        })

        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "vocab_size": tok.vocab_size,
            "epoch": epoch,
            "val_loss": val_loss,
            "encoder_kwargs": {
                "num_heads": num_heads,
                "num_layers": num_layers,
                "dim_feedforward": dim_feedforward,
                "dropout": dropout,
            },
            "train_mode": train_mode,
            "corrupt_ratio_range": ratio_range,
            "inject_ratio_range": resolved_inject_range,
        }
        torch.save(checkpoint, checkpoint_path)

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    return {"model": model, "tokenizer": tok, "splits": splits, "history": history, "device": device}


def load_prune_checkpoint(checkpoint_path: str, device: torch.device) -> tuple[PruneModel, CardTokenizer, dict]:
    """Mirrors training/evaluate.py's load_checkpoint, for PruneModel."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    tok = CardTokenizer.load_pretrained(TOKENIZER_PATH)

    struct_feats = tok.feature_matrix()
    text_feats = tok.text_embeddings(cache_path=TEXT_EMBEDDING_CACHE)
    pad_id = tok.token_to_id[PAD]

    card_embedding = HybridCardEmbedding(
        ckpt["vocab_size"], ckpt["embed_dim"], struct_feats, text_feats, pad_id
    ).to(device)
    encoder = PruneEncoder(ckpt["embed_dim"], **ckpt["encoder_kwargs"]).to(device)
    model = PruneModel(card_embedding, encoder, ckpt["hidden_dim"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, tok, ckpt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--train-mode", default="swap", choices=["swap", "inject"],
        help="swap (default): replace a small fraction of the real deck's own cards in "
             "place, deck stays normal size. inject: add extra cross-deck negatives on top "
             "of the intact real deck (prune_dataset.INJECT_RATIO_RANGE), so the deck is "
             "genuinely oversized -- simulates a real 'too many candidates, cut down to "
             "size' scenario. Validation always uses swap/default density regardless, so "
             "corruption_recovery stays comparable across checkpoints.",
    )
    parser.add_argument(
        "--corrupt-ratio-min", type=float, default=None,
        help="swap mode only. Overrides prune_dataset.CORRUPT_RATIO_RANGE's low end -- "
             "e.g. pass --corrupt-ratio-min 0.15 --corrupt-ratio-max 0.20 to test inject "
             "mode's density at swap mode's unchanged deck size (the density-vs-size "
             "ablation). Both min and max must be given together.",
    )
    parser.add_argument("--corrupt-ratio-max", type=float, default=None)
    parser.add_argument(
        "--inject-ratio-min", type=float, default=None,
        help="inject mode only. Overrides prune_dataset.INJECT_RATIO_RANGE's low end -- "
             "e.g. pass --inject-ratio-min 0.03 --inject-ratio-max 0.08 to test the ORIGINAL "
             "accepted checkpoint's density at inject mode's oversized deck (the other half "
             "of the density-vs-size ablation: isolates size at safe, already-proven density). "
             "Both min and max must be given together.",
    )
    parser.add_argument("--inject-ratio-max", type=float, default=None)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--checkpoint-path", default=CHECKPOINT_PATH)
    parser.add_argument("--history-path", default=HISTORY_PATH)
    args = parser.parse_args()

    if (args.corrupt_ratio_min is None) != (args.corrupt_ratio_max is None):
        raise SystemExit("--corrupt-ratio-min and --corrupt-ratio-max must be given together.")
    corrupt_ratio_range = (
        (args.corrupt_ratio_min, args.corrupt_ratio_max) if args.corrupt_ratio_min is not None else None
    )
    if (args.inject_ratio_min is None) != (args.inject_ratio_max is None):
        raise SystemExit("--inject-ratio-min and --inject-ratio-max must be given together.")
    inject_ratio_range = (
        (args.inject_ratio_min, args.inject_ratio_max) if args.inject_ratio_min is not None else None
    )

    run(
        epochs=args.epochs,
        batch_size=args.batch_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        train_mode=args.train_mode,
        corrupt_ratio_range=corrupt_ratio_range,
        inject_ratio_range=inject_ratio_range,
        limit_train_batches=args.limit_train_batches,
        checkpoint_path=args.checkpoint_path,
        history_path=args.history_path,
    )
