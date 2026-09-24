"""
train_prune_hardneg.py

Alternative pruner training: instead of prune_dataset.py's cross-deck
real-card negatives, mines negatives from the (frozen) SYNERGY GENERATOR's
own top mispredictions when asked to complete a deck missing a few real
cards -- closer to the actual mistakes the pruner needs to catch in
production, since a deck it prunes there was built by that same generator.
The original checkpoint's negatives (real cards pulled from OTHER decks)
teach general cross-deck fit; this teaches "does this look like one of
MY OWN generator's plausible-but-wrong picks."

Trains on prune_dataset.HardNegativeDeckDataset (train split only) --
evaluates on the ORIGINAL prune_dataset.PruneDataset (cross-deck
negatives, val split), the exact same benchmark data/prune_checkpoint.pt
was scored on, so corruption_recovery is directly comparable: different
training data, same yardstick.

Usage:
    python3 -m training.train_prune_hardneg
    python3 -m training.train_prune_hardneg --epochs 20
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tokenizer.mtg_tokenizer import SPECIAL_TOKENS, CardTokenizer, PAD
from training.complete_deck import legal_mask
from training.deck_dataset import load_decks_auto, split_decks
from training.ensemble import SYNERGY_CHECKPOINT
from training.evaluate import load_checkpoint
from training.model import HybridCardEmbedding, PruneEncoder, PruneModel
from training.prune_dataset import (
    HardNegativeDeckDataset,
    PruneDataset,
    make_hard_negative_collate_fn,
    make_prune_collate_fn,
)
from training.train import pick_device
from training.train_prune import evaluate_corruption_recovery, prune_loss

TOKENIZER_PATH = "./tokenizer.json"
TEXT_EMBEDDING_CACHE = "./tokenizer_text_embeddings.npy"
CHECKPOINT_PATH = "./data/prune_checkpoint_hardneg.pt"
HISTORY_PATH = "./data/prune_train_history_hardneg.json"


def commander_colors_for_batch(cmdr_ids: torch.Tensor, cmdr_mask: torch.Tensor, tok: CardTokenizer) -> list[frozenset]:
    colors_list = []
    for i in range(cmdr_ids.shape[0]):
        colors: set[str] = set()
        for j in range(cmdr_ids.shape[1]):
            if cmdr_mask[i, j]:
                card = tok.card_by_token[tok.id_to_token[int(cmdr_ids[i, j].item())]]
                colors |= set(card.color_identity)
        colors_list.append(frozenset(colors))
    return colors_list


def build_legal_mask_batch(
    colors_list: list[frozenset], tok: CardTokenizer, cache: dict[frozenset, torch.Tensor], device: torch.device,
) -> torch.Tensor:
    rows = []
    for colors in colors_list:
        cached = cache.get(colors)
        if cached is None:
            cached = legal_mask(tok, set(colors)).to(device)
            cache[colors] = cached
        rows.append(cached)
    return torch.stack(rows, dim=0)


def build_hard_negative_batch(
    synergy_model,
    real_ids: torch.Tensor,
    real_mask: torch.Tensor,
    held_out_ids: torch.Tensor,
    held_out_mask: torch.Tensor,
    n_corrupt: torch.Tensor,
    cmdr_ids: torch.Tensor,
    cmdr_mask: torch.Tensor,
    pad_id: int,
    invalid_mask: torch.Tensor,
    legal_mask_batch: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mines n_corrupt[i] genuine mispredictions per example: runs the
    frozen synergy generator on (real_ids, commander) -- the exact
    "context missing a few cards" shape it sees in real use -- restricted
    to legal, not-already-visible candidates, and excludes the TRUE
    held-out cards too (so a lucky correct guess can never get labeled a
    negative by mistake -- every mined substitute is a genuine
    misprediction). Assembles and returns (deck_ids, deck_mask, labels):
    real cards labeled 1, mined mispredictions labeled 0, appended after
    the real cards. A batch element whose legal pool runs out before
    n_corrupt is satisfied simply gets fewer negatives (deck_mask=False for
    the unfilled columns), same graceful-degradation pattern as
    prune_dataset.PruneDataset's cross-deck sampling."""
    was_training = synergy_model.training
    synergy_model.eval()
    with torch.no_grad():
        logits = synergy_model(real_ids, real_mask, cmdr_ids, cmdr_mask)  # (B, V)
        already_visible = torch.zeros_like(logits, dtype=torch.bool)
        already_visible.scatter_(1, real_ids, real_mask)
        already_visible.scatter_(1, cmdr_ids, cmdr_mask)
        already_visible.scatter_(1, held_out_ids, held_out_mask)
        logits = logits.masked_fill(already_visible, float("-inf"))
        logits = logits.masked_fill(invalid_mask.unsqueeze(0), float("-inf"))
        logits = logits.masked_fill(~legal_mask_batch, float("-inf"))

        max_k = max(int(n_corrupt.max().item()), 1)
        top_scores, top_idx = torch.topk(logits, max_k, dim=1)
    if was_training:
        synergy_model.train()

    B, L_real = real_ids.shape
    total_len = L_real + max_k
    deck_ids = torch.full((B, total_len), pad_id, dtype=torch.long, device=device)
    deck_mask = torch.zeros((B, total_len), dtype=torch.bool, device=device)
    labels = torch.zeros((B, total_len), dtype=torch.float32, device=device)

    deck_ids[:, :L_real] = real_ids
    deck_mask[:, :L_real] = real_mask
    labels[:, :L_real] = 1.0

    col = torch.arange(max_k, device=device).unsqueeze(0)
    neg_valid = (col < n_corrupt.unsqueeze(1)) & (top_scores > float("-inf"))
    deck_ids[:, L_real:] = torch.where(neg_valid, top_idx, pad_id)
    deck_mask[:, L_real:] = neg_valid
    # labels already 0.0 for the mined-negative columns

    return deck_ids, deck_mask, labels


def run(
    epochs: int = 20,
    batch_size: int = 256,
    embed_dim: int = 128,
    hidden_dim: int = 256,
    lr: float = 1e-3,
    num_heads: int = 4,
    num_layers: int = 3,
    dim_feedforward: int = 512,
    dropout: float = 0.1,
    synergy_checkpoint: str = SYNERGY_CHECKPOINT,
    log_every: int = 100,
    limit_train_batches: int | None = None,
    checkpoint_path: str = CHECKPOINT_PATH,
    history_path: str = HISTORY_PATH,
    device: torch.device | None = None,
) -> dict:
    device = device or pick_device()
    print(f"Device: {device}", flush=True)
    print(f"Synergy checkpoint (negative-mining oracle): {synergy_checkpoint}", flush=True)

    tok = CardTokenizer.load_pretrained(TOKENIZER_PATH)
    decks = load_decks_auto(tok)
    splits = split_decks(decks)
    print(f"Decks -- train: {len(splits['train'])}, val: {len(splits['val'])}, "
          f"test: {len(splits['test'])}, held_out_commander: {len(splits['held_out_commander'])}", flush=True)

    pad_id = tok.token_to_id[PAD]
    hardneg_collate = make_hard_negative_collate_fn(pad_id)
    prune_collate = make_prune_collate_fn(pad_id)

    train_ds = HardNegativeDeckDataset(splits["train"])
    val_ds = PruneDataset(splits["val"], tok, seed=1)  # same benchmark train_prune.py used

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=hardneg_collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=prune_collate)

    struct_feats = tok.feature_matrix()
    text_feats = tok.text_embeddings(cache_path=TEXT_EMBEDDING_CACHE)

    card_embedding = HybridCardEmbedding(tok.vocab_size, embed_dim, struct_feats, text_feats, pad_id).to(device)
    encoder = PruneEncoder(
        embed_dim, num_heads=num_heads, num_layers=num_layers,
        dim_feedforward=dim_feedforward, dropout=dropout,
    ).to(device)
    model = PruneModel(card_embedding, encoder, hidden_dim).to(device)

    synergy_model, _, _ = load_checkpoint(synergy_checkpoint, device)
    synergy_model.eval()
    for p in synergy_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}", flush=True)

    special_ids = [tok.token_to_id[t] for t in SPECIAL_TOKENS if t in tok.token_to_id]
    invalid_mask = torch.zeros(tok.vocab_size, dtype=torch.bool, device=device)
    invalid_mask[special_ids] = True
    legal_mask_cache: dict[frozenset, torch.Tensor] = {}

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running_loss, n_seen = 0.0, 0
        epoch_loss_sum, epoch_n = 0.0, 0

        for step, (real_ids, real_mask, held_out_ids, held_out_mask, n_corrupt, cmdr_ids, cmdr_mask) in enumerate(
            train_loader, start=1
        ):
            if limit_train_batches is not None and step > limit_train_batches:
                break

            real_ids, real_mask = real_ids.to(device), real_mask.to(device)
            held_out_ids, held_out_mask = held_out_ids.to(device), held_out_mask.to(device)
            cmdr_ids, cmdr_mask = cmdr_ids.to(device), cmdr_mask.to(device)
            n_corrupt = n_corrupt.to(device)

            colors_list = commander_colors_for_batch(cmdr_ids, cmdr_mask, tok)
            legal_mask_batch = build_legal_mask_batch(colors_list, tok, legal_mask_cache, device)

            deck_ids, deck_mask, labels = build_hard_negative_batch(
                synergy_model, real_ids, real_mask, held_out_ids, held_out_mask, n_corrupt,
                cmdr_ids, cmdr_mask, pad_id, invalid_mask, legal_mask_batch, device,
            )

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
            "negative_source": "generator_hard_negatives",
            "synergy_checkpoint": synergy_checkpoint,
        }
        torch.save(checkpoint, checkpoint_path)

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    return {"model": model, "tokenizer": tok, "splits": splits, "history": history, "device": device}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--synergy-checkpoint", default=SYNERGY_CHECKPOINT)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--checkpoint-path", default=CHECKPOINT_PATH)
    parser.add_argument("--history-path", default=HISTORY_PATH)
    args = parser.parse_args()

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
        synergy_checkpoint=args.synergy_checkpoint,
        limit_train_batches=args.limit_train_batches,
        checkpoint_path=args.checkpoint_path,
        history_path=args.history_path,
    )
