"""
evaluate.py

Evaluation for the deck-completion model: Recall@K on held-out (masked) cards across fixed
mask ratios, broken out by card popularity tier (since staples like Sol Ring
appear in almost every deck -- a model that just always predicts staples
regardless of context can otherwise look deceptively good on an aggregate
Recall@K number), plus a nearest-neighbor qualitative check on the learned
embeddings.

Usage:
    python3 -m training.evaluate
    python3 -m training.evaluate --split test --checkpoint ./data/deepsets_checkpoint.pt
"""

from __future__ import annotations

import argparse

import torch

from tokenizer.mtg_tokenizer import CardTokenizer, PAD
from training.deck_dataset import load_decks_auto, split_decks
from training.metrics import (
    STAPLE_COLOR_TIER_NAMES,
    card_tiers,
    nearest_neighbors,
    recall_at_k,
    staple_color_tiers,
)
from training.model import DeckCompletionModel, HybridCardEmbedding, TwoHeadDeckCompletionModel, build_encoder
from training.train import TEXT_EMBEDDING_CACHE, TOKENIZER_PATH, pick_device

DEFAULT_CHECKPOINT = "./data/deepsets_checkpoint.pt"


def load_checkpoint(checkpoint_path: str, device: torch.device):
    tok = CardTokenizer.load_pretrained(TOKENIZER_PATH)
    struct_feats = tok.feature_matrix()
    text_feats = tok.text_embeddings(cache_path=TEXT_EMBEDDING_CACHE)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    pad_id = tok.token_to_id[PAD]

    card_embedding = HybridCardEmbedding(
        ckpt["vocab_size"], ckpt["embed_dim"], struct_feats, text_feats, pad_id
    )
    # ckpt.get(...) defaults: checkpoints saved before the Set Transformer
    # addition have no encoder_type key at all and are all DeepSets.
    encoder_type = ckpt.get("encoder_type", "deepsets")
    encoder_kwargs = ckpt.get("encoder_kwargs", {}) or {}
    encoder = build_encoder(encoder_type, ckpt["embed_dim"], ckpt["hidden_dim"], **encoder_kwargs)
    # ckpt.get(...) default: checkpoints saved before the deck-completeness
    # feature was added have no such key and were all trained without it.
    model = DeckCompletionModel(
        card_embedding, encoder, ckpt["hidden_dim"],
        use_completeness_feature=ckpt.get("use_completeness_feature", False),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, tok, ckpt


def load_two_head_checkpoint(checkpoint_path: str, device: torch.device):
    """Sibling of load_checkpoint for a TwoHeadDeckCompletionModel checkpoint
    (training/train_two_head.py, ckpt["architecture"] == "two_head") --
    same tokenizer/feature-loading and encoder reconstruction, just a
    different model class. Kept separate rather than branched inside
    load_checkpoint since the two model classes have different forward
    signatures (one (B,V) logits tensor vs. a {"staple","synergy"} dict) --
    callers need to know which one they're getting."""
    tok = CardTokenizer.load_pretrained(TOKENIZER_PATH)
    struct_feats = tok.feature_matrix()
    text_feats = tok.text_embeddings(cache_path=TEXT_EMBEDDING_CACHE)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    pad_id = tok.token_to_id[PAD]

    card_embedding = HybridCardEmbedding(
        ckpt["vocab_size"], ckpt["embed_dim"], struct_feats, text_feats, pad_id
    )
    encoder = build_encoder(
        ckpt.get("encoder_type", "deepsets"), ckpt["embed_dim"], ckpt["hidden_dim"],
        **(ckpt.get("encoder_kwargs", {}) or {}),
    )
    model = TwoHeadDeckCompletionModel(card_embedding, encoder, ckpt["hidden_dim"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, tok, ckpt


def run(
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    split: str = "val",
    max_decks: int | None = None,
    tier_mode: str = "default",
):
    device = pick_device()
    model, tok, ckpt = load_checkpoint(checkpoint_path, device)
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")

    decks = load_decks_auto(tok)
    splits = split_decks(decks)
    if tier_mode == "color_split":
        tiers = staple_color_tiers(splits["train"], tok)
        tier_names = STAPLE_COLOR_TIER_NAMES
    else:
        tiers = card_tiers(splits["train"])
        tier_names = ("staple", "mid", "long_tail")

    for name in ("val", "test", "held_out_commander") if split == "all" else (split,):
        result = recall_at_k(
            model, splits[name], tiers,
            device=device, max_decks=max_decks, tier_names=tier_names,
        )
        print(f"\n=== {name} ({result['n_decks']} decks) ===")
        print("Recall@K by mask ratio:")
        for k, v in result["by_ratio"].items():
            print(f"  {k}: {v:.3f}")
        print("Recall@K by popularity tier (of true targets, how many found):")
        for k, v in result["by_tier"].items():
            print(f"  {k}: {v:.3f}")
        print("Precision@K by mask ratio:")
        for k, v in result["precision_by_ratio"].items():
            print(f"  {k}: {v:.3f}")
        print("Precision@K by popularity tier (of predictions in this tier, how many correct):")
        for k, v in result["precision_by_tier"].items():
            print(f"  {k}: {v:.3f}")
        print("MRR by mask ratio:")
        for k, v in result["mrr_by_ratio"].items():
            print(f"  ratio={k}: {v:.4f}")
        print("MRR by popularity tier:")
        for k, v in result["mrr_by_tier"].items():
            print(f"  {k}: {v:.4f}")

    print("\n=== Nearest-neighbor sanity check (learned embeddings) ===")
    for name in ("Sol Ring", "Swords to Plowshares", "Cyclonic Rift"):
        neighbors = nearest_neighbors(model, tok, name, device=device)
        print(f"{name}:")
        for n, sim in neighbors:
            print(f"   {n} (sim={sim:.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", default="val", choices=["train", "val", "test", "held_out_commander", "all"])
    parser.add_argument("--max-decks", type=int, default=None)
    parser.add_argument("--tier-mode", default="default", choices=["default", "color_split"],
                         help="'color_split' further splits 'staple' into 'universal' (colorless) "
                              "vs 'color_popular' (has a color identity) -- see staple_color_tiers().")
    args = parser.parse_args()
    run(checkpoint_path=args.checkpoint, split=args.split, max_decks=args.max_decks, tier_mode=args.tier_mode)
