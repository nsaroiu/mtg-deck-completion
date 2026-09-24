"""
model.py

Deck-completion model (see README.md's "Architecture" section): a hybrid
ID+content card embedding table, a permutation-invariant
set encoder that pools a variable-size set of card embeddings into one
vector, and a scoring head tied to the same embedding table (word2vec-style)
that scores the entire vocabulary in one forward pass.

Two set encoders are implemented, selectable via build_encoder() /
train.py's --encoder-type: DeepSetsEncoder (mean-pool + MLP -- the original
baseline, cheap enough that a bug is almost certainly in the data/training
pipeline rather than the architecture) and SetTransformerEncoder (self-
attention over the set + a CLS pooling token, compared against DeepSets on
Recall@K/MRR before being adopted as the synergy specialist below).
DeckCompletionModel itself is architecture-agnostic -- it takes
a constructed encoder rather than building one, so both share every other
piece of the pipeline (HybridCardEmbedding, the tied scoring head, training/
eval code) unchanged.

PruneModel (bottom of this file) is a different shape of model entirely:
every encoder above is permutation-INVARIANT (pools a set down to one
vector, then scores the whole vocab from it). PruneModel is permutation-
EQUIVARIANT -- one score per input card, not one pooled vector -- because
its job is "does this specific card, already in this specific deck, belong"
rather than "what's missing". See its own docstring and training/
train_prune.py / training/prune_dataset.py for the full design.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# Canonical Commander deck size (99 nonland/nonbasic cards + 1-2
# commanders, rounded), used only to normalize the optional completeness
# feature below into a roughly [0, 1] ratio.
DECK_SIZE = 100


class HybridCardEmbedding(nn.Module):
    """Per-card embedding = learned id embedding + a small MLP over
    structured features (CardTokenizer.feature_matrix()) + a small MLP over
    semantic text features (CardTokenizer.text_embeddings()), summed. Gives
    every card, even one with zero decklist occurrences, a meaningful
    starting representation instead of a near-random one.

    structured_features/text_features are registered as (non-trainable)
    buffers -- they're fixed per-card content, only the projection layers
    (and the id embedding) are learned.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        structured_features: np.ndarray,
        text_features: np.ndarray,
        pad_id: int,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.id_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        nn.init.normal_(self.id_embedding.weight, mean=0.0, std=0.02)

        # persistent=False: these are deterministically reproducible from the
        # tokenizer file (feature_matrix()/text_embeddings()), not learned --
        # excluding them from state_dict keeps checkpoints to just the
        # trainable parameters instead of needlessly duplicating ~92MB of
        # fixed content features into every saved checkpoint.
        self.register_buffer("structured_features", torch.tensor(structured_features, dtype=torch.float32), persistent=False)
        self.register_buffer("text_features", torch.tensor(text_features, dtype=torch.float32), persistent=False)

        self.structured_proj = nn.Sequential(
            nn.Linear(structured_features.shape[1], embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_features.shape[1], embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: any integer tensor shape (...) -> embeddings (..., embed_dim)."""
        id_e = self.id_embedding(ids)
        struct_e = self.structured_proj(self.structured_features[ids])
        text_e = self.text_proj(self.text_features[ids])
        return id_e + struct_e + text_e

    def full_table(self) -> torch.Tensor:
        """(vocab_size, embed_dim) embedding for every card in the vocab --
        used by the tied scoring head. Recomputed each call since the
        projection layers are being trained; cheap at this vocab size."""
        idx = torch.arange(self.vocab_size, device=self.id_embedding.weight.device)
        return self.forward(idx)


class DeepSetsEncoder(nn.Module):
    """Permutation-invariant set encoder: encode each element independently
    (already done by HybridCardEmbedding), mean-pool over the set, then an
    MLP on the pooled vector. Can't directly represent pairwise/combo
    interactions the way self-attention can (see SetTransformerEncoder
    below) -- an acceptable tradeoff for a first baseline, and still the
    stronger of the two at staple recognition specifically (see
    DeckCompletionModel's ensemble usage).
    """

    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, card_embeds: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """card_embeds: (B, L, D), mask: (B, L) bool (True = real card, not
        padding) -> pooled (B, D)."""
        mask_f = mask.unsqueeze(-1).to(card_embeds.dtype)
        summed = (card_embeds * mask_f).sum(dim=1)
        counts = mask_f.sum(dim=1).clamp(min=1.0)
        pooled = summed / counts
        return self.mlp(pooled)


class SetTransformerEncoder(nn.Module):
    """Permutation-invariant set encoder via self-attention: a learned CLS
    token attends over the (masked) set alongside every real element, and
    its output is the pooled vector -- the "Set Transformer" alternative to
    DeepSetsEncoder's mean-pool, named as the natural next architecture to
    try in this module's docstring above. No positional encoding, since set
    order carries no meaning.

    CLS-token pooling (not PMA/attention-pooling) specifically because the
    CLS token is always unmasked: a set that's entirely padding (e.g.
    complete_deck.py's commander-only completion, an all-False ctx_mask)
    still has one real, attendable token, so there's no all-masked-row edge
    case to special-case the way DeepSetsEncoder's mean-pool needs a
    denominator clamp for the same input.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN: trains more stably with no LR warmup
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, card_embeds: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """card_embeds: (B, L, D), mask: (B, L) bool (True = real card, not
        padding) -> pooled (B, D), the transformer's CLS output."""
        batch_size = card_embeds.size(0)
        cls = self.cls_token.expand(batch_size, 1, -1)
        x = torch.cat([cls, card_embeds], dim=1)
        full_mask = torch.cat([mask.new_ones(batch_size, 1), mask], dim=1)
        # nn.TransformerEncoder's key_padding_mask convention is inverted
        # from this codebase's (True = ignore, vs. True = real card here).
        out = self.transformer(x, src_key_padding_mask=~full_mask)
        return out[:, 0, :]


def build_encoder(encoder_type: str, embed_dim: int, hidden_dim: int, **kwargs) -> nn.Module:
    """Constructs a set encoder by name, from a checkpoint's ("encoder_type",
    "encoder_kwargs") or train.py's --encoder-type CLI flag -- the single
    place both train.py and evaluate.py go to reconstruct an encoder, so
    they can never disagree on how."""
    if encoder_type == "deepsets":
        return DeepSetsEncoder(embed_dim, hidden_dim)
    if encoder_type == "set_transformer":
        return SetTransformerEncoder(embed_dim, **kwargs)
    raise ValueError(f"Unknown encoder_type: {encoder_type!r}")


class DeckCompletionModel(nn.Module):
    """Commander(s) and context cards are each pooled by the same set
    encoder (a commander set is just a small card set), combined into one
    deck vector, then scored against the full card vocabulary via a dot
    product with the tied embedding table.

    Takes an already-constructed encoder (DeepSetsEncoder or
    SetTransformerEncoder, see build_encoder above) rather than building one
    itself, so this class stays architecture-agnostic -- deck_proj's job
    (project concatenated ctx+cmdr vectors to a deck vector) is identical
    either way.

    use_completeness_feature: appends one extra scalar (known-card count /
    DECK_SIZE) to deck_proj's input. Motivated by a read-only probe finding
    real signal: a linear regression of known_count against the raw pooled
    ctx_vec (before deck_proj) reached held-out R^2=0.21 on a production
    checkpoint that was never trained to preserve it, but R^2=0.05 against
    the post-deck_proj deck_vec actually used for scoring -- i.e. deck_proj
    already has a natural route to this information available (mean-
    pooling's per-dimension variance shrinks as more cards are averaged in)
    and mostly discards it, since nothing in training ever rewarded keeping
    it. In practice a real training run found this hurt more than it
    helped (regressed staple-tier MRR and held-out-commander recall with no
    compensating gain) -- kept here, off by default (False, so every
    existing checkpoint/caller is unaffected), for the record rather than
    wired into any production checkpoint."""

    def __init__(
        self,
        card_embedding: HybridCardEmbedding,
        encoder: nn.Module,
        hidden_dim: int,
        use_completeness_feature: bool = False,
    ):
        super().__init__()
        self.card_embedding = card_embedding
        embed_dim = card_embedding.embed_dim
        self.encoder = encoder
        self.use_completeness_feature = use_completeness_feature
        extra_dim = 1 if use_completeness_feature else 0
        self.deck_proj = nn.Sequential(
            nn.Linear(embed_dim * 2 + extra_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def deck_vector(
        self,
        ctx_ids: torch.Tensor,
        ctx_mask: torch.Tensor,
        cmdr_ids: torch.Tensor,
        cmdr_mask: torch.Tensor,
    ) -> torch.Tensor:
        ctx_vec = self.encoder(self.card_embedding(ctx_ids), ctx_mask)
        cmdr_vec = self.encoder(self.card_embedding(cmdr_ids), cmdr_mask)
        parts = [ctx_vec, cmdr_vec]
        if self.use_completeness_feature:
            known = (ctx_mask.sum(dim=1) + cmdr_mask.sum(dim=1)).to(ctx_vec.dtype)
            known_ratio = (known / DECK_SIZE).clamp(max=1.0).unsqueeze(-1)  # (B, 1)
            parts.append(known_ratio)
        return self.deck_proj(torch.cat(parts, dim=-1))

    def forward(
        self,
        ctx_ids: torch.Tensor,
        ctx_mask: torch.Tensor,
        cmdr_ids: torch.Tensor,
        cmdr_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Returns (B, vocab_size) logits -- one per card in the vocab."""
        deck_vec = self.deck_vector(ctx_ids, ctx_mask, cmdr_ids, cmdr_mask)
        full_table = self.card_embedding.full_table()  # (V, D)
        return deck_vec @ full_table.T


class TwoHeadDeckCompletionModel(nn.Module):
    """Two specialized scoring heads sharing one trunk (HybridCardEmbedding
    + a set encoder), instead of DeckCompletionModel's single deck_proj --
    built to replace training/ensemble.py's two-FULL-MODEL specialization
    (a staple-precision checkpoint + a synergy-MRR checkpoint, fused at
    generation time) with something that gets the same specialization at
    roughly one model's cost: the expensive part (embedding lookups, set
    encoding) is shared, only the small final projection-and-score step is
    duplicated. See training/ensemble.py's docstring for why one model
    under one loss couldn't hold both properties at once, and why this
    two-head attempt was itself eventually rejected in favor of two fully
    independent checkpoints (kept here for the record).

    staple_proj is meant to be trained with a pointwise loss (BCE,
    calibrated per-card precision); synergy_proj with a competitive/ranking
    loss (multi_positive_softmax_loss, sharp relative ordering) -- see
    training/train_two_head.py's build_loss_fn-equivalent. Both heads score
    against the SAME tied full_table(), so a card's identity/content
    embedding is shared between what the two heads say about it; only the
    deck-context-to-score projection differs.
    """

    def __init__(self, card_embedding: HybridCardEmbedding, encoder: nn.Module, hidden_dim: int):
        super().__init__()
        self.card_embedding = card_embedding
        self.encoder = encoder
        embed_dim = card_embedding.embed_dim

        def make_head() -> nn.Module:
            return nn.Sequential(
                nn.Linear(embed_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embed_dim),
            )

        self.staple_proj = make_head()
        self.synergy_proj = make_head()

    def _pool(
        self,
        ctx_ids: torch.Tensor,
        ctx_mask: torch.Tensor,
        cmdr_ids: torch.Tensor,
        cmdr_mask: torch.Tensor,
    ) -> torch.Tensor:
        ctx_vec = self.encoder(self.card_embedding(ctx_ids), ctx_mask)
        cmdr_vec = self.encoder(self.card_embedding(cmdr_ids), cmdr_mask)
        return torch.cat([ctx_vec, cmdr_vec], dim=-1)

    def forward(
        self,
        ctx_ids: torch.Tensor,
        ctx_mask: torch.Tensor,
        cmdr_ids: torch.Tensor,
        cmdr_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Returns {"staple": (B, V) logits, "synergy": (B, V) logits}."""
        pooled = self._pool(ctx_ids, ctx_mask, cmdr_ids, cmdr_mask)
        full_table = self.card_embedding.full_table()  # (V, D)
        return {
            "staple": self.staple_proj(pooled) @ full_table.T,
            "synergy": self.synergy_proj(pooled) @ full_table.T,
        }


class HeadView(nn.Module):
    """Adapts one head of a TwoHeadDeckCompletionModel to the single-tensor
    `model(ctx_ids, ctx_mask, cmdr_ids, cmdr_mask) -> (B, V) logits`
    interface training/metrics.py's recall_at_k/nearest_neighbors already
    expect -- so a two-head
    model's staple/synergy heads can be evaluated with the EXACT existing
    machinery, unchanged, as if each were its own checkpoint, rather than
    needing a parallel eval stack built just for this architecture."""

    def __init__(self, two_head_model: TwoHeadDeckCompletionModel, head: str):
        super().__init__()
        if head not in ("staple", "synergy"):
            raise ValueError(f"Unknown head: {head!r}")
        self.model = two_head_model
        self.head = head
        self.card_embedding = two_head_model.card_embedding

    def forward(self, *args) -> torch.Tensor:
        return self.model(*args)[self.head]


class PruneEncoder(nn.Module):
    """Permutation-EQUIVARIANT self-attention encoder: every position in the
    input set attends to every other position (commander(s) included) and
    the encoder returns one refined embedding PER POSITION, not one pooled
    vector -- see PruneModel below for why. Structurally identical to
    SetTransformerEncoder's transformer stack, minus the CLS token (there's
    nothing to pool to here) and minus the final [:, 0, :] slice (every
    position's output is kept, not just one). No positional encoding, same
    as every other set encoder in this file, since set order carries no
    meaning.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, embeds: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """embeds: (B, L, D), mask: (B, L) bool (True = real, not padding)
        -> (B, L, D), one refined embedding per input position. Every
        sequence passed in here always has at least one real (unmasked)
        position -- the commander(s), concatenated in by PruneModel below
        -- so there's no all-masked-row edge case to guard against, the same
        reasoning SetTransformerEncoder's CLS token relies on."""
        return self.transformer(embeds, src_key_padding_mask=~mask)


class PruneModel(nn.Module):
    """Scores each card in a candidate deck for whether it belongs, given
    the commander(s) and the REST of the deck -- the deck-pruning model
    (training/prune_dataset.py, training/train_prune.py). Trained as a
    real-vs-corrupted discriminator: take a real deck, swap a few of its
    cards for real cards pulled from OTHER decks, and predict per-card which
    ones are the swaps. Motivation, in short: every generation model in
    this file only ever ADDS cards; nothing evaluates a card already sitting in a
    completed list, which is the one deckbuilding move a human does on a
    final pass and this pipeline couldn't do at all before this model.

    Structurally the mirror image of DeckCompletionModel: that model pools
    a variable-size set down to ONE vector, then scores the whole vocab
    against it (permutation-invariant, one output shared across every
    candidate card). This model keeps one output PER input card
    (permutation-equivariant, via PruneEncoder's self-attention) -- there's
    no fixed vocab to score against here, only "does this specific card, in
    this specific spot, fit the other 99". A learned segment embedding
    distinguishes commander positions (context only, always real, never
    scored) from deck-card positions (scored, possibly corrupted).
    """

    def __init__(self, card_embedding: HybridCardEmbedding, encoder: PruneEncoder, hidden_dim: int):
        super().__init__()
        self.card_embedding = card_embedding
        self.encoder = encoder
        embed_dim = card_embedding.embed_dim
        # 0 = deck-card position (scored), 1 = commander position (context only)
        self.segment_embedding = nn.Embedding(2, embed_dim)
        self.score_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        deck_ids: torch.Tensor,
        deck_mask: torch.Tensor,
        cmdr_ids: torch.Tensor,
        cmdr_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Returns (B, L_deck) real-vs-corrupted logits, one per deck_ids
        position (higher = more confidently real). Padding positions'
        logits are meaningless -- callers must mask with deck_mask, same
        convention as every masked tensor elsewhere in this pipeline."""
        deck_embeds = self.card_embedding(deck_ids) + self.segment_embedding(
            torch.zeros_like(deck_ids)
        )
        cmdr_embeds = self.card_embedding(cmdr_ids) + self.segment_embedding(
            torch.ones_like(cmdr_ids)
        )
        combined = torch.cat([deck_embeds, cmdr_embeds], dim=1)
        combined_mask = torch.cat([deck_mask, cmdr_mask], dim=1)
        out = self.encoder(combined, combined_mask)
        deck_out = out[:, : deck_ids.shape[1], :]
        return self.score_head(deck_out).squeeze(-1)
