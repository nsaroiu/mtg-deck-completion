# mtg-deck-completion

A deck-completion model for Magic: The Gathering Commander decks, trained
on 122,571 real public Commander decklists
([dataset](https://huggingface.co/datasets/nsaroiu/moxfield-dump)).

Given a commander — and optionally a partial decklist — the trained model
suggests cards to complete the deck, grounded in hybrid ID + content card
embeddings (a learned per-card identity, summed with structured features
and a semantic embedding of the card's oracle text) rather than raw
co-occurrence counts alone.

```
$ python3 -m training.ensemble --commander "Vito, Thorn of the Dusk Rose" --top 10

Top 10 suggestions (2 staple-backfilled, 8 synergy):
  1.000  [synergy]  Sol Ring
  1.000  [synergy]  Sign in Blood
  1.000  [synergy]  Demonic Tutor
  0.999  [synergy]  Urborg, Tomb of Yawgmoth
  0.999  [synergy]  Swiftfoot Boots
  0.999  [synergy]  Nykthos, Shrine to Nyx
  0.999  [synergy]  Mana Vault
  0.999  [synergy]  Necropotence
  1.000  [staple ]  Vampiric Tutor
  1.000  [staple ]  Dark Ritual
```
*(Vito was never seen during training — this is a held-out-commander
completion, purely from color identity + card content.)*

Pretrained weights: [huggingface.co/nsaroiu/mtg-deck-completion](https://huggingface.co/nsaroiu/mtg-deck-completion).

## Architecture

Every card gets a hybrid embedding: a learned per-card vector, summed with
a small MLP over structured features (color identity, mana cost, type,
keywords) and a semantic sentence embedding of its oracle text — so even a
rarely-seen card starts from a meaningful representation instead of a
near-random one (`tokenizer/mtg_tokenizer.py`).

A deck is treated as an unordered *set*, not a sequence: a
permutation-invariant encoder (mean-pool or self-attention,
`training/model.py`) pools the known cards + commander(s) into one vector,
scored against the full card vocabulary via a tied dot product
(word2vec-style). Training is a masked-set-recovery task — hide a random
fraction of a real deck's cards, predict them back from the rest
(`training/deck_dataset.py`).

Production inference is an **ensemble of two specialized checkpoints**
(`training/ensemble.py`), not one model — a single model/loss was tried
(`training/train_two_head.py`) and found to trade off staple recognition
against synergy ranking, so this splits the job instead:

1. **Synergy phase** (Set Transformer encoder, self-attention): generates
   iteratively, a few picks at a time, re-conditioning on its own prior
   picks — this is what surfaces specific combo/synergy pieces rather than
   a generic "good stuff" pile.
2. **Staple backfill** (DeepSets encoder, mean-pool): tops up near-universal
   staples (Sol Ring, Command Tower, ...) to a realistic target count,
   since the synergy phase alone under-recommends them.
3. **Optional pruning pass** (`training/model.py`'s `PruneModel`): a
   separate discriminator, trained as a real-vs-corrupted classifier, that
   re-scores the assembled deck and cuts the weakest fit — the one step no
   generation-only model can do.

Color-identity legality (a card's colors must be a subset of the
commander's) is enforced as a deterministic post-filter on the model's raw
scores, not something the model is asked to learn.

## Setup

```bash
pip install -r requirements.txt
```

## Using the pretrained model

```python
from huggingface_hub import hf_hub_download

repo = "nsaroiu/mtg-deck-completion"
for f in ["deepsets_softmax_checkpoint.pt", "set_transformer_softmax_checkpoint.pt",
          "prune_checkpoint_inject_lowdensity.pt", "tokenizer.json",
          "tokenizer_text_embeddings.npy", "card_tiers.json"]:
    hf_hub_download(repo_id=repo, filename=f, local_dir=".")

from training.evaluate import load_checkpoint
from training.ensemble import ensemble_complete_deck, load_tiers
from training.train import pick_device

device = pick_device()
staple_model, tok, _ = load_checkpoint("deepsets_softmax_checkpoint.pt", device)
synergy_model, _, _ = load_checkpoint("set_transformer_softmax_checkpoint.pt", device)
tiers = load_tiers(tok, path="card_tiers.json")

# (name, score, source) triples, source in {"synergy", "staple"}
results = ensemble_complete_deck(
    ["Atraxa, Praetors' Voice"], [], 20, tok, device,
    staple_model, synergy_model, tiers,
)
for name, score, source in results:
    print(f"{score:.3f}  [{source:7}]  {name}")
```

Or from the command line, once the files above are downloaded:

```bash
python3 -m training.ensemble --commander "Atraxa, Praetors' Voice" --top 20
python3 -m training.ensemble --commander "Silas Renn, Seeker Adept" --commander "Rograkh, Son of Rohgahh" \
    --card "Sol Ring" --card "Command Tower" --top 15
```

`training/complete_deck.py` exposes a single specialist checkpoint at a
time, without the ensemble fusion, if that's all you need.

## Training from scratch

```bash
python3 -m tokenizer.fetch_oracle_cards      # fetch card data from Scryfall
python3 -m tokenizer.mtg_tokenizer           # build the card vocabulary
python3 -m training.train --epochs 40 --loss-type softmax --encoder-type deepsets
python3 -m training.train --epochs 40 --loss-type softmax --encoder-type set_transformer \
    --checkpoint-path ./data/set_transformer_softmax_checkpoint.pt
python3 -m training.export_card_tiers        # data/card_tiers.json, needed by ensemble.py
python3 -m training.evaluate --split all     # Recall/Precision/MRR @ K
```

Training data is loaded automatically: `training/deck_dataset.py`'s
`load_decks_auto` uses a local `data/dataset.jsonl.gz` if present, else
pulls straight from the published
[`nsaroiu/moxfield-dump`](https://huggingface.co/datasets/nsaroiu/moxfield-dump)
Hub dataset (needs the `datasets` package, already in `requirements.txt`).

The optional deck-pruning model trains separately:

```bash
python3 -m training.train_prune --epochs 20
```

## Evaluation

Recall@50 / Precision@50 / MRR on held-out real decks, by how much of the
deck is already known (mask ratio — low = mostly complete, high = mostly
empty), staple-recognition checkpoint, validation split:

| mask ratio | Recall@50 | Precision@50 | MRR |
|---|---|---|---|
| 0.1 (deck mostly complete) | 0.60 | 0.10 | 0.132 |
| 0.5 | 0.51 | 0.44 | 0.068 |
| 0.9 (deck mostly empty) | 0.38 | 0.59 | 0.046 |

Held-out-commander split (commanders never seen in training, testing
generalization via card content rather than memorized co-occurrence):
Recall@50 0.27–0.41 across the same ratio range — meaningfully above
chance, confirming the hybrid content embeddings carry real signal for
unfamiliar commanders.

## Known limitations

- **Staple-tier ranking is imprecise in a near-empty context.** The model
  confidently recognizes *that* a staple belongs (~90% precision when it
  predicts one) well before it reliably ranks *which specific* staple is
  the true target — the ensemble's backfill phase exists specifically to
  work around this.
- **Iterative generation (the synergy phase's default mode) is an
  experimental setting**, tuned on a sample of 11 commanders, not
  exhaustively validated.
- Trained only on real, currently-popular Commander decks — a format
  power-level/meta shift (new sets, ban changes) isn't reflected until
  retrained on fresher data.

## License

MIT (see `LICENSE`). The training data itself is a collection of public
Moxfield decklists with no asserted license — see the
[dataset card](https://huggingface.co/datasets/nsaroiu/moxfield-dump) for
details.
