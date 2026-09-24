"""
mtg_tokenizer.py

A card-level tokenizer for Magic: The Gathering Commander decklists.

Design summary:
- One token = one card, identified by Scryfall `oracle_id` (collapses reprints/foils/promos).
- Basic lands are excluded from the vocabulary entirely (handled by a separate rule-based step).
- Cards are kept in-vocab as long as they're paper-legal in some real sense -- see
  `commander_legality` below -- rather than requiring current Commander legality, so a real
  downloaded deck containing a since-banned card still resolves.
- Special tokens: [PAD], [MASK], [CMDR], [UNK]
- Each card token exposes two complementary feature sources for hybrid ID + content
  embeddings: `feature_matrix()` (cheap structured features: color identity, mana pips,
  type, dynamically-discovered keywords, CMC, legendary/legality flags) and
  `text_embeddings()` (semantic sentence-transformer embeddings of oracle_text -- the
  actual "content" half; structured features alone can't distinguish two same-cost,
  same-type, no-keyword cards with unrelated effects). Both give every card, even one
  with zero decklist occurrences, a meaningful starting representation.

Data source: Scryfall's "Oracle Cards" bulk data file.
https://scryfall.com/docs/api/bulk-data  ->  download the "Oracle Cards" JSON locally
and pass its path to CardTokenizer.from_scryfall_bulk() (or run fetch_oracle_cards.py
to fetch it automatically).

Usage:
    tok = CardTokenizer.from_scryfall_bulk("oracle-cards.json")
    ids = tok.encode_deck(commander="Atraxa, Praetors' Voice",
                           cards=["Sol Ring", "Arcane Signet", "Cyclonic Rift"])
    names = tok.decode(ids)
    feats = tok.feature_matrix()     # (vocab_size, feature_dim) structured features
    texts = tok.text_embeddings()    # (vocab_size, text_dim) semantic features (optional dep)
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import get_close_matches
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from scryfall_bulk import load_bulk_json

# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------

PAD = "[PAD]"
MASK = "[MASK]"
CMDR = "[CMDR]"
UNK = "[UNK]"
SPECIAL_TOKENS = [PAD, MASK, CMDR, UNK]

# These two are genuinely closed, stable enumerations (5 Magic colors; the
# real set of deck-buildable card types), unlike keyword abilities below --
# no need for them to be discovered dynamically from the data.
COLORS = ["W", "U", "B", "R", "G"]
CARD_TYPES = [
    "Creature", "Instant", "Sorcery", "Artifact", "Enchantment",
    "Planeswalker", "Land", "Battle", "Kindred",
]

BASIC_LAND_NAMES = {"Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"}


def _normalize_name(name: str) -> str:
    """Lowercase, strip accents/punctuation for robust name matching."""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9 ]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name


def _pip_counts(mana_cost: str) -> np.ndarray:
    """Count each color's occurrences across all mana symbols in the cost,
    including hybrid/Phyrexian symbols (e.g. {W/U} counts toward both W and
    U, {W/P} counts toward W) -- a coarse but cheap proxy for color intensity
    that plain color-identity membership doesn't capture (e.g. {W}{W} vs {W})."""
    counts = {c: 0.0 for c in COLORS}
    for symbol in re.findall(r"\{([^}]+)\}", mana_cost):
        for c in COLORS:
            if c in symbol:
                counts[c] += 1.0
    return np.array([counts[c] for c in COLORS], dtype=np.float32)


@dataclass
class CardRecord:
    oracle_id: str
    name: str
    mana_cost: str
    cmc: float
    type_line: str
    oracle_text: str
    color_identity: list = field(default_factory=list)
    keywords: list = field(default_factory=list)
    # Scryfall's commander legality value: "legal", "banned", or "not_legal"
    # (joke/un-set/non-paper cards, silver-bordered, etc). Deliberately NOT a
    # bool and NOT a vocab filter beyond excluding "not_legal" -- a card
    # banned in Commander is still a real card that can show up in an older
    # or house-ruled real deck, and legality is time-dependent (bans change),
    # so it's exposed as a feature instead (see CardTokenizer._feature_vector).
    commander_legality: str = "not_legal"
    # Face names for split/adventure/transform (double-faced) cards, e.g.
    # ["Fire", "Ice"] for "Fire // Ice" -- indexed alongside the full combined
    # name so a deck referencing a card by just one face's name still resolves.
    face_names: list = field(default_factory=list)


class CardTokenizer:
    """Maps between card names and integer token ids, and exposes structured
    + semantic features per card for hybrid embedding construction."""

    def __init__(
        self,
        cards: list[CardRecord],
        previous: Optional["CardTokenizer"] = None,
        min_keyword_count: int = 3,
    ):
        filtered = [
            c for c in cards
            if c.name not in BASIC_LAND_NAMES and c.commander_legality != "not_legal"
        ]

        if previous is not None:
            # Stable id assignment across rebuilds (e.g. periodic Scryfall
            # refreshes): cards already known to `previous`
            # keep their relative order, so their token id doesn't shift and
            # embeddings trained against `previous` stay valid. Newly
            # discovered cards are appended after, in file order.
            priority = {c.oracle_id: i for i, c in enumerate(previous.cards)}
            known = sorted(
                (c for c in filtered if c.oracle_id in priority),
                key=lambda c: priority[c.oracle_id],
            )
            new = [c for c in filtered if c.oracle_id not in priority]
            self.cards: list[CardRecord] = known + new
        else:
            self.cards = filtered

        # Keyword vocabulary is discovered from the actual data rather than a
        # hand-maintained whitelist, so it automatically covers every keyword
        # ability Scryfall tracks (including ones like "Partner", "Partner
        # with", "Mutate", "Equip", "Crew", "Companion" that a fixed list can
        # easily miss) instead of silently dropping anything not on the list.
        # Scryfall's `keywords` field also includes one-off flavor-named
        # abilities from joke/crossover sets (e.g. "RadAway", "Grenades!"
        # from a Fallout crossover) -- confirmed live that ~60% of distinct
        # keyword strings occur on exactly one card. A keyword that rare adds
        # a feature dimension with no generalizable signal (effectively just
        # re-deriving that one card's identity), so only keywords occurring
        # on at least `min_keyword_count` cards are included.
        keyword_counts: dict[str, int] = {}
        for c in self.cards:
            for kw in c.keywords:
                keyword_counts[kw] = keyword_counts.get(kw, 0) + 1
        qualifying = {k for k, n in keyword_counts.items() if n >= min_keyword_count}

        if previous is not None:
            # Once a keyword has a feature dimension, keep it as long as it
            # still occurs at all (even if it dips below the threshold on a
            # later rebuild) rather than reordering/dropping it -- avoids the
            # same id-shifting problem __init__ already avoids for cards.
            still_present = {k for k in previous.keyword_vocab if k in keyword_counts}
            newly_qualifying = sorted(qualifying - set(previous.keyword_vocab))
            self.keyword_vocab: list[str] = [
                k for k in previous.keyword_vocab if k in still_present
            ] + newly_qualifying
        else:
            self.keyword_vocab = sorted(qualifying)

        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self.oracle_id_to_token: dict[str, str] = {}
        self._name_index: dict[str, str] = {}  # normalized name -> oracle_id token
        self.card_by_token: dict[str, CardRecord] = {}

        self._build_vocab()

    # ------------------------------------------------------------------
    # Vocab construction
    # ------------------------------------------------------------------

    def _build_vocab(self):
        idx = 0
        for tok in SPECIAL_TOKENS:
            self.token_to_id[tok] = idx
            self.id_to_token[idx] = tok
            idx += 1

        for card in self.cards:
            token = card.oracle_id  # token identity = oracle_id, collapses reprints
            if token in self.token_to_id:
                continue  # duplicate printing, already registered
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            self.oracle_id_to_token[card.oracle_id] = token
            self.card_by_token[token] = card
            self._name_index[_normalize_name(card.name)] = token
            for face_name in card.face_names:
                # setdefault: never let a face name clobber another card's
                # own full-name registration, regardless of processing order
                # (rare but real risk -- some DFCs share generic face names).
                self._name_index.setdefault(_normalize_name(face_name), token)
            idx += 1

        self.vocab_size = len(self.token_to_id)

    @classmethod
    def from_scryfall_bulk(
        cls,
        path: str,
        previous_tokenizer_path: Optional[str] = None,
        min_keyword_count: int = 3,
    ) -> "CardTokenizer":
        """Build a tokenizer from Scryfall's 'Oracle Cards' bulk data file.

        Handles two formats transparently (see scryfall_bulk.load_bulk_json):
          - Standard Scryfall bulk export: a single JSON array `[{...}, {...}]`
          - JSON Lines: one JSON object per line (common if the file was
            re-exported/streamed by another tool, often named `.jsonl`)

        previous_tokenizer_path: path to a previously saved tokenizer.json, if
        one exists. When given, cards and keywords already known to it keep
        their existing ids/feature indices (see __init__) -- pass this
        whenever refreshing from newer Scryfall data (recommended
        periodically, since new sets keep being printed) to avoid silently
        invalidating embeddings trained against the previous vocab.

        min_keyword_count: minimum number of cards a keyword string must
        appear on to get its own feature dimension (see __init__).
        """
        raw = load_bulk_json(path)
        records = []
        for entry in raw:
            # Skip non-paper-relevant or memorabilia/token objects.
            if entry.get("layout") in ("token", "art_series", "double_faced_token"):
                continue

            legalities = entry.get("legalities", {})
            commander_legality = legalities.get("commander", "not_legal")

            # Multi-faced cards (MDFCs, split, adventure): Scryfall nests faces
            # under `card_faces`. We use the top-level oracle_text/mana_cost
            # when present, falling back to a concatenation of faces.
            oracle_text = entry.get("oracle_text", "")
            mana_cost = entry.get("mana_cost", "")
            type_line = entry.get("type_line", "")
            face_names = []
            if "card_faces" in entry:
                faces = entry["card_faces"]
                face_names = [f.get("name") for f in faces if f.get("name")]
                if not oracle_text:
                    oracle_text = " // ".join(f.get("oracle_text", "") for f in faces)
                mana_cost = mana_cost or faces[0].get("mana_cost", "")

            records.append(CardRecord(
                oracle_id=entry["oracle_id"],
                name=entry["name"],
                mana_cost=mana_cost,
                cmc=float(entry.get("cmc", 0.0)),
                type_line=type_line,
                oracle_text=oracle_text,
                color_identity=entry.get("color_identity", []),
                keywords=entry.get("keywords", []),
                commander_legality=commander_legality,
                face_names=face_names,
            ))

        previous = None
        if previous_tokenizer_path is not None and Path(previous_tokenizer_path).exists():
            try:
                previous = cls.load_pretrained(previous_tokenizer_path)
            except ValueError as e:
                # Format-version mismatch (e.g. this rebuild is what's
                # upgrading the schema) means there's no valid ordering to
                # inherit from it anyway -- fall back to a fresh build rather
                # than hard-failing on what's an expected, one-time event.
                print(f"Note: ignoring previous_tokenizer_path ({e})")

        return cls(records, previous=previous, min_keyword_count=min_keyword_count)


    # ------------------------------------------------------------------
    # Name resolution (handles user-typed, possibly imperfect, card names)
    # ------------------------------------------------------------------

    def resolve_name_with_confidence(self, name: str) -> Optional[tuple[str, bool]]:
        """Like resolve_name, but also reports whether the match was exact
        or fuzzy. Fuzzy matches (typo/partial-name fallback) carry real risk
        of matching the wrong card among near-duplicate names -- callers
        doing bulk/unattended resolution can use this to audit or reject
        low-confidence matches instead of silently trusting them the way
        plain resolve_name does.

        Returns (oracle_id_token, is_exact_match), or None if unresolvable.
        """
        norm = _normalize_name(name)
        if norm in self._name_index:
            return self._name_index[norm], True
        # Fuzzy fallback for typos/partial names.
        matches = get_close_matches(norm, self._name_index.keys(), n=1, cutoff=0.85)
        if matches:
            return self._name_index[matches[0]], False
        return None

    def resolve_name(self, name: str) -> Optional[str]:
        """Return the oracle_id token for a user-provided card name, or None."""
        result = self.resolve_name_with_confidence(name)
        return result[0] if result else None


    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    _FORMAT_VERSION = 2

    def save_pretrained(self, path: str) -> None:
        """Save the tokenizer to a JSON file for fast reloading later.

        This serializes the already-filtered, already-deduped, already-ordered
        card list and keyword vocabulary, so `load_pretrained` reproduces the
        exact same token_to_id mapping and feature indices -- important once
        you've trained embeddings against a particular id assignment.
        """
        payload = {
            "format_version": self._FORMAT_VERSION,
            "keyword_vocab": self.keyword_vocab,
            "cards": [
                {
                    "oracle_id": c.oracle_id,
                    "name": c.name,
                    "mana_cost": c.mana_cost,
                    "cmc": c.cmc,
                    "type_line": c.type_line,
                    "oracle_text": c.oracle_text,
                    "color_identity": c.color_identity,
                    "keywords": c.keywords,
                    "commander_legality": c.commander_legality,
                    "face_names": c.face_names,
                }
                for c in self.cards
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load_pretrained(cls, path: str) -> "CardTokenizer":
        """Load a tokenizer previously saved with `save_pretrained`.

        This skips the raw-bulk-data filtering step entirely (no re-parsing
        of a multi-hundred-MB Scryfall file, no re-checking legality/basic
        exclusion) -- it's a straight deserialization, so it's fast and
        deterministic. Vocab ids will exactly match the tokenizer that was
        saved.
        """
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        version = payload.get("format_version")
        if version != cls._FORMAT_VERSION:
            raise ValueError(
                f"{path} was saved with tokenizer format_version={version}, "
                f"but this code expects version={cls._FORMAT_VERSION}. "
                f"Rebuild the tokenizer from raw Scryfall data with "
                f"from_scryfall_bulk() and re-save."
            )

        records = [CardRecord(**c) for c in payload["cards"]]

        # Bypass __init__'s filtering/ordering (already done when originally
        # saved) to avoid re-applying legality/basic-land checks against a
        # possibly-stale definition, and to preserve exact registration order.
        tok = cls.__new__(cls)
        tok.cards = records
        tok.keyword_vocab = payload.get("keyword_vocab", [])
        tok.token_to_id = {}
        tok.id_to_token = {}
        tok.oracle_id_to_token = {}
        tok._name_index = {}
        tok.card_by_token = {}
        tok._build_vocab()
        return tok


    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def encode_deck(self, commander: str, cards: Iterable[str]) -> list[int]:
        """Encode a commander + partial decklist into token ids.
        Order: [CMDR token id, commander card id, card ids...]. Unresolvable
        names map to [UNK] rather than raising, so bad input degrades gracefully.
        """
        ids = [self.token_to_id[CMDR]]

        cmdr_token = self.resolve_name(commander)
        ids.append(self.token_to_id[cmdr_token] if cmdr_token else self.token_to_id[UNK])

        for name in cards:
            token = self.resolve_name(name)
            ids.append(self.token_to_id[token] if token else self.token_to_id[UNK])

        return ids

    def decode(self, ids: Iterable[int]) -> list[str]:
        names = []
        for i in ids:
            tok = self.id_to_token.get(i, UNK)
            if tok in SPECIAL_TOKENS:
                names.append(tok)
            else:
                names.append(self.card_by_token[tok].name)
        return names

    # ------------------------------------------------------------------
    # Structured features (for hybrid ID + content embeddings)
    # ------------------------------------------------------------------

    def _feature_vector(self, card: CardRecord) -> np.ndarray:
        color_vec = np.array([1.0 if c in card.color_identity else 0.0 for c in COLORS])
        pip_vec = _pip_counts(card.mana_cost) / 4.0  # soft-normalize typical pip counts
        type_vec = np.array([1.0 if t in card.type_line else 0.0 for t in CARD_TYPES])
        keyword_vec = np.array([1.0 if k in card.keywords else 0.0 for k in self.keyword_vocab])
        cmc_vec = np.array([card.cmc / 10.0])  # simple scaling, adjust as needed
        legendary_vec = np.array([1.0 if "Legendary" in card.type_line else 0.0])
        # Within the filtered vocab (not_legal already excluded), 0 here
        # unambiguously means "banned" rather than some other illegal reason.
        legal_vec = np.array([1.0 if card.commander_legality == "legal" else 0.0])
        return np.concatenate(
            [color_vec, pip_vec, type_vec, keyword_vec, cmc_vec, legendary_vec, legal_vec]
        ).astype(np.float32)

    @property
    def feature_dim(self) -> int:
        return (
            len(COLORS)        # color identity
            + len(COLORS)      # mana pip counts
            + len(CARD_TYPES)  # card type membership
            + len(self.keyword_vocab)  # dynamically discovered keyword abilities
            + 1                # cmc
            + 1                # is_legendary
            + 1                # is_commander_legal (vs. banned)
        )

    def feature_matrix(self) -> np.ndarray:
        """(vocab_size, feature_dim) array of cheap structured features.
        Special tokens get zero vectors. Feed this into a linear layer to
        initialize/condition content embeddings; combine with a learned
        nn.Embedding(vocab_size, id_dim) *and* text_embeddings() below for
        the full hybrid representation described in the design doc.
        """
        mat = np.zeros((self.vocab_size, self.feature_dim), dtype=np.float32)
        for token, i in self.token_to_id.items():
            if token in SPECIAL_TOKENS:
                continue
            mat[i] = self._feature_vector(self.card_by_token[token])
        return mat

    def oracle_texts(self) -> list[str]:
        """Ordered list of oracle text per vocab id, for feeding into a
        sentence-transformer to get text embeddings (empty string for
        special tokens)."""
        texts = []
        for i in range(self.vocab_size):
            token = self.id_to_token[i]
            if token in SPECIAL_TOKENS:
                texts.append("")
            else:
                texts.append(self.card_by_token[token].oracle_text)
        return texts

    def text_embeddings(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_path: Optional[str] = None,
        batch_size: int = 64,
    ) -> np.ndarray:
        """(vocab_size, text_dim) semantic embeddings of each card's oracle
        text -- the actual "content" half of hybrid ID+content embeddings.
        `feature_matrix()` alone can't distinguish, say, two same-cost
        same-type no-keyword cards with completely different effects (e.g. a
        vanilla counterspell vs. a vanilla cantrip); this is what fixes that,
        and is what gives brand-new/rarely-played cards a meaningful starting
        representation instead of a near-random one. Special tokens get zero
        vectors.

        Requires the optional `sentence-transformers` dependency -- only
        imported when this method is actually called, so basic tokenizer
        usage (encode/decode/feature_matrix) never needs it installed.

        cache_path: if given and the cached array's first dimension matches
        the current vocab_size, load it instead of recomputing. Embedding the
        whole vocab from scratch takes on the order of minutes on CPU -- not
        worth repeating every time this is called (e.g. once per training
        run start).
        """
        if cache_path is not None and Path(cache_path).exists():
            cached = np.load(cache_path)
            if cached.shape[0] == self.vocab_size:
                return cached

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "text_embeddings() requires the optional 'sentence-transformers' "
                "package: pip install sentence-transformers"
            ) from e

        model = SentenceTransformer(model_name)
        texts = self.oracle_texts()
        non_empty = [i for i, t in enumerate(texts) if t]

        # get_sentence_embedding_dimension was renamed get_embedding_dimension
        # in sentence-transformers 6.x; support either.
        if hasattr(model, "get_embedding_dimension"):
            text_dim = model.get_embedding_dimension()
        else:
            text_dim = model.get_sentence_embedding_dimension()

        embeddings = np.zeros((self.vocab_size, text_dim), dtype=np.float32)
        if non_empty:
            encoded = model.encode(
                [texts[i] for i in non_empty],
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
            for pos, i in enumerate(non_empty):
                embeddings[i] = encoded[pos]

        if cache_path is not None:
            np.save(cache_path, embeddings)

        return embeddings

    def __len__(self):
        return self.vocab_size


# ---------------------------------------------------------------------------
# Build tokenizer.json from oracle-cards.jsonl (fetch it first with
# tokenizer/fetch_oracle_cards.py), then smoke-test the result. An existing
# tokenizer.json is passed as previous_tokenizer_path, so a rebuild keeps
# every already-known card's token id and only appends new cards at the end.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    bulk_path = Path("./oracle-cards.jsonl")
    tokenizer_path = Path("./tokenizer.json")
    if not bulk_path.exists():
        raise SystemExit(f"{bulk_path} not found -- run `python3 -m tokenizer.fetch_oracle_cards` first.")
    previous = str(tokenizer_path) if tokenizer_path.exists() else None
    print(f"Building tokenizer from {bulk_path}" + (f" (reusing token ids from {previous})" if previous else ""))
    CardTokenizer.from_scryfall_bulk(str(bulk_path), previous_tokenizer_path=previous).save_pretrained(str(tokenizer_path))

    tok = CardTokenizer.load_pretrained(str(tokenizer_path))
    print(f"Vocab size (basics excluded): {tok.vocab_size}")
    print(f"Keyword vocab size: {len(tok.keyword_vocab)}")
    print(f"Structured feature_dim: {tok.feature_dim}")

    ids = tok.encode_deck(commander="Atraxa, Praetors Voice", cards=["sol ring", "Cyclonic rift", "Not A Real Card"])
    print("Encoded ids:", ids)
    print("Decoded:", tok.decode(ids))

    feats = tok.feature_matrix()
    print("Feature matrix shape:", feats.shape)

    try:
        texts = tok.text_embeddings(cache_path="./tokenizer_text_embeddings.npy")
        print("Text embedding matrix shape:", texts.shape)
    except ImportError as e:
        print(f"(skipping text_embeddings: {e})")
