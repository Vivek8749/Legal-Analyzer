"""Check that quotes attributed to the document actually appear in it.

Claude is instructed to quote verbatim, but a legal review must not show a fabricated or
altered quote as if it were the contract's wording. Every finding's quote is matched
against the source chunks; unmatched quotes are flagged in the report and down-weighted
in the risk score.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

from .models import Chunk

VERIFIED_THRESHOLD = 0.9
_QUOTE_CHARS = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"})
_ELLIPSIS = re.compile(r"\s*(?:\.\.\.|…|\[\.\.\.\])\s*")


def normalize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).translate(_QUOTE_CHARS).lower()
    return re.sub(r"[^a-z0-9$%]+", " ", text).split()


@dataclass
class QuoteMatch:
    verified: bool
    score: float
    chunk_id: int | None


def _coverage(quote_tokens: list[str], source_tokens: list[str]) -> float:
    """Fraction of the quote's tokens found, in order, in the source."""
    if not quote_tokens:
        return 0.0
    n = len(quote_tokens)
    for start in range(len(source_tokens) - n + 1):
        if source_tokens[start:start + n] == quote_tokens:
            return 1.0
    matcher = SequenceMatcher(None, quote_tokens, source_tokens, autojunk=False)
    return sum(block.size for block in matcher.get_matching_blocks()) / n


def match_quote(quote: str, chunks: dict[int, Chunk], preferred_ids: list[int]) -> QuoteMatch:
    """Find the chunk that best contains ``quote``; cited chunks are checked first."""
    # Quotes elided with "..." are verified part by part (each part must be in the same chunk).
    parts = [normalize(p) for p in _ELLIPSIS.split(quote)]
    parts = [p for p in parts if len(p) >= 3] or [normalize(quote)]
    if sum(len(p) for p in parts) < 3:
        return QuoteMatch(False, 0.0, None)

    total = sum(len(p) for p in parts)
    vocabulary = {token for part in parts for token in part}

    def score(chunk: Chunk) -> float:
        tokens = normalize(chunk.text)
        if len(vocabulary & set(tokens)) < 0.5 * len(vocabulary):
            return 0.0  # cheap pre-filter: most of the quote's words are absent
        return sum(_coverage(p, tokens) * len(p) for p in parts) / total

    best = QuoteMatch(False, 0.0, None)
    preferred = set(preferred_ids)
    ordered = [chunks[i] for i in preferred_ids if i in chunks]
    ordered += [c for i, c in chunks.items() if i not in preferred]
    for chunk in ordered:
        s = score(chunk)
        if s > best.score:
            best = QuoteMatch(s >= VERIFIED_THRESHOLD, s, chunk.id)
        if s >= 0.999:
            break
    return best
