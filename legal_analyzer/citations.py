"""Citation helpers shared by the model providers.

Claude cites passages natively (``search_result`` blocks). Other models are asked to cite
excerpts inline as ``[C12]``; :func:`parse_inline_citations` turns those markers into the
same (text, chunk ids) segments, so the rest of the app treats every provider alike.
"""

from __future__ import annotations

import re

from .chunker import split_sentences
from .grounding import normalize
from .models import Chunk

MAX_BLOCK_CHARS = 400

# "[C12]", "[C12, C14]", "[C12; c3]" (optionally preceded by whitespace, which is dropped)
_MARKER = re.compile(r"\s*\[\s*([Cc]\d+(?:\s*[,;]\s*[Cc]?\d+)*)\s*\]")


def citable_blocks(chunk: Chunk) -> list[str]:
    """Split a chunk into paragraph/sentence-sized passages, the unit a citation points at."""
    blocks: list[str] = []
    for paragraph in chunk.text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= MAX_BLOCK_CHARS:
            blocks.append(paragraph)
            continue
        current = ""
        for sentence in split_sentences(paragraph):
            if current and len(current) + len(sentence) + 1 > MAX_BLOCK_CHARS:
                blocks.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if current:
            blocks.append(current)
    return blocks or [chunk.text]


def parse_inline_citations(text: str, valid_ids: set[int]) -> list[tuple[str, list[int]]]:
    """Split an answer at ``[C12]`` markers into (text, cited chunk ids) segments.

    Markers citing excerpts the model was never given are dropped, so a hallucinated ID
    can't turn into a citation.
    """
    segments: list[tuple[str, list[int]]] = []
    cursor = 0
    for match in _MARKER.finditer(text):
        ids = [int(n) for n in re.findall(r"\d+", match.group(1))]
        ids = [i for i in dict.fromkeys(ids) if i in valid_ids]
        piece = text[cursor:match.start()]
        if not piece and segments:
            segments[-1][1].extend(i for i in ids if i not in segments[-1][1])
        else:
            segments.append((piece, ids))
        cursor = match.end()
    if cursor < len(text):
        segments.append((text[cursor:], []))
    return segments


def best_matching_passage(claim: str, chunk: Chunk) -> str | None:
    """The passage of ``chunk`` that best supports ``claim`` (by word overlap), if any."""
    claim_words = set(normalize(claim))
    if not claim_words:
        return None
    best, best_score = None, 0.0
    for block in citable_blocks(chunk):
        words = set(normalize(block))
        if not words:
            continue
        score = len(claim_words & words) / len(claim_words)
        if score > best_score:
            best, best_score = block, score
    return best if best_score >= 0.15 else None


def cited_passages(segments: list[tuple[str, list[int]]], chunks: dict[int, Chunk]) -> dict[int, list[str]]:
    """For inline citations, guess the passage each citation points at from the cited sentence."""
    passages: dict[int, list[str]] = {}
    for text, ids in segments:
        sentences = split_sentences(" ".join(text.split()))
        claim = sentences[-1] if sentences else text
        for chunk_id in ids:
            passage = best_matching_passage(claim, chunks[chunk_id]) if chunk_id in chunks else None
            if passage and passage not in passages.setdefault(chunk_id, []):
                passages[chunk_id].append(passage)
    return passages
