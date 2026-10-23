"""Clause-aware chunking for legal documents.

Legal text has structure that generic fixed-size chunking destroys: numbered clauses,
sub-clauses, and headings. This chunker

1. rebuilds paragraphs from PDF lines, starting a new paragraph at clause markers
   ("12.", "4.2", "(a)", "Section 9") and headings;
2. tracks the heading path ("12. Limitation of Liability › 12.2 Exclusions") so every
   chunk knows which section it belongs to;
3. packs paragraphs into chunks that end at clause boundaries where possible, splitting
   only over-long clauses at sentence boundaries (with a one-sentence overlap);
4. records exact page numbers, even for paragraphs that run across a page break.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import Chunk

MAX_CHARS = 1800  # ~400 tokens: comfortably inside the 512-token embedding window
MIN_CHARS = 300   # don't start a new chunk at a heading if the current one is a tiny stub
OVERLAP_SENTENCE_CHARS = 300

_CLAUSE_START = re.compile(
    r"""^(?:
        (?:section|sec\.|article|art\.|clause|schedule|exhibit|annex|appendix|part)\s+[0-9IVXLC]+[A-Z]?\b
      | \d{1,3}(?:\.\d{1,3}){1,4}\.?(?=\s)           # 1.1   2.3.4
      | \d{1,3}[.)](?=\s+\S)                         # 1.    2)
      | \((?:[a-z]{1,2}|[ivxlc]{1,5}|\d{1,2})\)(?=\s)  # (a)   (iv)   (3)
      | [a-z][.)](?=\s)                              # a.    b)
      | [•▪●◦‣∙·*\-–—](?=\s)                         # bullets
    )""",
    re.X | re.I,
)
_HEADING_NUMBERED = re.compile(
    r"^(?:(?:section|article|clause|part)\s+)?(\d{1,3}(?:\.\d{1,3})*|[IVXLC]{1,6})[.):]?\s+(.+)$", re.I
)
_INLINE_TITLE = re.compile(
    r"^((?:section\s+)?\d{1,3}(?:\.\d{1,3})*\.?)\s+([A-Z][A-Za-z ,&/'’\-]{2,60}?)\.\s+(?=[A-Z(\"“])"
)
_SMALL_WORDS = {"of", "and", "or", "the", "a", "an", "to", "in", "on", "for", "by", "with", "at", "from", "as", "vs", "v"}
_TERMINAL = (".", ":", ";", "?", "!")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?;])\s+(?=[\"“(]?[A-Z0-9])")
_ABBREVIATIONS = {
    "inc", "ltd", "co", "corp", "llc", "llp", "no", "nos", "sec", "secs", "art", "arts", "u.s", "u.k", "e.g",
    "i.e", "etc", "vs", "v", "mr", "mrs", "ms", "dr", "st", "para", "cl", "approx", "fig", "jan", "feb", "mar",
    "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}


def _is_all_caps(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return len(letters) >= 4 and sum(c.isupper() for c in letters) / len(letters) > 0.9


def _is_title_case(text: str) -> bool:
    words = re.findall(r"[A-Za-z][A-Za-z'’\-]*", text)
    if not words or len(words) > 12:
        return False
    significant = [w for w in words if w.lower() not in _SMALL_WORDS]
    if not significant:
        return False
    return sum(w[0].isupper() for w in significant) / len(significant) >= 0.8


def _level(number: str) -> int:
    return number.count(".") + 1 if number[0].isdigit() else 1


def detect_heading(line: str, at_boundary: bool = True) -> tuple[int, str] | None:
    """Return ``(level, title)`` if the line is a stand-alone heading.

    ``at_boundary`` says whether the previous line finished a sentence. Unnumbered ALL-CAPS
    lines only count as headings there; otherwise they are usually part of an ALL-CAPS
    disclaimer paragraph that wraps across lines.
    """
    s = line.strip()
    if not s or len(s) > 100 or s.endswith((",", ";")):
        return None
    m = _HEADING_NUMBERED.match(s)
    if m:
        rest = m.group(2).strip()
        if re.search(r"[.:]\s+\S", rest):
            return None  # "7. Fees. You must pay…": a titled clause, see detect_inline_title
        rest = rest.rstrip(".:")
        if rest and len(rest) <= 80 and (_is_all_caps(rest) or _is_title_case(rest)):
            return _level(m.group(1)), s.rstrip(".:")
        return None
    if at_boundary and _is_all_caps(s) and len(s.split()) <= 12 and not s.endswith(".") and s.count(",") <= 1:
        return 1, s
    return None


def detect_inline_title(text: str) -> tuple[int, str] | None:
    """'7. Limitation of Liability. In no event…' -> (1, '7. Limitation of Liability')."""
    m = _INLINE_TITLE.match(text)
    if m and _is_title_case(m.group(2)):
        label = m.group(1).rstrip(".")
        number = label.split()[-1]
        separator = " " if "." in number else ". "
        return _level(number), f"{label}{separator}{m.group(2)}"
    return None


@dataclass
class _Para:
    text: str = ""
    # (char offset in text, page number) at every point where a new page begins
    page_marks: list[tuple[int, int]] = field(default_factory=list)
    heading: tuple[int, str] | None = None

    def add_line(self, line: str, page: int) -> None:
        if not self.page_marks or self.page_marks[-1][1] != page:
            self.page_marks.append((len(self.text) + (1 if self.text else 0), page))
        self.text = f"{self.text} {line}" if self.text else line

    def pages_between(self, start: int, end: int) -> tuple[int, int]:
        first = last = self.page_marks[0][1]
        for offset, page in self.page_marks:
            if offset <= start:
                first = page
            if offset < end:
                last = page
        return first, max(first, last)


def _paragraphs(pages: list[str]) -> list[_Para]:
    paras: list[_Para] = []
    current = _Para()

    def flush() -> None:
        nonlocal current
        if current.text:
            paras.append(current)
        current = _Para()

    for page_no, page_text in enumerate(pages, start=1):
        for line in (ln.strip() for ln in page_text.split("\n")):
            if not line:
                flush()
                continue
            at_boundary = not current.text or current.text.endswith(_TERMINAL)
            heading = detect_heading(line, at_boundary=at_boundary)
            if heading:
                flush()
                para = _Para(heading=heading)
                para.add_line(line, page_no)
                paras.append(para)
                continue
            new_paragraph = bool(_CLAUSE_START.match(line)) or (
                current.text.endswith(_TERMINAL) and (line[0].isupper() or line[0] in "\"“(")
            )
            if new_paragraph:
                flush()
            current.add_line(line, page_no)
        # A paragraph that ends mid-sentence at the bottom of a page continues on the next.
        if current.text.endswith(_TERMINAL):
            flush()
    flush()
    return paras


def split_sentences(text: str) -> list[str]:
    merged: list[str] = []
    for part in _SENTENCE_BOUNDARY.split(text):
        if merged and _ends_with_abbreviation(merged[-1]):
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return [s for s in merged if s.strip()]


def _ends_with_abbreviation(sentence: str) -> bool:
    m = re.search(r"([A-Za-z][A-Za-z.]*)\.$", sentence)
    if not m:
        return False
    token = m.group(1).lower()
    return len(token) == 1 or token in _ABBREVIATIONS


@dataclass
class _Piece:
    text: str
    page_start: int
    page_end: int
    sections: tuple[str, ...] = ()  # heading path in effect for this piece
    is_heading: bool = False


def _split_long(para: _Para, max_chars: int) -> list[_Piece]:
    """Split an over-long paragraph into sentence windows with a one-sentence overlap."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for sentence in split_sentences(para.text):
        start = para.text.find(sentence, cursor)
        if start < 0:
            start = cursor
        end = start + len(sentence)
        # Hard-wrap sentences that are longer than a whole chunk.
        while end - start > max_chars:
            cut = para.text.rfind(" ", start, start + max_chars)
            cut = cut if cut > start else start + max_chars
            spans.append((start, cut))
            start = cut + 1
        spans.append((start, end))
        cursor = end

    pieces: list[_Piece] = []
    window: list[tuple[int, int]] = []
    for span in spans:
        if window and span[1] - window[0][0] > max_chars:
            pieces.append(_make_piece(para, window))
            last = window[-1]
            keep = last[1] - last[0] <= OVERLAP_SENTENCE_CHARS and span[1] - last[0] <= max_chars
            window = [last] if keep else []
        window.append(span)
    if window:
        pieces.append(_make_piece(para, window))
    return pieces


def _make_piece(para: _Para, window: list[tuple[int, int]]) -> _Piece:
    start, end = window[0][0], window[-1][1]
    first, last = para.pages_between(start, end)
    return _Piece(para.text[start:end].strip(), first, last)


def _common_section(pieces: list[_Piece]) -> str | None:
    """Label a chunk with the deepest heading path shared by all of its pieces."""
    paths = [p.sections for p in pieces if p.sections]
    if not paths:
        return None
    shared: list[str] = []
    for titles in zip(*paths, strict=False):  # stops at the shortest path: a common prefix
        if any(t != titles[0] for t in titles):
            break
        shared.append(titles[0])
    label = " › ".join(shared or paths[0])
    return label[:160]


def chunk_document(pages: list[str], max_chars: int = MAX_CHARS, min_chars: int = MIN_CHARS) -> list[Chunk]:
    chunks: list[Chunk] = []
    stack: list[tuple[int, str]] = []
    buffer: list[_Piece] = []

    def emit() -> None:
        nonlocal buffer
        if buffer:
            chunks.append(Chunk(
                id=len(chunks),
                text="\n".join(p.text for p in buffer),
                page_start=min(p.page_start for p in buffer),
                page_end=max(p.page_end for p in buffer),
                section=_common_section(buffer),
            ))
        buffer = []

    def has_body() -> bool:
        return any(not p.is_heading for p in buffer)

    for para in _paragraphs(pages):
        new_section = para.heading or detect_inline_title(para.text)
        if new_section:
            level, title = new_section
            # Top-level sections always start a new chunk; small sub-clauses may share one.
            if has_body() and (level == 1 or sum(len(p.text) + 1 for p in buffer) >= min_chars):
                emit()
            stack[:] = [entry for entry in stack if entry[0] < level] + [(level, title)]

        sections = tuple(title for _, title in stack)
        if len(para.text) > max_chars:
            pieces = _split_long(para, max_chars)
        else:
            pieces = [_Piece(para.text, *para.pages_between(0, len(para.text)))]
        for piece in pieces:
            piece.sections = sections
            piece.is_heading = para.heading is not None
            if buffer and sum(len(p.text) + 1 for p in buffer) + len(piece.text) > max_chars:
                emit()
            buffer.append(piece)
    emit()
    return chunks
