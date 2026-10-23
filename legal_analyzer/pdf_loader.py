"""Extract clean, page-numbered text from PDF (or plain-text) documents."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .models import format_pages

SUPPORTED_SUFFIXES = (".pdf", ".txt", ".md")


class DocumentError(Exception):
    """The document could not be read or contains no usable text."""


@dataclass
class LoadedDocument:
    pages: list[str]  # cleaned text; pages[0] is page 1
    title: str | None = None
    empty_pages: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return sum(len(p) for p in self.pages)


def load_document(path: Path) -> LoadedDocument:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        raw_pages, title = _read_pdf(path)
    elif suffix in (".txt", ".md"):
        raw_pages, title = _read_text(path), None
    else:
        raise DocumentError(f"Unsupported file type {suffix!r}; use one of {', '.join(SUPPORTED_SUFFIXES)}.")

    pages = clean_pages(raw_pages)
    empty = [i + 1 for i, text in enumerate(pages) if len(text.strip()) < 20]
    if not pages or len(empty) == len(pages):
        raise DocumentError(
            "No extractable text found. This usually means the PDF is a scanned image. "
            "Run OCR first (for example `ocrmypdf input.pdf output.pdf`) and try again."
        )
    warnings = []
    if empty:
        warnings.append(
            f"{len(empty)} of {len(pages)} pages had no extractable text and were skipped "
            f"({format_pages(empty)}); they may be scanned images."
        )
    return LoadedDocument(pages=pages, title=title, empty_pages=empty, warnings=warnings)


def _read_pdf(path: Path) -> tuple[list[str], str | None]:
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted and not reader.decrypt(""):
            raise DocumentError("The PDF is password-protected. Remove the password and try again.")
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:  # a single malformed page should not sink the whole document
                pages.append("")
        title = None
        if reader.metadata and reader.metadata.title:
            title = str(reader.metadata.title).strip() or None
        return pages, title
    except DocumentError:
        raise
    except Exception as exc:
        raise DocumentError(f"Could not read PDF: {exc}") from exc


def _read_text(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    # Form feeds mark page breaks in text exports; otherwise treat the file as one page.
    return text.split("\f") if "\f" in text else [text]


# --- Cleaning -------------------------------------------------------------------------

_INVISIBLE = dict.fromkeys(map(ord, "­​‌‍﻿"), None)
_PAGE_NUMBER_LINE = re.compile(r"^\W*(?:page\s*)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?\W*$", re.I)
_SPACES = re.compile(r"[ \t  -   　]+")


def _normalize_line(line: str) -> str:
    return re.sub(r"\d+", "#", " ".join(line.lower().split()))


def clean_pages(raw_pages: list[str]) -> list[str]:
    """Normalise unicode, drop repeated headers/footers and page numbers, repair hyphenation."""
    split_pages: list[list[str]] = []
    for raw in raw_pages:
        text = unicodedata.normalize("NFKC", raw).translate(_INVISIBLE)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = [_SPACES.sub(" ", line).strip() for line in text.split("\n")]
        split_pages.append(lines)

    repeated = _repeated_margin_lines(split_pages)
    cleaned = []
    for lines in split_pages:
        drop = {i for slot, i in _margin_slots(lines) if (slot, _normalize_line(lines[i])) in repeated}
        kept = [line for i, line in enumerate(lines) if i not in drop and not _PAGE_NUMBER_LINE.match(line)]
        cleaned.append(_dehyphenate("\n".join(kept)).strip())
    return cleaned


def _margin_slots(lines: list[str]) -> list[tuple[str, int]]:
    """(slot, line index) for the first two and last two non-empty lines of a page."""
    non_empty = [i for i, line in enumerate(lines) if line]
    slots = [(f"top{k}", i) for k, i in enumerate(non_empty[:2])]
    slots += [(f"bottom{k}", i) for k, i in enumerate(reversed(non_empty[-2:]))]
    return slots


def _repeated_margin_lines(pages: list[list[str]]) -> set[tuple[str, str]]:
    """Running headers/footers: the same line in the same margin position on most pages."""
    if len(pages) < 3:
        return set()
    counts: Counter[tuple[str, str]] = Counter()
    for lines in pages:
        counts.update({(slot, _normalize_line(lines[i])) for slot, i in _margin_slots(lines)})
    threshold = max(3, int(len(pages) * 0.5))
    return {key for key, n in counts.items() if n >= threshold and 0 < len(key[1]) < 120}


_HYPHEN_BREAK = re.compile(r"([A-Za-z]{2,})-\n([a-z]{2,})[ \t]*")
# Real compound words ("non-refundable", "third-party") keep their hyphen when rejoined.
_COMPOUND_PREFIXES = {
    "non", "co", "self", "pre", "post", "re", "anti", "multi", "sub", "inter", "cross", "third",
    "well", "long", "short", "full", "part", "first", "one", "two", "opt", "sign", "follow", "set",
}


def _dehyphenate(text: str) -> str:
    # "termi-\nnation" -> "termination\n" (the line break moves after the rejoined word)
    def join(m: re.Match[str]) -> str:
        head, tail = m.group(1), m.group(2)
        hyphen = "-" if head.lower() in _COMPOUND_PREFIXES else ""
        return f"{head}{hyphen}{tail}\n"

    return _HYPHEN_BREAK.sub(join, text)
