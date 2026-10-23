"""Helpers shared by the providers that speak OpenAI-style chat (OpenAI-compatible APIs, Ollama)."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

from ..citations import cited_passages, parse_inline_citations
from ..llm import ChatTurn
from ..models import Chunk
from ..prompts import render_chat_question


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """A self-contained JSON schema suitable for constrained decoding.

    ``$ref``s are inlined (not every server resolves them), every object is closed
    (``additionalProperties: false``) with all properties required, as OpenAI's strict mode
    requires, and ``title``/``default`` keywords are dropped.
    """
    schema = model.model_json_schema()
    definitions = schema.get("$defs", {})

    def clean(node: Any) -> Any:
        if isinstance(node, list):
            return [clean(item) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            target = clean(definitions[node["$ref"].rsplit("/", 1)[-1]])
            extra = {k: clean(v) for k, v in node.items() if k not in ("$ref", "title", "default")}
            return {**target, **extra}
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key in ("title", "default", "$defs"):
                continue
            # property names (one of ours is literally "title") are kept as they are
            out[key] = {name: clean(sub) for name, sub in value.items()} if key == "properties" else clean(value)
        if out.get("type") == "object":
            out["additionalProperties"] = False
            out["required"] = list(out.get("properties", {}))
        return out

    return clean(schema)


_THINK_BLOCK = re.compile(r"<think>.*?(?:</think>|\Z)", re.S)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def extract_json(text: str) -> str:
    """The JSON object in a model's answer, without reasoning tags or Markdown fences."""
    text = _FENCE.sub("", _THINK_BLOCK.sub("", text).strip()).strip()
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            text = text[start:end + 1]
    return text


class ReasoningFilter:
    """Hide ``<think>…</think>`` sections in streamed text. Some local reasoning models write
    their reasoning inline; tags may be split across stream chunks."""

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self) -> None:
        self._buffer = ""
        self._inside = False

    def feed(self, text: str) -> str:
        self._buffer += text
        visible: list[str] = []
        while True:
            tag = self.CLOSE if self._inside else self.OPEN
            position = self._buffer.find(tag)
            if position < 0:
                keep = self._partial_tag(tag)
                if not self._inside:
                    visible.append(self._buffer[:len(self._buffer) - keep])
                self._buffer = self._buffer[len(self._buffer) - keep:]
                return "".join(visible)
            if not self._inside:
                visible.append(self._buffer[:position])
            self._buffer = self._buffer[position + len(tag):]
            self._inside = not self._inside

    def flush(self) -> str:
        rest = "" if self._inside else self._buffer
        self._buffer = ""
        return rest

    def _partial_tag(self, tag: str) -> int:
        """Length of a suffix of the buffer that could be the start of ``tag``."""
        for size in range(min(len(tag) - 1, len(self._buffer)), 0, -1):
            if self._buffer.endswith(tag[:size]):
                return size
        return 0


def chat_messages(system: str, turns: list[ChatTurn], history_chars: int) -> list[dict[str, str]]:
    """OpenAI-style messages: earlier turns as plain Q&A (newest first, within ``history_chars``),
    then the current question with its retrieved excerpts."""
    history: list[dict[str, str]] = []
    used = 0
    for turn in reversed(turns[:-1]):
        answer = turn.reply if isinstance(turn.reply, str) else ""
        size = len(turn.question) + len(answer)
        if used + size > history_chars:
            break
        history[:0] = [{"role": "user", "content": turn.question}, {"role": "assistant", "content": answer}]
        used += size
    current = turns[-1]
    return [
        {"role": "system", "content": system},
        *history,
        {"role": "user", "content": render_chat_question(current.question, current.chunks)},
    ]


def inline_reply(answer: str, turns: list[ChatTurn]) -> tuple[list[tuple[str, list[int]]], dict[int, list[str]]]:
    """Segments and cited passages for an answer that cites excerpts inline as [C12]."""
    chunks: dict[int, Chunk] = {c.id: c for turn in turns for c in turn.chunks}
    segments = parse_inline_citations(answer, set(chunks))
    return segments, cited_passages(segments, chunks)
