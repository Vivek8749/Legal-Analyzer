"""Question answering over a document: hybrid retrieval + a cited answer from the model.

Each question retrieves the most relevant clauses; the provider passes them to the model
and returns the answer split into segments with the chunks each one cites (Claude cites
natively; other models cite inline as [C12]). Conversations are multi-turn.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import Settings
from .llm import LLM, ChatReply, ChatTurn, Usage
from .store import DocumentRecord


@dataclass
class Source:
    number: int
    chunk_id: int
    pages: list[int]
    location: str
    section: str | None
    cited_text: list[str] = field(default_factory=list)


@dataclass
class Answer:
    text: str  # answer with [n] citation markers
    segments: list[dict[str, Any]]  # [{"text": ..., "sources": [n, ...]}]
    sources: list[Source]
    retrieved_chunk_ids: list[int]
    model: str


class ChatSession:
    def __init__(self, record: DocumentRecord, llm: LLM, settings: Settings, k: int | None = None):
        self.record = record
        self.llm = llm
        self.settings = settings
        # Small local context windows get fewer, but still the most relevant, excerpts.
        self.k = k or (8 if llm.context_window >= 64_000 else 5)
        self.turns: list[ChatTurn] = []
        self.usage = Usage()
        self._chunks = {c.id: c for c in record.chunks}
        self._lock = threading.Lock()

    def ask(self, question: str, on_text: Callable[[str], None] | None = None) -> Answer:
        question = question.strip()
        if not question:
            raise ValueError("The question is empty.")
        with self._lock:
            # Follow-ups ("what about refunds?") often lean on the previous question for context.
            queries, weights = [question], [1.0]
            if self.turns:
                queries.append(self.turns[-1].question)
                weights.append(0.5)
            hits = self.record.index.search(queries, k=self.k, weights=weights)
            shown = {chunk_id for turn in self.turns for chunk_id in turn.new_chunk_ids}
            turn = ChatTurn(
                question=question,
                chunks=[h.chunk for h in hits],
                new_chunk_ids=[h.chunk.id for h in hits if h.chunk.id not in shown],
            )
            reply = self.llm.chat(
                document_name=self.record.info.filename,
                turns=self.turns + [turn],
                effort=self.settings.chat_effort,
                on_text=on_text,
            )
            # Record the turn only once the request succeeded, keeping the history valid.
            turn.reply = reply.raw
            self.turns.append(turn)
            self.usage.add(reply.usage)
            return self._answer(reply, [h.chunk.id for h in hits])

    def _answer(self, reply: ChatReply, retrieved: list[int]) -> Answer:
        numbers: dict[int, int] = {}
        sources: list[Source] = []
        segments: list[dict[str, Any]] = []
        for text, chunk_ids in reply.segments:
            refs: list[int] = []
            for chunk_id in chunk_ids:
                chunk = self._chunks.get(chunk_id)
                if chunk is None:
                    continue
                if chunk_id not in numbers:
                    numbers[chunk_id] = len(numbers) + 1
                    sources.append(Source(numbers[chunk_id], chunk_id, chunk.pages, chunk.location, chunk.section,
                                          list(reply.cited_text.get(chunk_id, []))))
                if numbers[chunk_id] not in refs:
                    refs.append(numbers[chunk_id])
            segments.append({"text": text, "sources": refs})
        text = "".join(s["text"] + "".join(f"[{n}]" for n in s["sources"]) for s in segments)
        return Answer(text=text, segments=segments, sources=sources, retrieved_chunk_ids=retrieved, model=reply.model)
