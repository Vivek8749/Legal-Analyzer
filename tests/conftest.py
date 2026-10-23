from __future__ import annotations

import re
import zlib
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from legal_analyzer.config import Settings
from legal_analyzer.llm import ChatReply, ChatTurn, StructuredResult, Usage
from legal_analyzer.schemas import LLMFinding, LLMGap, LLMParty, LLMTerm, PassOutput, SynthesisOutput
from legal_analyzer.store import DocumentStore
from legal_analyzer.taxonomy import GROUPS

SAMPLE_PDF = Path(__file__).resolve().parent.parent / "samples" / "sample_terms_of_service.pdf"


class HashingEmbedder:
    """Deterministic stand-in for the real embedding model (bag of hashed words)."""

    name = "test-hashing-embedder"

    def _embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), 256), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z]+", text.lower()):
                matrix[row, zlib.crc32(word.encode()) % 256] += 1.0
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.where(norms == 0, 1, norms)

    embed_documents = _embed
    embed_queries = _embed


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", embedding_backend="none", concurrency=3)


@pytest.fixture
def store(settings: Settings) -> DocumentStore:
    return DocumentStore(settings)


@pytest.fixture
def sample_record(store: DocumentStore):
    return store.ingest(SAMPLE_PDF)


def make_finding(**overrides: Any) -> LLMFinding:
    values = dict(
        category_id="limitation_of_liability",
        title="Liability capped at $10",
        chunk_ids=[11],
        quote="WILL NOT EXCEED THE GREATER OF TEN U.S. DOLLARS (US $10)",
        issue_type="one_sided_term",
        severity="critical",
        favors="drafter",
        plain_english="If Acme causes you a loss, the most you can recover is about $10.",
        why_it_matters="Losing your stored files could cost far more than $10.",
        loophole_scenario="Acme loses a year of your files and offers you $10.",
        what_to_do="Ask for a higher cap or keep independent backups.",
        confidence="high",
    )
    values.update(overrides)
    return LLMFinding(**values)


def default_pass_outputs() -> dict[str, PassOutput]:
    """Canned per-pass results, keyed by group id, quoting the real sample document."""
    empty = PassOutput(findings=[], key_terms=[], gaps=[])
    outputs = {group.id: empty for group in GROUPS}
    outputs["liability"] = PassOutput(
        findings=[
            make_finding(),
            # duplicate of the first (same category, near-identical quote): must be merged
            make_finding(severity="high", quote="WILL NOT EXCEED THE GREATER OF TEN U.S. DOLLARS (US $10) OR THE FEES"),
            make_finding(
                category_id="indemnification", title="You pay even for Acme's negligence", chunk_ids=[3],  # wrong chunk id
                quote="including claims caused in part by Acme's negligence", severity="high",
            ),
        ],
        key_terms=[LLMTerm(term="Indemnify", definition="Promise to pay someone else's losses.", chunk_ids=[12])],
        gaps=[],
    )
    outputs["disputes"] = PassOutput(
        findings=[make_finding(
            category_id="arbitration_class_waiver", title="Seven days to opt out of arbitration", chunk_ids=[13],
            quote="within seven (7) days after you first accept these Terms", severity="high",
        )],
        key_terms=[LLMTerm(term="indemnify", definition="Duplicate entry, should merge.", chunk_ids=[13])],
        gaps=[LLMGap(category_id="governing_law_venue", observation="No clause found that lets you sue locally.")],
    )
    outputs["drafting"] = PassOutput(
        findings=[make_finding(
            category_id="vague_discretion", title="Fabricated quote", chunk_ids=[2],
            quote="Acme shall pay a penalty of one million dollars to every user", severity="medium",
        )],
        key_terms=[],
        gaps=[],
    )
    return outputs


def default_synthesis() -> SynthesisOutput:
    return SynthesisOutput(
        document_type="Terms of service",
        parties=[LLMParty(name="Acme Cloud Storage, Inc.", role="Service provider"), LLMParty(name="You", role="User")],
        favoured_party="Acme Cloud Storage, Inc.",
        governing_law="Delaware",
        what_this_document_does="Sets the rules for using Acme's storage service.",
        headline="Acme keeps almost all the power under these terms.",
        overall_assessment="The terms heavily favour Acme.",
        top_concern_ids=["F1", "F99", "F1"],
        recommendations=["Keep your own backups."],
        questions_to_ask=["Will you raise the liability cap?"],
    )


class FakeLLM:
    """Implements the LLM provider interface without network calls."""

    provider = "fake"
    model = "fake-model"
    label = "Fake · fake-model"
    local = False
    max_output_tokens = 4_000
    max_concurrency = 8

    def __init__(self, pass_outputs: dict[str, PassOutput] | None = None,
                 synthesis: SynthesisOutput | None = None,
                 errors: dict[str, Exception] | None = None,
                 chat_reply: Callable[[list[ChatTurn]], str] | None = None,
                 context_window: int = 200_000):
        self.pass_outputs = pass_outputs if pass_outputs is not None else default_pass_outputs()
        self.synthesis = synthesis or default_synthesis()
        self.errors = errors or {}
        self.chat_reply = chat_reply
        self.context_window = context_window
        self.calls: list[dict[str, Any]] = []

    def status(self) -> str | None:
        return None

    def structured(self, *, system, blocks, output_type, effort, on_stream_start=None):
        if on_stream_start is not None:
            on_stream_start()
        usage = Usage(requests=1, input_tokens=100, output_tokens=50, cache_read_input_tokens=1000)
        if output_type is PassOutput:
            task = blocks[1].text
            group = next(g for g in GROUPS if task.startswith(f"Review pass: {g.name}."))
            self.calls.append({"kind": "pass", "group": group.id, "blocks": blocks, "system": system})
            if group.id in self.errors:
                raise self.errors[group.id]
            return StructuredResult(self.pass_outputs[group.id], usage, "fake-model")
        self.calls.append({"kind": "synthesis", "blocks": blocks})
        if "synthesis" in self.errors:
            raise self.errors["synthesis"]
        return StructuredResult(self.synthesis, usage, "fake-model")

    def chat(self, *, document_name, turns, effort, on_text=None):
        """Answers with ``chat_reply(turns)``, citing excerpts inline like the non-Claude providers."""
        from legal_analyzer.providers.common import inline_reply

        self.calls.append({"kind": "chat", "turns": list(turns)})
        answer = self.chat_reply(turns) if self.chat_reply else "No."
        if on_text:
            on_text(answer)
        segments, cited = inline_reply(answer, turns)
        return ChatReply(segments, cited, raw=answer, usage=Usage(requests=1, input_tokens=10, output_tokens=5),
                         model="fake-model")
