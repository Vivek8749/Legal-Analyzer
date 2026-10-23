"""Provider-neutral interface to the language models the analyzer can use.

Two kinds of calls are made:

* ``structured`` – an answer that must match a Pydantic schema, used for the review passes
  and the summary. Every provider constrains decoding to the JSON schema.
* ``chat`` – a streamed, cited answer to a question about the document.

Providers live in ``legal_analyzer.providers``: Claude (Anthropic API), OpenAI-compatible
APIs (OpenAI, Gemini, OpenRouter, Groq, LM Studio, …), and local models through Ollama.
They translate provider errors into the exceptions below, so callers never need to know
which provider is in use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Protocol, TypeVar

from pydantic import BaseModel

from .config import PRICING, Settings
from .models import Chunk, UsageSummary

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """A model request failed; the analysis reports it and carries on with the other passes."""


class FatalLLMError(LLMError):
    """Every request would fail the same way (bad key, model missing, server down): stop."""


class MissingCredentialsError(FatalLLMError):
    pass


class ProviderUnavailableError(FatalLLMError):
    pass


class ModelNotFoundError(FatalLLMError):
    pass


class RefusalError(LLMError):
    pass


class TruncatedOutputError(LLMError):
    pass


@dataclass
class Usage:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def add(self, other: Usage) -> None:
        self.requests += other.requests
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_creation_input_tokens += other.cache_creation_input_tokens
        self.cache_read_input_tokens += other.cache_read_input_tokens

    def summary(self, model: str, local: bool = False) -> UsageSummary:
        price = PRICING.get(model)
        cost: float | None = 0.0 if local else None
        if price and not local:
            cost = round((
                self.input_tokens * price["input"]
                + self.output_tokens * price["output"]
                + self.cache_creation_input_tokens * price["cache_write"]
                + self.cache_read_input_tokens * price["cache_read"]
            ) / 1_000_000, 4)
        return UsageSummary(
            requests=self.requests,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens,
            estimated_cost_usd=cost,
        )


@dataclass(frozen=True)
class PromptBlock:
    """A piece of the user prompt. ``cacheable`` marks the end of a prefix shared across
    requests, where providers with explicit prompt caching place a cache breakpoint."""

    text: str
    cacheable: bool = False


@dataclass
class StructuredResult(Generic[T]):
    output: T
    usage: Usage
    model: str


@dataclass
class ChatTurn:
    question: str
    chunks: list[Chunk]  # excerpts retrieved for this question, best first
    new_chunk_ids: list[int]  # the ones not shown earlier in the conversation
    reply: Any = None  # provider-specific record of the answer, used to replay history


@dataclass
class ChatReply:
    segments: list[tuple[str, list[int]]]  # answer text, split where citations attach (chunk ids)
    cited_text: dict[int, list[str]] = field(default_factory=dict)  # chunk id -> cited passages
    raw: Any = None  # stored on ChatTurn.reply
    usage: Usage = field(default_factory=Usage)
    model: str = ""


class LLM(Protocol):
    provider: str
    model: str
    label: str
    local: bool
    context_window: int  # tokens the model can attend to (prompt + output)
    max_output_tokens: int
    max_concurrency: int

    def status(self) -> str | None:
        """None when ready to use, otherwise a message saying what to fix."""

    def structured(
        self,
        *,
        system: str,
        blocks: list[PromptBlock],
        output_type: type[T],
        effort: str,
        on_stream_start: Callable[[], None] | None = None,
    ) -> StructuredResult[T]: ...

    def chat(
        self,
        *,
        document_name: str,
        turns: list[ChatTurn],
        effort: str,
        on_text: Callable[[str], None] | None = None,
    ) -> ChatReply: ...


def create_llm(settings: Settings, provider: str | None = None, model: str | None = None) -> LLM:
    """Build the client for ``provider`` (default: the configured one) and ``model``."""
    provider = (provider or settings.provider).strip().lower()
    model = (model or "").strip() or settings.model_for(provider)
    if provider == "anthropic":
        from .providers.claude import ClaudeLLM

        return ClaudeLLM(settings, model)
    if provider in ("openai", "gemini"):
        from .providers.openai_compat import OpenAICompatLLM

        return OpenAICompatLLM(settings, model, provider=provider)
    if provider == "ollama":
        from .providers.ollama_local import OllamaLLM

        return OllamaLLM(settings, model)
    raise ValueError(f"Unknown provider {provider!r}; use anthropic, openai, gemini, or ollama.")
