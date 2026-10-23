"""Local models through Ollama (https://ollama.com).

Everything runs on your machine: no API key, no per-token cost, and the document never
leaves it. Structured answers use Ollama's JSON-schema ``format`` (constrained decoding),
and retrieved clauses are cited inline as [C12].

Two settings matter most for local models:

* ``num_ctx`` – Ollama's default context window is small and longer prompts are silently
  truncated, so it is always set explicitly (LEGAL_ANALYZER_OLLAMA_NUM_CTX, default 16384,
  capped at what the model supports). The analyzer sizes its excerpts to fit.
* ``think`` – reasoning models (qwen3.5, deepseek-r1, gpt-oss, …) can think before
  answering. That improves quality but is much slower, so it is off unless you set
  LEGAL_ANALYZER_OLLAMA_THINK=true (or low/medium/high for models that take a level).
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Callable, Iterator

import httpx
import ollama
from pydantic import ValidationError

from ..config import Settings
from ..llm import (
    ChatReply,
    ChatTurn,
    LLMError,
    ModelNotFoundError,
    PromptBlock,
    ProviderUnavailableError,
    StructuredResult,
    T,
    TruncatedOutputError,
    Usage,
)
from ..prompts import chat_system_prompt
from .common import ReasoningFilter, chat_messages, extract_json, inline_reply, strict_json_schema

DEFAULT_HOST = "http://localhost:11434"
CHARS_PER_TOKEN = 3.2


def _usage(final: Any) -> Usage:
    if final is None:
        return Usage(requests=1)
    return Usage(requests=1, input_tokens=final.prompt_eval_count or 0, output_tokens=final.eval_count or 0)


def has_model(installed: set[str], model: str) -> bool:
    return model in installed or (":" not in model and f"{model}:latest" in installed)


class OllamaLLM:
    provider = "ollama"
    local = True

    def __init__(self, settings: Settings, model: str | None = None, client: ollama.Client | None = None):
        self.settings = settings
        self.model = model or settings.model_for("ollama")
        self.label = f"Ollama · {self.model}"
        self.host = settings.ollama_host or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST
        self.max_output_tokens = settings.ollama_num_predict
        self.max_concurrency = settings.ollama_parallel
        # No read timeout: a large prompt on a small GPU can take minutes before the first token.
        self.client = client or ollama.Client(host=self.host, timeout=httpx.Timeout(None, connect=5.0))
        self._info: tuple[set[str], int | None] | None = None

    # --- model information ------------------------------------------------------------

    @contextmanager
    def _translate_errors(self) -> Iterator[None]:
        try:
            yield
        except (ConnectionError, httpx.ConnectError) as exc:
            raise ProviderUnavailableError(
                f"Could not connect to Ollama at {self.host}. Install it from https://ollama.com and make sure it "
                "is running (`ollama serve`)."
            ) from exc
        except ollama.ResponseError as exc:
            if exc.status_code == 404:
                raise ModelNotFoundError(f"Model {self.model!r} isn't downloaded. Run: ollama pull {self.model}") from exc
            raise LLMError(f"Ollama error: {exc.error}") from exc
        except httpx.TimeoutException as exc:
            raise LLMError("The request to Ollama timed out.") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"The request to Ollama failed: {exc}") from exc

    def installed_models(self) -> list[str]:
        with self._translate_errors():
            listed = self.client.list()
        return sorted(m.model for m in listed.models if m.model)

    def _model_info(self) -> tuple[set[str], int | None]:
        """(capabilities, maximum context length) of the model, from ``ollama show``."""
        if self._info is None:
            with self._translate_errors():
                shown = self.client.show(self.model)
            capabilities = set(shown.capabilities or [])
            info = shown.modelinfo or {}
            longest = next((int(v) for k, v in info.items() if k.endswith(".context_length") and v), None)
            self._info = (capabilities, longest)
        return self._info

    @property
    def context_window(self) -> int:
        _, longest = self._model_info()
        return min(self.settings.ollama_num_ctx, longest) if longest else self.settings.ollama_num_ctx

    def status(self) -> str | None:
        try:
            installed = set(self.installed_models())
        except ProviderUnavailableError as exc:
            return str(exc)
        except LLMError as exc:
            return str(exc)
        if not has_model(installed, self.model):
            return f"Model {self.model!r} isn't downloaded. Run: ollama pull {self.model}"
        return None

    def _think(self) -> bool | str | None:
        capabilities, _ = self._model_info()
        if "thinking" not in capabilities:
            return None  # sending `think` to a model without the capability is an error
        setting = (self.settings.ollama_think or "false").strip().lower()
        if setting in ("false", "off", "no", "0"):
            return False
        if setting in ("true", "on", "yes", "1"):
            return True
        return setting  # "low" | "medium" | "high" for models that take a level

    def _stream(self, messages: list[dict[str, str]], *, temperature: float, schema: dict | None = None) -> Iterator[Any]:
        return self.client.chat(
            model=self.model,
            messages=messages,
            stream=True,
            format=schema,
            think=self._think(),
            keep_alive=self.settings.ollama_keep_alive,
            options={"num_ctx": self.context_window, "num_predict": self.max_output_tokens, "temperature": temperature},
        )

    # --- requests -----------------------------------------------------------------------

    def structured(
        self,
        *,
        system: str,
        blocks: list[PromptBlock],
        output_type: type[T],
        effort: str,
        on_stream_start: Callable[[], None] | None = None,
    ) -> StructuredResult[T]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(b.text for b in blocks)},
        ]
        parts: list[str] = []
        final = None
        with self._translate_errors():
            for part in self._stream(messages, temperature=0.2, schema=strict_json_schema(output_type)):
                if on_stream_start is not None:
                    on_stream_start()
                    on_stream_start = None
                if part.message is not None and part.message.content:
                    parts.append(part.message.content)
                if part.done:
                    final = part
        if final is not None and final.done_reason == "length":
            raise TruncatedOutputError(
                "The local model's answer was cut off. Raise LEGAL_ANALYZER_OLLAMA_NUM_PREDICT (and NUM_CTX if needed)."
            )
        try:
            output = output_type.model_validate_json(extract_json("".join(parts)))
        except ValidationError as exc:
            raise LLMError(f"{self.label} returned output that did not match the expected format.") from exc
        return StructuredResult(output, _usage(final), self.model)

    def chat(
        self,
        *,
        document_name: str,
        turns: list[ChatTurn],
        effort: str,
        on_text: Callable[[str], None] | None = None,
    ) -> ChatReply:
        history_chars = int(self.context_window * CHARS_PER_TOKEN * 0.2)
        messages = chat_messages(chat_system_prompt(document_name, inline_citations=True), turns, history_chars)
        reasoning = ReasoningFilter()
        parts: list[str] = []
        final = None

        def emit(text: str) -> None:
            if text:
                parts.append(text)
                if on_text is not None:
                    on_text(text)

        with self._translate_errors():
            for part in self._stream(messages, temperature=0.3):
                if part.message is not None and part.message.content:
                    emit(reasoning.feed(part.message.content))
                if part.done:
                    final = part
        emit(reasoning.flush())
        segments, cited = inline_reply("".join(parts).strip(), turns)
        return ChatReply(segments, cited, raw="".join(parts).strip(), usage=_usage(final), model=self.model)
