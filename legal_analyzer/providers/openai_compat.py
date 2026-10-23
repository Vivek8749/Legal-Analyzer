"""OpenAI and OpenAI-compatible chat APIs.

One adapter covers OpenAI itself, Google Gemini (through its OpenAI-compatible endpoint),
and any server that implements the Chat Completions API: OpenRouter, Groq, DeepSeek,
Together, LM Studio, vLLM, and so on (set LEGAL_ANALYZER_OPENAI_BASE_URL).

Structured answers use ``response_format`` with a strict JSON schema. Retrieved clauses are
cited inline as [C12] and parsed back into citations.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Callable, Iterator

import openai
from pydantic import ValidationError

from ..config import GEMINI_BASE_URL, PROVIDER_LABELS, Settings
from ..llm import (
    ChatReply,
    ChatTurn,
    FatalLLMError,
    LLMError,
    MissingCredentialsError,
    ModelNotFoundError,
    PromptBlock,
    ProviderUnavailableError,
    RefusalError,
    StructuredResult,
    T,
    TruncatedOutputError,
    Usage,
)
from ..prompts import chat_system_prompt
from .common import ReasoningFilter, chat_messages, extract_json, inline_reply, strict_json_schema

CHARS_PER_TOKEN = 3.2


def _usage(usage: Any) -> Usage:
    if usage is None:
        return Usage(requests=1)
    details = getattr(usage, "prompt_tokens_details", None)
    cached = (getattr(details, "cached_tokens", None) or 0) if details else 0
    prompt = getattr(usage, "prompt_tokens", 0) or 0
    return Usage(
        requests=1,
        input_tokens=max(0, prompt - cached),
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        cache_read_input_tokens=cached,
    )


class OpenAICompatLLM:
    local = False

    def __init__(self, settings: Settings, model: str | None = None, provider: str = "openai",
                 client: openai.OpenAI | None = None):
        self.settings = settings
        self.provider = provider
        self.model = model or settings.model_for(provider)
        if provider == "gemini":
            self.base_url: str | None = GEMINI_BASE_URL
            self._key_vars = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
        else:
            self.base_url = settings.openai_base_url
            self._key_vars = ("OPENAI_API_KEY",)
        self.label = f"{PROVIDER_LABELS[provider]} · {self.model}"
        self.context_window = settings.context_window or (1_000_000 if provider == "gemini" else 128_000)
        self.max_output_tokens = 16_000
        self.max_concurrency = settings.concurrency
        self._client = client

    def _api_key(self) -> str | None:
        return next((os.environ[var] for var in self._key_vars if os.environ.get(var)), None)

    def status(self) -> str | None:
        if self._client is not None or self._api_key():
            return None
        if self.base_url and self.provider == "openai":
            return None  # custom endpoints (LM Studio, vLLM, …) often need no key
        return f"Set {self._key_vars[0]} to use {PROVIDER_LABELS[self.provider]} models."

    def client(self) -> openai.OpenAI:
        if self._client is None:
            key = self._api_key()
            if not key and not (self.base_url and self.provider == "openai"):
                raise MissingCredentialsError(f"Set {self._key_vars[0]} to use {PROVIDER_LABELS[self.provider]} models.")
            self._client = openai.OpenAI(
                api_key=key or "not-needed",
                base_url=self.base_url,
                max_retries=self.settings.max_retries,
                timeout=600.0,
            )
        return self._client

    @contextmanager
    def _translate_errors(self) -> Iterator[None]:
        name = PROVIDER_LABELS[self.provider]
        try:
            yield
        except openai.AuthenticationError as exc:
            raise FatalLLMError(f"The {name} API key was rejected. Check {self._key_vars[0]}.") from exc
        except openai.PermissionDeniedError as exc:
            raise FatalLLMError(f"This {name} API key does not have access to {self.model}.") from exc
        except openai.NotFoundError as exc:
            raise ModelNotFoundError(
                f"Model {self.model!r} was not found. Set LEGAL_ANALYZER_{self.provider.upper()}_MODEL to a model you can use."
            ) from exc
        except openai.BadRequestError as exc:
            raise FatalLLMError(f"{name} rejected the request: {exc.message}") from exc
        except openai.RateLimitError as exc:
            raise LLMError(f"Rate limited by {name}. Wait a minute and try again.") from exc
        except openai.APITimeoutError as exc:
            raise LLMError(f"The request to {name} timed out.") from exc
        except openai.APIConnectionError as exc:
            raise ProviderUnavailableError(f"Could not reach {self.base_url or 'the OpenAI API'}.") from exc
        except openai.APIStatusError as exc:
            raise LLMError(f"{name} API error {exc.status_code}: {exc.message}") from exc

    def _extra_params(self) -> dict[str, Any]:
        effort = self.settings.openai_reasoning_effort
        return {"reasoning_effort": effort} if effort else {}

    def structured(
        self,
        *,
        system: str,
        blocks: list[PromptBlock],
        output_type: type[T],
        effort: str,
        on_stream_start: Callable[[], None] | None = None,
    ) -> StructuredResult[T]:
        if on_stream_start is not None:
            on_stream_start()  # no explicit prompt cache to warm up first
        schema = {"name": output_type.__name__, "schema": strict_json_schema(output_type), "strict": True}
        client = self.client()
        with self._translate_errors():
            completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": "\n\n".join(b.text for b in blocks)},
                ],
                response_format={"type": "json_schema", "json_schema": schema},
                **self._extra_params(),
            )
        if not completion.choices:
            raise LLMError(f"{self.label} returned no answer.")
        choice = completion.choices[0]
        refusal = getattr(choice.message, "refusal", None)
        if refusal:
            raise RefusalError(f"The model declined this request: {refusal}")
        if choice.finish_reason == "content_filter":
            raise RefusalError("The provider's content filter blocked this answer.")
        if choice.finish_reason == "length":
            raise TruncatedOutputError("The model's answer was cut off by its output token limit.")
        try:
            output = output_type.model_validate_json(extract_json(choice.message.content or ""))
        except ValidationError as exc:
            raise LLMError(f"{self.label} returned output that did not match the expected format.") from exc
        return StructuredResult(output, _usage(completion.usage), completion.model or self.model)

    def chat(
        self,
        *,
        document_name: str,
        turns: list[ChatTurn],
        effort: str,
        on_text: Callable[[str], None] | None = None,
    ) -> ChatReply:
        history_chars = int(self.context_window * CHARS_PER_TOKEN * 0.25)
        messages = chat_messages(chat_system_prompt(document_name, inline_citations=True), turns, history_chars)
        params: dict[str, Any] = {"model": self.model, "messages": messages, "stream": True, **self._extra_params()}
        if self.base_url is None:
            params["stream_options"] = {"include_usage": True}  # not every compatible server accepts this

        reasoning = ReasoningFilter()
        parts: list[str] = []
        usage = Usage(requests=1)
        finish_reason = refusal = None

        def emit(text: str) -> None:
            if text:
                parts.append(text)
                if on_text is not None:
                    on_text(text)

        client = self.client()
        with self._translate_errors():
            for chunk in client.chat.completions.create(**params):
                if getattr(chunk, "usage", None):
                    usage = _usage(chunk.usage)
                for choice in chunk.choices or []:
                    delta = choice.delta
                    if delta is not None and delta.content:
                        emit(reasoning.feed(delta.content))
                    if delta is not None and getattr(delta, "refusal", None):
                        refusal = (refusal or "") + delta.refusal
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason
        emit(reasoning.flush())

        answer = "".join(parts).strip()
        if not answer and (refusal or finish_reason == "content_filter"):
            raise RefusalError(f"The model declined to answer{': ' + refusal if refusal else '.'}")
        segments, cited = inline_reply(answer, turns)
        return ChatReply(segments, cited, raw=answer, usage=usage, model=self.model)
