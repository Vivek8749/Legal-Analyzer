"""Claude through the Anthropic API.

Uses adaptive thinking with an explicit effort level, streams every request (long outputs
never hit HTTP timeouts), caches the shared prompt prefix across review passes, and cites
retrieved clauses natively through ``search_result`` content blocks. Requests to models
that support it opt into the server-side refusal fallback: if a safety classifier declines
a request, the API re-runs it on Anthropic's recommended fallback model instead of failing
(disable with LEGAL_ANALYZER_FALLBACKS=off).
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Any, Callable, Iterator

import anthropic
from pydantic import ValidationError

from ..config import Settings
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
from ..models import Chunk
from ..prompts import chat_system_prompt
from ..citations import citable_blocks

FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Models that accept the server-side ``fallbacks: "default"`` parameter.
FALLBACK_MODELS = {"claude-fable-5-1", "claude-fable-5", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
_SOURCE_ID = re.compile(r"/chunk/(\d+)$")


def _usage(message: Any) -> Usage:
    usage = message.usage
    return Usage(
        requests=1,
        input_tokens=usage.input_tokens or 0,
        output_tokens=usage.output_tokens or 0,
        cache_creation_input_tokens=usage.cache_creation_input_tokens or 0,
        cache_read_input_tokens=usage.cache_read_input_tokens or 0,
    )


def _refusal_message(message: Any) -> str:
    details = getattr(message, "stop_details", None)
    category = getattr(details, "category", None) if details else None
    suffix = f" (category: {category})" if category else ""
    return f"Claude declined this request{suffix}."


def chunk_label(chunk: Chunk) -> str:
    return f"[C{chunk.id}] {chunk.location}" + (f" — {chunk.section}" if chunk.section else "")


def _search_result(chunk: Chunk) -> dict[str, Any]:
    return {
        "type": "search_result",
        "source": f"doc://chunk/{chunk.id}",
        "title": chunk_label(chunk),
        # Claude cites whole content blocks, so small blocks give precise citations.
        "content": [{"type": "text", "text": block} for block in citable_blocks(chunk)],
        "citations": {"enabled": True},
    }


def _with_cache_breakpoints(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark the end of the newest user turn (cache write) and of the previous one (cache read).

    Cited answers are split into many text blocks, which can push the previous turn's cache
    entry out of the 20-block lookback window; an explicit marker on the previous user turn
    keeps the read position exact. Markers are applied to copies, never to stored history.
    """
    user_turns = [i for i, m in enumerate(messages) if m["role"] == "user"][-2:]
    marked = []
    for i, message in enumerate(messages):
        if i in user_turns:
            content = list(message["content"])
            content[-1] = {**content[-1], "cache_control": {"type": "ephemeral"}}
            message = {**message, "content": content}
        marked.append(message)
    return marked


def _replayable(content: list[Any]) -> list[Any]:
    """Prepare assistant content for the next request (append-only history).

    After a mid-answer refusal fallback, thinking and tool blocks that precede the last
    ``fallback`` marker belong to the declined attempt and must not be echoed back.
    """
    fallback_positions = [i for i, block in enumerate(content) if getattr(block, "type", None) == "fallback"]
    if not fallback_positions:
        return content
    boundary = fallback_positions[-1]
    dropped = {"thinking", "redacted_thinking", "tool_use"}
    return [b for i, b in enumerate(content) if i > boundary or getattr(b, "type", None) not in dropped]


class ClaudeLLM:
    provider = "anthropic"
    local = False
    context_window = 1_000_000
    max_output_tokens = 48_000

    def __init__(self, settings: Settings, model: str | None = None, client: anthropic.Anthropic | None = None):
        self.settings = settings
        self.model = model or settings.model_for("anthropic")
        self.label = f"Claude · {self.model}"
        self.max_concurrency = settings.concurrency
        self.client = client or anthropic.Anthropic(max_retries=settings.max_retries)

    def status(self) -> str | None:
        client = self.client
        if getattr(client, "api_key", None) or getattr(client, "auth_token", None) or getattr(client, "credentials", None):
            return None
        return "Set ANTHROPIC_API_KEY (for example in a .env file) to use Claude."

    def _base_params(self, effort: str) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": effort},
        }
        if self.settings.use_fallbacks and self.model in FALLBACK_MODELS:
            params["betas"] = [FALLBACK_BETA]
            params["fallbacks"] = "default"
        return params

    @contextmanager
    def _translate_errors(self) -> Iterator[None]:
        try:
            yield
        except TypeError as exc:
            # The SDK raises TypeError("Could not resolve authentication method…") without a key.
            if "authentication" in str(exc).lower():
                raise MissingCredentialsError(
                    "No Anthropic credentials found. Set ANTHROPIC_API_KEY (for example in a .env file) and try again."
                ) from exc
            raise
        except anthropic.AuthenticationError as exc:
            raise FatalLLMError("The Anthropic API key was rejected. Check ANTHROPIC_API_KEY.") from exc
        except anthropic.PermissionDeniedError as exc:
            raise FatalLLMError(f"This Anthropic API key does not have access to {self.model}.") from exc
        except anthropic.NotFoundError as exc:
            raise ModelNotFoundError(f"Model {self.model!r} was not found. Check LEGAL_ANALYZER_ANTHROPIC_MODEL.") from exc
        except anthropic.BadRequestError as exc:
            raise FatalLLMError(f"The Anthropic API rejected the request: {exc.message}") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("Rate limited by the Anthropic API. Wait a minute and try again.") from exc
        except anthropic.APITimeoutError as exc:
            raise LLMError("The request to the Anthropic API timed out.") from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderUnavailableError("Could not reach the Anthropic API. Check your network connection.") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc

    def structured(
        self,
        *,
        system: str,
        blocks: list[PromptBlock],
        output_type: type[T],
        effort: str,
        on_stream_start: Callable[[], None] | None = None,
    ) -> StructuredResult[T]:
        """Run a streamed request whose final answer is parsed into ``output_type``.

        ``on_stream_start`` fires on the first stream event. By then the prompt has been
        processed and written to the prompt cache, so parallel requests sharing the same
        prefix can be released and will read it instead of paying to write it again.
        """
        content = [
            {"type": "text", "text": b.text, **({"cache_control": {"type": "ephemeral"}} if b.cacheable else {})}
            for b in blocks
        ]
        try:
            with self._translate_errors(), self.client.beta.messages.stream(
                max_tokens=self.max_output_tokens,
                system=[{"type": "text", "text": system}],
                messages=[{"role": "user", "content": content}],
                output_format=output_type,
                **self._base_params(effort),
            ) as stream:
                for _event in stream:
                    if on_stream_start is not None:
                        on_stream_start()
                        on_stream_start = None
                message = stream.get_final_message()
        except ValidationError as exc:
            raise LLMError(f"Claude's answer was incomplete or did not match the expected format: {exc}") from exc

        if message.stop_reason == "refusal":
            raise RefusalError(_refusal_message(message))
        if message.stop_reason == "max_tokens":
            raise TruncatedOutputError("Claude's answer was cut off by the max_tokens limit.")
        parsed = message.parsed_output
        if parsed is None:
            raise LLMError("Claude returned no structured output.")
        return StructuredResult(parsed, _usage(message), message.model)

    def chat(
        self,
        *,
        document_name: str,
        turns: list[ChatTurn],
        effort: str,
        on_text: Callable[[str], None] | None = None,
    ) -> ChatReply:
        """Stream a cited answer. Excerpts are sent as search results; each conversation is
        append-only, and excerpts already shown earlier are referred to rather than resent."""
        messages: list[dict[str, Any]] = []
        result_chunk_ids: list[int] = []  # chunk behind each search_result block, in request order
        for turn in turns:
            new_ids = set(turn.new_chunk_ids)
            new = [c for c in turn.chunks if c.id in new_ids]
            earlier = [c for c in turn.chunks if c.id not in new_ids]
            content: list[dict[str, Any]] = [_search_result(c) for c in new]
            result_chunk_ids += [c.id for c in new]
            prompt = turn.question
            if earlier:
                prompt += "\n\n(Also relevant: excerpts shown earlier in this conversation: " + "; ".join(
                    chunk_label(c) for c in earlier) + ".)"
            if not turn.chunks:
                prompt += "\n\n(No excerpts from the document matched this question.)"
            content.append({"type": "text", "text": prompt})
            messages.append({"role": "user", "content": content})
            if turn.reply is not None:
                messages.append({"role": "assistant", "content": turn.reply})

        with self._translate_errors(), self.client.beta.messages.stream(
            max_tokens=16_000,
            system=[{"type": "text", "text": chat_system_prompt(document_name, inline_citations=False)}],
            messages=_with_cache_breakpoints(messages),
            **self._base_params(effort),
        ) as stream:
            for event in stream:
                if event.type == "text" and on_text is not None:
                    on_text(event.text)
            message = stream.get_final_message()

        if message.stop_reason == "refusal":
            raise RefusalError(_refusal_message(message))

        known = {c.id for turn in turns for c in turn.chunks}
        segments: list[tuple[str, list[int]]] = []
        cited: dict[int, list[str]] = {}
        for block in message.content:
            if getattr(block, "type", None) != "text":
                continue
            ids: list[int] = []
            for citation in getattr(block, "citations", None) or []:
                chunk_id = self._citation_chunk(citation, result_chunk_ids, known)
                if chunk_id is None:
                    continue
                if chunk_id not in ids:
                    ids.append(chunk_id)
                passage = " ".join((getattr(citation, "cited_text", "") or "").split())
                if passage and passage not in cited.setdefault(chunk_id, []):
                    cited[chunk_id].append(passage)
            segments.append((block.text, ids))
        return ChatReply(segments, cited, _replayable(list(message.content)), _usage(message), message.model)

    @staticmethod
    def _citation_chunk(citation: Any, result_chunk_ids: list[int], known: set[int]) -> int | None:
        if getattr(citation, "type", None) != "search_result_location":
            return None
        match = _SOURCE_ID.search(getattr(citation, "source", "") or "")
        if match and int(match.group(1)) in known:
            return int(match.group(1))
        index = getattr(citation, "search_result_index", None)
        if isinstance(index, int) and 0 <= index < len(result_chunk_ids):
            return result_chunk_ids[index]
        return None
