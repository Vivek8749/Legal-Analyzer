"""Exercise the Claude provider and ChatSession against the real Anthropic SDK with a mocked HTTP transport.

The mock returns server-sent event streams shaped like the Messages API's, so these tests
check the exact request bodies the app sends and how it reads streamed, structured, and
cited responses, all without an API key or network access.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import anthropic
import httpx2
import pytest

from legal_analyzer.config import Settings
from legal_analyzer.llm import LLMError, MissingCredentialsError, PromptBlock, RefusalError
from legal_analyzer.providers.claude import FALLBACK_BETA, ClaudeLLM
from legal_analyzer.qa import ChatSession
from legal_analyzer.schemas import PassOutput

from .conftest import make_finding


def sse(events: list[dict[str, Any]]) -> bytes:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def stream_body(blocks: list[dict[str, Any]], stop_reason: str = "end_turn", stop_details: dict | None = None) -> bytes:
    events: list[dict[str, Any]] = [{
        "type": "message_start",
        "message": {
            "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-opus-5-5", "content": [],
            "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 20, "output_tokens": 1, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 1500},
        },
    }]
    for i, block in enumerate(blocks):
        if block["type"] == "thinking":
            events += [
                {"type": "content_block_start", "index": i, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
                {"type": "content_block_delta", "index": i, "delta": {"type": "signature_delta", "signature": block["signature"]}},
                {"type": "content_block_stop", "index": i},
            ]
            continue
        events.append({"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": "", "citations": None}})
        text = block["text"]
        events += [{"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": text[j:j + 40]}}
                   for j in range(0, len(text), 40)]
        events += [{"type": "content_block_delta", "index": i, "delta": {"type": "citations_delta", "citation": c}}
                   for c in block.get("citations", [])]
        events.append({"type": "content_block_stop", "index": i})
    events += [
        {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None, "stop_details": stop_details},
         "usage": {"output_tokens": 42}},
        {"type": "message_stop"},
    ]
    return sse(events)


class MockAPI:
    def __init__(self, respond: Callable[[dict[str, Any]], bytes] | list[bytes]):
        self.respond = respond
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        self.requests.append({"headers": dict(request.headers), "json": body})
        content = self.respond(body) if callable(self.respond) else self.respond.pop(0)
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=content)

    def client(self, **kwargs: Any) -> anthropic.Anthropic:
        transport = httpx2.MockTransport(self)
        return anthropic.Anthropic(max_retries=0, http_client=anthropic.DefaultHttpxClient(transport=transport),
                                   **{"api_key": "sk-test", **kwargs})


PASS_REQUEST = dict(
    system="SYSTEM",
    blocks=[PromptBlock("DOCUMENT", cacheable=True), PromptBlock("TASK")],
    output_type=PassOutput,
    effort="high",
)


def test_structured_request_shape_and_parsing():
    expected = PassOutput(findings=[make_finding()], key_terms=[], gaps=[])
    api = MockAPI([stream_body([{"type": "thinking", "signature": "sig"}, {"type": "text", "text": expected.model_dump_json()}])])
    started: list[bool] = []
    result = ClaudeLLM(Settings(), client=api.client()).structured(**PASS_REQUEST, on_stream_start=lambda: started.append(True))

    assert result.output == expected
    assert started == [True]
    assert result.usage.cache_read_input_tokens == 1500 and result.usage.output_tokens == 42
    request = api.requests[0]
    body = request["json"]
    assert body["model"] == "claude-opus-5-5"
    assert body["stream"] is True
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"]["effort"] == "high"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert "limitation_of_liability" in json.dumps(body["output_config"]["format"]["schema"])
    assert body["fallbacks"] == "default"
    assert FALLBACK_BETA in request["headers"]["anthropic-beta"]
    assert body["system"] == [{"type": "text", "text": "SYSTEM"}]
    assert body["messages"][0]["content"][0] == {"type": "text", "text": "DOCUMENT", "cache_control": {"type": "ephemeral"}}
    assert body["messages"][0]["content"][1] == {"type": "text", "text": "TASK"}


def test_fallbacks_only_sent_to_models_that_support_them():
    expected = PassOutput(findings=[], key_terms=[], gaps=[])
    api = MockAPI([stream_body([{"type": "text", "text": expected.model_dump_json()}])])
    ClaudeLLM(Settings(), "claude-haiku-5-5", client=api.client()).structured(**PASS_REQUEST)
    assert api.requests[0]["json"]["model"] == "claude-haiku-5-5"
    assert "fallbacks" not in api.requests[0]["json"]


def test_fallbacks_can_be_disabled():
    expected = PassOutput(findings=[], key_terms=[], gaps=[])
    api = MockAPI([stream_body([{"type": "text", "text": expected.model_dump_json()}])])
    ClaudeLLM(Settings(use_fallbacks=False), client=api.client()).structured(**PASS_REQUEST)
    assert "fallbacks" not in api.requests[0]["json"]
    assert FALLBACK_BETA not in api.requests[0]["headers"].get("anthropic-beta", "")


def test_refusal_raises():
    api = MockAPI([stream_body([], stop_reason="refusal", stop_details={"type": "refusal", "category": "cyber", "explanation": None})])
    with pytest.raises(RefusalError, match="cyber"):
        ClaudeLLM(Settings(), client=api.client()).structured(**PASS_REQUEST)


def test_truncated_structured_output_raises():
    api = MockAPI([stream_body([{"type": "text", "text": '{"findings": [{"category_id": "limitation_of'}], stop_reason="max_tokens")])
    with pytest.raises(LLMError):
        ClaudeLLM(Settings(), client=api.client()).structured(**PASS_REQUEST)


def test_missing_credentials(monkeypatch, tmp_path):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    api = MockAPI([])
    client = anthropic.Anthropic(max_retries=0, http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(api)))
    llm = ClaudeLLM(Settings(), client=client)
    assert "ANTHROPIC_API_KEY" in (llm.status() or "")
    with pytest.raises(MissingCredentialsError):
        llm.structured(**PASS_REQUEST)
    assert api.requests == []


def test_chat_citations_and_append_only_history(sample_record, settings):
    def respond(body: dict[str, Any]) -> bytes:
        latest = body["messages"][-1]["content"]
        results = [b for b in latest if b["type"] == "search_result"]
        all_results = [b for m in body["messages"] if m["role"] == "user" for b in m["content"] if b["type"] == "search_result"]
        target = (results or all_results)[0]
        citation = {
            "type": "search_result_location", "source": target["source"], "title": target["title"],
            "search_result_index": all_results.index(target), "start_block_index": 0, "end_block_index": 1,
            "cited_text": target["content"][0]["text"],
        }
        return stream_body([
            {"type": "thinking", "signature": f"sig-{len(body['messages'])}"},
            {"type": "text", "text": "Yes. "},
            {"type": "text", "text": "Acme can end your account at any time", "citations": [citation]},
            {"type": "text", "text": "."},
        ])

    api = MockAPI(respond)
    session = ChatSession(sample_record, ClaudeLLM(settings, client=api.client()), settings, k=4)
    streamed: list[str] = []
    answer = session.ask("Can Acme close my account without notice?", on_text=streamed.append)

    first = api.requests[0]["json"]
    content = first["messages"][0]["content"]
    search_results = [b for b in content if b["type"] == "search_result"]
    assert len(search_results) == 4
    assert all(b["citations"] == {"enabled": True} for b in search_results)
    assert content[-1]["type"] == "text" and content[-1]["cache_control"] == {"type": "ephemeral"}
    assert first["output_config"] == {"effort": "medium"}
    cited_chunk = int(search_results[0]["source"].rsplit("/", 1)[1])
    assert answer.sources[0].chunk_id == cited_chunk and answer.sources[0].number == 1
    assert answer.text == "Yes. Acme can end your account at any time[1]."
    assert "".join(streamed) == "Yes. Acme can end your account at any time."

    session.ask("And will I get a refund?")
    messages = api.requests[1]["json"]["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assistant = messages[1]["content"]
    assert assistant[0] == {"type": "thinking", "thinking": "", "signature": "sig-1"}  # replayed unchanged
    assert assistant[2]["citations"][0]["type"] == "search_result_location"
    # cache read point on the previous user turn, write point on the new one
    assert messages[0]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert messages[2]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    # the first user turn is byte-identical to what was sent the first time (append-only history)
    first_turn = {k: v for k, v in messages[0]["content"][-1].items() if k != "cache_control"}
    assert first_turn == {k: v for k, v in content[-1].items() if k != "cache_control"}
    assert messages[0]["content"][:-1] == content[:-1]
    # chunks already in the conversation are not sent again
    first_sources = {b["source"] for b in messages[0]["content"] if b["type"] == "search_result"}
    second_sources = {b["source"] for b in messages[2]["content"] if b["type"] == "search_result"}
    assert not first_sources & second_sources
