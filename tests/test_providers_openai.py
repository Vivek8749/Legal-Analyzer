"""The OpenAI-compatible provider against the real openai SDK with a mocked HTTP transport."""

from __future__ import annotations

import json
from typing import Any

import httpx2
import openai
import pytest

from legal_analyzer.config import Settings
from legal_analyzer.llm import (
    ChatTurn,
    FatalLLMError,
    ModelNotFoundError,
    PromptBlock,
    RefusalError,
    TruncatedOutputError,
)
from legal_analyzer.models import Chunk
from legal_analyzer.providers.openai_compat import OpenAICompatLLM
from legal_analyzer.schemas import PassOutput

from .conftest import make_finding

REQUEST = dict(system="SYSTEM", blocks=[PromptBlock("DOCUMENT", cacheable=True), PromptBlock("TASK")],
               output_type=PassOutput, effort="high")


def completion(content: str | None, finish_reason: str = "stop", refusal: str | None = None) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": "gpt-5",
        "choices": [{"index": 0, "finish_reason": finish_reason,
                     "message": {"role": "assistant", "content": content, "refusal": refusal}}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200,
                  "prompt_tokens_details": {"cached_tokens": 400}},
    }


def sse(chunks: list[dict[str, Any]]) -> bytes:
    return ("".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n").encode()


def delta(text: str, finish: str | None = None) -> dict[str, Any]:
    return {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "gpt-5",
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": finish}]}


class MockAPI:
    def __init__(self, *responses: httpx2.Response):
        self.responses = list(responses)
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.responses.pop(0)

    def body(self, n: int = 0) -> dict[str, Any]:
        return json.loads(self.requests[n].content)

    def llm(self, settings: Settings | None = None, provider: str = "openai") -> OpenAICompatLLM:
        client = openai.OpenAI(api_key="sk-test", max_retries=0,
                               http_client=openai.DefaultHttpxClient(transport=httpx2.MockTransport(self)))
        return OpenAICompatLLM(settings or Settings(), provider=provider, client=client)


def test_structured_request_uses_strict_json_schema():
    expected = PassOutput(findings=[make_finding()], key_terms=[], gaps=[])
    api = MockAPI(httpx2.Response(200, json=completion(expected.model_dump_json())))
    result = api.llm().structured(**REQUEST)

    assert result.output == expected
    assert (result.usage.input_tokens, result.usage.cache_read_input_tokens, result.usage.output_tokens) == (600, 400, 200)
    body = api.body()
    assert api.requests[0].url.path.endswith("/chat/completions")
    assert body["model"] == "gpt-5"
    assert body["messages"] == [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "DOCUMENT\n\nTASK"}]
    fmt = body["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"]["additionalProperties"] is False
    assert "$ref" not in json.dumps(fmt)
    assert "reasoning_effort" not in body and "temperature" not in body


def test_reasoning_effort_is_opt_in():
    api = MockAPI(httpx2.Response(200, json=completion(PassOutput(findings=[], key_terms=[], gaps=[]).model_dump_json())))
    api.llm(Settings(openai_reasoning_effort="low")).structured(**REQUEST)
    assert api.body()["reasoning_effort"] == "low"


@pytest.mark.parametrize("response, error", [
    (httpx2.Response(200, json=completion(None, refusal="I can't help with that.")), RefusalError),
    (httpx2.Response(200, json=completion('{"findings": [', finish_reason="length")), TruncatedOutputError),
    (httpx2.Response(401, json={"error": {"message": "bad key"}}), FatalLLMError),
    (httpx2.Response(404, json={"error": {"message": "no such model"}}), ModelNotFoundError),
])
def test_structured_errors(response, error):
    with pytest.raises(error):
        MockAPI(response).llm().structured(**REQUEST)


def test_chat_streams_filtered_text_and_inline_citations():
    chunks = [delta("<think>let me"), delta(" look</think>Yes. Acme can "), delta("terminate at any time [C"),
              delta("9]. Fees stay non-refundable [C4][C99]."), delta("", finish="stop"),
              {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "gpt-5", "choices": [],
               "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70}}]
    api = MockAPI(httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=sse(chunks)))
    turn = ChatTurn("Can they close my account?", [
        Chunk(id=9, text="9.1 Acme may terminate your account at any time.", page_start=2, page_end=2),
        Chunk(id=4, text="4.3 All fees are non-refundable.", page_start=1, page_end=1),
    ], [9, 4])
    streamed: list[str] = []
    reply = api.llm().chat(document_name="terms.pdf", turns=[turn], effort="medium", on_text=streamed.append)

    assert "".join(streamed) == "Yes. Acme can terminate at any time [C9]. Fees stay non-refundable [C4][C99]."
    assert reply.segments == [("Yes. Acme can terminate at any time", [9]), (". Fees stay non-refundable", [4]), (".", [])]
    assert reply.cited_text[9] == ["9.1 Acme may terminate your account at any time."]
    assert reply.raw.startswith("Yes.") and reply.usage.output_tokens == 20
    body = api.body()
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert "How to cite" in body["messages"][0]["content"]  # inline citation instructions
    assert '<excerpt id="C9"' in body["messages"][-1]["content"]


def test_status_and_endpoint_presets(monkeypatch):
    for var in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert "OPENAI_API_KEY" in OpenAICompatLLM(Settings()).status()
    # self-hosted OpenAI-compatible servers (LM Studio, vLLM, …) usually need no key
    local = OpenAICompatLLM(Settings(openai_base_url="http://localhost:1234/v1"))
    assert local.status() is None and str(local.client().base_url).startswith("http://localhost:1234/v1")

    monkeypatch.setenv("GEMINI_API_KEY", "gm-test")
    gemini = OpenAICompatLLM(Settings(), provider="gemini")
    assert gemini.status() is None and gemini.model == "gemini-flash-latest"
    assert str(gemini.client().base_url) == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert gemini.client().api_key == "gm-test"
