"""The Ollama provider against the real ollama client with a mocked HTTP transport."""

from __future__ import annotations

import json
from typing import Any

import httpx
import ollama
import pytest

from legal_analyzer.config import Settings
from legal_analyzer.llm import (
    ChatTurn,
    LLMError,
    ModelNotFoundError,
    PromptBlock,
    ProviderUnavailableError,
    TruncatedOutputError,
)
from legal_analyzer.models import Chunk
from legal_analyzer.providers.ollama_local import OllamaLLM
from legal_analyzer.schemas import PassOutput

from .conftest import make_finding

REQUEST = dict(system="SYSTEM", blocks=[PromptBlock("DOCUMENT", cacheable=True), PromptBlock("TASK")],
               output_type=PassOutput, effort="high")


def ndjson(pieces: list[str], done_reason: str = "stop") -> bytes:
    lines = [{"model": "qwen3.5:4b", "created_at": "2026-10-23T00:00:00Z",
              "message": {"role": "assistant", "content": p}, "done": False} for p in pieces]
    lines.append({"model": "qwen3.5:4b", "created_at": "2026-10-23T00:00:01Z", "message": {"role": "assistant", "content": ""},
                  "done": True, "done_reason": done_reason, "prompt_eval_count": 1200, "eval_count": 300})
    return "\n".join(json.dumps(line) for line in lines).encode() + b"\n"


class MockOllama:
    def __init__(self, chat: bytes | None = None, capabilities: list[str] | None = None, context_length: int = 262_144,
                 installed: list[str] | None = None, show_status: int = 200):
        self.chat = chat
        self.capabilities = capabilities if capabilities is not None else ["completion", "thinking"]
        self.context_length = context_length
        self.installed = installed if installed is not None else ["qwen3.5:4b"]
        self.show_status = show_status
        self.requests: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        self.requests.append((request.url.path, body))
        if request.url.path == "/api/show":
            if self.show_status != 200:
                return httpx.Response(self.show_status, json={"error": f"model '{body.get('model')}' not found"})
            return httpx.Response(200, json={"capabilities": self.capabilities,
                                             "model_info": {"qwen35.context_length": self.context_length}})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": m, "model": m} for m in self.installed]})
        if request.url.path == "/api/chat":
            return httpx.Response(200, headers={"content-type": "application/x-ndjson"}, content=self.chat or b"")
        return httpx.Response(404, json={"error": "not found"})

    def chat_body(self) -> dict[str, Any]:
        return next(body for path, body in self.requests if path == "/api/chat")

    def llm(self, settings: Settings | None = None, model: str = "qwen3.5:4b") -> OllamaLLM:
        client = ollama.Client(host="http://ollama.test", transport=httpx.MockTransport(self))
        return OllamaLLM(settings or Settings(), model, client=client)


def test_structured_request_sets_context_schema_and_thinking():
    expected = PassOutput(findings=[make_finding()], key_terms=[], gaps=[])
    text = expected.model_dump_json()
    server = MockOllama(chat=ndjson([text[:50], text[50:]]))
    started: list[bool] = []
    result = server.llm().structured(**REQUEST, on_stream_start=lambda: started.append(True))

    assert result.output == expected and started == [True]
    assert (result.usage.input_tokens, result.usage.output_tokens) == (1200, 300)
    body = server.chat_body()
    assert body["model"] == "qwen3.5:4b" and body["stream"] is True
    assert body["messages"] == [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "DOCUMENT\n\nTASK"}]
    assert body["format"]["additionalProperties"] is False and "$ref" not in json.dumps(body["format"])
    assert body["options"] == {"num_ctx": 16_384, "num_predict": 4_096, "temperature": 0.2}
    assert body["think"] is False  # reasoning off by default for thinking-capable models
    assert body["keep_alive"] == "10m"


def test_think_omitted_for_models_without_thinking_and_context_capped():
    text = PassOutput(findings=[], key_terms=[], gaps=[]).model_dump_json()
    server = MockOllama(chat=ndjson([text]), capabilities=["completion"], context_length=8_192)
    llm = server.llm(Settings(ollama_think="true"))
    assert llm.context_window == 8_192  # the model supports less than the configured 16k
    llm.structured(**REQUEST)
    body = server.chat_body()
    assert "think" not in body and body["options"]["num_ctx"] == 8_192


def test_thinking_can_be_enabled():
    text = PassOutput(findings=[], key_terms=[], gaps=[]).model_dump_json()
    server = MockOllama(chat=ndjson(["<think>reasoning</think>", text]))
    result = server.llm(Settings(ollama_think="high")).structured(**REQUEST)
    assert server.chat_body()["think"] == "high"
    assert result.output.findings == []


def test_truncated_answer_raises():
    with pytest.raises(TruncatedOutputError, match="NUM_PREDICT"):
        MockOllama(chat=ndjson(['{"findings": ['], done_reason="length")).llm().structured(**REQUEST)


def test_invalid_json_raises_llm_error():
    with pytest.raises(LLMError, match="did not match"):
        MockOllama(chat=ndjson(["not json at all"])).llm().structured(**REQUEST)


def test_missing_model_and_server_down():
    server = MockOllama(show_status=404, installed=["llama3.2:3b"])
    llm = server.llm()
    assert llm.status() == "Model 'qwen3.5:4b' isn't downloaded. Run: ollama pull qwen3.5:4b"
    with pytest.raises(ModelNotFoundError, match="ollama pull qwen3.5:4b"):
        llm.structured(**REQUEST)

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    down = OllamaLLM(Settings(), "qwen3.5:4b", client=ollama.Client(host="http://ollama.test", transport=httpx.MockTransport(refuse)))
    assert "Could not connect to Ollama" in down.status()
    with pytest.raises(ProviderUnavailableError):
        down.structured(**REQUEST)


def test_chat_hides_reasoning_and_parses_citations():
    server = MockOllama(chat=ndjson(["<thi", "nk>planning</think>", "Yes [C2]", ". You get no refund [C5]."]))
    turn = ChatTurn("Can they cancel?", [
        Chunk(id=2, text="9.1 Acme may suspend or terminate your account at any time.", page_start=2, page_end=2),
        Chunk(id=5, text="9.4 No fees will be refunded upon termination.", page_start=2, page_end=2),
    ], [2, 5])
    streamed: list[str] = []
    reply = server.llm().chat(document_name="terms.pdf", turns=[turn], effort="medium", on_text=streamed.append)
    assert "".join(streamed) == "Yes [C2]. You get no refund [C5]."
    assert reply.segments == [("Yes", [2]), (". You get no refund", [5]), (".", [])]
    assert reply.raw == "Yes [C2]. You get no refund [C5]."
    body = server.chat_body()
    assert body["options"]["temperature"] == 0.3 and "format" not in body
    assert "How to cite" in body["messages"][0]["content"]
