import json
import time
import pytest
from fastapi.testclient import TestClient

from legal_analyzer.web.app import create_app

from .conftest import SAMPLE_PDF, FakeLLM


def cited_reply(turns):
    return f"Yes, at any time [C{turns[-1].chunks[0].id}]."


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings, llm=FakeLLM(chat_reply=cited_reply))) as test_client:
        yield test_client


def upload(client, name="terms.pdf"):
    with open(SAMPLE_PDF, "rb") as f:
        return client.post("/api/documents", files={"file": (name, f, "application/pdf")})


def test_upload_list_search_and_chunks(client):
    response = upload(client)
    assert response.status_code == 201
    doc = response.json()
    assert doc["filename"] == "terms.pdf" and doc["chunk_count"] == 22 and not doc["has_report"]
    assert client.get("/api/documents").json()[0]["id"] == doc["id"]
    assert upload(client).json()["id"] == doc["id"]  # same bytes -> same document
    hits = client.get(f"/api/documents/{doc['id']}/search", params={"q": "class action waiver"}).json()
    assert hits[0]["section"].startswith("13.")
    assert client.get(f"/api/documents/{doc['id']}/chunks/0").json()["location"] == "p. 1"
    assert client.get(f"/api/documents/{doc['id']}/chunks/999").status_code == 404


def test_rejects_bad_uploads_and_unknown_ids(client):
    assert client.post("/api/documents", files={"file": ("x.exe", b"MZ", "application/octet-stream")}).status_code == 400
    assert client.post("/api/documents", files={"file": ("x.pdf", b"not a pdf", "application/pdf")}).status_code == 400
    assert client.get("/api/documents/not-a-real-id").status_code == 404
    assert client.get("/api/documents/0123456789abcdef").status_code == 404


def test_analysis_job_and_reports(client):
    doc = upload(client).json()
    started = client.post(f"/api/documents/{doc['id']}/analyze", json={"perspective": "the customer"})
    assert started.status_code == 202
    deadline = time.time() + 10
    while True:
        data = client.get(f"/api/documents/{doc['id']}/analysis").json()
        if data["job"]["status"] != "running" or time.time() > deadline:
            break
        time.sleep(0.05)
    assert data["job"]["status"] == "done", data["job"]
    assert data["report"]["perspective"] == "the customer"
    assert data["report"]["findings"][0]["id"] == "F1"
    html = client.get(f"/api/documents/{doc['id']}/report.html")
    assert html.status_code == 200 and "Liability capped at $10" in html.text
    assert "# Legal document analysis" in client.get(f"/api/documents/{doc['id']}/report.md").text
    assert client.get("/api/documents").json()[0]["has_report"] is True


def test_chat_streams_answer_with_citations(client):
    doc = upload(client).json()
    with client.stream("POST", f"/api/documents/{doc['id']}/chat", json={"question": "Can they terminate me?"}) as r:
        events = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]
    assert [e["type"] for e in events] == ["session", "text", "answer"]
    assert events[0]["model"] == "Fake · fake-model"
    answer = events[-1]
    assert answer["text"] == "Yes, at any time[1]."
    assert answer["sources"][0]["number"] == 1 and answer["sources"][0]["cited_text"]

    session_id = events[0]["session_id"]
    with client.stream("POST", f"/api/documents/{doc['id']}/chat",
                       json={"question": "And refunds?", "session_id": session_id}) as r:
        follow_up = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]
    assert follow_up[0]["session_id"] == session_id


def test_delete_document(client):
    doc = upload(client).json()
    assert client.delete(f"/api/documents/{doc['id']}").status_code == 204
    assert client.get(f"/api/documents/{doc['id']}").status_code == 404


def test_analysis_and_chat_require_credentials(settings):
    class NoKeyLLM(FakeLLM):
        def status(self):
            return "Set ANTHROPIC_API_KEY to use Claude."

    with TestClient(create_app(settings, llm=NoKeyLLM())) as client:
        health = client.get("/api/health").json()
        assert health["ready"] is False and "ANTHROPIC_API_KEY" in health["message"]
        doc = upload(client).json()
        analyze = client.post(f"/api/documents/{doc['id']}/analyze", json={})
        assert analyze.status_code == 400 and "ANTHROPIC_API_KEY" in analyze.json()["detail"]
        assert client.post(f"/api/documents/{doc['id']}/chat", json={"question": "hi"}).status_code == 400


def test_models_endpoint_reports_every_provider(settings, monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    # point Ollama at a closed port so the test never touches a real server
    local = settings.with_overrides(ollama_host="http://127.0.0.1:9")
    with TestClient(create_app(local)) as client:
        data = client.get("/api/models").json()
        providers = {p["id"]: p for p in data["providers"]}
        assert set(providers) == {"anthropic", "openai", "gemini", "ollama"}
        assert data["default"] == {"provider": "anthropic", "model": "claude-opus-5-5"}
        assert providers["openai"]["ready"] is True and providers["openai"]["models"] == ["gpt-5"]
        assert providers["gemini"]["ready"] is False and "GEMINI_API_KEY" in providers["gemini"]["message"]
        assert providers["ollama"]["ready"] is False and providers["ollama"]["local"] is True
        assert "claude-sonnet-5-5" in providers["anthropic"]["models"]
        bad = client.post("/api/documents/0123456789abcdef/analyze", json={"provider": "nope"})
        assert bad.status_code == 404  # unknown document is checked first
        upload_id = upload(client).json()["id"]
        bad = client.post(f"/api/documents/{upload_id}/analyze", json={"provider": "nope"})
        assert bad.status_code == 400
        not_ready = client.post(f"/api/documents/{upload_id}/analyze", json={"provider": "ollama"})
        assert not_ready.status_code == 400 and "Ollama" in not_ready.json()["detail"]


def test_index_page_served(client):
    page = client.get("/")
    assert page.status_code == 200 and "Legal Document Analyzer" in page.text
    assert client.get("/static/app.js").status_code == 200
