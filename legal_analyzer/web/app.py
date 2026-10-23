"""FastAPI web app: upload documents, run analyses in the background, and chat with citations."""

from __future__ import annotations

import json
import logging
import queue
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..analyzer import DEFAULT_PERSPECTIVE, AnalysisError, AnalysisOptions, Analyzer
from ..config import ANTHROPIC_MODELS, PROVIDER_LABELS, PROVIDERS, Settings
from ..llm import LLM, LLMError, RefusalError, create_llm
from ..models import DocumentInfo
from ..pdf_loader import SUPPORTED_SUFFIXES, DocumentError
from ..qa import Answer, ChatSession
from ..report import to_html, to_markdown
from ..store import DocumentNotFound, DocumentStore

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_SESSIONS = 200


class ModelChoice(BaseModel):
    provider: str | None = Field(default=None, max_length=20)
    model: str | None = Field(default=None, max_length=200)


class AnalyzeRequest(ModelChoice):
    perspective: str | None = Field(default=None, max_length=300)
    jurisdiction: str | None = Field(default=None, max_length=120)


class ChatRequest(ModelChoice):
    question: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, max_length=64)


@dataclass
class Job:
    status: str = "running"  # running | done | error
    stage: str = "queued"
    message: str = "Starting"
    completed: int = 0
    total: int = 0
    error: str | None = None
    started_at: float = 0.0
    finished_at: float | None = None
    model: str = ""


def friendly_error(exc: BaseException) -> str:
    # Providers translate their SDK errors into LLMError subclasses with readable messages.
    if isinstance(exc, (LLMError, AnalysisError, DocumentError, ValueError)):
        return str(exc)
    return f"Unexpected error: {exc}"


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _answer_payload(answer: Answer) -> dict[str, Any]:
    return {
        "text": answer.text,
        "segments": answer.segments,
        "sources": [asdict(s) for s in answer.sources],
        "retrieved_chunk_ids": answer.retrieved_chunk_ids,
        "model": answer.model,
    }


def create_app(settings: Settings | None = None, llm: LLM | None = None) -> FastAPI:
    """``llm`` pins every request to one model client (used by tests); otherwise each request
    may pick a provider and model, defaulting to the configured ones."""
    settings = settings or Settings.from_env()
    store = DocumentStore(settings)
    clients: dict[tuple[str, str], LLM] = {}
    clients_lock = threading.Lock()
    jobs: dict[str, Job] = {}
    jobs_lock = threading.Lock()
    sessions: OrderedDict[str, ChatSession] = OrderedDict()
    sessions_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Load (and on first run download) the embedding model in the background so the
        # first upload doesn't wait for it.
        threading.Thread(target=store.embedder, daemon=True).start()
        yield

    app = FastAPI(title="Legal Document Analyzer", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    def get_llm(provider: str | None = None, model: str | None = None) -> LLM:
        if llm is not None:
            return llm
        provider = (provider or settings.provider).strip().lower()
        if provider not in PROVIDERS:
            raise HTTPException(400, f"Unknown provider {provider!r}.")
        model = (model or "").strip() or settings.model_for(provider)
        with clients_lock:
            if (provider, model) not in clients:
                clients[(provider, model)] = create_llm(settings, provider, model)
            return clients[(provider, model)]

    def ready_llm(choice: ModelChoice) -> LLM:
        client = get_llm(choice.provider, choice.model)
        problem = client.status()
        if problem:
            raise HTTPException(400, f"{client.label} is not ready. {problem}")
        return client

    def provider_entry(client: LLM) -> dict[str, Any]:
        problem = client.status()
        models = [client.model]
        if client.provider == "anthropic":
            models = list(dict.fromkeys([client.model, *ANTHROPIC_MODELS]))
        elif client.provider == "ollama" and problem is None:
            try:
                models = client.installed_models()  # type: ignore[attr-defined]
            except LLMError as exc:  # Ollama stopped in the meantime
                problem = str(exc)
        return {
            "id": client.provider,
            "label": PROVIDER_LABELS.get(client.provider, client.provider),
            "local": client.local,
            "ready": problem is None,
            "message": problem,
            "default_model": client.model,
            "models": models,
        }

    def record_or_404(doc_id: str):
        try:
            return store.get(doc_id)
        except DocumentNotFound:
            raise HTTPException(404, "Document not found.") from None

    def job_dict(doc_id: str) -> dict[str, Any] | None:
        with jobs_lock:
            job = jobs.get(doc_id)
            return asdict(job) if job else None

    def doc_payload(info: DocumentInfo) -> dict[str, Any]:
        has_report = (settings.documents_dir / info.id / "analysis.json").exists()
        return {**info.model_dump(), "has_report": has_report, "job": job_dict(info.id)}

    # --- pages ---------------------------------------------------------------------------

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        client = get_llm()
        problem = client.status()
        return {
            "provider": client.provider,
            "model": client.model,
            "label": client.label,
            "ready": problem is None,
            "message": problem,
            "effort": settings.analysis_effort,
            "embeddings": settings.embedding_backend != "none",
            "default_perspective": DEFAULT_PERSPECTIVE,
        }

    @app.get("/api/models")
    def models() -> dict[str, Any]:
        """Every provider with its readiness and models (installed ones, for Ollama)."""
        default = get_llm()
        clients_to_show = [default] if llm is not None else [get_llm(p) for p in PROVIDERS]
        return {
            "default": {"provider": default.provider, "model": default.model},
            "providers": [provider_entry(c) for c in clients_to_show],
        }

    # --- documents -----------------------------------------------------------------------

    @app.get("/api/documents")
    def list_documents() -> list[dict[str, Any]]:
        return [doc_payload(info) for info in store.list_documents()]

    @app.post("/api/documents", status_code=201)
    async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
        name = Path(file.filename or "document.pdf").name
        suffix = Path(name).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise HTTPException(400, f"Unsupported file type. Upload one of: {', '.join(SUPPORTED_SUFFIXES)}.")
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "File is larger than 50 MB.")
        if suffix == ".pdf" and b"%PDF-" not in data[:1024]:
            raise HTTPException(400, "This file does not look like a PDF.")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"upload{suffix}"
            path.write_bytes(data)
            try:
                record = await run_in_threadpool(store.ingest, path, name)
            except DocumentError as exc:
                raise HTTPException(422, str(exc)) from exc
        return doc_payload(record.info)

    @app.get("/api/documents/{doc_id}")
    def get_document(doc_id: str) -> dict[str, Any]:
        record = record_or_404(doc_id)
        return {**doc_payload(record.info), "search_mode": "hybrid" if record.index.uses_embeddings else "keyword"}

    @app.delete("/api/documents/{doc_id}", status_code=204)
    def delete_document(doc_id: str) -> None:
        with jobs_lock:
            if doc_id in jobs and jobs[doc_id].status == "running":
                raise HTTPException(409, "An analysis of this document is still running.")
            jobs.pop(doc_id, None)
        try:
            store.delete(doc_id)
        except DocumentNotFound:
            raise HTTPException(404, "Document not found.") from None
        with sessions_lock:
            for sid in [sid for sid, s in sessions.items() if s.record.info.id == doc_id]:
                sessions.pop(sid)

    @app.get("/api/documents/{doc_id}/chunks/{chunk_id}")
    def get_chunk(doc_id: str, chunk_id: int) -> dict[str, Any]:
        record = record_or_404(doc_id)
        if not 0 <= chunk_id < len(record.chunks):
            raise HTTPException(404, "Chunk not found.")
        chunk = record.chunks[chunk_id]
        return {**chunk.model_dump(), "location": chunk.location}

    @app.get("/api/documents/{doc_id}/search")
    def search(doc_id: str, q: str = Query(min_length=1, max_length=500), k: int = Query(5, ge=1, le=20)) -> list[dict[str, Any]]:
        record = record_or_404(doc_id)
        return [
            {**hit.chunk.model_dump(), "location": hit.chunk.location, "score": round(hit.score, 5),
             "keyword_rank": hit.keyword_rank, "semantic_rank": hit.semantic_rank}
            for hit in record.index.search(q, k=k)
        ]

    # --- analysis ------------------------------------------------------------------------

    def run_job(doc_id: str, options: AnalysisOptions, client: LLM) -> None:
        job = jobs[doc_id]

        def progress(event: dict[str, Any]) -> None:
            with jobs_lock:
                job.stage = event["stage"]
                job.message = event["message"]
                job.completed = event["completed"]
                job.total = event["total"]

        try:
            report = Analyzer(client, settings).analyze(store.get(doc_id), options, progress)
            store.save_report(report)
            with jobs_lock:
                job.status = "done"
        except Exception as exc:  # reported to the UI through the job status
            log.exception("Analysis of %s failed", doc_id)
            with jobs_lock:
                job.status, job.error = "error", friendly_error(exc)
        finally:
            with jobs_lock:
                job.finished_at = time.time()

    @app.post("/api/documents/{doc_id}/analyze", status_code=202)
    def analyze(doc_id: str, body: AnalyzeRequest) -> dict[str, Any]:
        record_or_404(doc_id)
        client = ready_llm(body)
        options = AnalysisOptions(
            perspective=(body.perspective or "").strip() or DEFAULT_PERSPECTIVE,
            jurisdiction=(body.jurisdiction or "").strip() or None,
        )
        with jobs_lock:
            existing = jobs.get(doc_id)
            if existing and existing.status == "running":
                return asdict(existing)
            jobs[doc_id] = Job(started_at=time.time(), model=client.label)
        threading.Thread(target=run_job, args=(doc_id, options, client), daemon=True).start()
        return job_dict(doc_id) or {}

    @app.get("/api/documents/{doc_id}/analysis")
    def analysis(doc_id: str) -> dict[str, Any]:
        record_or_404(doc_id)
        report = store.load_report(doc_id)
        return {"job": job_dict(doc_id), "report": report.model_dump() if report else None}

    def report_or_404(doc_id: str):
        record_or_404(doc_id)
        report = store.load_report(doc_id)
        if report is None:
            raise HTTPException(404, "This document has not been analysed yet.")
        return report

    @app.get("/api/documents/{doc_id}/report.html", response_class=HTMLResponse)
    def report_html(doc_id: str, download: bool = False) -> HTMLResponse:
        report = report_or_404(doc_id)
        headers = {"Content-Disposition": f'attachment; filename="{doc_id}-analysis.html"'} if download else None
        return HTMLResponse(to_html(report), headers=headers)

    @app.get("/api/documents/{doc_id}/report.md", response_class=PlainTextResponse)
    def report_md(doc_id: str) -> PlainTextResponse:
        report = report_or_404(doc_id)
        return PlainTextResponse(
            to_markdown(report), media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{doc_id}-analysis.md"'},
        )

    # --- chat ----------------------------------------------------------------------------

    def get_session(doc_id: str, session_id: str | None, client: LLM) -> tuple[str, ChatSession]:
        """Reuse the conversation unless the document or the model changed."""
        with sessions_lock:
            session = sessions.get(session_id or "")
            if session is not None and session.record.info.id == doc_id and session.llm is client:
                sessions.move_to_end(session_id)  # type: ignore[arg-type]
                return session_id, session  # type: ignore[return-value]
            new_id = uuid.uuid4().hex
            sessions[new_id] = ChatSession(store.get(doc_id), client, settings)
            while len(sessions) > MAX_SESSIONS:
                sessions.popitem(last=False)
            return new_id, sessions[new_id]

    @app.post("/api/documents/{doc_id}/chat")
    def chat(doc_id: str, body: ChatRequest) -> StreamingResponse:
        record_or_404(doc_id)
        client = ready_llm(body)
        session_id, session = get_session(doc_id, body.session_id, client)

        def events() -> Iterator[str]:
            channel: queue.Queue[tuple[str, Any]] = queue.Queue()

            def worker() -> None:
                try:
                    answer = session.ask(body.question, on_text=lambda text: channel.put(("text", text)))
                    channel.put(("answer", _answer_payload(answer)))
                except RefusalError as exc:
                    channel.put(("error", f"{exc} Try rephrasing the question."))
                except Exception as exc:  # surfaced to the user in the chat
                    log.exception("Chat request failed")
                    channel.put(("error", friendly_error(exc)))
                finally:
                    channel.put(("end", None))

            threading.Thread(target=worker, daemon=True).start()
            yield _sse({"type": "session", "session_id": session_id, "model": client.label})
            while True:
                kind, payload = channel.get()
                if kind == "end":
                    break
                if kind == "text":
                    yield _sse({"type": "text", "text": payload})
                elif kind == "answer":
                    yield _sse({"type": "answer", **payload})
                else:
                    yield _sse({"type": "error", "message": payload})

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    return app
