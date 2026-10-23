"""On-disk store of ingested documents: extracted chunks, embeddings, and analysis reports.

Layout (one folder per document, keyed by a hash of the file's bytes):

    data/documents/<doc_id>/
        document.json     metadata (written last: its presence marks a complete ingest)
        source.pdf        copy of the original file
        chunks.json       clause-aware chunks with page numbers and section headings
        embeddings.npy    normalised embedding matrix (absent in keyword-only mode)
        analysis.json     latest analysis report, if one has been run
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .chunker import chunk_document
from .config import Settings
from .models import AnalysisReport, Chunk, DocumentInfo
from .pdf_loader import DocumentError, load_document
from .retrieval import Embedder, HybridIndex, get_embedder

_DOC_ID = re.compile(r"^[0-9a-f]{16}$")


class DocumentNotFound(KeyError):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class DocumentRecord:
    info: DocumentInfo
    chunks: list[Chunk]
    index: HybridIndex


class DocumentStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.root = settings.documents_dir
        self._cache: dict[str, DocumentRecord] = {}
        self._lock = threading.RLock()

    def embedder(self) -> Embedder | None:
        return get_embedder(self.settings.embedding_backend, self.settings.embedding_model, self.settings.models_dir)

    def _folder(self, doc_id: str) -> Path:
        if not _DOC_ID.match(doc_id):
            raise DocumentNotFound(doc_id)
        return self.root / doc_id

    # --- ingest --------------------------------------------------------------------------

    def ingest(self, path: Path, filename: str | None = None) -> DocumentRecord:
        """Extract, chunk, and index a document. Re-ingesting the same file returns the existing record."""
        path = Path(path)
        data = path.read_bytes()
        sha256 = hashlib.sha256(data).hexdigest()
        doc_id = sha256[:16]
        if (self.root / doc_id / "document.json").exists():
            return self.get(doc_id)

        loaded = load_document(path)
        chunks = chunk_document(loaded.pages)
        if not chunks:
            raise DocumentError("No text could be extracted from the document.")
        embedder = self.embedder()
        embeddings = embedder.embed_documents([c.index_text() for c in chunks]) if embedder else None

        info = DocumentInfo(
            id=doc_id,
            filename=filename or path.name,
            sha256=sha256,
            page_count=len(loaded.pages),
            chunk_count=len(chunks),
            char_count=loaded.char_count,
            title=loaded.title,
            empty_pages=loaded.empty_pages,
            warnings=loaded.warnings,
            embedding_model=embedder.name if embedder else None,
            created_at=now_iso(),
        )

        self.root.mkdir(parents=True, exist_ok=True)
        staging = self.root / f".staging-{doc_id}-{uuid.uuid4().hex[:8]}"
        staging.mkdir()
        try:
            shutil.copyfile(path, staging / f"source{path.suffix.lower()}")
            (staging / "chunks.json").write_text(
                json.dumps([c.model_dump() for c in chunks], ensure_ascii=False), encoding="utf-8"
            )
            if embeddings is not None:
                np.save(staging / "embeddings.npy", embeddings)
            (staging / "document.json").write_text(info.model_dump_json(indent=2), encoding="utf-8")
            target = self.root / doc_id
            if target.exists() and not (target / "document.json").exists():
                shutil.rmtree(target)  # leftovers of an interrupted ingest
            os.replace(staging, target)
        except OSError:
            shutil.rmtree(staging, ignore_errors=True)
            if not (self.root / doc_id / "document.json").exists():  # lost a race with an identical upload
                raise

        record = DocumentRecord(info, chunks, HybridIndex(chunks, embeddings, embedder))
        with self._lock:
            self._cache[doc_id] = record
        return record

    # --- read ----------------------------------------------------------------------------

    def get(self, doc_id: str) -> DocumentRecord:
        folder = self._folder(doc_id)
        with self._lock:
            if doc_id in self._cache:
                return self._cache[doc_id]
            if not (folder / "document.json").exists():
                raise DocumentNotFound(doc_id)
            info = DocumentInfo.model_validate_json((folder / "document.json").read_text(encoding="utf-8"))
            chunks = [Chunk(**c) for c in json.loads((folder / "chunks.json").read_text(encoding="utf-8"))]
            embedder = self.embedder()
            embeddings = None
            if embedder is not None:
                path = folder / "embeddings.npy"
                if path.exists() and info.embedding_model == embedder.name:
                    embeddings = np.load(path)
                if embeddings is None or embeddings.shape[0] != len(chunks):
                    # Indexed in keyword-only mode or with another model: embed now and keep the result.
                    embeddings = embedder.embed_documents([c.index_text() for c in chunks])
                    np.save(path, embeddings)
                    info.embedding_model = embedder.name
                    (folder / "document.json").write_text(info.model_dump_json(indent=2), encoding="utf-8")
            record = DocumentRecord(info, chunks, HybridIndex(chunks, embeddings, embedder))
            self._cache[doc_id] = record
            return record

    def list_documents(self) -> list[DocumentInfo]:
        if not self.root.exists():
            return []
        infos = []
        for meta in self.root.glob("*/document.json"):
            try:
                infos.append(DocumentInfo.model_validate_json(meta.read_text(encoding="utf-8")))
            except ValueError:
                continue
        return sorted(infos, key=lambda info: info.created_at, reverse=True)

    def delete(self, doc_id: str) -> None:
        folder = self._folder(doc_id)
        with self._lock:
            self._cache.pop(doc_id, None)
            if not folder.exists():
                raise DocumentNotFound(doc_id)
            shutil.rmtree(folder)

    # --- reports -------------------------------------------------------------------------

    def save_report(self, report: AnalysisReport) -> Path:
        path = self._folder(report.document.id) / "analysis.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, path)
        return path

    def load_report(self, doc_id: str) -> AnalysisReport | None:
        path = self._folder(doc_id) / "analysis.json"
        if not path.exists():
            return None
        return AnalysisReport.model_validate_json(path.read_text(encoding="utf-8"))
