"""Hybrid retrieval: BM25 keyword search + dense embeddings, fused with Reciprocal Rank Fusion.

Legal language is precise ("indemnify", "arbitration", "sole discretion"), which keyword
search handles well; paraphrased questions ("can they close my account without warning?")
need semantic search. RRF combines both rankings without having to calibrate their scores.
"""

from __future__ import annotations

import logging
import math
import re
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import snowballstemmer

from .models import Chunk

log = logging.getLogger(__name__)

RRF_K = 60
CANDIDATES_PER_RANKING = 50

# Common function words. Modal verbs and negations ("may", "shall", "not", "without") are
# kept on purpose: in legal text they carry the meaning.
_STOPWORDS = frozenset(
    "a an the and or of to in on at by for from with as is are was were be been being this that these those "
    "it its into than then there their them they he she his her we our us you your i me my which who whom "
    "what when where how do does did has have had".split()
)
_TOKEN = re.compile(r"[a-z0-9]+")
_stemmer = snowballstemmer.stemmer("english")


@lru_cache(maxsize=50_000)
def _stem(word: str) -> str:
    return _stemmer.stemWord(word)


def tokenize(text: str) -> list[str]:
    return [_stem(w) for w in _TOKEN.findall(text.lower()) if w not in _STOPWORDS]


class BM25:
    """Okapi BM25 over pre-tokenized documents, using an inverted index."""

    def __init__(self, corpus: Sequence[list[str]], k1: float = 1.5, b: float = 0.75):
        self.size = len(corpus)
        lengths = np.array([len(doc) for doc in corpus], dtype=np.float64)
        avg = lengths.mean() if self.size else 0.0
        self._norm = k1 * (1 - b + b * lengths / avg) if avg else np.full(self.size, k1)
        self._k1 = k1
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc_id, doc in enumerate(corpus):
            for term, tf in Counter(doc).items():
                postings[term].append((doc_id, tf))
        self._postings = {
            term: (np.array([d for d, _ in entries]), np.array([tf for _, tf in entries], dtype=np.float64))
            for term, entries in postings.items()
        }
        self._idf = {
            term: math.log(1 + (self.size - len(entries) + 0.5) / (len(entries) + 0.5))
            for term, entries in postings.items()
        }

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self.size)
        for term in set(query_tokens):
            if term not in self._postings:
                continue
            docs, tf = self._postings[term]
            scores[docs] += self._idf[term] * tf * (self._k1 + 1) / (tf + self._norm[docs])
        return scores


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_queries(self, texts: list[str]) -> np.ndarray: ...


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.where(norms == 0, 1, norms)).astype(np.float32)


class FastEmbedEmbedder:
    """Local ONNX embeddings via fastembed (no API key; ~70 MB model downloaded once)."""

    def __init__(self, model_name: str, cache_dir: Path):
        from fastembed import TextEmbedding

        cache_dir.mkdir(parents=True, exist_ok=True)
        self.name = model_name
        self._model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir))
        self._lock = threading.Lock()

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        with self._lock:
            vectors = list(self._model.embed(texts, batch_size=32))
        return _normalize_rows(np.array(vectors))

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        with self._lock:
            vectors = list(self._model.query_embed(texts))
        return _normalize_rows(np.array(vectors))


_embedder_lock = threading.Lock()
_embedders: dict[tuple[str, str], Embedder | None] = {}


def get_embedder(backend: str, model_name: str, cache_dir: Path) -> Embedder | None:
    """Return a shared embedder, or None for keyword-only retrieval (backend 'none' or load failure)."""
    if backend == "none":
        return None
    key = (backend, model_name)
    with _embedder_lock:
        if key not in _embedders:
            try:
                if backend != "fastembed":
                    raise ValueError(f"unknown embedding backend {backend!r}")
                _embedders[key] = FastEmbedEmbedder(model_name, cache_dir)
            except Exception as exc:  # retrieval still works with BM25 alone
                log.warning("Embeddings unavailable (%s); falling back to keyword-only retrieval.", exc)
                _embedders[key] = None
        return _embedders[key]


@dataclass
class Hit:
    chunk: Chunk
    score: float
    keyword_rank: int | None = None
    semantic_rank: int | None = None


def reciprocal_rank_fusion(rankings: list[list[int]], weights: list[float] | None = None) -> dict[int, float]:
    fused: dict[int, float] = defaultdict(float)
    for i, ranking in enumerate(rankings):
        weight = weights[i] if weights else 1.0
        for rank, doc_id in enumerate(ranking):
            fused[doc_id] += weight / (RRF_K + rank + 1)
    return dict(fused)


class HybridIndex:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray | None = None, embedder: Embedder | None = None):
        self.chunks = chunks
        self.embeddings = embeddings if embedder is not None else None
        self.embedder = embedder
        self._bm25 = BM25([tokenize(c.index_text()) for c in chunks])

    @property
    def uses_embeddings(self) -> bool:
        return self.embeddings is not None and self.embedder is not None

    def search(self, queries: str | Sequence[str], k: int = 8, weights: Sequence[float] | None = None) -> list[Hit]:
        """Search with one or more queries; all rankings are fused into a single top-k list."""
        if isinstance(queries, str):
            queries = [queries]
        if not self.chunks or not queries:
            return []
        query_weights = list(weights) if weights else [1.0] * len(queries)
        if len(query_weights) != len(queries):
            raise ValueError("weights must have one entry per query")

        rankings: list[list[int]] = []
        ranking_weights: list[float] = []
        best_keyword: dict[int, int] = {}
        best_semantic: dict[int, int] = {}

        def record(best: dict[int, int], order: list[int]) -> None:
            for rank, doc_id in enumerate(order, start=1):
                best[doc_id] = min(best.get(doc_id, rank), rank)

        for query, weight in zip(queries, query_weights, strict=True):
            scores = self._bm25.scores(tokenize(query))
            order = [int(i) for i in np.argsort(-scores, kind="stable")[:CANDIDATES_PER_RANKING] if scores[i] > 0]
            rankings.append(order)
            ranking_weights.append(weight)
            record(best_keyword, order)

        if self.uses_embeddings:
            query_vectors = self.embedder.embed_queries(list(queries))  # type: ignore[union-attr]
            similarities = query_vectors @ self.embeddings.T  # type: ignore[union-attr]
            for row, weight in zip(similarities, query_weights, strict=True):
                order = [int(i) for i in np.argsort(-row, kind="stable")[:CANDIDATES_PER_RANKING]]
                rankings.append(order)
                ranking_weights.append(weight)
                record(best_semantic, order)

        fused = reciprocal_rank_fusion(rankings, ranking_weights)
        top = sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:k]
        return [Hit(self.chunks[i], score, best_keyword.get(i), best_semantic.get(i)) for i, score in top]
