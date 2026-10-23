"""The analysis pipeline: retrieve → review passes with a language model → ground quotes → summarise.

1. Retrieval. For every risk category, hybrid search pulls the clauses most likely to
   matter. The union of those excerpts (plus the document's opening, which names the
   parties) becomes the shared context for the review.
2. Review passes. Categories are grouped into themed passes that run in parallel. All
   passes share a byte-identical prefix (system prompt + excerpts): with Claude the first
   pass writes it to the prompt cache and the rest read it at a fraction of the price; a
   local Ollama model reuses its KV cache for the shared prefix. Each pass returns
   schema-validated findings via structured outputs. The excerpts are sized to fit the
   model's context window, so small local models get the most relevant clauses only.
3. Grounding. Every quoted clause is matched back to the source text; findings whose
   quote can't be found are flagged as unverified.
4. Synthesis. A final call writes the executive summary from the verified findings.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from .config import Settings
from .grounding import match_quote, normalize
from .llm import LLM, FatalLLMError, LLMError, PromptBlock, Usage
from .models import (
    SEVERITY_ORDER,
    AnalysisReport,
    Chunk,
    ExecutiveSummary,
    Finding,
    Gap,
    GlossaryTerm,
    Party,
    RetrievalStats,
    RiskScore,
)
from .prompts import (
    ANALYSIS_SYSTEM,
    SYNTHESIS_SYSTEM,
    render_document_block,
    render_excerpt,
    render_pass_task,
    render_synthesis_request,
)
from .schemas import PassOutput, SynthesisOutput
from .store import DocumentRecord
from .taxonomy import CATEGORIES, CATEGORY_BY_ID, GROUP_BY_ID, GROUPS, OVERVIEW_QUERIES, Group, categories_in

log = logging.getLogger(__name__)

DEFAULT_PERSPECTIVE = "the party who did not draft this document (for example the customer, user, tenant, or employee)"

# Rough size of a token in legal English, kept low so prompts stay inside the context window.
CHARS_PER_TOKEN = 3.2
# System prompt, review context, and per-pass task, in tokens (with margin).
PROMPT_OVERHEAD_TOKENS = 3_000

SEVERITY_WEIGHTS = {"critical": 25, "high": 12, "medium": 5, "low": 1}
RISK_METHOD = (
    "Score = 100 × (1 − e^(−W/60)), where W adds up finding weights (critical 25, high 12, medium 5, low 1). "
    "Findings whose quote could not be verified count half; findings that are balanced or favour the reviewed "
    "party count zero."
)

ProgressCallback = Callable[[dict[str, Any]], None]


class AnalysisError(RuntimeError):
    pass


@dataclass
class AnalysisOptions:
    perspective: str = DEFAULT_PERSPECTIVE
    jurisdiction: str | None = None
    per_category_k: int = 6
    max_context_chunks: int = 120


@dataclass
class _PassResult:
    group: Group
    output: PassOutput | None = None
    error: str | None = None
    usage: Usage = field(default_factory=Usage)


def compute_risk(findings: list[Finding]) -> RiskScore:
    weight = 0.0
    for f in findings:
        if f.favors in ("balanced", "reviewed_party"):
            continue
        weight += SEVERITY_WEIGHTS[f.severity] * (1.0 if f.verified else 0.5)
    score = round(100 * (1 - math.exp(-weight / 60)))
    level = "low" if score < 25 else "moderate" if score < 50 else "high" if score < 75 else "severe"
    counts = {s: sum(1 for f in findings if f.severity == s) for s in ("critical", "high", "medium", "low")}
    return RiskScore(score=score, level=level, counts=counts, method=RISK_METHOD)


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0


class Analyzer:
    def __init__(self, llm: LLM, settings: Settings):
        self.llm = llm
        self.settings = settings

    # --- public ---------------------------------------------------------------------------

    def analyze(
        self,
        record: DocumentRecord,
        options: AnalysisOptions | None = None,
        progress: ProgressCallback | None = None,
    ) -> AnalysisReport:
        options = options or AnalysisOptions()
        notify = progress or (lambda event: None)
        started = time.monotonic()
        chunk_map = {c.id: c for c in record.chunks}

        notify({"stage": "retrieval", "message": "Retrieving relevant clauses", "completed": 0, "total": len(GROUPS)})
        ranked = {
            category.id: [hit.chunk.id for hit in record.index.search(category.queries, k=options.per_category_k)]
            for category in CATEGORIES
        }
        opening_ids = [c.id for c in record.chunks[:3]]
        overview_ids = opening_ids + [h.chunk.id for h in record.index.search(OVERVIEW_QUERIES, k=4)]
        budget = self._context_budget_chars()
        context_ids = self._select_context(ranked, overview_ids, options.max_context_chunks, chunk_map, budget)
        context = [chunk_map[i] for i in sorted(context_ids)]
        # Only point the passes at excerpts that made it into the context.
        ranked_in_context = {cat: [i for i in ids if i in context_ids] for cat, ids in ranked.items()}

        document_block = render_document_block(
            record.info.filename, options.perspective, options.jurisdiction, context, len(record.chunks)
        )
        results = self._run_passes(document_block, ranked_in_context, notify)

        usage = Usage()
        errors: list[str] = []
        for result in results:
            usage.add(result.usage)
            if result.error:
                errors.append(f"{result.group.name}: {result.error}")
        if all(r.output is None for r in results):
            raise AnalysisError("All review passes failed. " + " | ".join(errors))

        findings, glossary, gaps = self._assemble(results, chunk_map)

        summary = None
        notify({"stage": "synthesis", "message": "Writing the summary", "completed": len(GROUPS), "total": len(GROUPS)})
        try:
            opening = self._fit([chunk_map[i] for i in sorted(set(overview_ids))], budget * 0.3)
            summary, synthesis_usage = self._synthesize(record, options, opening, findings, gaps, budget * 0.5)
            usage.add(synthesis_usage)
        except FatalLLMError:
            raise
        except LLMError as exc:
            errors.append(f"Summary: {exc}")

        report = AnalysisReport(
            document=record.info,
            perspective=options.perspective,
            jurisdiction=options.jurisdiction,
            provider=self.llm.provider,
            model=self.llm.model,
            effort=self.settings.analysis_effort if self.llm.provider == "anthropic" else None,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            duration_seconds=round(time.monotonic() - started, 1),
            summary=summary,
            risk=compute_risk(findings),
            findings=findings,
            glossary=glossary,
            gaps=gaps,
            retrieval=RetrievalStats(
                context_chunks=len(context),
                total_chunks=len(record.chunks),
                per_category=ranked,
                embeddings=record.index.uses_embeddings,
            ),
            usage=usage.summary(self.llm.model, local=self.llm.local),
            errors=errors,
        )
        notify({"stage": "done", "message": "Analysis complete", "completed": len(GROUPS), "total": len(GROUPS)})
        return report

    # --- retrieval ------------------------------------------------------------------------

    def _context_budget_chars(self) -> int:
        """How much excerpt text fits in the model's context next to the prompt and the answer."""
        tokens = self.llm.context_window - self.llm.max_output_tokens - PROMPT_OVERHEAD_TOKENS
        return int(max(tokens, 1_500) * CHARS_PER_TOKEN)

    @staticmethod
    def _select_context(ranked: dict[str, list[int]], overview_ids: list[int], limit: int,
                        chunk_map: dict[int, Chunk], budget_chars: int) -> set[int]:
        """Union of the per-category results; if too large, keep the most relevant chunks."""
        priority: dict[int, float] = defaultdict(float)
        for ids in ranked.values():
            for rank, chunk_id in enumerate(ids):
                priority[chunk_id] += 1 / (rank + 1)
        for chunk_id in overview_ids:
            priority[chunk_id] += 10  # the opening names the parties: always keep it
        selected: set[int] = set()
        used = 0
        for chunk_id in sorted(priority, key=lambda i: (-priority[i], i)):
            size = len(render_excerpt(chunk_map[chunk_id])) + 2
            if len(selected) >= limit or (selected and used + size > budget_chars):
                continue
            selected.add(chunk_id)
            used += size
        return selected

    @staticmethod
    def _fit(chunks: list[Chunk], budget_chars: float) -> list[Chunk]:
        kept, used = [], 0
        for chunk in chunks:
            size = len(render_excerpt(chunk)) + 2
            if kept and used + size > budget_chars:
                break
            kept.append(chunk)
            used += size
        return kept

    # --- review passes --------------------------------------------------------------------

    def _run_pass(self, group: Group, document_block: str, ranked: dict[str, list[int]],
                  on_stream_start: Callable[[], None] | None) -> _PassResult:
        blocks = [
            # Identical across passes: the end of the shared, cacheable prefix.
            PromptBlock(document_block, cacheable=True),
            PromptBlock(render_pass_task(group, categories_in(group.id), ranked)),
        ]
        result = _PassResult(group)
        try:
            structured = self.llm.structured(
                system=ANALYSIS_SYSTEM,
                blocks=blocks,
                output_type=PassOutput,
                effort=self.settings.analysis_effort,
                on_stream_start=on_stream_start,
            )
            result.output, result.usage = structured.output, structured.usage
        except FatalLLMError:
            raise
        except LLMError as exc:
            log.warning("Review pass %s failed: %s", group.id, exc)
            result.error = str(exc)
        return result

    def _run_passes(self, document_block: str, ranked: dict[str, list[int]], notify: ProgressCallback) -> list[_PassResult]:
        groups = list(GROUPS)
        total = len(groups)
        done = 0
        done_lock = threading.Lock()
        prefix_cached = threading.Event()

        def run(group: Group, signal: Callable[[], None] | None) -> _PassResult:
            nonlocal done
            try:
                return self._run_pass(group, document_block, ranked, signal)
            finally:
                with done_lock:
                    done += 1
                    notify({"stage": "analysis", "message": f"Reviewed: {group.name}", "completed": done, "total": total})

        notify({"stage": "analysis", "message": f"Reviewing clauses with {self.llm.label}", "completed": 0, "total": total})
        workers = max(1, min(self.settings.concurrency, self.llm.max_concurrency))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # Start one pass and wait until it is streaming: its prompt is then in the cache,
            # so the parallel passes read the shared prefix instead of each writing it.
            # The done-callback also releases the wait if the first pass ends without streaming.
            first: Future[_PassResult] = pool.submit(run, groups[0], prefix_cached.set)
            first.add_done_callback(lambda _: prefix_cached.set())
            prefix_cached.wait()
            if first.done() and first.exception() is not None:
                raise first.exception()  # type: ignore[misc]  # fatal: don't start the other passes
            futures = [first] + [pool.submit(run, group, None) for group in groups[1:]]
            return [future.result() for future in futures]

    # --- assembly -------------------------------------------------------------------------

    def _assemble(self, results: list[_PassResult], chunk_map: dict[int, Chunk]
                  ) -> tuple[list[Finding], list[GlossaryTerm], list[Gap]]:
        candidates: list[Finding] = []
        terms: dict[str, GlossaryTerm] = {}
        gaps: dict[tuple[str, str], Gap] = {}

        for result in results:
            if result.output is None:
                continue
            for item in result.output.findings:
                category = CATEGORY_BY_ID[item.category_id]
                cited = [i for i in dict.fromkeys(item.chunk_ids) if i in chunk_map]
                match = match_quote(item.quote, chunk_map, cited)
                chunk_ids = cited
                if match.chunk_id is not None and match.score >= 0.6:
                    chunk_ids = [match.chunk_id] + [i for i in cited if i != match.chunk_id]
                primary = chunk_map[chunk_ids[0]] if chunk_ids else None
                candidates.append(Finding(
                    id="",
                    category_id=category.id,
                    category=category.name,
                    group=GROUP_BY_ID[category.group].name,
                    title=item.title.strip(),
                    quote=item.quote.strip(),
                    issue_type=item.issue_type,
                    severity=item.severity,
                    favors=item.favors,
                    plain_english=item.plain_english.strip(),
                    why_it_matters=item.why_it_matters.strip(),
                    loophole_scenario=item.loophole_scenario.strip(),
                    what_to_do=item.what_to_do.strip(),
                    confidence=item.confidence,
                    chunk_ids=chunk_ids,
                    pages=primary.pages if primary else [],
                    section=primary.section if primary else None,
                    verified=match.verified,
                    match_score=round(match.score, 3),
                ))
            for term in result.output.key_terms:
                key = " ".join(term.term.lower().split())
                pages = sorted({p for i in term.chunk_ids if i in chunk_map for p in chunk_map[i].pages})
                if key and key not in terms:
                    terms[key] = GlossaryTerm(term=term.term.strip(), definition=term.definition.strip(), pages=pages)
                elif key:
                    terms[key].pages = sorted(set(terms[key].pages) | set(pages))
            for gap in result.output.gaps:
                key = (gap.category_id, " ".join(gap.observation.lower().split()))
                gaps.setdefault(key, Gap(
                    category_id=gap.category_id,
                    category=CATEGORY_BY_ID[gap.category_id].name,
                    observation=gap.observation.strip(),
                ))

        findings = self._dedupe(candidates)
        findings.sort(key=lambda f: (SEVERITY_ORDER[f.severity], not f.verified, f.pages[:1] or [10**6]))
        for number, finding in enumerate(findings, start=1):
            finding.id = f"F{number}"
        glossary = sorted(terms.values(), key=lambda t: t.term.lower())
        return findings, glossary, list(gaps.values())

    @staticmethod
    def _dedupe(findings: list[Finding]) -> list[Finding]:
        kept: list[Finding] = []
        for finding in findings:
            tokens = normalize(finding.quote)
            duplicate = next(
                (k for k in kept if k.category_id == finding.category_id and _jaccard(tokens, normalize(k.quote)) >= 0.7),
                None,
            )
            if duplicate is None:
                kept.append(finding)
            elif (SEVERITY_ORDER[finding.severity], not finding.verified) < (
                SEVERITY_ORDER[duplicate.severity], not duplicate.verified
            ):
                kept[kept.index(duplicate)] = finding
        return kept

    # --- synthesis ------------------------------------------------------------------------

    def _synthesize(self, record: DocumentRecord, options: AnalysisOptions, opening: list[Chunk],
                    findings: list[Finding], gaps: list[Gap], budget_chars: float) -> tuple[ExecutiveSummary, Usage]:
        # Findings are sorted most serious first; small context windows get the top ones.
        shown, used = [], 0
        for finding in findings:
            used += len(finding.title) + len(finding.plain_english) + len(finding.why_it_matters) + 150
            if shown and used > budget_chars:
                break
            shown.append(finding)
        request = render_synthesis_request(
            record.info.filename, options.perspective, options.jurisdiction, opening, shown, gaps
        )
        result = self.llm.structured(
            system=SYNTHESIS_SYSTEM,
            blocks=[PromptBlock(request)],
            output_type=SynthesisOutput,
            effort=self.settings.synthesis_effort,
        )
        out = result.output
        valid_ids = {f.id for f in findings}
        summary = ExecutiveSummary(
            document_type=out.document_type.strip(),
            parties=[Party(name=p.name.strip(), role=p.role.strip()) for p in out.parties],
            favoured_party=out.favoured_party.strip(),
            governing_law=out.governing_law.strip(),
            what_this_document_does=out.what_this_document_does.strip(),
            headline=out.headline.strip(),
            overall_assessment=out.overall_assessment.strip(),
            top_concern_ids=[i for i in dict.fromkeys(out.top_concern_ids) if i in valid_ids][:5],
            recommendations=[r.strip() for r in out.recommendations if r.strip()],
            questions_to_ask=[q.strip() for q in out.questions_to_ask if q.strip()],
        )
        return summary, result.usage
