import re
import threading
import time

import pytest

from legal_analyzer.analyzer import AnalysisError, AnalysisOptions, Analyzer, compute_risk
from legal_analyzer.llm import LLMError, MissingCredentialsError, ModelNotFoundError
from legal_analyzer.models import Finding
from legal_analyzer.taxonomy import GROUPS

from .conftest import FakeLLM


def test_pipeline_assembles_grounded_findings(sample_record, settings):
    llm = FakeLLM()
    events = []
    report = Analyzer(llm, settings).analyze(sample_record, AnalysisOptions(perspective="the customer"), events.append)

    assert [c["kind"] for c in llm.calls].count("pass") == len(GROUPS)
    titles = [f.title for f in report.findings]
    # near-duplicate findings merge, keeping the more severe one
    assert titles.count("Liability capped at $10") == 1
    cap = next(f for f in report.findings if f.title == "Liability capped at $10")
    assert cap.severity == "critical" and cap.verified
    assert cap.pages == [3] and cap.section == "11. LIMITATION OF LIABILITY"
    # a wrong chunk id is corrected by locating the quote
    indemnity = next(f for f in report.findings if f.category_id == "indemnification")
    assert indemnity.verified and indemnity.section == "12. INDEMNIFICATION"
    # a fabricated quote is flagged, not presented as verified
    assert not next(f for f in report.findings if f.title == "Fabricated quote").verified
    assert report.findings[0].id == "F1" and report.findings[0].severity == "critical"
    assert [t.term for t in report.glossary] == ["Indemnify"]
    assert report.gaps[0].category == "Governing law & venue"
    assert report.summary.top_concern_ids == ["F1"]  # unknown and duplicate ids dropped
    assert report.risk.counts["critical"] == 1 and report.risk.score > 0
    assert report.usage.requests == len(GROUPS) + 1
    assert report.perspective == "the customer"
    assert events[-1]["stage"] == "done"


def test_passes_share_a_byte_identical_cached_prefix(sample_record, settings):
    llm = FakeLLM()
    Analyzer(llm, settings).analyze(sample_record)
    passes = [c for c in llm.calls if c["kind"] == "pass"]
    assert len({c["blocks"][0].text for c in passes}) == 1
    assert all(c["blocks"][0].cacheable and not c["blocks"][1].cacheable for c in passes)
    assert len({c["blocks"][1].text for c in passes}) == len(GROUPS)
    assert len({c["system"] for c in passes}) == 1


def test_first_pass_is_streaming_before_the_rest_start(sample_record, settings):
    class OrderedLLM(FakeLLM):
        def __init__(self):
            super().__init__()
            self.log: list[str] = []
            self.lock = threading.Lock()

        def structured(self, *, on_stream_start=None, **kwargs):
            with self.lock:
                self.log.append("start")
            if on_stream_start is not None:
                time.sleep(0.2)
                with self.lock:
                    self.log.append("streaming")
                on_stream_start()
            return super().structured(on_stream_start=None, **kwargs)

    llm = OrderedLLM()
    Analyzer(llm, settings).analyze(sample_record)
    assert llm.log[:2] == ["start", "streaming"]
    assert llm.log.count("start") == len(GROUPS) + 1  # passes + synthesis


def test_failed_pass_is_reported_and_analysis_continues(sample_record, settings):
    report = Analyzer(FakeLLM(errors={"money": LLMError("cut off")}), settings).analyze(sample_record)
    assert any(e.startswith("Fees, payments & penalties") for e in report.errors)
    assert report.findings


def test_fatal_error_aborts(sample_record, settings):
    llm = FakeLLM(errors={g.id: MissingCredentialsError("no key") for g in GROUPS})
    with pytest.raises(MissingCredentialsError):
        Analyzer(llm, settings).analyze(sample_record)
    assert [c["kind"] for c in llm.calls] == ["pass"]  # stopped after the first pass


def test_all_passes_failing_raises(sample_record, settings):
    with pytest.raises(AnalysisError):
        Analyzer(FakeLLM(errors={g.id: LLMError("boom") for g in GROUPS}), settings).analyze(sample_record)


def test_summary_failure_keeps_findings(sample_record, settings):
    report = Analyzer(FakeLLM(errors={"synthesis": LLMError("declined")}), settings).analyze(sample_record)
    assert report.summary is None and report.findings
    assert any(e.startswith("Summary") for e in report.errors)


def _finding(severity: str, verified: bool = True, favors: str = "drafter") -> Finding:
    return Finding(id="F1", category_id="x", category="x", group="x", title="t", quote="q", issue_type="loophole",
                   severity=severity, favors=favors, plain_english="", why_it_matters="", verified=verified)


def test_risk_score():
    assert compute_risk([]).score == 0 and compute_risk([]).level == "low"
    severe = compute_risk([_finding("critical")] * 6)
    assert severe.level == "severe" and severe.score <= 100
    assert compute_risk([_finding("high", verified=False)]).score < compute_risk([_finding("high")]).score
    assert compute_risk([_finding("critical", favors="balanced")]).score == 0


def test_fatal_provider_errors_abort(sample_record, settings):
    llm = FakeLLM(errors={g.id: ModelNotFoundError("ollama pull qwen3.5:4b") for g in GROUPS})
    with pytest.raises(ModelNotFoundError):
        Analyzer(llm, settings).analyze(sample_record)


def test_small_context_window_limits_excerpts(sample_record, settings):
    big = FakeLLM()
    Analyzer(big, settings).analyze(sample_record)
    small = FakeLLM(context_window=10_000)  # e.g. a local model with num_ctx 10k
    report = Analyzer(small, settings).analyze(sample_record)

    def excerpts(llm):
        block = next(c for c in llm.calls if c["kind"] == "pass")["blocks"][0].text
        return block.count("<excerpt ")

    assert excerpts(small) < excerpts(big)
    assert len(next(c for c in small.calls if c["kind"] == "pass")["blocks"][0].text) <= (10_000 - 4_000 - 3_000) * 3.2 + 800
    assert report.retrieval.context_chunks == excerpts(small)
    # passes only point at excerpts that are actually in the context
    first_pass = next(c for c in small.calls if c["kind"] == "pass")
    shown = {int(i) for i in re.findall(r'<excerpt id="C(\d+)"', first_pass["blocks"][0].text)}
    pointed = {int(i) for line in re.findall(r"most relevant: (.*)", first_pass["blocks"][1].text)
               for i in re.findall(r"C(\d+)", line)}
    assert pointed and pointed <= shown


def test_report_records_provider_and_local_cost(sample_record, settings):
    llm = FakeLLM()
    llm.provider, llm.local, llm.model = "ollama", True, "qwen3.5:4b"
    report = Analyzer(llm, settings).analyze(sample_record)
    assert report.provider == "ollama" and report.model == "qwen3.5:4b" and report.effort is None
    assert report.usage.estimated_cost_usd == 0.0
