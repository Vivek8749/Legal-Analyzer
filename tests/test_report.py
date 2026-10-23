from legal_analyzer.analyzer import Analyzer
from legal_analyzer.report import to_html, to_markdown
from legal_analyzer.schemas import PassOutput

from .conftest import FakeLLM, make_finding


def _report(sample_record, settings, **finding_overrides):
    outputs = {g: PassOutput(findings=[], key_terms=[], gaps=[]) for g in
               ("changes", "liability", "disputes", "money", "data_ip", "obligations", "drafting")}
    outputs["liability"] = PassOutput(findings=[make_finding(**finding_overrides)], key_terms=[], gaps=[])
    return Analyzer(FakeLLM(pass_outputs=outputs), settings).analyze(sample_record)


def test_markdown_report_sections(sample_record, settings):
    md = to_markdown(_report(sample_record, settings))
    for heading in ("# Legal document analysis", "## Summary", "## Findings", "## About this analysis"):
        assert heading in md
    assert "Liability capped at $10" in md and "quote verified" in md


def test_html_report_escapes_untrusted_text(sample_record, settings):
    html = to_html(_report(sample_record, settings, title="<script>alert(1)</script>"))
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
