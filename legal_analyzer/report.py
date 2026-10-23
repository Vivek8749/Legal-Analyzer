"""Render an AnalysisReport as Markdown or as a self-contained HTML page."""

from __future__ import annotations

from html import escape

from .models import AnalysisReport, Finding, format_pages

SEVERITY_LABEL = {"critical": "Critical", "high": "High", "medium": "Medium", "low": "Low"}
ISSUE_LABEL = {
    "one_sided_term": "One-sided term",
    "loophole": "Loophole",
    "waived_right": "Waived right",
    "ambiguity": "Ambiguity",
    "hidden_cost_or_obligation": "Hidden cost or obligation",
    "missing_protection": "Missing protection",
}
FAVORS_LABEL = {
    "drafter": "Favours the drafter",
    "reviewed_party": "Favours you",
    "balanced": "Balanced",
    "unclear": "Unclear",
}


PROVIDER_NAMES = {"anthropic": "Claude", "openai": "OpenAI", "gemini": "Gemini", "ollama": "Ollama, local"}


def model_description(report: AnalysisReport) -> str:
    """'claude-opus-5-5 (Claude, effort high)' or 'qwen3.5:4b (Ollama, local)'."""
    details = [PROVIDER_NAMES.get(report.provider, report.provider)]
    if report.effort:
        details.append(f"effort {report.effort}")
    return f"{report.model} ({', '.join(details)})"


def _cost_text(cost: float | None) -> str:
    if cost is None:
        return ""
    return ", no API cost (local model)" if cost == 0 else f", estimated cost ${cost:.2f}"


def _counts_line(report: AnalysisReport) -> str:
    counts = report.risk.counts
    return " · ".join(f"{counts.get(s, 0)} {s}" for s in ("critical", "high", "medium", "low"))


def _verification(f: Finding) -> str:
    return "quote verified" if f.verified else "quote not found in source; check manually"


# --- Markdown -----------------------------------------------------------------------------

def to_markdown(report: AnalysisReport) -> str:
    s = report.summary
    doc = report.document
    out: list[str] = [f"# Legal document analysis: {doc.filename}", "", f"> {report.disclaimer}", ""]
    out.append(f"**Risk score:** {report.risk.score}/100 ({report.risk.level}) · {_counts_line(report)}")
    if s:
        parties = "; ".join(f"{p.name} ({p.role})" for p in s.parties) or "not identified"
        out.append(f"**Document type:** {s.document_type or 'unknown'} · **Parties:** {parties}")
        out.append(f"**Favours:** {s.favoured_party or 'n/a'} · **Governing law:** {s.governing_law or 'not stated'}")
    out.append(f"**Reviewed for:** {report.perspective}" + (f" · **Jurisdiction:** {report.jurisdiction}" if report.jurisdiction else ""))
    out.append(f"**Model:** {model_description(report)} · **Date:** {report.created_at}")
    out.append("")

    if s:
        out += ["## Summary", "", f"**{s.headline}**", "", s.what_this_document_does, "", s.overall_assessment, ""]
        by_id = {f.id: f for f in report.findings}
        if s.top_concern_ids:
            out += ["### Top concerns", ""]
            for n, fid in enumerate(s.top_concern_ids, 1):
                f = by_id[fid]
                out.append(f"{n}. **{f.id}: {f.title}** ({f.severity}, {f.location})")
            out.append("")
        if s.recommendations:
            out += ["### Recommendations", ""] + [f"- {r}" for r in s.recommendations] + [""]
        if s.questions_to_ask:
            out += ["### Questions to ask", ""] + [f"- {q}" for q in s.questions_to_ask] + [""]

    out += ["## Findings", ""]
    if not report.findings:
        out += ["No problematic terms were flagged.", ""]
    for f in report.findings:
        out.append(f"### {f.id} · {SEVERITY_LABEL[f.severity]} · {f.category}: {f.title}")
        meta = [f.location]
        if f.section:
            meta.append(f.section)
        meta += [ISSUE_LABEL.get(f.issue_type, f.issue_type), FAVORS_LABEL.get(f.favors, f.favors), _verification(f)]
        out += [f"*{' · '.join(meta)}*", "", f"> {f.quote}", ""]
        out.append(f"**In plain English:** {f.plain_english}")
        out.append("")
        out.append(f"**Why it matters:** {f.why_it_matters}")
        out.append("")
        if f.loophole_scenario:
            out += [f"**How it could be used against you:** {f.loophole_scenario}", ""]
        if f.what_to_do:
            out += [f"**What you can do:** {f.what_to_do}", ""]

    if report.gaps:
        out += ["## Missing protections", "", "*Based on the excerpts reviewed; confirm against the full document.*", ""]
        out += [f"- **{g.category}:** {g.observation}" for g in report.gaps] + [""]

    if report.glossary:
        out += ["## Glossary", ""]
        for t in report.glossary:
            where = f" ({format_pages(t.pages)})" if t.pages else ""
            out.append(f"- **{t.term}**{where}: {t.definition}")
        out.append("")

    out += ["## About this analysis", ""]
    r = report.retrieval
    out.append(f"- Retrieval: {r.context_chunks} of {r.total_chunks} chunks used as context "
               f"({'hybrid keyword + semantic' if r.embeddings else 'keyword-only'} search).")
    u = report.usage
    cost = _cost_text(u.estimated_cost_usd)
    out.append(f"- Claude usage: {u.requests} requests, {u.input_tokens:,} input tokens, "
               f"{u.cache_read_input_tokens:,} cached input tokens read, {u.cache_creation_input_tokens:,} written, "
               f"{u.output_tokens:,} output tokens{cost}. Took {report.duration_seconds:.0f}s.")
    out.append(f"- Risk score: {report.risk.method}")
    for error in report.errors:
        out.append(f"- Warning: {error}")
    out.append("")
    return "\n".join(out)


# --- HTML ---------------------------------------------------------------------------------

_CSS = """
:root { --bg:#f6f7f9; --card:#fff; --text:#1d2433; --muted:#5b6475; --border:#e3e6ec; --accent:#2f5bea;
  --critical:#b42318; --high:#c4320a; --medium:#b54708; --low:#475467; --ok:#067647; --warn:#b54708; }
@media (prefers-color-scheme: dark) { :root { --bg:#0f1218; --card:#171b23; --text:#e7eaf0; --muted:#9aa3b2;
  --border:#2a303c; --accent:#7c9cff; --critical:#f97066; --high:#fb6514; --medium:#fdb022; --low:#98a2b3;
  --ok:#47cd89; --warn:#fdb022; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 920px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 26px; margin: 0 0 4px; } h2 { font-size: 20px; margin: 36px 0 12px; } h3 { font-size: 16px; margin: 0; }
.muted { color: var(--muted); } .small { font-size: 13px; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 18px 20px; margin: 12px 0; }
.card > :first-child { margin-top: 0; } .card > :last-child { margin-bottom: 0; }
.disclaimer { border-left: 4px solid var(--warn); }
.score { display:flex; gap:20px; align-items:center; flex-wrap:wrap; }
.score .num { font-size: 44px; font-weight: 700; line-height: 1; }
.pill { display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: 12px; font-weight: 600;
  border: 1px solid currentColor; margin-right: 6px; }
.sev-critical { color: var(--critical); } .sev-high { color: var(--high); } .sev-medium { color: var(--medium); }
.sev-low { color: var(--low); } .ok { color: var(--ok); } .warn { color: var(--warn); }
.finding { border-left: 4px solid var(--border); }
.finding.critical { border-left-color: var(--critical); } .finding.high { border-left-color: var(--high); }
.finding.medium { border-left-color: var(--medium); } .finding.low { border-left-color: var(--low); }
blockquote { margin: 12px 0; padding: 10px 14px; background: var(--bg); border-radius: 8px; font-family: Georgia, serif; }
dl { display:grid; grid-template-columns: max-content 1fr; gap: 6px 14px; margin: 0; }
dt { font-weight: 600; } dd { margin: 0; }
@media (max-width: 640px) { dl { grid-template-columns: 1fr; } dd { margin-bottom: 6px; } }
ul, ol { padding-left: 22px; } li { margin: 4px 0; }
@media print { body { background: #fff; } .card { break-inside: avoid; } }
"""


def _finding_html(f: Finding) -> str:
    meta = [escape(f.location)]
    if f.section:
        meta.append(escape(f.section))
    meta.append(escape(ISSUE_LABEL.get(f.issue_type, f.issue_type)))
    meta.append(escape(FAVORS_LABEL.get(f.favors, f.favors)))
    check = '<span class="ok">✓ quote verified</span>' if f.verified else '<span class="warn">⚠ quote not found in source</span>'
    rows = [("In plain English", f.plain_english), ("Why it matters", f.why_it_matters)]
    if f.loophole_scenario:
        rows.append(("How it could be used against you", f.loophole_scenario))
    if f.what_to_do:
        rows.append(("What you can do", f.what_to_do))
    details = "".join(f"<dt>{escape(k)}</dt><dd>{escape(v)}</dd>" for k, v in rows)
    return (
        f'<section class="card finding {f.severity}" id="{f.id}">'
        f'<div><span class="pill sev-{f.severity}">{SEVERITY_LABEL[f.severity]}</span>'
        f'<span class="muted small">{escape(f.id)} · {escape(f.category)}</span></div>'
        f"<h3 style='margin-top:8px'>{escape(f.title)}</h3>"
        f'<div class="muted small">{" · ".join(meta)} · {check}</div>'
        f"<blockquote>{escape(f.quote)}</blockquote><dl>{details}</dl></section>"
    )


def to_html(report: AnalysisReport) -> str:
    s = report.summary
    doc = report.document
    parts: list[str] = []
    parts.append(f"<h1>Legal document analysis</h1><div class='muted'>{escape(doc.filename)} · {doc.page_count} pages · "
                 f"analysed {escape(report.created_at)}</div>")
    parts.append(f"<div class='card disclaimer small'>{escape(report.disclaimer)}</div>")

    level_class = {"low": "sev-low", "moderate": "sev-medium", "high": "sev-high", "severe": "sev-critical"}[report.risk.level]
    counts = "".join(
        f'<span class="pill sev-{k}">{report.risk.counts.get(k, 0)} {k}</span>' for k in ("critical", "high", "medium", "low")
    )
    parts.append(
        f"<div class='card score'><div><div class='num {level_class}'>{report.risk.score}</div>"
        f"<div class='muted small'>risk score / 100</div></div><div><div><strong class='{level_class}'>"
        f"{report.risk.level.title()} risk</strong></div><div style='margin-top:6px'>{counts}</div></div></div>"
    )

    if s:
        parties = "; ".join(f"{escape(p.name)} ({escape(p.role)})" for p in s.parties) or "not identified"
        facts = [("Document type", escape(s.document_type or "unknown")), ("Parties", parties),
                 ("Favours", escape(s.favoured_party or "n/a")), ("Governing law", escape(s.governing_law or "not stated")),
                 ("Reviewed for", escape(report.perspective))]
        if report.jurisdiction:
            facts.append(("Jurisdiction", escape(report.jurisdiction)))
        fact_rows = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in facts)
        parts.append(f"<h2>Summary</h2><div class='card'><p><strong>{escape(s.headline)}</strong></p>"
                     f"<p>{escape(s.what_this_document_does)}</p><p>{escape(s.overall_assessment)}</p>"
                     f"<dl>{fact_rows}</dl></div>")
        by_id = {f.id: f for f in report.findings}
        if s.top_concern_ids:
            items = "".join(
                f"<li><a href='#{fid}'>{escape(fid)}</a>: {escape(by_id[fid].title)} "
                f"<span class='muted small'>({by_id[fid].severity}, {escape(by_id[fid].location)})</span></li>"
                for fid in s.top_concern_ids
            )
            parts.append(f"<div class='card'><h3>Top concerns</h3><ol>{items}</ol></div>")
        if s.recommendations:
            parts.append("<div class='card'><h3>Recommendations</h3><ul>"
                         + "".join(f"<li>{escape(r)}</li>" for r in s.recommendations) + "</ul></div>")
        if s.questions_to_ask:
            parts.append("<div class='card'><h3>Questions to ask</h3><ul>"
                         + "".join(f"<li>{escape(q)}</li>" for q in s.questions_to_ask) + "</ul></div>")

    parts.append(f"<h2>Findings ({len(report.findings)})</h2>")
    parts.append("".join(_finding_html(f) for f in report.findings) or "<div class='card'>No problematic terms were flagged.</div>")

    if report.gaps:
        items = "".join(f"<li><strong>{escape(g.category)}:</strong> {escape(g.observation)}</li>" for g in report.gaps)
        parts.append(f"<h2>Missing protections</h2><div class='card'><p class='muted small'>Based on the excerpts "
                     f"reviewed; confirm against the full document.</p><ul>{items}</ul></div>")

    if report.glossary:
        rows = "".join(
            f"<dt>{escape(t.term)}</dt><dd>{escape(t.definition)}"
            + (f" <span class='muted small'>({escape(format_pages(t.pages))})</span>" if t.pages else "") + "</dd>"
            for t in report.glossary
        )
        parts.append(f"<h2>Glossary</h2><div class='card'><dl>{rows}</dl></div>")

    u, r = report.usage, report.retrieval
    cost = _cost_text(u.estimated_cost_usd)
    about = [
        f"Retrieval: {r.context_chunks} of {r.total_chunks} chunks used as context "
        f"({'hybrid keyword + semantic' if r.embeddings else 'keyword-only'} search).",
        f"Model: {model_description(report)}. {u.requests} requests, {u.input_tokens:,} input tokens, "
        f"{u.cache_read_input_tokens:,} cached tokens read, {u.output_tokens:,} output tokens{cost}. "
        f"Took {report.duration_seconds:.0f}s.",
        f"Risk score: {report.risk.method}",
    ] + [f"Warning: {e}" for e in report.errors]
    parts.append("<h2>About this analysis</h2><div class='card small'><ul>"
                 + "".join(f"<li>{escape(a)}</li>" for a in about) + "</ul></div>")

    title = escape(f"Analysis: {doc.filename}")
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{title}</title><style>{_CSS}</style></head><body><main>{''.join(parts)}</main></body></html>"
    )
