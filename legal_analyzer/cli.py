"""Command-line interface: ``legal-analyzer <command>`` (or ``python -m legal_analyzer``)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .config import EFFORT_LEVELS, PROVIDER_LABELS, PROVIDERS, Settings
from .models import AnalysisReport
from .pdf_loader import DocumentError
from .store import DocumentNotFound, DocumentRecord, DocumentStore

console = Console()
SEVERITY_STYLE = {"critical": "bold red", "high": "red", "medium": "yellow", "low": "dim"}
LEVEL_STYLE = {"low": "green", "moderate": "yellow", "high": "red", "severe": "bold red"}


def _settings(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env()
    return settings.with_overrides(
        analysis_effort=getattr(args, "effort", None),
        data_dir=Path(args.data_dir) if getattr(args, "data_dir", None) else None,
    )


def _llm(args: argparse.Namespace, settings: Settings):
    """The model client chosen by --provider/--model (or the configuration), checked for readiness."""
    from .llm import create_llm

    llm = create_llm(settings, getattr(args, "provider", None), getattr(args, "model", None))
    problem = llm.status()
    if problem:
        raise SystemExit(f"error: {llm.label} is not ready. {problem}")
    return llm


def _resolve(store: DocumentStore, target: str) -> DocumentRecord:
    """Accept a file path (ingested on first use) or the ID of an already-ingested document."""
    path = Path(target)
    if path.exists():
        with console.status(f"Indexing {path.name} (the first run downloads a ~70 MB embedding model)…"):
            return store.ingest(path)
    try:
        return store.get(target)
    except DocumentNotFound:
        raise SystemExit(f"error: {target!r} is neither a file nor the ID of an ingested document.") from None


def _print_info(record: DocumentRecord) -> None:
    info = record.info
    search = "hybrid (keyword + semantic)" if record.index.uses_embeddings else "keyword-only"
    console.print(f"[bold]{info.filename}[/bold]  id=[cyan]{info.id}[/cyan]  pages={info.page_count}  "
                  f"chunks={info.chunk_count}  search={search}")
    for warning in info.warnings:
        console.print(f"[yellow]warning:[/yellow] {warning}")


# --- commands -----------------------------------------------------------------------------

def cmd_ingest(args: argparse.Namespace) -> None:
    store = DocumentStore(_settings(args))
    _print_info(_resolve(store, args.document))


def cmd_list(args: argparse.Namespace) -> None:
    store = DocumentStore(_settings(args))
    table = Table("ID", "File", "Pages", "Chunks", "Analysed", "Added")
    for info in store.list_documents():
        analysed = "yes" if store.load_report(info.id) else ""
        table.add_row(info.id, info.filename, str(info.page_count), str(info.chunk_count), analysed, info.created_at)
    console.print(table)


def cmd_search(args: argparse.Namespace) -> None:
    store = DocumentStore(_settings(args))
    record = _resolve(store, args.document)
    for rank, hit in enumerate(record.index.search(args.query, k=args.k), start=1):
        ranks = f"keyword #{hit.keyword_rank or '-'} · semantic #{hit.semantic_rank or '-'}"
        title = f"{rank}. C{hit.chunk.id} · {hit.chunk.location} · {hit.chunk.section or 'no section'}"
        console.print(Panel(hit.chunk.text, title=title, subtitle=ranks, title_align="left"))


def _print_report(report: AnalysisReport) -> None:
    risk = report.risk
    style = LEVEL_STYLE[risk.level]
    counts = "  ".join(f"[{SEVERITY_STYLE[s]}]{risk.counts.get(s, 0)} {s}[/]" for s in ("critical", "high", "medium", "low"))
    header = f"[{style}]Risk score {risk.score}/100 ({risk.level})[/]   {counts}"
    s = report.summary
    body = header
    if s:
        body += f"\n\n[bold]{s.headline}[/bold]\n\n{s.what_this_document_does}\n\n{s.overall_assessment}"
    console.print(Panel(body, title=f"{report.document.filename}", title_align="left"))

    table = Table("ID", "Severity", "Category", "Finding", "Where", "Quote", show_lines=False)
    for f in report.findings:
        check = "[green]✓[/green]" if f.verified else "[yellow]⚠ unverified[/yellow]"
        table.add_row(f.id, f"[{SEVERITY_STYLE[f.severity]}]{f.severity}[/]", f.category, f.title, f.location, check)
    console.print(table)
    if s and s.recommendations:
        console.print("[bold]Recommendations[/bold]")
        for r in s.recommendations:
            console.print(f"  • {r}")
    for error in report.errors:
        console.print(f"[yellow]warning:[/yellow] {error}")
    u = report.usage
    if u.estimated_cost_usd is None:
        cost = ""
    elif u.estimated_cost_usd == 0:
        cost = " · local model, no API cost"
    else:
        cost = f" · est. ${u.estimated_cost_usd:.2f}"
    console.print(f"[dim]{u.requests} requests · {u.input_tokens:,} in · {u.cache_read_input_tokens:,} cached · "
                  f"{u.output_tokens:,} out{cost} · {report.duration_seconds:.0f}s[/dim]")


def cmd_analyze(args: argparse.Namespace) -> None:
    from .analyzer import AnalysisOptions, Analyzer, DEFAULT_PERSPECTIVE
    from .report import to_html, to_markdown

    settings = _settings(args)
    store = DocumentStore(settings)
    record = _resolve(store, args.document)
    _print_info(record)
    llm = _llm(args, settings)

    options = AnalysisOptions(perspective=args.perspective or DEFAULT_PERSPECTIVE, jurisdiction=args.jurisdiction)
    detail = f" (effort {settings.analysis_effort})" if llm.provider == "anthropic" else ""
    pace = "local models can take a while on a small GPU" if llm.local else "this usually takes a few minutes"
    console.print(f"Analysing with [bold]{llm.label}[/bold]{detail}; {pace}.")
    with console.status("Starting…") as status:
        def progress(event: dict) -> None:
            status.update(f"{event['message']} ({event['completed']}/{event['total']} review passes)")

        report = Analyzer(llm, settings).analyze(record, options, progress)
    store.save_report(report)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(record.info.filename).stem
    paths = {
        "json": out_dir / f"{stem}.analysis.json",
        "md": out_dir / f"{stem}.analysis.md",
        "html": out_dir / f"{stem}.analysis.html",
    }
    paths["json"].write_text(report.model_dump_json(indent=2), encoding="utf-8")
    paths["md"].write_text(to_markdown(report), encoding="utf-8")
    paths["html"].write_text(to_html(report), encoding="utf-8")
    _print_report(report)
    console.print("Reports: " + ", ".join(str(p) for p in paths.values()))


def _ask(session, question: str) -> None:  # session: qa.ChatSession
    buffer: list[str] = []
    with Live(Markdown(""), console=console, refresh_per_second=12) as live:
        def on_text(delta: str) -> None:
            buffer.append(delta)
            live.update(Markdown("".join(buffer)))

        answer = session.ask(question, on_text=on_text)
        live.update(Markdown(answer.text))
    for source in answer.sources:
        section = f" — {source.section}" if source.section else ""
        quote = " … ".join(source.cited_text)
        quote = quote if len(quote) <= 300 else quote[:297] + "…"
        console.print(f"[cyan][{source.number}][/cyan] {source.location}{section}\n    [dim]“{quote}”[/dim]")


def _chat_session(args: argparse.Namespace):
    from .qa import ChatSession

    settings = _settings(args)
    store = DocumentStore(settings)
    record = _resolve(store, args.document)
    llm = _llm(args, settings)
    return record, ChatSession(record, llm, settings)


def cmd_ask(args: argparse.Namespace) -> None:
    _, session = _chat_session(args)
    _ask(session, args.question)


def cmd_chat(args: argparse.Namespace) -> None:
    record, session = _chat_session(args)
    _print_info(record)
    console.print(f"Model: [bold]{session.llm.label}[/bold]")
    console.print("Ask questions about the document. Type [bold]exit[/bold] or press Ctrl-D to quit.\n")
    while True:
        try:
            question = console.input("[bold green]you>[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if question.lower() in {"exit", "quit", ":q"}:
            break
        if question:
            _ask(session, question)
            console.print()


def cmd_models(args: argparse.Namespace) -> None:
    from .llm import create_llm
    from .providers.ollama_local import OllamaLLM

    settings = _settings(args)
    table = Table("Provider", "Model", "Status")
    for provider in PROVIDERS:
        llm = create_llm(settings, provider)
        problem = llm.status()
        models = llm.model
        if isinstance(llm, OllamaLLM) and problem is None:
            models = ", ".join(llm.installed_models()) or llm.model
        default = " [dim](default)[/dim]" if provider == settings.provider else ""
        state = f"[yellow]{problem}[/yellow]" if problem else "[green]ready[/green]"
        table.add_row(PROVIDER_LABELS[provider] + default, models, state)
    console.print(table)
    console.print("[dim]Choose with --provider/--model, or LEGAL_ANALYZER_PROVIDER and LEGAL_ANALYZER_<PROVIDER>_MODEL.[/dim]")


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from .web.app import create_app

    settings = _settings(args)
    console.print(f"Legal Document Analyzer running at [bold]http://{args.host}:{args.port}[/bold] "
                  f"(default provider: {PROVIDER_LABELS[settings.provider]}; switch models in the page header)")
    uvicorn.run(create_app(settings), host=args.host, port=args.port, log_level="warning")


# --- entry point --------------------------------------------------------------------------

def _add_model_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--provider", choices=PROVIDERS,
                   help="anthropic (Claude), openai, gemini, or ollama for a local model (default: anthropic)")
    p.add_argument("--model", help="model ID for that provider, e.g. claude-opus-5-5, gpt-5, or qwen3.5:4b")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="legal-analyzer",
        description="RAG-based legal document analyzer: finds one-sided terms and loopholes and explains them.",
    )
    parser.add_argument("--data-dir", help="where indexes and reports are stored (default: ./data)")
    parser.add_argument("-v", "--verbose", action="store_true", help="show debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="extract, chunk, and index a document (no API key needed)")
    p.add_argument("document", help="path to a PDF/TXT file")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("list", help="list ingested documents")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("search", help="show the clauses retrieval finds for a query (no API key needed)")
    p.add_argument("document", help="file path or document ID")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5, help="number of results (default 5)")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("analyze", help="find biased terms and loopholes and write a report")
    p.add_argument("document", help="file path or document ID")
    p.add_argument("--perspective", help="whose interests to review for, e.g. 'the tenant' or 'the freelancer'")
    p.add_argument("--jurisdiction", help="your country/state, used for context, e.g. 'California, USA'")
    p.add_argument("--effort", choices=EFFORT_LEVELS, help="Claude effort for the review passes (default: high)")
    _add_model_options(p)
    p.add_argument("--out", default="reports", help="directory for JSON/Markdown/HTML reports (default: ./reports)")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("ask", help="ask one question; the answer cites the clauses it relies on")
    p.add_argument("document", help="file path or document ID")
    p.add_argument("question")
    _add_model_options(p)
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("chat", help="interactive Q&A about a document")
    p.add_argument("document", help="file path or document ID")
    _add_model_options(p)
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("models", help="show which model providers are ready (and your local Ollama models)")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("serve", help="start the web interface")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    from .analyzer import AnalysisError
    from .llm import LLMError

    try:
        args.func(args)
    except (DocumentError, LLMError, AnalysisError, ValueError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
