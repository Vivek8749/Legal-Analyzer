# Legal Document Analyzer

A RAG-based analyzer for contracts, terms of service, leases, and employment agreements. It finds
**one-sided (biased) terms, loopholes, waived rights, hidden costs, and vague wording**, explains each
in plain English with the exact quote and page number, builds a glossary of the legal jargon, and
answers follow-up questions with citations to the clauses it relies on.

It runs as a **web app** (upload → analyze → explore findings → ask questions) or as a **CLI**, on a
**frontier model** (Claude, GPT, Gemini, or any OpenAI-compatible API) or a **local model through
Ollama**, so a confidential contract never has to leave your computer.

> Informational only. This is not legal advice. Check quotes against the original and consult a lawyer
> before relying on an analysis.

## How it works

```
PDF ──► text extraction ──► clause-aware chunking ──► hybrid index (BM25 + local embeddings)
         (pypdf, header/footer      (numbered clauses,           │
          & hyphenation cleanup)     headings, page numbers)     │
                                                                 ▼
             21 risk categories ──► retrieval per category ──► shared excerpt context
                                                                 │
           7 review passes with Claude, GPT, Gemini, or a local Ollama model ◄──┘
                         (schema-constrained JSON output)
                                                                 │
             quote grounding (every quote matched back to the source text)
                                                                 │
             summary pass ──► risk score, findings, glossary, missing protections ──► JSON / Markdown / HTML
```

**Retrieval.** Each of 21 risk categories (unilateral changes, termination, auto-renewal, limitation
of liability, indemnification, arbitration and class-action waivers, data sharing, IP licences,
non-competes, vague discretion, and more; see `legal_analyzer/taxonomy.py`) has its own retrieval
queries. Keyword search (BM25 with stemming) and semantic search (local `bge-small` embeddings via
fastembed) are fused with Reciprocal Rank Fusion. The union of the retrieved clauses becomes the
review context, so long documents are reviewed through their most relevant clauses instead of in full.
The context is sized to the model: Claude's 1M-token window takes up to 120 clauses, while a local
model with a 16k window gets only the most relevant ones that fit.

**Review passes.** Categories are grouped into 7 themed passes (in parallel on cloud models, one
after another on a local GPU). Every pass returns findings in a fixed JSON schema that every provider
enforces during decoding: category, quote, severity, which party it favours, a plain-English
explanation, why it matters, a concrete "how this could be used against you" scenario, and what to do
about it. All passes share one prompt prefix, so Claude reads it from the prompt cache and Ollama
reuses its KV cache instead of reprocessing the document seven times.

**Grounding.** The model is told to quote verbatim, and the app checks it: each quote is matched
against the source text. Findings whose quote can't be found are marked **"quote not found in
source"** and count half in the risk score. This matters most for small local models.

**Q&A.** Questions retrieve the relevant clauses and every claim in the answer links to the exact
passage and page. Claude cites natively (`search_result` blocks); other models cite excerpts inline as
`[C12]`, which the app turns into the same clickable citations.

## Choosing a model

| Provider | Setup | Notes |
|---|---|---|
| **Claude** (default) | `ANTHROPIC_API_KEY` | `claude-opus-5-5` by default; Sonnet 5.5 and Haiku 5.5 in the picker. Prompt caching, native citations, adaptive thinking. |
| **OpenAI** | `OPENAI_API_KEY` | `gpt-5` by default; set `LEGAL_ANALYZER_OPENAI_MODEL` to any model with structured outputs. |
| **Gemini** | `GEMINI_API_KEY` | Through Google's OpenAI-compatible endpoint; `gemini-flash-latest` by default. |
| **Any OpenAI-compatible API** | `LEGAL_ANALYZER_OPENAI_BASE_URL` (+ key if needed) | OpenRouter, Groq, DeepSeek, Together, LM Studio, vLLM… |
| **Ollama** (local) | [Install Ollama](https://ollama.com), `ollama pull qwen3.5:4b` | No key, no cost, nothing leaves your machine. |

In the web app, pick the model in the header. The list shows which providers are set up and every
model you have pulled into Ollama, and "Other … model…" accepts any model ID. On the command line,
use `--provider` and `--model`, or set defaults in `.env`. `legal-analyzer models` shows what is ready.

**Local models with Ollama.** Pick a model that fits your GPU. The context window (KV cache) needs
memory on top of the weights:

| GPU memory | Try |
|---|---|
| 6–8 GB | `qwen3.5:4b` (default), `phi4-mini`, `llama3.2:3b` |
| 12–16 GB | 8–14B models such as `qwen3:14b` |
| 24 GB+ | larger models such as `gpt-oss:20b` or 30B-class models |

The app always sets Ollama's context window explicitly (`LEGAL_ANALYZER_OLLAMA_NUM_CTX`, default
16384, capped at what the model supports), because Ollama's own default is small and silently cuts
long prompts. Raise it if you have the memory: the analysis then sees more clauses. Reasoning models
(qwen3.5, deepseek-r1, gpt-oss) think before answering when `LEGAL_ANALYZER_OLLAMA_THINK=true` (or
`low`/`medium`/`high`); this improves results but is much slower, so it is off by default. Expect a
local analysis to take several minutes and to miss subtler issues a frontier model catches. Every
quote is still checked against the document.

## Setup

Requires Python 3.10+, plus either an API key for a cloud provider or [Ollama](https://ollama.com)
for local models.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env        # then add a key (ANTHROPIC_API_KEY=…) or set LEGAL_ANALYZER_PROVIDER=ollama
```

For a fully local setup: install Ollama, run `ollama pull qwen3.5:4b`, and set
`LEGAL_ANALYZER_PROVIDER=ollama` in `.env` (or just pick the model in the web app's header).

The first time a document is indexed, a ~70 MB embedding model is downloaded to `data/models/`.
Indexing and search work without any model; analysis and Q&A need a cloud key or Ollama.

## Usage

### Web app

```bash
legal-analyzer serve            # http://127.0.0.1:8000
```

Upload a PDF, optionally say whose side you're on ("the tenant", "the freelancer") and your
jurisdiction, then click **Analyze document**. You get:

- **Overview**: risk score (0–100), headline, parties, governing law, top concerns, recommendations,
  questions to ask, and missing protections
- **Findings**: filterable cards with the quote, page and section, plain-English explanation, loophole
  scenario, and next steps, with a "View clause in context" link that highlights the passage
- **Glossary**: legal terms used in the document, defined in plain English
- **Ask a question**: multi-turn chat with streamed answers and clickable citations

Reports can be opened or downloaded as HTML and Markdown.

### Command line

```bash
legal-analyzer analyze contract.pdf --perspective "the tenant" --jurisdiction "Ontario, Canada"
legal-analyzer analyze contract.pdf --provider ollama --model qwen3.5:4b   # fully local
legal-analyzer analyze contract.pdf --provider openai --model gpt-5
legal-analyzer ask contract.pdf "Can the landlord keep my deposit?"
legal-analyzer chat contract.pdf                         # interactive Q&A
legal-analyzer search contract.pdf "early termination fee"   # inspect retrieval (no API key)
legal-analyzer ingest contract.pdf                       # index only (no API key)
legal-analyzer list
legal-analyzer models                                    # which providers/models are ready
```

`analyze` writes `reports/<name>.analysis.{json,md,html}`. Documents can be referred to by path or by
the ID shown by `list`. Try everything on `samples/sample_terms_of_service.pdf`, a deliberately
one-sided terms of service for a fictional company (regenerate it with `python samples/make_sample_pdf.py`).

## Configuration

Set in the environment or in `.env` (read from the current directory).

| Variable | Default | Meaning |
|---|---|---|
| `LEGAL_ANALYZER_PROVIDER` | `anthropic` | Default provider: `anthropic`, `openai`, `gemini`, or `ollama` |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY` | (none) | API keys for the cloud providers you use |
| `LEGAL_ANALYZER_ANTHROPIC_MODEL` | `claude-opus-5-5` | Claude model (`LEGAL_ANALYZER_MODEL` overrides the default provider's model) |
| `LEGAL_ANALYZER_OPENAI_MODEL` | `gpt-5` | OpenAI / OpenAI-compatible model |
| `LEGAL_ANALYZER_GEMINI_MODEL` | `gemini-flash-latest` | Gemini model |
| `LEGAL_ANALYZER_OLLAMA_MODEL` | `qwen3.5:4b` | Local model |
| `LEGAL_ANALYZER_OPENAI_BASE_URL` | (OpenAI) | Any OpenAI-compatible endpoint |
| `LEGAL_ANALYZER_OPENAI_REASONING_EFFORT` | (unset) | `reasoning_effort` for OpenAI reasoning models |
| `LEGAL_ANALYZER_CONTEXT_WINDOW` | 128k (OpenAI), 1M (Gemini) | Context window to assume for OpenAI-compatible models |
| `LEGAL_ANALYZER_OLLAMA_HOST` | `http://localhost:11434` | Ollama server (`OLLAMA_HOST` also works) |
| `LEGAL_ANALYZER_OLLAMA_NUM_CTX` | `16384` | Ollama context window in tokens |
| `LEGAL_ANALYZER_OLLAMA_NUM_PREDICT` | `4096` | Longest answer a local model may write |
| `LEGAL_ANALYZER_OLLAMA_THINK` | `false` | `true` / `low` / `medium` / `high` to let reasoning models think |
| `LEGAL_ANALYZER_OLLAMA_PARALLEL` | `1` | Parallel requests to Ollama (each needs its own KV cache) |
| `LEGAL_ANALYZER_EFFORT` | `high` | Claude effort for the review passes (`low` … `max`); also `--effort` |
| `LEGAL_ANALYZER_SYNTHESIS_EFFORT` | `medium` | Claude effort for the summary |
| `LEGAL_ANALYZER_CHAT_EFFORT` | `medium` | Claude effort for Q&A |
| `LEGAL_ANALYZER_CONCURRENCY` | `4` | Review passes run in parallel on cloud models |
| `LEGAL_ANALYZER_EMBEDDINGS` | `fastembed` | `none` for keyword-only retrieval (no model download) |
| `LEGAL_ANALYZER_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Any fastembed text model |
| `LEGAL_ANALYZER_FALLBACKS` | `on` | Server-side refusal fallback (see below) |
| `LEGAL_ANALYZER_DATA_DIR` | `./data` | Where indexes, embeddings, and reports are stored |

**Refusal fallback.** Claude requests opt into the API's server-side fallback (`fallbacks: "default"`,
on models that support it): if a safety classifier declines a request, which is rare for legal text,
the API retries it on Anthropic's recommended fallback model instead of failing. Set
`LEGAL_ANALYZER_FALLBACKS=off` to disable.

**Cost.** Local models cost nothing per request. An analysis is 8 requests. With Claude, most input
tokens are served from the prompt cache, so cost is dominated by output and thinking tokens: expect
roughly $0.50–$3 per document at the default `high` effort, more for long contracts. `--effort medium`
is cheaper and faster. Each report ends with the measured token usage (and a cost estimate for Claude
models). Each Q&A turn is one request.

## Project layout

```
legal_analyzer/
  pdf_loader.py   PDF/text extraction and cleanup
  chunker.py      clause-aware chunking with section paths and page tracking
  retrieval.py    BM25, embeddings, Reciprocal Rank Fusion
  taxonomy.py     the 21 risk categories, their review briefs and retrieval queries
  prompts.py      system prompts and prompt rendering
  schemas.py      structured-output schemas the model fills in
  llm.py          provider-neutral interface, errors, and the provider factory
  providers/      claude.py (Anthropic API), openai_compat.py (OpenAI, Gemini, compatible APIs),
                  ollama_local.py (local models), common.py (shared helpers)
  citations.py    citation parsing and passage matching
  analyzer.py     the analysis pipeline
  grounding.py    quote verification
  qa.py           cited question answering
  report.py       Markdown and HTML reports
  store.py        on-disk document store
  cli.py          command-line interface
  web/            FastAPI app and the single-page UI
samples/          sample document and its generator
tests/            unit, pipeline, SDK-contract, and web tests (no API key needed)
```

## Tests

```bash
pytest
```

The tests use a fake LLM for the pipeline and web app. Each provider is also tested through its real
client library (`anthropic`, `openai`, `ollama`) against a mocked HTTP server, which checks the exact
request bodies (JSON schemas, caching, context size, thinking, citations) and response handling
without network access or API keys.

## Limitations

- **Scanned PDFs** have no text layer. Run OCR first (for example `ocrmypdf in.pdf out.pdf`).
- **Retrieval can miss clauses** in very long documents. The review sees the top clauses per category
  (shown under "About this analysis"), and "missing protections" are relative to those clauses.
- **The taxonomy is English and common-law flavoured.** Edit `taxonomy.py` to add categories or
  retrieval queries for your documents.
- **Page numbers are per chunk.** A finding in a chunk that spans a page break shows both pages.
- **Smaller models find less.** A 4B local model catches the obvious problems; subtle loopholes that
  depend on how clauses interact are where frontier models earn their cost.
- **It is not a lawyer.** Severity and enforceability notes are general, not jurisdiction-specific advice.
