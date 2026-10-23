"use strict";

const SEVERITIES = ["critical", "high", "medium", "low"];
const SEVERITY_LABEL = { critical: "Critical", high: "High", medium: "Medium", low: "Low" };
const LEVEL_COLOR = { low: "var(--ok)", moderate: "var(--medium)", high: "var(--high)", severe: "var(--critical)" };
const ISSUE_LABEL = {
  one_sided_term: "One-sided term",
  loophole: "Loophole",
  waived_right: "Waived right",
  ambiguity: "Ambiguity",
  hidden_cost_or_obligation: "Hidden cost or obligation",
  missing_protection: "Missing protection",
};
const FAVORS_LABEL = { drafter: "Favours the drafter", reviewed_party: "Favours you", balanced: "Balanced", unclear: "Unclear who it favours" };
const SUGGESTIONS = [
  "Can they change these terms without telling me?",
  "How do I cancel, and will I get a refund?",
  "What happens to my data if my account is closed?",
  "What rights am I giving up if there is a dispute?",
  "What is the most money I could lose under this agreement?",
];
const PROVIDER_NAMES = { anthropic: "Claude", openai: "OpenAI", gemini: "Gemini", ollama: "Ollama, local" };
const MODEL_STORAGE_KEY = "legal-analyzer:model";
const UPLOAD_TITLE = "Upload a contract";
const UPLOAD_HINT = "PDF or text file · drag & drop or click";

const state = {
  models: null, // providers, their readiness, and their models (from /api/models)
  choice: null, // { provider, model } used for analysis and Q&A
  modelBanner: false,
  docs: [],
  docId: null,
  doc: null,
  report: null,
  job: null,
  pollTimer: null,
  sessionId: null,
  chatBusy: false,
  filters: { severities: new Set(), group: "", query: "" },
};

const $ = (id) => document.getElementById(id);

// --- helpers --------------------------------------------------------------------------

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let message = res.statusText;
    try {
      const body = await res.json();
      message = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch { /* not JSON */ }
    throw new Error(message || `Request failed (${res.status})`);
  }
  return res.status === 204 ? null : res.json();
}

function formatPages(pages) {
  if (!pages || !pages.length) return "page unknown";
  const sorted = [...new Set(pages)].sort((a, b) => a - b);
  const ranges = [];
  let start = sorted[0];
  let prev = sorted[0];
  for (const p of sorted.slice(1).concat([null])) {
    if (p !== null && p === prev + 1) { prev = p; continue; }
    ranges.push(start === prev ? `${start}` : `${start}–${prev}`);
    if (p !== null) { start = p; prev = p; }
  }
  return `${sorted.length === 1 ? "p." : "pp."} ${ranges.join(", ")}`;
}

const num = (n) => Number(n || 0).toLocaleString();
const capitalize = (s) => s.charAt(0).toUpperCase() + s.slice(1);

function card(title, ...children) {
  return el("div", { class: "card" }, el("h3", {}, title), ...children);
}

function list(items, ordered = false) {
  return el(ordered ? "ol" : "ul", {}, ...items.map((item) => el("li", {}, item)));
}

function showBanner(message, { transient = true } = {}) {
  const banner = $("banner");
  banner.textContent = message;
  banner.hidden = false;
  clearTimeout(showBanner.timer);
  if (transient) showBanner.timer = setTimeout(() => { banner.hidden = true; }, 8000);
}

// Minimal, safe Markdown: everything is HTML-escaped first, then a few inline/block rules apply.
function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function inlineMarkdown(text) {
  return text
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*([^*\s][^*]*?)\*(?!\*)/g, "$1<em>$2</em>");
}

function renderMarkdown(text) {
  const out = [];
  let listTag = null;
  let paragraph = [];
  const flush = () => {
    if (paragraph.length) out.push(`<p>${inlineMarkdown(paragraph.join(" "))}</p>`);
    paragraph = [];
  };
  const closeList = () => {
    if (listTag) out.push(`</${listTag}>`);
    listTag = null;
  };
  for (const line of escapeHtml(text).split(/\r?\n/)) {
    const t = line.trim();
    let m;
    if (!t) { flush(); closeList(); continue; }
    if ((m = t.match(/^#{1,6}\s+(.*)$/))) { flush(); closeList(); out.push(`<h4>${inlineMarkdown(m[1])}</h4>`); continue; }
    if ((m = t.match(/^[-*•]\s+(.*)$/)) || (m = t.match(/^\d+[.)]\s+(.*)$/))) {
      const tag = /^\d/.test(t) ? "ol" : "ul";
      flush();
      if (listTag !== tag) { closeList(); out.push(`<${tag}>`); listTag = tag; }
      out.push(`<li>${inlineMarkdown(m[1])}</li>`);
      continue;
    }
    closeList();
    paragraph.push(t);
  }
  flush();
  closeList();
  return out.join("");
}

function highlightText(text, phrases) {
  const ranges = [];
  for (const phrase of phrases || []) {
    const words = phrase.split(/\s+/).filter(Boolean).map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
    if (!words.length) continue;
    const match = new RegExp(words.join("\\s+"), "i").exec(text);
    if (match) ranges.push([match.index, match.index + match[0].length]);
  }
  ranges.sort((a, b) => a[0] - b[0]);
  const nodes = [];
  let pos = 0;
  for (const [start, end] of ranges) {
    if (start < pos) continue;
    nodes.push(document.createTextNode(text.slice(pos, start)), el("mark", {}, text.slice(start, end)));
    pos = end;
  }
  nodes.push(document.createTextNode(text.slice(pos)));
  return nodes;
}

// --- health & documents ---------------------------------------------------------------

// --- model picker ---------------------------------------------------------------------

function providerInfo(id) {
  return state.models ? state.models.providers.find((p) => p.id === id) : null;
}

// Why the chosen model can't be used yet, or null when it is ready.
function choiceProblem() {
  const provider = providerInfo(state.choice && state.choice.provider);
  if (!provider) return "Unknown provider.";
  if (!provider.ready) return provider.message;
  const model = state.choice.model;
  const installed = provider.models.includes(model) || (!model.includes(":") && provider.models.includes(`${model}:latest`));
  if (provider.id === "ollama" && !installed) return `Model '${model}' isn't downloaded. Run: ollama pull ${model}`;
  return null;
}

function choiceReady() {
  return choiceProblem() === null;
}

function modelChoice() {
  return state.choice ? { provider: state.choice.provider, model: state.choice.model } : {};
}

function describeModel(report) {
  const details = [PROVIDER_NAMES[report.provider] || report.provider];
  if (report.effort) details.push(`effort ${report.effort}`);
  return `${report.model} (${details.join(", ")})`;
}

async function loadModels() {
  state.models = await api("/api/models");
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(MODEL_STORAGE_KEY) || "null"); } catch { /* storage unavailable */ }
  const provider = saved && providerInfo(saved.provider);
  const usable = provider && (saved.custom || provider.models.includes(saved.model));
  state.choice = usable ? saved : (state.choice || { ...state.models.default });
  renderModelPicker();
}

function renderModelPicker() {
  const select = el("select", { id: "model-select", "aria-label": "Model used for analysis and questions" });
  for (const provider of state.models.providers) {
    const group = el("optgroup", { label: provider.ready ? provider.label : `${provider.label} (not set up)` });
    const models = [...provider.models];
    if (state.choice.provider === provider.id && !models.includes(state.choice.model)) models.push(state.choice.model);
    for (const model of models) {
      const selected = state.choice.provider === provider.id && state.choice.model === model;
      group.append(el("option", { value: `${provider.id}::${model}`, selected }, model));
    }
    group.append(el("option", { value: `${provider.id}::` }, `Other ${provider.label} model…`));
    select.append(group);
  }
  select.addEventListener("change", () => {
    const [providerId, model] = select.value.split("::");
    if (model) {
      setChoice({ provider: providerId, model });
      return;
    }
    const provider = providerInfo(providerId);
    const typed = prompt(`${provider.label} model ID (for example ${provider.default_model}):`);
    if (typed && typed.trim()) setChoice({ provider: providerId, model: typed.trim(), custom: true });
    else renderModelPicker();
  });

  const provider = providerInfo(state.choice.provider);
  const problem = choiceProblem();
  const chip = problem === null
    ? el("span", { class: "chip ok", title: provider.local ? "Runs on this computer" : "Cloud API" }, provider.local ? "Local ✓" : "Ready ✓")
    : el("span", { class: "chip warn", title: problem }, provider && provider.ready ? "Not downloaded" : "Not set up");
  $("status").replaceChildren(select, chip);

  if (problem !== null) {
    showBanner(`${state.choice.model} can't be used yet: ${problem} You can still upload and search documents.`, { transient: false });
    state.modelBanner = true;
  } else if (state.modelBanner) {
    $("banner").hidden = true;
    state.modelBanner = false;
  }
}

function setChoice(choice) {
  state.choice = choice;
  try { localStorage.setItem(MODEL_STORAGE_KEY, JSON.stringify(choice)); } catch { /* storage unavailable */ }
  renderModelPicker();
  if (state.sessionId || $("chat-thread").children.length) {
    state.sessionId = null; // a conversation stays with one model
    $("chat-thread").append(el("div", { class: "note" }, `Switched to ${choice.model}. New questions start a fresh conversation.`));
  }
}

function notReadyMessage(action) {
  return `${action} needs a model that is ready. ${choiceProblem()} Or pick another model in the header.`;
}

async function loadDocs() {
  state.docs = await api("/api/documents");
  renderDocList();
}

function renderDocList() {
  const listNode = $("doc-list");
  if (!state.docs.length) {
    listNode.replaceChildren(el("li", { class: "doc-empty" }, "No documents yet."));
    return;
  }
  listNode.replaceChildren(...state.docs.map((doc) => el("li", {
    class: `doc-item${doc.id === state.docId ? " active" : ""}`,
    tabindex: "0",
    onclick: () => selectDoc(doc.id),
    onkeydown: (e) => { if (e.key === "Enter") selectDoc(doc.id); },
  },
  el("div", { class: "doc-text" },
    el("div", { class: "doc-name", title: doc.filename }, doc.filename),
    el("div", { class: "doc-sub" }, `${doc.page_count} pages · ${doc.job && doc.job.status === "running" ? "analysing…" : doc.has_report ? "analysed" : "not analysed"}`)),
  el("button", {
    type: "button", class: "icon-btn", title: "Delete", "aria-label": `Delete ${doc.filename}`,
    onclick: (e) => { e.stopPropagation(); deleteDoc(doc); },
  }, "×"))));
}

function setupUpload() {
  const zone = $("dropzone");
  const input = $("file-input");
  zone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); }
  });
  input.addEventListener("change", () => {
    if (input.files[0]) upload(input.files[0]);
    input.value = "";
  });
  for (const type of ["dragenter", "dragover"]) {
    zone.addEventListener(type, (e) => { e.preventDefault(); zone.classList.add("dragging"); });
  }
  for (const type of ["dragleave", "drop"]) {
    zone.addEventListener(type, (e) => { e.preventDefault(); zone.classList.remove("dragging"); });
  }
  zone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) upload(file);
  });
}

async function upload(file) {
  const zone = $("dropzone");
  if (zone.classList.contains("busy")) return;
  zone.classList.add("busy");
  $("dropzone-title").textContent = `Indexing ${file.name}…`;
  $("dropzone-hint").textContent = "Extracting clauses and building the search index";
  try {
    const form = new FormData();
    form.append("file", file);
    const doc = await api("/api/documents", { method: "POST", body: form });
    await loadDocs();
    await selectDoc(doc.id);
  } catch (err) {
    showBanner(`Upload failed: ${err.message}`);
  } finally {
    zone.classList.remove("busy");
    $("dropzone-title").textContent = UPLOAD_TITLE;
    $("dropzone-hint").textContent = UPLOAD_HINT;
  }
}

async function deleteDoc(doc) {
  if (!confirm(`Delete “${doc.filename}” and its analysis?`)) return;
  try {
    await api(`/api/documents/${doc.id}`, { method: "DELETE" });
    if (state.docId === doc.id) {
      stopPolling();
      Object.assign(state, { docId: null, doc: null, report: null, job: null, sessionId: null });
      $("doc-view").hidden = true;
      $("welcome").hidden = false;
    }
    await loadDocs();
  } catch (err) {
    showBanner(err.message);
  }
}

async function selectDoc(id) {
  stopPolling();
  Object.assign(state, { docId: id, doc: null, report: null, job: null, sessionId: null });
  state.filters = { severities: new Set(), group: "", query: "" };
  $("finding-search").value = "";
  $("glossary-search").value = "";
  renderDocList();
  $("welcome").hidden = true;
  $("doc-view").hidden = false;
  resetChat();
  try {
    const [doc, analysis] = await Promise.all([api(`/api/documents/${id}`), api(`/api/documents/${id}/analysis`)]);
    if (state.docId !== id) return; // the user picked another document meanwhile
    state.doc = doc;
    state.report = analysis.report;
    state.job = analysis.job;
    renderDocHeader();
    renderReport();
    if (state.job && state.job.status === "running") startPolling();
  } catch (err) {
    showBanner(err.message);
  }
}

function renderDocHeader() {
  const doc = state.doc;
  $("doc-title").textContent = doc.filename;
  const search = doc.search_mode === "hybrid" ? "keyword + semantic search" : "keyword search";
  $("doc-meta").textContent = `${doc.page_count} pages · ${doc.chunk_count} clauses indexed · ${search}`;
  $("doc-warnings").replaceChildren(...doc.warnings.map((w) => el("div", { class: "notice" }, w)));
}

function renderActions() {
  const actions = $("doc-actions");
  if (!state.report) { actions.replaceChildren(); return; }
  const base = `/api/documents/${state.docId}`;
  actions.replaceChildren(
    el("a", { class: "btn small", href: `${base}/report.html`, target: "_blank", rel: "noopener" }, "Open report"),
    el("a", { class: "btn small", href: `${base}/report.html?download=true` }, "Download HTML"),
    el("a", { class: "btn small", href: `${base}/report.md` }, "Download Markdown"),
  );
}

// --- analysis -------------------------------------------------------------------------

function setupAnalyzeForm() {
  $("analyze-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!state.docId) return;
    if (!choiceReady()) {
      showBanner(notReadyMessage("Analysis"));
      return;
    }
    try {
      state.job = await api(`/api/documents/${state.docId}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...modelChoice(), perspective: $("perspective").value, jurisdiction: $("jurisdiction").value }),
      });
      renderJob();
      startPolling();
      loadDocs();
    } catch (err) {
      showBanner(err.message);
    }
  });
}

function startPolling() {
  stopPolling();
  state.pollTimer = setInterval(poll, 1500);
}

function stopPolling() {
  if (state.pollTimer) clearInterval(state.pollTimer);
  state.pollTimer = null;
}

async function poll() {
  const id = state.docId;
  try {
    const analysis = await api(`/api/documents/${id}/analysis`);
    if (id !== state.docId) return;
    state.job = analysis.job;
    if (!state.job || state.job.status !== "running") {
      stopPolling();
      if (state.job && state.job.status === "done") {
        state.report = analysis.report;
        renderReport();
      }
      loadDocs();
    }
    renderJob();
  } catch {
    // transient network error: keep polling
  }
}

function renderJob() {
  const job = state.job;
  const running = Boolean(job && job.status === "running");
  const button = $("analyze-btn");
  button.disabled = running;
  button.textContent = running ? "Analyzing…" : state.report ? "Re-run analysis" : "Analyze document";
  $("progress").hidden = !running;
  if (running) {
    const steps = (job.total || 7) + 1; // review passes + summary
    const done = job.completed + (job.stage === "synthesis" ? 0.5 : 0);
    $("progress-fill").style.width = `${Math.max(4, Math.round((done / steps) * 100))}%`;
    $("progress-text").textContent = job.total ? `${job.message} · ${job.completed}/${job.total} review passes` : job.message;
  }
  const old = document.getElementById("job-error");
  if (old) old.remove();
  if (job && job.status === "error") {
    $("analyze-form").after(el("div", { class: "notice error", id: "job-error" }, `Analysis failed: ${job.error}`));
  }
}

// --- report ---------------------------------------------------------------------------

function renderReport() {
  const report = state.report;
  $("findings-count").textContent = report ? String(report.findings.length) : "";
  $("glossary-count").textContent = report ? String(report.glossary.length) : "";
  renderActions();
  renderOverview();
  renderFindingFilters();
  renderFindings();
  renderGlossary();
  renderJob();
}

function findingById(id) {
  return state.report.findings.find((f) => f.id === id);
}

function renderOverview() {
  const panel = $("tab-overview");
  const report = state.report;
  if (!report) {
    panel.replaceChildren(el("div", { class: "card placeholder" },
      el("p", {}, "Run an analysis to see the risk score, biased terms, loopholes, and plain-English explanations."),
      el("p", { class: "small" }, "You can already ask questions about the document in the “Ask a question” tab.")));
    return;
  }
  const s = report.summary;
  const risk = report.risk;
  const color = LEVEL_COLOR[risk.level];
  const scoreCard = el("div", { class: "card score-card" },
    el("div", { class: "score-ring", style: `--value:${risk.score};--color:${color}` },
      el("div", {}, el("div", {}, el("div", { class: "score-num" }, String(risk.score)), el("div", { class: "muted small" }, "risk / 100")))),
    el("div", { class: "score-level", style: `color:${color}` }, `${capitalize(risk.level)} risk`),
    el("div", { class: "score-counts" }, ...SEVERITIES.map((sev) => el("span", { class: `pill sev-${sev}` }, `${risk.counts[sev] || 0} ${sev}`))),
    el("details", { class: "small muted" }, el("summary", {}, "How is this scored?"), el("p", {}, risk.method)));

  let summaryCard;
  if (s) {
    const parties = s.parties.map((p) => `${p.name} (${p.role})`).join("; ") || "Not identified";
    const facts = [
      ["Document type", s.document_type || "Unknown"],
      ["Parties", parties],
      ["Favours", s.favoured_party || "n/a"],
      ["Governing law", s.governing_law || "Not stated"],
      ["Reviewed for", report.perspective + (report.jurisdiction ? ` (${report.jurisdiction})` : "")],
    ];
    summaryCard = el("div", { class: "card" },
      el("p", { class: "headline" }, s.headline),
      el("p", {}, s.what_this_document_does),
      el("p", {}, s.overall_assessment),
      el("dl", { class: "facts" }, ...facts.flatMap(([k, v]) => [el("dt", {}, k), el("dd", {}, v)])));
  } else {
    summaryCard = el("div", { class: "card" }, el("p", {}, "The summary could not be generated. The findings are still available."));
  }

  const nodes = [el("div", { class: "overview-grid" }, scoreCard, summaryCard)];
  if (s && s.top_concern_ids.length) {
    nodes.push(card("Top concerns", el("ol", {}, ...s.top_concern_ids.map((id) => {
      const f = findingById(id);
      return el("li", {},
        el("button", { type: "button", class: "link-btn", onclick: () => showFinding(id) }, f.title), " ",
        el("span", { class: `pill sev-${f.severity}` }, f.severity),
        el("span", { class: "muted small" }, ` ${formatPages(f.pages)}`));
    }))));
  }
  if (s && (s.recommendations.length || s.questions_to_ask.length)) {
    nodes.push(el("div", { class: "two-col" },
      card("What you can do", list(s.recommendations)),
      card("Questions to ask", list(s.questions_to_ask))));
  }
  if (report.gaps.length) {
    nodes.push(card("Missing protections",
      el("p", { class: "muted small" }, "Based on the clauses reviewed; confirm against the full document."),
      el("ul", {}, ...report.gaps.map((g) => el("li", {}, el("strong", {}, `${g.category}: `), g.observation)))));
  }
  const u = report.usage;
  const r = report.retrieval;
  const cost = u.estimated_cost_usd == null ? "" : u.estimated_cost_usd === 0 ? ", no API cost (local model)"
    : `, about $${u.estimated_cost_usd.toFixed(2)}`;
  nodes.push(card("About this analysis", el("ul", { class: "small" },
    el("li", {}, `${r.context_chunks} of ${r.total_chunks} clauses retrieved as context using ${r.embeddings ? "hybrid keyword + semantic" : "keyword"} search.`),
    el("li", {}, `${describeModel(report)}: ${u.requests} requests, ${num(u.input_tokens)} input tokens, ` +
      `${num(u.cache_read_input_tokens)} read from cache, ${num(u.output_tokens)} output tokens${cost}.`),
    el("li", {}, `Completed in ${Math.round(report.duration_seconds)}s on ${new Date(report.created_at).toLocaleString()}.`),
    ...report.errors.map((e) => el("li", { class: "unverified" }, `Warning: ${e}`)))));
  panel.replaceChildren(...nodes);
}

function renderFindingFilters() {
  const report = state.report;
  const chips = $("severity-filters");
  const select = $("group-filter");
  if (!report) {
    chips.replaceChildren();
    select.replaceChildren(el("option", { value: "" }, "All areas"));
    return;
  }
  chips.replaceChildren(...SEVERITIES.map((sev) => {
    const count = report.findings.filter((f) => f.severity === sev).length;
    return el("button", {
      type: "button",
      class: `chip${state.filters.severities.has(sev) ? " active" : ""}`,
      "aria-pressed": state.filters.severities.has(sev) ? "true" : "false",
      onclick: () => {
        const set = state.filters.severities;
        if (set.has(sev)) set.delete(sev); else set.add(sev);
        renderFindingFilters();
        renderFindings();
      },
    }, el("span", { class: `pill sev-${sev}` }, SEVERITY_LABEL[sev]), String(count));
  }));
  const groups = [...new Set(report.findings.map((f) => f.group))];
  select.replaceChildren(el("option", { value: "" }, "All areas"),
    ...groups.map((g) => el("option", { value: g, selected: g === state.filters.group }, g)));
}

function renderFindings() {
  const container = $("findings-list");
  const report = state.report;
  if (!report) {
    container.replaceChildren(el("div", { class: "card placeholder" }, "No analysis yet."));
    return;
  }
  const { severities, group, query } = state.filters;
  const q = query.trim().toLowerCase();
  const items = report.findings.filter((f) =>
    (!severities.size || severities.has(f.severity)) &&
    (!group || f.group === group) &&
    (!q || [f.title, f.quote, f.plain_english, f.why_it_matters, f.category].join(" ").toLowerCase().includes(q)));
  if (!items.length) {
    container.replaceChildren(el("div", { class: "card placeholder" },
      report.findings.length ? "No findings match these filters." : "No problematic terms were flagged."));
    return;
  }
  container.replaceChildren(...items.map(findingCard));
}

function findingCard(f) {
  const rows = [["In plain English", f.plain_english], ["Why it matters", f.why_it_matters]];
  if (f.loophole_scenario) rows.push(["How it could be used against you", f.loophole_scenario]);
  if (f.what_to_do) rows.push(["What you can do", f.what_to_do]);
  return el("article", { class: `card finding ${f.severity}`, id: `finding-${f.id}` },
    el("div", { class: "finding-top" },
      el("span", { class: `pill sev-${f.severity}` }, SEVERITY_LABEL[f.severity]),
      el("span", {}, `${f.id} · ${f.category} · ${ISSUE_LABEL[f.issue_type] || f.issue_type}`)),
    el("h3", {}, f.title),
    el("div", { class: "meta" },
      el("span", {}, formatPages(f.pages)),
      f.section ? el("span", {}, f.section) : null,
      el("span", {}, FAVORS_LABEL[f.favors] || f.favors),
      f.verified
        ? el("span", { class: "verified" }, "✓ Quote verified in source")
        : el("span", { class: "unverified", title: "The quoted text could not be matched to the document. Check it manually." }, "⚠ Quote not found in source")),
    el("blockquote", { class: "quote" }, f.quote),
    el("dl", { class: "explain" }, ...rows.flatMap(([k, v]) => [el("dt", {}, k), el("dd", {}, v)])),
    f.chunk_ids.length
      ? el("div", { class: "finding-actions" },
        el("button", { type: "button", class: "btn small", onclick: () => openSource(f.chunk_ids[0], [f.quote]) }, "View clause in context"))
      : null);
}

function showFinding(id) {
  state.filters = { severities: new Set(), group: "", query: "" };
  $("finding-search").value = "";
  renderFindingFilters();
  renderFindings();
  switchTab("findings");
  const node = document.getElementById(`finding-${id}`);
  if (!node) return;
  node.scrollIntoView({ behavior: "smooth", block: "start" });
  node.classList.remove("flash");
  void node.offsetWidth; // restart the animation
  node.classList.add("flash");
}

function renderGlossary() {
  const container = $("glossary-list");
  const report = state.report;
  if (!report) {
    container.replaceChildren(el("div", { class: "card placeholder" }, "Run an analysis to build a glossary of the legal terms used in this document."));
    return;
  }
  const q = $("glossary-search").value.trim().toLowerCase();
  const terms = report.glossary.filter((t) => !q || `${t.term} ${t.definition}`.toLowerCase().includes(q));
  if (!terms.length) {
    container.replaceChildren(el("div", { class: "card placeholder" }, report.glossary.length ? "No terms match." : "No legal terms were extracted."));
    return;
  }
  container.replaceChildren(el("div", { class: "glossary" }, ...terms.map((t) => el("div", { class: "card term" },
    el("strong", {}, t.term),
    el("div", {}, t.definition),
    t.pages.length ? el("div", { class: "muted small" }, formatPages(t.pages)) : null))));
}

function setupFilters() {
  $("group-filter").addEventListener("change", (e) => { state.filters.group = e.target.value; renderFindings(); });
  $("finding-search").addEventListener("input", (e) => { state.filters.query = e.target.value; renderFindings(); });
  $("glossary-search").addEventListener("input", renderGlossary);
}

// --- tabs & source viewer -------------------------------------------------------------

function switchTab(name) {
  document.querySelectorAll(".tabs button").forEach((b) => {
    const active = b.dataset.tab === name;
    b.classList.toggle("active", active);
    b.setAttribute("aria-selected", active ? "true" : "false");
  });
  for (const tab of ["overview", "findings", "glossary", "ask"]) $(`tab-${tab}`).hidden = tab !== name;
  if (name === "ask") $("chat-input").focus();
}

function setupTabs() {
  document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => switchTab(b.dataset.tab)));
}

async function openSource(chunkId, highlights) {
  const dialog = $("source-dialog");
  $("source-title").textContent = "Loading…";
  $("source-meta").textContent = "";
  $("source-body").replaceChildren();
  if (!dialog.open) dialog.showModal();
  try {
    const chunk = await api(`/api/documents/${state.docId}/chunks/${chunkId}`);
    $("source-title").textContent = chunk.section || "Document excerpt";
    $("source-meta").textContent = `${chunk.location} · clause C${chunk.id}`;
    $("source-body").replaceChildren(...highlightText(chunk.text, highlights));
    const mark = $("source-body").querySelector("mark");
    if (mark) mark.scrollIntoView({ block: "center" });
  } catch (err) {
    $("source-title").textContent = "Could not load the excerpt";
    $("source-body").textContent = err.message;
  }
}

function setupDialog() {
  const dialog = $("source-dialog");
  $("source-close").addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", (e) => { if (e.target === dialog) dialog.close(); });
}

// --- chat -----------------------------------------------------------------------------

function resetChat() {
  $("chat-thread").replaceChildren();
  $("suggestions").hidden = false;
  $("suggestions").replaceChildren(...SUGGESTIONS.map((q) => el("button", { type: "button", class: "chip", onclick: () => ask(q) }, q)));
}

function setupChat() {
  $("chat-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const question = $("chat-input").value.trim();
    if (question) ask(question);
  });
  $("chat-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("chat-form").requestSubmit(); }
  });
}

async function readServerSentEvents(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const raw = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const data = raw.split("\n").filter((l) => l.startsWith("data:")).map((l) => l.slice(5).trimStart()).join("\n");
      if (data) onEvent(JSON.parse(data));
    }
  }
}

async function ask(question) {
  if (state.chatBusy || !state.docId) return;
  if (!choiceReady()) {
    showBanner(notReadyMessage("Q&A"));
    return;
  }
  const docId = state.docId;
  state.chatBusy = true;
  $("chat-btn").disabled = true;
  $("chat-input").value = "";
  $("suggestions").hidden = true;

  const thread = $("chat-thread");
  thread.append(el("div", { class: "msg user" }, question));
  const answerNode = el("div", { class: "answer" }, el("span", { class: "typing" }, "Reading the relevant clauses"));
  const bubble = el("div", { class: "msg assistant streaming" }, answerNode);
  thread.append(bubble);
  bubble.scrollIntoView({ block: "end", behavior: "smooth" });

  let streamed = "";
  let answeredBy = "";
  try {
    const res = await fetch(`/api/documents/${docId}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, session_id: state.sessionId, ...modelChoice() }),
    });
    if (!res.ok) {
      let message = res.statusText;
      try { message = (await res.json()).detail; } catch { /* not JSON */ }
      throw new Error(message);
    }
    await readServerSentEvents(res, (event) => {
      if (event.type === "session") {
        if (state.docId === docId) state.sessionId = event.session_id;
        answeredBy = event.model || "";
      } else if (event.type === "text") {
        streamed += event.text;
        answerNode.textContent = streamed;
      } else if (event.type === "answer") {
        bubble.classList.remove("streaming");
        renderAnswer(bubble, answerNode, event);
        if (answeredBy) bubble.append(el("div", { class: "answered-by" }, `Answered by ${answeredBy}`));
      } else if (event.type === "error") {
        bubble.classList.remove("streaming");
        answerNode.replaceChildren(el("span", { class: "error" }, event.message));
      }
    });
  } catch (err) {
    bubble.classList.remove("streaming");
    answerNode.replaceChildren(el("span", { class: "error" }, `Something went wrong: ${err.message}`));
  } finally {
    state.chatBusy = false;
    $("chat-btn").disabled = false;
  }
}

function renderAnswer(bubble, answerNode, answer) {
  // Put citation markers before any trailing whitespace so they stay attached to their sentence.
  const marked = answer.segments.map((seg) => {
    const markers = seg.sources.map((n) => `${n}`).join("");
    const trailing = seg.text.match(/\s*$/)[0];
    return seg.text.slice(0, seg.text.length - trailing.length) + markers + trailing;
  }).join("");
  answerNode.innerHTML = renderMarkdown(marked).replace(
    /(\d+)/g,
    (_, n) => `<button type="button" class="cite" data-n="${n}" title="Show source ${n}">${n}</button>`,
  );
  const byNumber = new Map(answer.sources.map((s) => [s.number, s]));
  answerNode.querySelectorAll(".cite").forEach((button) => button.addEventListener("click", () => {
    const source = byNumber.get(Number(button.dataset.n));
    if (source) openSource(source.chunk_id, source.cited_text);
  }));
  if (!answer.sources.length) return;
  bubble.append(el("div", { class: "sources" },
    el("div", { class: "muted small" }, "Sources"),
    ...answer.sources.map((s) => el("button", { type: "button", class: "source", onclick: () => openSource(s.chunk_id, s.cited_text) },
      el("span", { class: "n" }, `[${s.number}]`),
      el("span", {},
        el("div", { class: "where" }, `${s.location}${s.section ? ` — ${s.section}` : ""}`),
        el("div", { class: "cited" }, s.cited_text.join(" … ")))))));
}

// --- start ----------------------------------------------------------------------------

async function init() {
  setupUpload();
  setupTabs();
  setupFilters();
  setupDialog();
  setupAnalyzeForm();
  setupChat();
  try {
    await loadModels();
    await loadDocs();
  } catch (err) {
    showBanner(`Could not reach the server: ${err.message}`, { transient: false });
    return;
  }
  if (state.docs.length) selectDoc(state.docs[0].id);
  // Pick up models pulled into Ollama (or keys added) while the page was open.
  window.addEventListener("focus", () => { loadModels().catch(() => {}); });
}

init();
