const form = $("#ask-form");
const questionEl = $("#question");
const askBtn = $("#ask-btn");
const topKEl = $("#top-k");
const minRelEl = $("#min-rel");
const llmEl = $("#llm-model");
const answerWrap = $("#answer-wrap");
const sourcesEl = $("#sources");
const filterBtn = $("#filter-btn");
const filterMenu = $("#filter-menu");

let mode = "answer";
let docs = [];
let selected = new Set();   // empty = all papers
let hits = [];
let busy = false;

// ---------- paper filter ----------

async function loadDocs() {
  docs = (await fetch("/api/documents").then(r => r.json())).documents;
  const pre = new URLSearchParams(location.search).get("doc");
  if (pre && docs.some(d => d.id === pre)) selected.add(pre);
  renderFilter();
}

function renderFilter() {
  filterMenu.innerHTML = `
    <label><input type="checkbox" data-all ${selected.size ? "" : "checked"}> <b>All papers</b></label>
    ${docs.map(d => `
      <label><input type="checkbox" value="${d.id}" ${selected.has(d.id) ? "checked" : ""}> ${esc(d.title)}</label>`).join("")}
    ${docs.length ? "" : `<div class="empty small">No papers yet — <a href="/library">upload some</a>.</div>`}`;

  if (!selected.size) filterBtn.textContent = "All papers ▾";
  else if (selected.size === 1) {
    const t = docs.find(d => selected.has(d.id))?.title ?? "1 paper";
    filterBtn.textContent = (t.length > 32 ? t.slice(0, 30) + "…" : t) + " ▾";
  } else filterBtn.textContent = `${selected.size} papers ▾`;
}

filterBtn.addEventListener("click", () => filterMenu.classList.toggle("open"));
document.addEventListener("click", e => {
  if (!e.target.closest(".filter")) filterMenu.classList.remove("open");
});
filterMenu.addEventListener("change", e => {
  if (e.target.dataset.all !== undefined) selected.clear();
  else e.target.checked ? selected.add(e.target.value) : selected.delete(e.target.value);
  renderFilter();
});

// ---------- mode + input ----------

$("#mode").addEventListener("click", e => {
  const m = e.target.dataset.mode;
  if (!m) return;
  mode = m;
  document.querySelectorAll("#mode button").forEach(b => b.classList.toggle("on", b.dataset.mode === m));
});

questionEl.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});
questionEl.addEventListener("input", () => {
  questionEl.style.height = "auto";
  questionEl.style.height = Math.min(questionEl.scrollHeight, 220) + "px";
});

$("#examples").addEventListener("click", e => {
  if (!e.target.classList.contains("chip")) return;
  questionEl.value = e.target.textContent;
  form.requestSubmit();
});

// ---------- rendering ----------

function inline(s) {
  return esc(s)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*(?!\s)(.+?)\*/g, "$1<em>$2</em>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\[(\d+(?:\s*[,;]\s*\d+)*)\]/g, (m, nums) =>
      nums.split(/[,;]/).map(n => {
        n = n.trim();
        const h = hits.find(x => x.ref === +n);
        if (!h) return "";   // drop citations to sources that don't exist
        const tip = `${h.title}${h.page ? ` — p. ${h.page}` : ""}`;
        return `<button class="cite" data-ref="${n}" title="${esc(tip)}">${n}</button>`;
      }).join(""));
}

function renderMarkdown(text) {
  const out = [];
  let para = [], list = null;
  const flushPara = () => { if (para.length) out.push(`<p>${inline(para.join(" "))}</p>`); para = []; };
  const flushList = () => { if (list) out.push(`<${list.tag}>${list.items.map(i => `<li>${inline(i)}</li>`).join("")}</${list.tag}>`); list = null; };

  for (const raw of text.split("\n")) {
    const line = raw.trimEnd();
    let m;
    if (!line.trim()) { flushPara(); flushList(); }
    else if ((m = line.match(/^#{1,6}\s+(.*)/))) { flushPara(); flushList(); out.push(`<h3>${inline(m[1])}</h3>`); }
    else if ((m = line.match(/^\s*[-*•]\s+(.*)/)) || (m = line.match(/^\s*\d+[.)]\s+(.*)/))) {
      flushPara();
      const tag = /^\s*\d/.test(line) ? "ol" : "ul";
      if (!list || list.tag !== tag) { flushList(); list = { tag, items: [] }; }
      list.items.push(m[1]);
    } else if (list && /^\s{2,}/.test(raw)) list.items[list.items.length - 1] += " " + line.trim();
    else { flushList(); para.push(line.trim()); }
  }
  flushPara(); flushList();
  return out.join("");
}

function renderSources() {
  if (!hits.length) { sourcesEl.innerHTML = ""; return; }
  sourcesEl.innerHTML = `<h2>Sources</h2>` + hits.map(h => {
    const link = `/api/documents/${h.doc_id}/file${h.is_pdf && h.page ? `#page=${h.page}` : ""}`;
    const ranks = [h.dense_rank && `semantic #${h.dense_rank}`, h.bm25_rank && `keyword #${h.bm25_rank}`].filter(Boolean).join(" · ");
    return `
      <div class="card source" id="src-${h.ref}">
        <div class="source-top">
          <div class="source-num">${h.ref}</div>
          <div>
            <div class="source-title">${esc(h.title)}</div>
            <div class="source-meta">
              ${h.page ? `<span>page ${h.page}</span>` : ""}
              <span title="Reranker relevance"><span class="bar"><i style="width:${Math.round(h.relevance * 100)}%"></i></span> ${Math.round(h.relevance * 100)}%</span>
              ${ranks ? `<span>${ranks}</span>` : ""}
            </div>
          </div>
        </div>
        <div class="source-text">${esc(h.text)}</div>
        <div class="source-actions">
          <button data-expand>Show more</button>
          <a href="${link}" target="_blank">Open ${h.page ? `at p. ${h.page}` : "file"} ↗</a>
          <button data-copy-src="${h.ref}">Copy citation</button>
        </div>
      </div>`;
  }).join("");
}

function citationLine(h) {
  return `[${h.ref}] ${h.title}${h.page ? `, p. ${h.page}` : ""} (${h.filename})`;
}

function renderAnswer({ text = "", streaming = false, note = "", meta = "" }) {
  answerWrap.innerHTML = `
    <div class="card answer">
      <div class="answer-head">
        ${streaming ? `<span class="spinner"></span> ${text ? "Writing…" : "Reading sources…"}` : `<span>${meta}</span>`}
        <span class="spacer"></span>
        ${!streaming && text ? `<button class="btn ghost small" id="copy-answer">Copy with sources</button>` : ""}
      </div>
      <div class="answer-body">${renderMarkdown(text)}${streaming ? `<span class="caret"></span>` : ""}</div>
      ${note ? `<div class="notice">${note}</div>` : ""}
    </div>`;
}

// citation click -> jump to the source card
document.addEventListener("click", e => {
  const cite = e.target.closest(".cite");
  if (cite) {
    const card = $(`#src-${cite.dataset.ref}`);
    card.scrollIntoView({ behavior: "smooth", block: "center" });
    card.classList.add("flash");
    setTimeout(() => card.classList.remove("flash"), 1400);
    return;
  }
  if (e.target.matches("[data-expand]")) {
    const card = e.target.closest(".source");
    card.classList.toggle("open");
    e.target.textContent = card.classList.contains("open") ? "Show less" : "Show more";
  }
  if (e.target.dataset.copySrc) {
    navigator.clipboard.writeText(citationLine(hits.find(h => h.ref === +e.target.dataset.copySrc)));
    toast("Citation copied");
  }
  if (e.target.id === "copy-answer") {
    const body = answerWrap.dataset.raw || "";
    const used = hits.filter(h => body.includes(`[${h.ref}]`) || new RegExp(`\\[[\\d,\\s;]*\\b${h.ref}\\b`).test(body));
    navigator.clipboard.writeText(`${body}\n\nSources:\n${(used.length ? used : hits).map(citationLine).join("\n")}`);
    toast("Answer copied");
  }
});

// ---------- history ----------

function renderHistory() {
  const items = store.get("rag-history", []);
  $("#history").innerHTML = items.length
    ? `<h2>Recent questions</h2>` + items.map(q => `<button data-q="${esc(q)}">${esc(q)}</button>`).join("")
    : "";
}
function remember(q) {
  store.set("rag-history", [q, ...store.get("rag-history", []).filter(x => x !== q)].slice(0, 8));
  renderHistory();
}
$("#history").addEventListener("click", e => {
  if (!e.target.dataset.q) return;
  questionEl.value = e.target.dataset.q;
  form.requestSubmit();
});

// ---------- ask ----------

function payload(question) {
  return JSON.stringify({
    question,
    top_k: +topKEl.value,
    min_relevance: +minRelEl.value,
    llm_model: llmEl.value || null,
    doc_ids: selected.size ? [...selected] : null,
  });
}

// answer model: whatever is pulled in Ollama, defaulting to the configured one
async function loadModels() {
  const { models, default: def } = await fetch("/api/llm/models").then(r => r.json());
  const names = models.includes(def) || models.some(m => m.split(":")[0] === def) ? models : [def, ...models];
  llmEl.innerHTML = names.map(m => `<option value="${esc(m)}">${esc(m)}${m.split(":")[0] === def.split(":")[0] ? " (default)" : ""}</option>`).join("");

  const saved = store.get("rag-llm", null);
  const fallback = names.find(m => m.split(":")[0] === def.split(":")[0]) || def;
  llmEl.value = names.includes(saved) ? saved : fallback;
  llmEl.hidden = names.length < 2;   // nothing to choose from
}
llmEl.addEventListener("change", () => store.set("rag-llm", llmEl.value));

async function postJSON(url, body) {
  const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body });
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `HTTP ${res.status}`);
  return res;
}

// remember the threshold between visits
minRelEl.value = store.get("rag-min-rel", minRelEl.value);
minRelEl.addEventListener("change", () => store.set("rag-min-rel", minRelEl.value));

async function runPassages(question) {
  const data = await (await postJSON("/api/search", payload(question))).json();
  hits = data.hits;
  renderSources();
  renderAnswer({
    meta: `${hits.length} passages passed the relevance filter · ${data.elapsed_ms} ms`,
    note: data.message ? esc(data.message) : "",
  });
}

async function runAnswer(question) {
  hits = [];
  renderSources();
  renderAnswer({ streaming: true });

  const res = await postJSON("/api/ask", payload(question));
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "", text = "", note = "", retrievalMs = 0, totalMs = 0, pending = false, usedModel = "";

  const paint = () => {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; renderAnswer({ text, streaming: true }); });
  };

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop();
    for (const line of lines) {
      if (!line.trim()) continue;
      const ev = JSON.parse(line);
      if (ev.type === "sources") { hits = ev.hits; retrievalMs = ev.retrieval_ms; usedModel = ev.model; renderSources(); }
      else if (ev.type === "token") { text += ev.text; paint(); }
      else if (ev.type === "error") note = esc(ev.message);
      else if (ev.type === "done") totalMs = ev.total_ms;
    }
  }

  if (note && hits.length && !text) note += " The retrieved passages are still shown on the right.";
  answerWrap.dataset.raw = text;
  renderAnswer({
    text,
    note,
    meta: text ? `Answered from ${hits.length} relevant source${hits.length === 1 ? "" : "s"} · ${esc(usedModel)} · retrieval ${retrievalMs} ms · total ${(totalMs / 1000).toFixed(1)} s` : "",
  });
}

form.addEventListener("submit", async e => {
  e.preventDefault();
  const q = questionEl.value.trim();
  if (!q || busy) return;

  busy = true;
  askBtn.disabled = true;
  $("#examples").style.display = "none";
  remember(q);

  try {
    await (mode === "answer" ? runAnswer(q) : runPassages(q));
  } catch (err) {
    renderAnswer({ note: `Something went wrong: ${esc(err.message)}` });
  } finally {
    busy = false;
    askBtn.disabled = false;
  }
});

loadDocs();
loadModels();
renderHistory();
