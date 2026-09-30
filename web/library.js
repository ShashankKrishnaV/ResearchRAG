const dropzone = $("#dropzone");
const fileInput = $("#file-input");
const queue = $("#queue");
const docList = $("#doc-list");
const docFilter = $("#doc-filter");

const ALLOWED = [".pdf", ".docx", ".txt", ".md"];
let documents = [];

// ---------- upload ----------

["dragenter", "dragover"].forEach(ev => dropzone.addEventListener(ev, e => {
  e.preventDefault();
  dropzone.classList.add("over");
}));
["dragleave", "drop"].forEach(ev => dropzone.addEventListener(ev, e => {
  e.preventDefault();
  dropzone.classList.remove("over");
}));
dropzone.addEventListener("drop", e => uploadAll([...e.dataTransfer.files]));
fileInput.addEventListener("change", () => {
  uploadAll([...fileInput.files]);
  fileInput.value = "";
});

function queueRow(file) {
  const row = document.createElement("div");
  row.className = "queue-item";
  row.innerHTML = `
    <span class="name">${esc(file.name)}</span>
    <span class="muted small">${fmtBytes(file.size)}</span>
    <span class="state"><span class="spinner"></span></span>`;
  queue.prepend(row);
  return row.querySelector(".state");
}

function setState(el, cls, html) {
  el.className = `state ${cls}`;
  el.innerHTML = html;
}

async function uploadAll(files) {
  // one request per file so each row reports its own progress
  for (const file of files) {
    const state = queueRow(file);
    const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
    if (!ALLOWED.includes(ext)) {
      setState(state, "err", "unsupported type");
      continue;
    }

    setState(state, "", `<span class="spinner"></span> indexing…`);
    const form = new FormData();
    form.append("files", file);

    try {
      const res = await fetch("/api/documents", { method: "POST", body: form });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `HTTP ${res.status}`);
      const r = (await res.json()).results[0];

      if (r.status === "indexed") setState(state, "ok", `✓ ${r.doc.chunks} chunks`);
      else if (r.status === "duplicate") setState(state, "warn", "already in library");
      else setState(state, "err", esc(r.error));
    } catch (err) {
      setState(state, "err", esc(err.message));
    }
  }
  await loadDocuments();
  refreshStatus();
}

// ---------- library list ----------

async function loadDocuments() {
  const data = await fetch("/api/documents").then(r => r.json());
  documents = data.documents;
  renderDocuments();
}

function renderDocuments() {
  const q = docFilter.value.trim().toLowerCase();
  const shown = documents.filter(d => !q || d.title.toLowerCase().includes(q) || d.filename.toLowerCase().includes(q));
  $("#doc-count").textContent = documents.length ? `(${documents.length})` : "";

  if (!documents.length) {
    docList.innerHTML = `<div class="empty">No papers yet. Drop a few above to get started.</div>`;
    return;
  }
  if (!shown.length) {
    docList.innerHTML = `<div class="empty">Nothing matches “${esc(q)}”.</div>`;
    return;
  }

  docList.innerHTML = shown.map(d => {
    const ext = d.filename.split(".").pop().toUpperCase();
    const meta = [
      esc(d.filename),
      d.pages ? `${d.pages} pages` : null,
      `${d.chunks} chunks`,
      fmtBytes(d.size_bytes),
      fmtDate(d.uploaded_at),
    ].filter(Boolean).join(" · ");
    return `
      <div class="doc-row">
        <div class="doc-icon">${esc(ext)}</div>
        <div class="doc-main">
          <div class="doc-title" title="${esc(d.title)}">${esc(d.title)}</div>
          <div class="doc-meta">${meta}</div>
        </div>
        <a class="btn ghost" href="/ask?doc=${d.id}" title="Ask questions about this paper only">Ask</a>
        <a class="btn ghost" href="/api/documents/${d.id}/file" target="_blank">Open</a>
        <button class="btn ghost" data-rename="${d.id}">Rename</button>
        <button class="btn ghost danger" data-del="${d.id}">Delete</button>
      </div>`;
  }).join("");
}

docList.addEventListener("click", async e => {
  const renameId = e.target.dataset.rename;
  if (renameId) {
    const doc = documents.find(d => d.id === renameId);
    const title = prompt("Paper title", doc.title)?.trim();
    if (!title || title === doc.title) return;
    const res = await fetch(`/api/documents/${renameId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    });
    toast(res.ok ? "Title updated" : (await res.json()).detail);
    return loadDocuments();
  }

  const id = e.target.dataset.del;
  if (!id) return;
  const doc = documents.find(d => d.id === id);
  if (!confirm(`Remove “${doc.title}” from the library?`)) return;
  const res = await fetch(`/api/documents/${id}`, { method: "DELETE" });
  toast(res.ok ? "Paper removed" : (await res.json()).detail);
  await loadDocuments();
  refreshStatus();
});

docFilter.addEventListener("input", renderDocuments);

// ---------- re-index ----------

const banner = $("#reindex-banner");
const rebuildBtn = $("#rebuild-btn");
let pollTimer = null;

function renderReindex(s) {
  const busy = s.running;
  dropzone.classList.toggle("disabled", busy || s.needs_reindex);
  rebuildBtn.disabled = busy || !s.documents;

  if (busy) {
    const pct = s.total ? Math.round((s.done / s.total) * 100) : 0;
    const idle = s.updated_at ? Math.round(s.now - s.updated_at) : 0;
    const elapsed = s.started_at ? Math.round(s.now - s.started_at) : 0;
    const hint = s.offline
      ? `<div class="small" style="color:var(--warn)">Lost contact with the server — retrying…</div>`
      : idle > 90
        ? `<div class="small" style="color:var(--warn)">No progress for ${idle}s — check the terminal running ./run.sh</div>`
        : "";
    banner.className = "card banner busy";
    banner.innerHTML = `
      <span class="spinner"></span>
      <div class="grow">
        <b>Rebuilding index with ${esc(s.config_model)}</b> — paper ${Math.min(s.done + 1, s.total)} of ${s.total}
        <span class="muted small">· ${elapsed}s</span>
        ${s.current ? `<div class="muted small">${esc(s.current)}</div>` : ""}
        <div class="progress"><i style="width:${pct}%"></i></div>
        ${hint}
      </div>`;
  } else if (s.error) {
    banner.className = "card banner failed";
    banner.innerHTML = `
      <div class="grow"><b>Rebuild failed</b> — your previous index was kept.<div class="muted small">${esc(s.error)}</div></div>
      <button class="btn primary" data-rebuild>Try again</button>`;
  } else if (s.needs_reindex) {
    banner.className = "card banner";
    banner.innerHTML = `
      <div class="grow">
        <b>Embedding model changed.</b> Your library was indexed with <code>${esc(s.index_model)}</code>,
        but the app is now set to <code>${esc(s.config_model)}</code>. Rebuild to keep searching and uploading —
        titles and files are kept.
      </div>
      <button class="btn primary" data-rebuild>Rebuild index (${s.documents} papers)</button>`;
  } else {
    hideBanner();
    return;
  }
  clearTimeout(hideTimer);
  banner.hidden = false;
}

let hideTimer = null;

function hideBanner() {
  banner.hidden = true;
  banner.innerHTML = "";   // drop the spinner/progress so nothing keeps animating
}

function showDone(prev, s) {
  const secs = prev.started_at ? Math.round(s.now - prev.started_at) : null;
  banner.className = "card banner done";
  banner.innerHTML = `
    <span class="check">✓</span>
    <div class="grow"><b>Index rebuilt</b> with ${esc(s.config_model)} · ${s.documents} papers${secs !== null ? ` in ${secs}s` : ""}.
      Search and uploads are back on.</div>`;
  banner.hidden = false;
  clearTimeout(hideTimer);
  hideTimer = setTimeout(hideBanner, 6000);
}

let lastStatus = null;

async function checkReindex() {
  clearTimeout(pollTimer);
  let s;
  try {
    const res = await fetch("/api/reindex", { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    s = await res.json();
  } catch {
    // a failed poll must never freeze the banner: keep retrying while a rebuild is in flight
    if (lastStatus?.running) {
      renderReindex({ ...lastStatus, offline: true });
      pollTimer = setTimeout(checkReindex, 3000);
    }
    return;
  }

  const justFinished = lastStatus?.running && !s.running;
  renderReindex(s);
  if (s.running) {
    pollTimer = setTimeout(checkReindex, 1000);
  } else if (justFinished) {
    if (!s.error && !s.needs_reindex) showDone(lastStatus, s);   // errors keep their own banner
    loadDocuments();
    refreshStatus();
  }
  lastStatus = s;
}

// coming back to a background tab: refresh right away instead of waiting for a throttled timer
document.addEventListener("visibilitychange", () => { if (!document.hidden && lastStatus?.running) checkReindex(); });

async function startReindex() {
  const res = await fetch("/api/reindex", { method: "POST" });
  if (!res.ok) toast((await res.json()).detail);
  checkReindex();
}

banner.addEventListener("click", e => { if (e.target.dataset.rebuild !== undefined) startReindex(); });
rebuildBtn.addEventListener("click", () => {
  if (confirm("Re-parse and re-embed every paper with the current settings? Search is paused until it finishes.")) startReindex();
});

loadDocuments();
checkReindex();
