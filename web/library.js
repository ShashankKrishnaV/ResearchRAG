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
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
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
        <button class="btn ghost danger" data-del="${d.id}">Delete</button>
      </div>`;
  }).join("");
}

docList.addEventListener("click", async e => {
  const id = e.target.dataset.del;
  if (!id) return;
  const doc = documents.find(d => d.id === id);
  if (!confirm(`Remove “${doc.title}” from the library?`)) return;
  await fetch(`/api/documents/${id}`, { method: "DELETE" });
  toast("Paper removed");
  await loadDocuments();
  refreshStatus();
});

docFilter.addEventListener("input", renderDocuments);
loadDocuments();
