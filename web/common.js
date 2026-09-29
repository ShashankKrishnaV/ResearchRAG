// shared helpers for both pages

const $ = (sel, root = document) => root.querySelector(sel);

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function fmtBytes(n) {
  if (!n) return "";
  const u = ["B", "KB", "MB", "GB"];
  const i = Math.min(Math.floor(Math.log(n) / Math.log(1024)), u.length - 1);
  return `${(n / 1024 ** i).toFixed(i ? 1 : 0)} ${u[i]}`;
}

function fmtDate(iso) {
  return new Date(iso).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

function toast(msg) {
  let el = $(".toast");
  if (!el) {
    el = document.createElement("div");
    el.className = "toast";
    document.body.appendChild(el);
  }
  el.textContent = msg;
  el.classList.add("show");
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.remove("show"), 2200);
}

const store = {
  get(key, fallback) {
    try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; }
  },
  set(key, val) {
    try { localStorage.setItem(key, JSON.stringify(val)); } catch { /* private mode etc. */ }
  },
};

// header status pills: paper count + LLM availability
async function refreshStatus() {
  const box = $("#status");
  if (!box) return;
  try {
    const h = await fetch("/api/health").then(r => r.json());
    const llm = h.llm.model_ready
      ? `<span class="dot ok"></span>${esc(h.llm.model)}`
      : h.llm.reachable
        ? `<span class="dot warn"></span>run: ollama pull ${esc(h.llm.model)}`
        : `<span class="dot err"></span>LLM offline`;
    box.innerHTML = `
      <span class="pill">${h.index.documents} papers · ${h.index.chunks} chunks</span>
      <span class="pill" title="Answers are generated locally via Ollama">${llm}</span>`;
  } catch {
    box.innerHTML = `<span class="pill"><span class="dot err"></span>server offline</span>`;
  }
}

document.addEventListener("DOMContentLoaded", refreshStatus);
