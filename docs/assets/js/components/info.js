// Model info card: opens on click of any model name. Holds what the compact tables leave out —
// checkpoint steps and relative paths (copyable), lineage, and every eval that loaded the model.
import { fmtDate, h, methodColor } from "../lib/dom.js";

let current = null;

function copyButton(text) {
  const btn = h("button", { class: "btn small ghost copy", type: "button", "aria-label": "Copy path" }, "Copy");
  btn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(text);
      btn.textContent = "Copied";
    } catch {
      // Clipboard API needs a secure context; fall back to selecting the path.
      const code = btn.previousElementSibling;
      const range = document.createRange();
      range.selectNodeContents(code);
      getSelection().removeAllRanges();
      getSelection().addRange(range);
      btn.textContent = "Selected";
    }
    setTimeout(() => { btn.textContent = "Copy"; }, 1400);
  });
  return btn;
}

export function closeInfo() {
  current?.close();
}

export function openInfo(exp, m, anchor) {
  if (current?.model === m.id) { closeInfo(); return; }
  closeInfo();
  const pre = m.pretrain || {};
  const method = exp.methodsById[m.method];
  const sets = (exp.manifest.sets || []).filter((s) => s.models.includes(m.id)).map((s) => s.label);
  const evals = exp.evals.filter((e) => e.model === m.id).sort((a, b) => (b.date || "").localeCompare(a.date || ""));
  const row = (k, ...v) => h("div", { class: "info-row" }, h("span", { class: "k" }, k), h("span", { class: "v" }, ...v));
  const path = (p) => p ? [h("code", { class: "path" }, p), copyButton(p)] : [h("span", { class: "tt-muted" }, "unknown")];

  const card = h("div", { class: "popover info-card", role: "dialog", "aria-label": `${m.label} details` },
    h("header", {},
      h("div", { class: "eyebrow" }, [method?.label || m.method, m.code && `code ${m.code}`].filter(Boolean).join(" · ")),
      h("div", { class: "info-title" }, h("span", { class: "dot", style: { background: methodColor(method) } }), m.label),
      m.description && h("p", { class: "section-note", style: { margin: 0 } }, m.description),
      m.note && h("p", { class: "section-note", style: { margin: 0 } }, m.note)),
    h("div", { class: "info-body" },
      h("div", { class: "info-group" }, "Checkpoints"),
      row("SFT", `step ${m.step ?? "?"}`, !m.sft.resolved && h("span", { class: "tag warn" }, "folder missing")),
      row("", ...path(m.sft.path)),
      row("Pretrain", pre.step != null ? `step ${pre.step}` : "unknown"),
      row("", ...path(pre.path)),
      h("div", { class: "info-group" }, "Training"),
      row("Tags", m.tags.length ? m.tags.join(", ") : "—"),
      row("SFT data", `${m.data ?? "?"} · ${m.anchors == null ? "anchors unknown" : m.anchors ? "with anchors" : "no anchors"}`),
      row("Trainer", pre.trainer ? `${pre.trainer} · reflection weight ${pre.reflection_weight}` : "—"),
      row("Run name", h("code", { class: "path" }, m.abbrev)),
      sets.length && row("Sets", sets.join(", ")),
      h("div", { class: "info-group" }, `Eval runs (${evals.length})`),
      evals.map((e) => row(fmtDate(e.date),
        h("code", { class: "path" }, e.label),
        h("span", { class: "tt-muted" }, ` · ${exp.splitsById[e.split]?.short || e.split} · ${e.prompts.length > 1 ? `${e.prompts.length} prompts` : e.prompts[0]}`)))));

  const r = anchor.getBoundingClientRect();
  const width = Math.min(560, window.innerWidth - 32);
  card.style.width = `${width}px`;
  card.style.top = `${r.bottom + window.scrollY + 8}px`;
  card.style.left = `${Math.max(16, Math.min(r.left + window.scrollX, window.scrollX + window.innerWidth - 16 - width))}px`;
  document.body.append(card);
  document.getElementById("tooltip").hidden = true;

  const outside = (e) => { if (!card.contains(e.target) && !anchor.contains(e.target)) close(); };
  const esc = (e) => { if (e.key === "Escape") { close(); anchor.focus(); } };
  function close() {
    card.remove();
    document.removeEventListener("mousedown", outside, true);
    document.removeEventListener("keydown", esc);
    current = null;
  }
  document.addEventListener("mousedown", outside, true);
  document.addEventListener("keydown", esc);
  current = { model: m.id, close };
}
