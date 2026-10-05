// Shared controls: tooltip, segmented buttons, select, legend, model chip.
import { h, methodColor } from "../lib/dom.js";
import { openInfo } from "./info.js";

/* ------------------------------------------------------------------ tooltip */

const tipEl = () => document.getElementById("tooltip");

function place(e) {
  const t = tipEl();
  const pad = 14;
  const { innerWidth: W, innerHeight: H } = window;
  const r = t.getBoundingClientRect();
  let x = e.clientX + pad, y = e.clientY + pad;
  if (x + r.width > W - 8) x = e.clientX - r.width - pad;
  if (y + r.height > H - 8) y = e.clientY - r.height - pad;
  t.style.left = `${Math.max(8, x)}px`;
  t.style.top = `${Math.max(8, y)}px`;
}

/** Attach a hover/focus tooltip; `render` returns a Node (built lazily on hover). */
export function tip(el, render) {
  const show = (e) => {
    const t = tipEl();
    t.replaceChildren(render());
    t.hidden = false;
    if (e.clientX != null) place(e);
    else {
      const r = el.getBoundingClientRect();
      place({ clientX: r.left, clientY: r.bottom });
    }
  };
  const hide = () => { tipEl().hidden = true; };
  el.addEventListener("mouseenter", show);
  el.addEventListener("mousemove", (e) => { if (!tipEl().hidden) place(e); });
  el.addEventListener("mouseleave", hide);
  el.addEventListener("focus", show);
  el.addEventListener("blur", hide);
  return el;
}

export function tipRows(title, value, rows = [], foot) {
  return h("div", {},
    h("div", { class: "tt-title" }, title),
    value != null && h("div", { class: "tt-value" }, value),
    rows.map(([k, v]) => h("div", { class: "tt-row" }, h("span", {}, k), h("span", {}, v))),
    foot && h("div", { class: "tt-muted", style: { marginTop: "6px" } }, foot),
  );
}

/* ------------------------------------------------------------------ controls */

export function segmented(options, value, onChange, label) {
  return h("div", { class: "seg", role: "group", "aria-label": label },
    options.map((o) => h("button", {
      type: "button", "aria-pressed": String(o.id === value), title: o.title || null,
      onclick: () => o.id !== value && onChange(o.id),
    }, o.label)));
}

export function select(options, value, onChange, label) {
  const groups = new Map();
  for (const o of options) {
    const g = o.group || "";
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push(o);
  }
  const optEls = (os) => os.map((o) => h("option", { value: o.id, selected: o.id === value }, o.label));
  const el = h("select", { class: "select", "aria-label": label, onchange: (e) => onChange(e.target.value) },
    [...groups].map(([g, os]) => (g ? h("optgroup", { label: g }, optEls(os)) : optEls(os))));
  return el;
}

export const field = (label, control) => h("label", { class: "field" }, h("span", { class: "label" }, label), control);

/** Colour key for baseline-centred cells: red below the baseline, grey at it, blue above. */
export function legend(measure, centredOn) {
  if (!measure) return null;
  const [lo, hi] = measure.legend || ["lower", "higher"];
  const c = measure.format === "pct" ? `${Math.round((measure.center ?? 0.5) * 100)}%` : String(measure.center ?? 0);
  return h("div", { class: "legend", title: "Cell shading" },
    h("span", {}, lo), h("span", { class: "ramp div" }), h("span", {}, hi),
    h("span", { class: "tt-muted" }, `· centred on ${centredOn || c}`));
}

/* ------------------------------------------------------------------ model display */

/** A model's display name; clicking it opens the info card (checkpoints, paths, evals). */
export function modelName(exp, m) {
  const method = exp.methodsById[m.method];
  const el = h("button", { class: "model-name model-link", type: "button", "aria-haspopup": "dialog" },
    h("span", { class: "dot", style: { background: methodColor(method) } }),
    h("span", { class: "label" }, m.label));
  el.addEventListener("click", (e) => { e.stopPropagation(); openInfo(exp, m, el); });
  return tip(el, () => modelTip(exp, m));
}

export function modelTip(exp, m) {
  const pre = m.pretrain || {};
  return tipRows(m.label, null, [
    ["Method", [exp.methodsById[m.method]?.label || m.method, m.code].filter(Boolean).join(" · ")],
    m.tags.length && ["Tags", m.tags.join(", ")],
    ["SFT step", m.step ?? "—"],
    ["Pretrain step", pre.step ?? "—"],
  ].filter(Boolean), "Click the name for checkpoint paths and eval runs.");
}

export function emptyState(title, body, action) {
  return h("div", { class: "empty" }, h("h3", {}, title), h("p", {}, body), action);
}
