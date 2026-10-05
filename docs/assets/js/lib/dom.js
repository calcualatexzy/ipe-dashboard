// Tiny DOM + formatting helpers shared by every view.

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else if (k === "html") el.innerHTML = v;
    else if (k in el && typeof v !== "string") el[k] = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  append(el, children);
  return el;
}

function append(el, children) {
  for (const c of children.flat(Infinity)) {
    if (c == null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
}

export const clear = (el) => { el.replaceChildren(); return el; };

export function fmtValue(v, format) {
  if (v == null || Number.isNaN(v)) return "—";
  if (format === "pct") return (v * 100).toFixed(1);
  return (v >= 0 ? "" : "−") + Math.abs(v).toFixed(3);
}

export function fmtDelta(d, format) {
  if (d == null || Number.isNaN(d)) return "";
  const abs = Math.abs(d);
  if (format === "pct") return `${(abs * 100).toFixed(1)} pt`;
  return abs.toFixed(3);
}

export function fmtDate(iso, withTime = false) {
  if (!iso) return "—";
  const d = new Date(iso.endsWith("Z") ? iso : iso + "Z");
  const opts = { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" };
  if (withTime) Object.assign(opts, { hour: "2-digit", minute: "2-digit", hour12: false });
  return d.toLocaleString("en-GB", opts);
}

/** Cell background diverging around `center` (the baseline): blue above, red below, grey at it. */
export function heat(v, center, range) {
  if (v == null || center == null) return null;
  const MAX = 58; // keep ink readable on the strongest tint
  const t = Math.max(-1, Math.min(1, (v - center) / (range || 1)));
  const pole = t >= 0 ? "var(--div-pos)" : "var(--div-neg)";
  return `color-mix(in oklab, ${pole} ${Math.round(Math.abs(t) * MAX)}%, var(--div-mid))`;
}

export const methodColor = (method) => `var(--s${method?.color ?? 0})`;

export function mean(xs) {
  const v = xs.filter((x) => x != null && !Number.isNaN(x));
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
}

export function std(xs) {
  const v = xs.filter((x) => x != null);
  if (v.length < 2) return null;
  const m = mean(v);
  return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / (v.length - 1));
}

export const chevron = () => {
  const s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  s.setAttribute("viewBox", "0 0 16 16"); s.setAttribute("width", "12"); s.setAttribute("height", "12");
  s.innerHTML = '<path d="M6 3l5 5-5 5" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>';
  return s;
};
