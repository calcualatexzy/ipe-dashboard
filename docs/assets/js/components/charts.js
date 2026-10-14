// Small SVG charts for the MSM Training and Likelihood tabs: a line chart with a crosshair, and dot + interval rows.
// Marks wear the categorical slots (--s1 … --s8); text stays in ink tokens. Both redraw on resize.
import { h } from "../lib/dom.js";
import { hideTip, showTip } from "./ui.js";

const NS = "http://www.w3.org/2000/svg";

export function svg(tag, attrs = {}, ...children) {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null) el.setAttribute(k, v);
  for (const c of children.flat()) if (c != null) el.append(c);
  return el;
}

/** Calls `draw(width)` now and on every resize, until the host leaves the page. */
export function responsive(host, draw) {
  const ro = new ResizeObserver(() => {
    if (!host.isConnected) return ro.disconnect(); // the view was re-rendered
    if (host.clientWidth) draw(host.clientWidth);
  });
  ro.observe(host);
}

/** About `count` round tick values covering [lo, hi]. */
export function niceTicks(lo, hi, count = 4) {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((x) => x * mag).find((x) => x >= raw) || raw;
  const out = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out;
}

/** "−0.25" with a true minus; `signed` adds "+" to positives. */
export function num(v, digits = 3, signed = false) {
  if (v == null || Number.isNaN(v)) return "—";
  const s = Math.abs(v).toFixed(digits);
  return v < 0 && Number(s) !== 0 ? `−${s}` : signed && Number(s) !== 0 ? `+${s}` : s;
}

export const pct = (v, digits = 1) => (v == null ? "—" : `${(v * 100).toFixed(digits)}%`);

/** Tick text: short, the axis carries the unit. */
function tickText(v, format, step) {
  if (format === "pct") return `${Math.round(v * 100)}%`;
  const d = [0, 1, 2, 3].find((k) => Math.abs(step * 10 ** k - Math.round(step * 10 ** k)) < 1e-6) ?? 3;
  return num(v, d);
}

/** Pads a [lo, hi] extent by 6% (a flat line gets ±1) and, with `zero`, stretches it to include 0. */
export function extent(values, { zero = false, domain = null } = {}) {
  if (domain) return domain;
  const v = values.filter((x) => x != null && Number.isFinite(x));
  if (!v.length) return [0, 1];
  let lo = Math.min(...v), hi = Math.max(...v);
  if (zero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
  const pad = (hi - lo) * 0.06 || Math.abs(hi) * 0.1 || 1;
  return [lo - pad, hi + pad];
}

/**
 * Line chart. `series`: [{label, slot, points: [[x, y], …]}]; `x`: {max, label}; `y`: {domain, format, label}.
 * Hovering (or arrow keys on focus) moves a crosshair to the nearest logged x and lists every series there.
 */
export function lineChart(host, { series, x, y, height = 190, title }) {
  responsive(host, (W) => {
    const H = height, M = { top: 10, right: 14, bottom: 30, left: 50 };
    const pw = W - M.left - M.right, ph = H - M.top - M.bottom;
    const [y0, y1] = y.domain;
    const xMax = x.max || Math.max(1, ...series.flatMap((s) => s.points.map((p) => p[0])));
    const sx = (v) => M.left + (v / xMax) * pw;
    const sy = (v) => M.top + ph * (1 - (Math.max(y0, Math.min(y1, v)) - y0) / (y1 - y0));
    const root = svg("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": title });

    const yt = niceTicks(y0, y1, 4);
    const ystep = yt.length > 1 ? yt[1] - yt[0] : 1;
    for (const t of yt) {
      if (t < y0 || t > y1) continue;
      root.append(svg("line", { x1: M.left, x2: W - M.right, y1: sy(t), y2: sy(t), class: t === 0 && y0 < 0 ? "axis" : "grid" }),
        svg("text", { x: M.left - 8, y: sy(t) + 4, class: "tick", "text-anchor": "end" }, tickText(t, y.format, ystep)));
    }
    root.append(svg("line", { x1: M.left, x2: W - M.right, y1: M.top + ph, y2: M.top + ph, class: "axis" }));
    for (const t of niceTicks(0, xMax, Math.max(2, Math.floor(pw / 90)))) {
      root.append(svg("text", { x: sx(t), y: H - M.bottom + 16, class: "tick", "text-anchor": "middle" }, t.toLocaleString("en-US")));
    }
    if (x.label) root.append(svg("text", { x: W - M.right, y: H - 2, class: "xsub", "text-anchor": "end" }, x.label));

    for (const s of series) {
      const pts = s.points.filter((p) => p[1] != null);
      if (!pts.length) continue;
      root.append(svg("path", { d: pts.map((p, i) => `${i ? "L" : "M"}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join(""),
        class: "line", style: `stroke: var(--s${s.slot})` }));
      const [lx, ly] = pts[pts.length - 1];
      root.append(svg("circle", { cx: sx(lx), cy: sy(ly), r: 3.5, class: "end-dot", style: `fill: var(--s${s.slot})` }));
    }

    // Crosshair: snaps to the nearest logged x of any series.
    const xs = [...new Set(series.flatMap((s) => s.points.filter((p) => p[1] != null).map((p) => p[0])))].sort((a, b) => a - b);
    const cross = svg("line", { y1: M.top, y2: M.top + ph, class: "crosshair", visibility: "hidden" });
    const marks = svg("g", { visibility: "hidden" });
    const overlay = svg("rect", { x: M.left, y: M.top, width: pw, height: ph, class: "overlay", tabindex: "0" });
    let at = -1;
    const readout = (i, e) => {
      if (!xs.length) return;
      at = Math.max(0, Math.min(xs.length - 1, i));
      const xv = xs[at];
      cross.setAttribute("x1", sx(xv)); cross.setAttribute("x2", sx(xv)); cross.setAttribute("visibility", "visible");
      marks.replaceChildren();
      const rows = [];
      for (const s of series) {
        const p = s.points.find((q) => q[0] === xv);
        if (!p || p[1] == null) continue;
        marks.append(svg("circle", { cx: sx(xv), cy: sy(p[1]), r: 4, class: "end-dot", style: `fill: var(--s${s.slot})` }));
        rows.push(h("div", { class: "tt-row" },
          h("span", { class: "tt-key" }, h("i", { class: "line-key", style: { background: `var(--s${s.slot})` } }), s.label),
          h("b", {}, y.format === "pct" ? pct(p[1]) : num(p[1], 3))));
      }
      marks.setAttribute("visibility", "visible");
      showTip(h("div", {}, h("div", { class: "tt-title" }, `${title} · ${x.unit || "step"} ${xv.toLocaleString("en-US")}`), rows), e, overlay);
    };
    const nearest = (e) => {
      const r = overlay.getBoundingClientRect();
      const xv = ((e.clientX - r.left) / r.width) * xMax;
      let best = 0;
      xs.forEach((v, i) => { if (Math.abs(v - xv) < Math.abs(xs[best] - xv)) best = i; });
      return best;
    };
    const leave = () => { cross.setAttribute("visibility", "hidden"); marks.setAttribute("visibility", "hidden"); hideTip(); };
    overlay.addEventListener("pointermove", (e) => readout(nearest(e), e));
    overlay.addEventListener("pointerleave", leave);
    overlay.addEventListener("focus", () => readout(at < 0 ? xs.length - 1 : at));
    overlay.addEventListener("blur", leave);
    overlay.addEventListener("keydown", (e) => {
      if (e.key === "ArrowLeft" || e.key === "ArrowRight") { e.preventDefault(); readout(at + (e.key === "ArrowLeft" ? -1 : 1)); }
    });
    root.append(cross, marks, overlay);
    host.replaceChildren(root);
  });
}

/**
 * Dot + interval rows on one horizontal axis with a zero line. `rows`: [{label, bold, points: [{x, lo, hi, slot,
 * hollow, tip}]}], where `tip()` returns the tooltip node. `x`: {label, format}.
 */
export function intervalRows(host, { rows, x, title, labelWidth = 220 }) {
  responsive(host, (W) => {
    const rowH = 26;
    const lw = Math.min(labelWidth, Math.round(W * 0.45));
    const M = { top: 6, right: 16, bottom: 34, left: lw + 8 };
    const H = M.top + rows.length * rowH + M.bottom;
    const pw = W - M.left - M.right;
    const vals = rows.flatMap((r) => r.points.flatMap((p) => [p.x, p.lo, p.hi]));
    const [x0, x1] = extent([...vals, 0], { zero: true });
    const sx = (v) => M.left + ((v - x0) / (x1 - x0)) * pw;
    const root = svg("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": title });
    const ticks = niceTicks(x0, x1, Math.max(2, Math.floor(pw / 80)));
    const step = ticks.length > 1 ? ticks[1] - ticks[0] : 1;
    const base = M.top + rows.length * rowH;
    for (const t of ticks) {
      if (t < x0 || t > x1) continue;
      root.append(svg("line", { x1: sx(t), x2: sx(t), y1: M.top, y2: base, class: t === 0 ? "axis" : "grid" }),
        svg("text", { x: sx(t), y: base + 16, class: "tick", "text-anchor": "middle" }, tickText(t, x.format, step)));
    }
    if (x.label) root.append(svg("text", { x: W - M.right, y: H - 2, class: "xsub", "text-anchor": "end" }, x.label));
    rows.forEach((r, i) => {
      const cy = M.top + i * rowH + rowH / 2;
      // Labels that won't fit the label column are cut (about 6.6px a character); the tooltip has the full name.
      const fit = Math.floor(lw / 6.6);
      const text = r.label.length > fit ? `${r.label.slice(0, fit - 1).trimEnd()}…` : r.label;
      root.append(svg("text", { x: lw, y: cy + 4, class: r.bold ? "row-label strong" : "row-label", "text-anchor": "end" }, text,
        text !== r.label ? svg("title", {}, r.label) : null));
      for (const p of r.points) {
        if (p.lo != null && p.hi != null) {
          root.append(svg("line", { x1: sx(p.lo), x2: sx(p.hi), y1: cy, y2: cy, class: "interval", style: `stroke: var(--s${p.slot})` }));
        }
        root.append(svg("circle", { cx: sx(p.x), cy, r: 4.5, class: p.hollow ? "dot-mark hollow" : "dot-mark",
          style: p.hollow ? `stroke: var(--s${p.slot})` : `fill: var(--s${p.slot})` }));
      }
      // Hit target: the whole row.
      const hit = svg("rect", { x: 0, y: cy - rowH / 2, width: W, height: rowH, class: "hit", tabindex: "0" });
      const show = (e) => r.points[0]?.tip && showTip(r.points[0].tip(), e, hit);
      hit.addEventListener("pointermove", show);
      hit.addEventListener("pointerleave", hideTip);
      hit.addEventListener("focus", show);
      hit.addEventListener("blur", hideTip);
      root.append(hit);
    });
    host.replaceChildren(root);
  });
}

/** Legend item for a line series: a short stroke of its colour, then its name. */
export const lineKey = (slot, label) => h("span", { class: "item" }, h("i", { class: "line-key", style: { background: `var(--s${slot})` } }), label);
