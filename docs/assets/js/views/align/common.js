// Shared pieces of the Alignment Distillation tabs: formats, the stacked and grouped bar charts, and the run tooltip.
import { h } from "../../lib/dom.js";
import { hideTip, showTip, tipRows } from "../../components/ui.js";
import { responsive, svg } from "../../components/charts.js";
import { put, section, withExtra } from "../distill/common.js";

export { put, section };

/** "21.6": half rounds away from zero, as in the collector's texts. */
export const f1 = (v) => (v == null ? "—" : (v * 100 + (v >= 0 ? 1e-9 : -1e-9)).toFixed(1).replace("-", "−"));
/** "+10.6", in points. */
export const pts = (v) => (v == null ? "—" : `${v > 0 ? "+" : ""}${f1(v)}`);
export const nats = (v, digits = 2) => (v == null ? "—" : v.toFixed(digits).replace("-", "−"));
export const count = (v) => (v == null ? "—" : Number(v).toLocaleString("en-US"));

/** Draws once all the named extra files have loaded. */
export function withAll(root, exp, names, draw) {
  const step = (i, got) => (i === names.length ? draw(...got)
    : withExtra(root, exp, names[i], `the ${names[i]} data`, (d) => step(i + 1, [...got, d])));
  step(0, []);
}

export const note = (...xs) => h("p", { class: "section-note" }, xs);

export function runTip(run, scope = "all") {
  const s = run.scopes[scope];
  return tipRows(run.label, `${f1(s.am)}% misaligned`, [
    scope === "all" && ["95% over replies", `${f1(run.ci[0])} – ${f1(run.ci[1])}`],
    scope === "all" && ["95% over evals", `${f1(run.ci_evals[0])} – ${f1(run.ci_evals[1])}`],
    ["Safe, uses tools", f1(s.safe_call)], ["Safe, no tool call", f1(s.safe_nocall)],
    ["Answers the colleague", f1(s.colleague)], ["Replies", count(s.n)],
  ].filter(Boolean), run.what);
}

/** Legend for bar segments. */
export const barKey = (items) => h("div", { class: "al-key" }, items.map(([fill, label]) =>
  h("span", { class: "item" }, h("i", { style: { background: fill } }), label)));

function labelText(text, x, y, width, cls) {
  const fit = Math.floor(width / 6.4);
  const shown = text.length > fit ? `${text.slice(0, fit - 1).trimEnd()}…` : text;
  return svg("text", { x, y, class: cls, "text-anchor": "end" }, shown, shown !== text ? svg("title", {}, text) : null);
}

function hover(el, render) {
  if (!render) return;
  const show = (e) => showTip(render(), e, el);
  el.addEventListener("pointermove", show);
  el.addEventListener("pointerleave", hideTip);
  el.addEventListener("focus", show);
  el.addEventListener("blur", hideTip);
}

/**
 * Stacked bars of shares that sum to 1. `items`: {head} for a group's header, or {label, strong, parts: [{v, fill}],
 * value, tip}. The value is printed right of the bar.
 */
export function stackRows(host, { items, title, valueLabel }) {
  responsive(host, (W) => {
    const lw = Math.min(300, Math.round(W * 0.44)), x0 = lw + 10, pw = Math.max(40, W - x0 - 52);
    const height = (it) => (it.head ? 34 : 23);
    const root = svg("svg", { width: W, height: 22 + items.reduce((a, it) => a + height(it), 0), role: "img", "aria-label": title });
    root.append(svg("text", { x: W - 4, y: 14, class: "tick", "text-anchor": "end" }, valueLabel));
    let y = 22;
    for (const it of items) {
      if (it.head) {
        root.append(svg("text", { x: 0, y: y + 23, class: "row-label strong" }, it.head));
      } else {
        root.append(labelText(it.label, lw, y + 15, lw, it.strong ? "row-label strong" : "row-label"));
        let x = x0;
        for (const p of it.parts) {
          const w = Math.max(0, (p.v || 0) * pw);
          if (w > 0.4) root.append(svg("rect", { x, y: y + 4, width: Math.max(0.6, w - 2), height: 14, rx: 2, style: `fill: ${p.fill}` }));
          x += w;
        }
        root.append(svg("text", { x: W - 4, y: y + 15, class: "bar-num strong", "text-anchor": "end" }, it.value));
        const hit = svg("rect", { x: 0, y, width: W, height: 23, class: "hit", tabindex: "0" });
        hover(hit, it.tip);
        root.append(hit);
      }
      y += height(it);
    }
    host.replaceChildren(root);
  });
}

/** Grouped horizontal bars from zero. `rows`: [{label, bars: [{v, fill}]}]; every bar's value is printed at its end. */
export function barRows(host, { rows, max = 1, format = f1, title, labelWidth = 250 }) {
  responsive(host, (W) => {
    const lw = Math.min(labelWidth, Math.round(W * 0.44)), x0 = lw + 10, pw = Math.max(40, W - x0 - 46);
    const height = (r) => r.bars.length * 16 + 14;
    const root = svg("svg", { width: W, height: rows.reduce((a, r) => a + height(r), 4), role: "img", "aria-label": title });
    let y = 4;
    for (const r of rows) {
      root.append(labelText(r.label, lw, y + (r.bars.length * 16) / 2 + 4, lw, r.strong ? "row-label strong" : "row-label"));
      r.bars.forEach((b, i) => {
        const w = b.v == null ? 0 : Math.max(1, (Math.max(0, b.v) / max) * pw);
        if (b.v != null) root.append(svg("rect", { x: x0, y: y + i * 16, width: w, height: 12, rx: 2, style: `fill: ${b.fill}` }));
        root.append(svg("text", { x: x0 + w + 5, y: y + i * 16 + 10, class: "bar-num" }, format(b.v)));
      });
      y += height(r);
    }
    host.replaceChildren(root);
  });
}

/** A plain table: `head` cells, `rows` of cells; a cell may be {v, cls}. Every column but the first `left` is numeric. */
export function table(head, rows, { left = 1, cls = "" } = {}) {
  const cell = (tag, c, i) => {
    const obj = c != null && typeof c === "object" && !(c instanceof Node) && !Array.isArray(c);
    return h(tag, { class: [i >= left ? "num" : "", obj ? c.cls : ""].filter(Boolean).join(" ") || null }, obj ? c.v : c);
  };
  return h("div", { class: "table-wrap" }, h("table", { class: `grid compact al-table ${cls}` },
    h("thead", {}, h("tr", {}, head.map((c, i) => cell("th", c, i)))),
    h("tbody", {}, rows.map((r) => (r.group
      ? h("tr", { class: "group-row" }, h("td", { colspan: head.length }, r.group))
      : h("tr", {}, r.map((c, i) => cell("td", c, i))))))));
}
