// Shared pieces of the SPP Evals tabs: run labels and colours, the two-benchmark dot chart, a scatter, formats.
import { h } from "../../lib/dom.js";
import { hideTip, showTip, tipRows } from "../../components/ui.js";
import { niceTicks, responsive, svg } from "../../components/charts.js";
import { count, f1, note, pts, put, section, table, withAll } from "../align/common.js";

export { count, f1, note, pts, put, section, table, withAll };

/** "0.87", with a true minus. */
export const num = (v, digits = 2) => (v == null ? "—" : v.toFixed(digits).replace("-", "−"));
/** McNemar p: "0.18", "3e-4", "<1e-10". */
export const pval = (p) => (p == null ? "—" : p >= 0.01 ? p.toFixed(2) : p >= 1e-10 ? p.toExponential(0).replace("e-", "e−") : "<1e−10");

export const specOf = (spp, run) => spp.specs.find((s) => s.id === run.spec) || spp.specs[0];
/** A run's colour: the sign of its spec (+ blue, − red), q amber, trained purple, none grey. */
export const slotOf = (spp, run) => specOf(spp, run).color;
export const groupOf = (spp, run) => spp.groups.find((g) => g.id === run.group);
/** "Instruct + SPP constitution", "Pretrained · No spec", "MSM + AFT, no CoT (paper)". */
export function fullLabel(spp, run) {
  if (run.group === "trained") return run.label;
  const g = groupOf(spp, run).short;
  return run.spec === "none" ? `${g}, no spec` : `${g} + ${run.label}`;
}

/** Runs by group, in registry order, each group with its runs that have a result on any benchmark. */
export const grouped = (spp, keep = () => true) => spp.groups
  .map((g) => ({ g, runs: spp.runs.filter((r) => r.group === g.id && keep(r)) })).filter((x) => x.runs.length);

/** The hover card of a run: both benchmarks with their intervals, and what the run is. */
export function runTip(spp, run) {
  const spec = specOf(spp, run);
  const rows = spp.benches.flatMap((b) => {
    const res = run.results[b.id];
    return res ? [[b.label, `${f1(res.value)}% ${b.better === "high" ? "right" : "risky"} [${f1(res.ci[0])}, ${f1(res.ci[1])}]`],
      [`${b.label} format`, `${f1(res.format)}%`]] : [[b.label, "not scored"]];
  });
  const what = [run.what, spec.what, run.para && "Then the values paragraph: follow the constitution (or the Model Spec) above."].filter(Boolean).join(" ");
  return tipRows(fullLabel(spp, run), null, rows, what || null);
}

export const keyItem = (slot, label, hollow = false) => h("span", { class: "item" },
  h("i", { class: "round", style: hollow ? { background: "transparent", boxShadow: `inset 0 0 0 2px var(--s${slot})` } : { background: `var(--s${slot})` } }), label);

/** The legend of the colours by spec. */
export const specKey = () => h("div", { class: "al-key" },
  keyItem(1, "Spec with the values we want (SPP, pro-align)"), keyItem(8, "Its mirror (neg-SPP, neg-align)"),
  keyItem(0, "No spec"), keyItem(4, "KD target q"), keyItem(7, "Trained, no spec"), keyItem(0, "+ values paragraph", true),
  h("span", { class: "item" }, h("i", { class: "line-key dashed-key" }), "Instruct, no spec"));

function cut(text, width) {
  const fit = Math.floor(width / 6.4);
  return text.length > fit ? `${text.slice(0, fit - 1).trimEnd()}…` : text;
}

function hover(el, render) {
  const show = (e) => showTip(render(), e, el);
  el.addEventListener("pointermove", show);
  el.addEventListener("pointerleave", hideTip);
  el.addEventListener("focus", show);
  el.addEventListener("blur", hideTip);
}

/**
 * Dot + interval rows in one panel per benchmark, sharing the row labels. `groups`: [{label, rows: [{label, strong,
 * values: {<panel key>: {v, lo, hi, slot, hollow}}, tip}]}]; `panels`: [{key, label, sub, domain, ref, format}]. A panel's
 * `ref` draws a dashed line (a reference value, or 0 for differences). Values are printed beside their interval.
 */
export function dualDots(host, { groups, panels, title, labelWidth = 230, format = f1 }) {
  responsive(host, (W) => {
    const lw = Math.min(labelWidth, Math.round(W * 0.34)), gap = 22, rowH = 22, headH = 24, top = 34, bottom = 26;
    const pw = Math.max(80, (W - lw - 10 - gap * panels.length) / panels.length);
    const x0 = (i) => lw + 10 + gap * i + pw * i;
    const head = (g) => (g.label ? headH : 0);
    const H = top + groups.reduce((a, g) => a + head(g) + g.rows.length * rowH, 0) + bottom;
    const root = svg("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": title });
    const scales = panels.map((p, i) => (v) => x0(i) + 6 + ((Math.max(p.domain[0], Math.min(p.domain[1], v)) - p.domain[0]) / (p.domain[1] - p.domain[0])) * (pw - 12));
    const base = H - bottom;
    panels.forEach((p, i) => {
      root.append(svg("text", { x: x0(i), y: 12, class: "row-label strong" }, p.label),
        svg("text", { x: x0(i), y: 26, class: "tick" }, p.sub || ""));
      for (const t of niceTicks(p.domain[0], p.domain[1], Math.max(2, Math.floor(pw / 70)))) {
        if (t < p.domain[0] || t > p.domain[1]) continue;
        root.append(svg("line", { x1: scales[i](t), x2: scales[i](t), y1: top, y2: base, class: "grid" }),
          svg("text", { x: scales[i](t), y: base + 15, class: "tick", "text-anchor": "middle" }, (p.format || ((v) => `${Math.round(v * 100)}`))(t)));
      }
      if (p.ref != null) root.append(svg("line", { x1: scales[i](p.ref), x2: scales[i](p.ref), y1: top, y2: base, class: "ref-line" }));
    });
    let y = top;
    for (const g of groups) {
      if (g.label) root.append(svg("text", { x: 0, y: y + 17, class: "row-label strong" }, g.label));
      y += head(g);
      for (const r of g.rows) {
        const cy = y + rowH / 2;
        root.append(svg("text", { x: lw, y: cy + 4, class: r.strong ? "row-label strong" : "row-label", "text-anchor": "end" },
          cut(r.label, lw), r.label.length > Math.floor(lw / 6.4) ? svg("title", {}, r.label) : null));
        panels.forEach((p, i) => {
          const v = r.values[p.key];
          if (!v || v.v == null) {
            root.append(svg("text", { x: x0(i) + 6, y: cy + 4, class: "tick" }, "not scored"));
            return;
          }
          const sx = scales[i];
          if (v.lo != null) root.append(svg("line", { x1: sx(v.lo), x2: sx(v.hi), y1: cy, y2: cy, class: "interval", style: `stroke: var(--s${v.slot})` }));
          root.append(svg("circle", { cx: sx(v.v), cy, r: 4.5, class: v.hollow ? "dot-mark hollow" : "dot-mark",
            style: v.hollow ? `stroke: var(--s${v.slot})` : `fill: var(--s${v.slot})` }));
          const end = sx(v.hi ?? v.v) + 6, text = format(v.v);
          const right = end + text.length * 6.5 < x0(i) + pw;
          root.append(svg("text", { x: right ? end : sx(v.lo ?? v.v) - 6, y: cy + 4, class: "bar-num halo", "text-anchor": right ? "start" : "end" }, text));
        });
        const hit = svg("rect", { x: 0, y, width: W, height: rowH, class: "hit", tabindex: "0" });
        if (r.tip) hover(hit, r.tip);
        root.append(hit);
        y += rowH;
      }
    }
    host.replaceChildren(root);
  });
}

/** A square scatter with the diagonal and both zero lines. `points`: [[x, y]]; both axes share the domain. */
export function scatter(host, { points, slot, title, xLabel, yLabel, size = 300 }) {
  responsive(host, (W) => {
    const S = Math.min(size, W), M = { top: 8, right: 8, bottom: 34, left: 44 };
    const p = S - M.left - M.right;
    const all = points.flat();
    const lo = Math.min(0, ...all), hi = Math.max(0, ...all), pad = (hi - lo) * 0.04;
    const d = [lo - pad, hi + pad];
    const sc = (v) => ((v - d[0]) / (d[1] - d[0])) * p;
    const sx = (v) => M.left + sc(v), sy = (v) => M.top + p - sc(v);
    const root = svg("svg", { width: S, height: S, viewBox: `0 0 ${S} ${S}`, role: "img", "aria-label": title });
    for (const t of niceTicks(d[0], d[1], 4)) {
      if (t < d[0] || t > d[1]) continue;
      root.append(svg("line", { x1: sx(t), x2: sx(t), y1: M.top, y2: M.top + p, class: t === 0 ? "axis" : "grid" }),
        svg("line", { x1: M.left, x2: M.left + p, y1: sy(t), y2: sy(t), class: t === 0 ? "axis" : "grid" }),
        svg("text", { x: sx(t), y: M.top + p + 14, class: "tick", "text-anchor": "middle" }, num(t, 0)),
        svg("text", { x: M.left - 6, y: sy(t) + 4, class: "tick", "text-anchor": "end" }, num(t, 0)));
    }
    root.append(svg("line", { x1: sx(d[0]), y1: sy(d[0]), x2: sx(d[1]), y2: sy(d[1]), class: "ref-line" }));
    const g = svg("g", { class: "scatter-dots", style: `fill: var(--s${slot})` });
    for (const [x, y] of points) g.append(svg("circle", { cx: sx(x), cy: sy(y), r: 2.2 }));
    root.append(g,
      svg("text", { x: M.left + p, y: S - 4, class: "xsub", "text-anchor": "end" }, xLabel),
      svg("text", { x: 12, y: M.top, class: "xsub", transform: `rotate(-90 12 ${M.top})`, "text-anchor": "end" }, yLabel));
    host.replaceChildren(root);
  });
}

/** A benchmark switch's options. */
export const benchOptions = (spp) => spp.benches.map((b) => ({ id: b.id, label: b.label }));
