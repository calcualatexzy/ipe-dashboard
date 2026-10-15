// Level bar chart: absolute percentages per level for chosen models, with whiskers from the
// per-sample estimates (each eval samples every question k times; estimate i uses the i-th response).
// It has its own model / split / prompt / level / measure controls, kept in the URL as pm, ps, pp (+ppf), pl, pmeas, perr.
import { fmtValue, h, mean, std } from "../lib/dom.js";
import { setParams } from "../lib/state.js";
import { field, segmented, select, tip, tipRows } from "../components/ui.js";
import { cellValue, levelToggle, meanFamily, perSample, pickLevels, promptControl, promptLabel, resolvePrompt } from "./common.js";

const MAX_SERIES = 8; // categorical palette slots; a 9th model never gets a generated colour
const T975 = { 1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262 };
const SVG = "http://www.w3.org/2000/svg";

function svg(tag, attrs = {}, ...children) {
  const el = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null) el.setAttribute(k, v);
  for (const c of children.flat()) if (c != null) el.append(c);
  return el;
}

/** "id.slot,id.slot" <-> [{id, slot}]; slots stay with their model when others are added or removed. */
const decode = (str) => (str || "").split(",").filter(Boolean).map((x) => {
  const [id, slot] = x.split(".");
  return { id, slot: Number(slot) };
});
const encode = (list) => list.map((x) => `${x.id}.${x.slot}`).join(",") || "none";

function stats(xs) {
  const v = xs.filter((x) => x != null);
  if (!v.length) return null;
  const sd = std(v);
  return { mean: mean(v), sd, ci: sd == null ? null : (T975[v.length - 1] || 1.96) * sd / Math.sqrt(v.length), n: v.length, values: v };
}

export function levelBars(exp, s, params) {
  const m = exp.manifest;
  const T = exp.terms;
  const pctMeasures = m.measures.filter((x) => x.format === "pct");
  const measure = exp.measuresById[params.pmeas] || (s.measure.format === "pct" ? s.measure : pctMeasures[0]);
  const split = exp.splitsById[params.ps] || (s.split !== "all" ? exp.splitsById[s.split] : m.splits[0]);
  const pick = params.pp ? resolvePrompt(exp, params.pp, params.ppf) : { prompt: s.prompt, family: s.family };
  const { prompt } = pick;
  const fam = meanFamily(exp, prompt);
  const errMode = ["sd", "ci", "none"].includes(params.perr) ? params.perr : "sd";
  const levels = pickLevels(m.levels, params.pl);
  const poolIds = new Set(s.pool.map((x) => x.id));
  const series = (params.pm ? decode(params.pm) : (m.plots?.[s.set.id] || s.pool.map((x) => x.id)).slice(0, MAX_SERIES).map((id, i) => ({ id, slot: i + 1 })))
    .filter((x) => poolIds.has(x.id) && x.slot >= 1 && x.slot <= MAX_SERIES)
    .slice(0, MAX_SERIES);

  // ------------------------------------------------------------------ data
  /** Per-sample estimates of `measure` for one model and level, plus the table's pooled value. */
  function bar(model, level) {
    const cv = cellValue(exp, s, model.id, split.id, level.id, null, { prompt, measure });
    if (!cv) return null;
    if (!measure.counts) return { mean: cv.value, pooled: cv.value, deterministic: true };
    const st = stats(perSample(exp, cv, prompt, level.id, measure));
    return st ? { ...st, pooled: cv.value } : { mean: cv.value, pooled: cv.value, deterministic: true };
  }

  const models = series.map((x) => ({ ...x, model: exp.modelsById[x.id] }));
  const data = models.map((x) => ({ ...x, bars: levels.map((l) => bar(x.model, l)) }));

  // ------------------------------------------------------------------ controls
  const chips = h("div", { class: "chips" }, s.pool.map((x) => {
    const on = series.find((y) => y.id === x.id);
    const full = !on && series.length >= MAX_SERIES;
    return h("button", {
      class: "chip", type: "button", "aria-pressed": String(!!on), disabled: full || null,
      title: full ? `Up to ${MAX_SERIES} models` : null,
      onclick: () => {
        if (on) return setParams({ pm: encode(series.filter((y) => y.id !== x.id)) });
        const used = new Set(series.map((y) => y.slot));
        const slot = [...Array(MAX_SERIES).keys()].map((i) => i + 1).find((i) => !used.has(i));
        setParams({ pm: encode([...series, { id: x.id, slot }]) });
      },
    }, on && h("span", { class: "swatch", style: { background: `var(--s${on.slot})` } }), x.label);
  }));

  const controls = h("div", { class: "chart-controls" },
    m.splits.length > 1 && field("Split", segmented(m.splits.map((x) => ({ id: x.id, label: x.short || x.label, title: x.label })), split.id,
      (v) => setParams({ ps: v }), "Chart split")),
    m.prompts.length > 1 && field("Prompt", promptControl(exp, pick, (v) => setParams({ pp: v.prompt, ppf: v.family }), { all: false })),
    field(T.levels, levelToggle(m.levels, levels, "pl", T.levels)),
    field("Value", select(pctMeasures.map((x) => ({ id: x.id, label: x.label, group: x.group })), measure.id,
      (v) => setParams({ pmeas: v }), "Chart measure")),
    field("Whiskers", segmented([{ id: "sd", label: "±1 SD" }, { id: "ci", label: "95% CI" }, { id: "none", label: "None" }], errMode,
      (v) => setParams({ perr: v }), "Whiskers")));

  const legendRow = h("div", { class: "chart-legend" }, models.map((x) => h("span", { class: "item" },
    h("span", { class: "swatch", style: { background: `var(--s${x.slot})` } }), x.model.label)));

  const host = h("div", { class: "chart-host" });
  const k = Math.max(0, ...data.flatMap((x) => x.bars.map((b) => b?.n || 0)));
  const caption = h("p", { class: "section-note", style: { marginTop: "10px" } },
    `${measure.label} on ${split.label}`, fam ? `, mean over the ${fam.label} prompts` : m.prompts.length > 1 ? `, prompt “${promptLabel(exp, prompt)}”` : "",
    levels.length < m.levels.length ? `, ${T.levels.toLowerCase()} ${levels.map((l) => l.id).join(", ")}. ` : ". ",
    measure.counts
      ? [`Bars are the mean of the ${k || "per"}-sample estimates, where estimate i uses the i-th sampled response to every question`,
        fam ? ", pooled over the family’s prompt variants" : "",
        `; whiskers show ${errMode === "sd" ? "±1 SD across samples" : errMode === "ci" ? "a 95% t-interval of that mean" : "nothing"}. `,
        "Decided rates can differ from the table’s pooled value by a fraction of a point."]
      : "This measure is deterministic (no sampling), so bars have no whiskers.");

  const section = h("section", { class: "chart-section" },
    h("h2", { class: "section" }, `Across ${T.levels.toLowerCase()}`),
    h("p", { class: "section-note" }, `Absolute values per ${T.level.toLowerCase()}. Pick up to eight models; each keeps its colour while others come and go.`),
    h("div", { class: "chart-models" }, chips),
    controls,
    models.length ? [legendRow, host, caption] : h("div", { class: "empty" }, h("h3", {}, "No models in the chart"), h("p", {}, "Pick models above.")));

  if (models.length) {
    const ro = new ResizeObserver(() => {
      if (!host.isConnected) return ro.disconnect(); // the view was re-rendered
      draw(host, data, measure, errMode, split, promptLabel(exp, prompt), levels, T);
    });
    ro.observe(host);
  }
  return section;
}

// ------------------------------------------------------------------ drawing

function draw(host, data, measure, errMode, split, promptText, levels, T) {
  const W = host.clientWidth;
  if (!W) return;
  const H = 340, M = { top: 12, right: 8, bottom: 46, left: 44 };
  const pw = W - M.left - M.right, ph = H - M.top - M.bottom;
  const y = (v) => M.top + ph * (1 - Math.max(0, Math.min(1, v)));
  const groupW = pw / levels.length;
  const gap = 2;
  const nb = data.length;
  const bw = Math.max(2, Math.min(56, (groupW * 0.8 - gap * (nb - 1)) / nb)); // capped so few levels don't give slabs
  const inner = nb * bw + gap * (nb - 1);
  const r = Math.min(4, bw / 2);
  const pct = (v) => fmtValue(v, "pct");

  const root = svg("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": `${measure.label} by ${T.level.toLowerCase()}, ${data.length} models` });

  // Grid + y ticks (one axis, 0–100%).
  for (let t = 0; t <= 1.0001; t += 0.2) {
    root.append(svg("line", { x1: M.left, x2: W - M.right, y1: y(t), y2: y(t), class: t === 0 ? "axis" : "grid" }),
      svg("text", { x: M.left - 8, y: y(t) + 4, class: "tick", "text-anchor": "end" }, `${Math.round(t * 100)}%`));
  }

  levels.forEach((level, li) => {
    const gx = M.left + li * groupW + (groupW - inner) / 2;
    root.append(
      svg("text", { x: M.left + li * groupW + groupW / 2, y: H - M.bottom + 18, class: "xlabel", "text-anchor": "middle" }, level.id),
      svg("text", { x: M.left + li * groupW + groupW / 2, y: H - M.bottom + 33, class: "xsub", "text-anchor": "middle" }, level.label));
    data.forEach((series, si) => {
      const b = series.bars[li];
      const x = gx + si * (bw + gap);
      if (!b) return;
      const top = y(b.mean), base = y(0);
      const hgt = base - top;
      const rr = Math.min(r, hgt);
      root.append(svg("path", {
        d: `M${x},${base} V${top + rr} Q${x},${top} ${x + rr},${top} H${x + bw - rr} Q${x + bw},${top} ${x + bw},${top + rr} V${base} Z`,
        style: `fill: var(--s${series.slot})`, class: "bar",
      }));
      const e = errMode === "sd" ? b.sd : errMode === "ci" ? b.ci : null;
      if (e != null && !b.deterministic) {
        const cx = x + bw / 2, cap = Math.min(8, bw * 0.6) / 2;
        root.append(svg("path", {
          d: `M${cx},${y(b.mean - e)} V${y(b.mean + e)} M${cx - cap},${y(b.mean - e)} H${cx + cap} M${cx - cap},${y(b.mean + e)} H${cx + cap}`,
          class: "whisker",
        }));
      }
      // Hit target: the full column, wider than the mark.
      const hit = svg("rect", { x: x - gap / 2, y: M.top, width: bw + gap, height: ph, class: "hit", tabindex: "0" });
      tip(hit, () => tipRows(`${series.model.label} · ${level.id} ${level.label}`, `${pct(b.mean)}%`, [
        b.deterministic ? ["Value", "deterministic"] : ["Mean of", `${b.n} samples`],
        b.sd != null && ["SD", `${pct(b.sd)} pt`],
        b.ci != null && ["95% CI", `± ${pct(b.ci)} pt`],
        b.values && ["Samples", b.values.map(pct).join(" · ")],
        !b.deterministic && ["Pooled (table)", `${pct(b.pooled)}%`],
      ].filter(Boolean), `${split.short || split.label} · ${promptText}`));
      root.append(hit);
    });
  });

  host.replaceChildren(root);
}
