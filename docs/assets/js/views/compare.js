// Compare: a reference model against the others — lineage, results with deltas, config diff.
import { fmtDelta, fmtValue, h, heat, mean } from "../lib/dom.js";
import { setParams } from "../lib/state.js";
import { emptyState, field, modelName, modelTip, segmented, select, tip, tipRows } from "../components/ui.js";
import { ALL_PROMPTS, baselineValue, cellValue, measureHelp, resolve, setNote, spread, toolbar } from "./common.js";

const MAX_MODELS = 8;

export function render(root, exp, params) {
  const s = resolve(exp, params);
  const models = s.models.slice(0, MAX_MODELS);
  const ref = models.find((x) => x.id === params.ref) || models[0];
  const tooMany = !params.models && s.models.length > MAX_MODELS;
  const refSelect = models.length > 1 && !tooMany && field("Reference", select(models.map((x) => ({ id: x.id, label: x.label })), ref.id,
    (v) => setParams({ ref: v }), "Reference model"));
  root.append(toolbar(exp, s, { split: "single", prompt: true, levels: true, measure: true }, refSelect));
  root.append(setNote(s), measureHelp(s.measure, s));

  if (models.length < 2) {
    root.append(emptyState("Pick two or more models",
      `Use the model picker to select between 2 and ${MAX_MODELS} models. The first one becomes the reference; deltas are measured against it.`));
    return;
  }
  if (tooMany) {
    root.append(emptyState(`${s.set.label} has ${s.models.length} models`,
      `Pick up to ${MAX_MODELS} with the model picker (presets help), or switch to a smaller set. The first one becomes the reference.`));
    return;
  }
  if (s.models.length > MAX_MODELS) {
    root.append(h("p", { class: "section-note", style: { marginTop: "14px" } },
      `Showing the first ${MAX_MODELS} of ${s.models.length} selected models. Narrow the selection to compare others.`));
  }
  const ordered = [ref, ...models.filter((x) => x !== ref)];

  root.append(h("h2", { class: "section" }, "Lineage"),
    h("p", { class: "section-note" }, "Where each checkpoint comes from. The outlined card is the reference."),
    h("div", { class: "lineage-cards" }, ordered.map((x) => lineageCard(exp, x, x === ref))));

  root.append(resultsTable(exp, s, ordered, ref));
  root.append(configDiff(exp, params, ordered, ref));
}

function lineageCard(exp, x, isRef) {
  const pre = x.pretrain || {};
  const step = (k, v) => h("div", { class: "step" }, h("span", { class: "k" }, k), h("span", { class: "v" }, v ?? "—"));
  return h("div", { class: "lineage-card" + (isRef ? " ref" : "") },
    h("div", {}, modelName(exp, x), isRef && h("span", { class: "tag accent" }, "reference")),
    x.description && h("p", { class: "section-note", style: { margin: 0 } }, x.description),
    h("div", { class: "chain" },
      step("Method", [exp.methodsById[x.method]?.label || x.method, x.code && `code ${x.code}`].filter(Boolean).join(" · ")),
      step("Tags", x.tags.join(", ") || "—"),
      step("SFT", `step ${x.step ?? "?"}${x.sft.resolved ? "" : " · folder missing"}`),
      h("div", { class: "step" }, h("span", {}), h("code", { class: "path" }, x.sft.path)),
      step("Pretrain", pre.step != null ? `step ${pre.step}` : "unknown"),
      pre.path && h("div", { class: "step" }, h("span", {}), h("code", { class: "path" }, pre.path)),
      step("SFT data", `${x.data ?? "?"} · ${x.anchors == null ? "anchors ?" : x.anchors ? "with anchors" : "no anchors"}`),
      step("Trainer", pre.trainer ? `${pre.trainer}${pre.reflection_weight != null ? ` · rw ${pre.reflection_weight}` : ""}` : "—")),
    x.note && h("p", { class: "section-note", style: { margin: 0 } }, x.note));
}

function resultsTable(exp, s, ordered, ref) {
  const m = exp.manifest;
  const sp = exp.splitsById[s.split === "all" ? m.splits[0].id : s.split];
  const unit = s.measure.format === "pct" ? "%" : "";
  const val = (x, l) => cellValue(exp, s, x.id, sp.id, l.id)?.value;
  const base = (l) => baselineValue(exp, s, sp.id, l.id);
  const rows = [...s.levels.map((l) => ({ id: l.id, label: l.label, get: (x) => val(x, l), centre: base(l) })),
    { id: "Mean", label: s.levels.map((l) => l.id).join("+"), get: (x) => mean(s.levels.map((l) => val(x, l))), mean: true,
      centre: s.baseline ? mean(s.levels.map((l) => base(l))) : null }];
  const range = spread(ordered.flatMap((x) => rows.map((r) => [r.get(x), r.centre])), s.measure);

  const table = h("table", { class: "grid" },
    h("thead", {}, h("tr", {}, h("th", { class: "model-cell", style: { minWidth: "150px" } }, "Level"),
      ordered.map((x) => h("th", { class: "num" }, tip(h("span", {}, x.label), () => modelTip(exp, x)),
        h("span", { class: "sub" }, x === ref ? "reference" : "Δ vs reference"))))),
    h("tbody", {}, rows.map((r) => h("tr", {},
      h("td", { class: "model-cell" + (r.mean ? " mean" : ""), style: { minWidth: "150px", borderLeft: 0 } }, h("b", { style: { fontWeight: 500 } }, r.id), " ",
        h("span", { class: "tt-muted" }, r.label)),
      ordered.map((x) => {
        const v = r.get(x), rv = r.get(ref);
        if (v == null) return h("td", { class: "num cell missing" }, "—");
        const d = x === ref || rv == null ? null : v - rv;
        const td = h("td", { class: "num cell heat" + (r.mean ? " mean" : ""), style: { background: heat(v, r.centre ?? s.measure.center ?? 0.5, range) } },
          fmtValue(v, s.measure.format),
          d != null && h("span", { class: "delta " + (d > 0 ? "up" : d < 0 ? "down" : "") }, fmtDelta(d, s.measure.format) || "0"));
        return tip(td, () => tipRows(`${x.label} · ${r.id}`, `${fmtValue(v, s.measure.format)}${unit}`,
          d != null ? [["Reference", `${fmtValue(rv, s.measure.format)}${unit}`], ["Difference", `${d > 0 ? "+" : d < 0 ? "−" : ""}${fmtDelta(d, s.measure.format)}`]] : [],
          `${sp.label} · ${s.prompt === ALL_PROMPTS ? "all prompts (mean)" : `prompt ${s.prompt}`}`));
      })))));

  return h("div", {},
    h("h2", { class: "section" }, "Results"),
    h("p", { class: "section-note" }, `${s.measure.label} on ${sp.label}, ${s.prompt === ALL_PROMPTS ? "mean over all prompt variants" : `prompt “${s.prompt}”`}. Deltas against the reference are in percentage points for rates; colours are centred on the baseline.`),
    h("div", { class: "table-wrap" }, table));
}

const SOURCES = [
  { id: "pretrain", label: "Pretrain", get: (x) => x.pretrain?.config },
  { id: "sft", label: "SFT", get: (x) => x.sft?.config },
];

function configDiff(exp, params, ordered, ref) {
  const src = SOURCES.find((x) => x.id === params.cfg) || SOURCES[0];
  const onlyDiff = params.diff !== "all";
  const configs = ordered.map((x) => src.get(x) || null);
  const keys = [...new Set(configs.flatMap((c) => Object.keys(c || {})))].sort();
  const show = (v) => (v === undefined ? "" : typeof v === "object" ? JSON.stringify(v) : String(v));
  const rows = keys.map((k) => ({ k, vals: configs.map((c) => (c ? c[k] : undefined)) }))
    .map((r) => ({ ...r, differs: new Set(r.vals.map(show)).size > 1 }))
    .filter((r) => !onlyDiff || r.differs);

  const controls = h("div", { class: "toolbar", style: { position: "static", borderBottom: 0, paddingBottom: 4 } },
    segmented(SOURCES.map((x) => ({ id: x.id, label: x.label })), src.id, (v) => setParams({ cfg: v }), "Config source"),
    h("label", { class: "check" }, h("input", { type: "checkbox", checked: onlyDiff,
      onchange: (e) => setParams({ diff: e.target.checked ? null : "all" }) }), "Only differences"),
    h("span", { class: "grow" }),
    h("span", { class: "section-note", style: { margin: 0 } }, `${rows.length} of ${keys.length} keys`));

  let prevGroup = null;
  const body = h("tbody");
  for (const r of rows) {
    const group = r.k.split(".")[0];
    if (group !== prevGroup) {
      body.append(h("tr", { class: "group-row" }, h("td", { colspan: ordered.length + 1 }, group)));
      prevGroup = group;
    }
    const refVal = show(r.vals[0]);
    body.append(h("tr", {}, h("td", { class: "key" }, r.k.slice(group.length + 1) || group),
      r.vals.map((v, i) => {
        const text = configs[i] ? show(v) : "unavailable";
        return h("td", { class: "val" + (i > 0 && show(v) !== refVal ? " diff" : ""), title: text }, text || "—");
      })));
  }
  const table = h("table", { class: "grid kv-diff" },
    h("thead", {}, h("tr", {}, h("th", {}, "Key"), ordered.map((x) => h("th", {}, x.label)))), body);

  return h("div", {},
    h("h2", { class: "section" }, "Configuration"),
    h("p", { class: "section-note" }, "Training configs side by side. Shaded cells differ from the reference. Paths are shown relative to the IPE repo."),
    controls,
    rows.length ? h("div", { class: "table-wrap" }, table)
      : h("p", { class: "section-note" }, onlyDiff ? "No differences in this config." : "No config available."));
}
