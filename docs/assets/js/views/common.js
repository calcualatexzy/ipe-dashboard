// State resolution, the shared toolbar (set, models, protocol, split, prompt, measure),
// and the cell-value / baseline helpers every results view uses.
import { h, mean, std } from "../lib/dom.js";
import { setParams } from "../lib/state.js";
import { modelPicker } from "../components/picker.js";
import { field, legend, segmented, select, tip } from "../components/ui.js";

export const ALL_PROMPTS = "all";

/** Resolve URL params against the experiment, filling defaults. */
export function resolve(exp, params) {
  const m = exp.manifest;
  const pick = (v, list, fallback) => (list.some((x) => x.id === v) ? v : fallback);
  // A set scopes every view; `pool` is its models in the set's own order.
  const sets = m.sets?.length ? m.sets : [{ id: "all", label: "All models", models: exp.models.map((x) => x.id) }];
  const set = sets.find((x) => x.id === params.set) || sets[0];
  const pool = set.models.map((id) => exp.modelsById[id]).filter(Boolean);
  const ids = new Set(params.models ? params.models.split(",") : pool.map((x) => x.id));
  return {
    sets, set, pool,
    models: pool.filter((x) => ids.has(x.id)),
    protocol: pick(params.protocol, m.protocols, m.protocols[0]?.id),
    split: params.split === "all" ? "all" : pick(params.split, m.splits, m.splits[0]?.id),
    prompt: params.prompt === ALL_PROMPTS && m.prompts.length > 1 ? ALL_PROMPTS : pick(params.prompt, m.prompts, m.prompts[0]?.id),
    measure: exp.measuresById[params.measure] || m.measures[0],
    levels: pickLevels(m.levels, params.lv),
    baseline: exp.modelsById[m.baseline] || null,
  };
}

/** Levels named in a comma list param (manifest order); all of them when the param is absent or empty. */
export function pickLevels(levels, param) {
  const want = new Set((param || "").split(",").filter(Boolean));
  const picked = levels.filter((l) => want.has(l.id));
  return picked.length ? picked : levels;
}

/** Multi-select level buttons; at least one stays on. Writes `key` (null when all are on). */
export function levelToggle(all, selected, key) {
  const on = new Set(selected.map((l) => l.id));
  const write = (next) => setParams({ [key]: next.size === all.length ? null : all.filter((l) => next.has(l.id)).map((l) => l.id).join(",") });
  return h("div", { class: "seg", role: "group", "aria-label": "Levels" },
    h("button", { type: "button", "aria-pressed": String(on.size === all.length), title: "All levels",
      onclick: () => write(new Set(all.map((l) => l.id))) }, "All"),
    all.map((l) => h("button", {
      type: "button", "aria-pressed": String(on.size < all.length && on.has(l.id)), title: l.label,
      onclick: () => {
        // From "all", a click isolates that level; otherwise it toggles, keeping at least one.
        const next = on.size === all.length ? new Set([l.id]) : new Set(on);
        if (on.size < all.length) next.has(l.id) ? next.delete(l.id) : next.add(l.id);
        if (next.size) write(next);
      },
    }, l.id)));
}

export function setModels(s, ids) {
  setParams({ models: ids.length === s.pool.length ? null : ids.join(",") || "none" });
}

/**
 * Value of one cell. With prompt "all" it is the mean over the prompt variants of the model's latest
 * multi-prompt run (so every variant comes from the same run); `perPrompt` holds the parts.
 */
export function cellValue(exp, s, modelId, splitId, levelId, topic = null, { prompt = s.prompt, measure = s.measure } = {}) {
  if (prompt !== ALL_PROMPTS) {
    const row = exp.get(modelId, splitId, s.protocol, prompt, levelId, topic, measure.id);
    return row ? { value: row.value, row } : null;
  }
  const ev = exp.multiRun(modelId, splitId, s.protocol);
  if (!ev) return null;
  const perPrompt = exp.manifest.prompts
    .map((p) => ({ prompt: p.id, row: exp.inEval(ev.id, p.id, levelId, topic, measure.id) }))
    .filter((x) => x.row);
  if (!perPrompt.length) return null;
  const vals = perPrompt.map((x) => x.row.value);
  return { value: mean(vals), sd: std(vals), perPrompt, eval: ev };
}

/** The baseline's value for the same cell: the centre of the colour scale. */
export function baselineValue(exp, s, splitId, levelId, topic = null, opts = {}) {
  return s.baseline ? cellValue(exp, s, s.baseline.id, splitId, levelId, topic, opts)?.value ?? null : null;
}

/** Colour range: the largest distance from the centre among the shaded cells (at least 2 pt). */
export function spread(pairs, measure) {
  const d = pairs.filter(([v, c]) => v != null && c != null).map(([v, c]) => Math.abs(v - c));
  return Math.max(measure.format === "pct" ? 0.02 : 1e-3, ...d);
}

/** Prompt buttons; hovering one shows its exact template. */
export function promptControl(exp, value, onChange, { all = true } = {}) {
  const m = exp.manifest;
  const opts = [...(all ? [{ id: ALL_PROMPTS, label: "All" }] : []), ...m.prompts.map((p) => ({ id: p.id, label: p.label }))];
  const seg = segmented(opts, value, onChange, "Prompt variant");
  [...seg.children].forEach((btn, i) => {
    const o = opts[i];
    const p = exp.promptsById[o.id];
    tip(btn, () => h("div", {},
      h("div", { class: "tt-title" }, o.id === ALL_PROMPTS ? "All prompt variants" : `Prompt · ${p.label}`),
      o.id === ALL_PROMPTS
        ? h("div", { class: "tt-muted" }, `Mean over ${m.prompts.map((x) => x.label).join(", ")}, read from each model’s multi-prompt (_pall) run. Models without one show no value.`)
        : h("pre", { class: "tt-pre" }, p.template)));
  });
  return seg;
}

/**
 * Builds the sticky toolbar. `opts` toggles controls: { split: true|"single", prompt, measure }.
 * Extra nodes (view-specific controls) go at the end.
 */
export function toolbar(exp, s, opts = {}, ...extra) {
  const m = exp.manifest;
  const bar = h("div", { class: "toolbar" });
  if (s.sets.length > 1) {
    bar.append(field("Set", segmented(s.sets.map((x) => ({ id: x.id, label: x.label, title: x.summary })), s.set.id,
      (v) => setParams({ set: v, models: null, ref: null, sort: null, dir: null, pm: null }), "Model set")));
  }
  bar.append(modelPicker(exp, s.pool, new Set(s.models.map((x) => x.id)), (ids) => setModels(s, ids)));
  if (m.protocols.length > 1) {
    bar.append(field("Protocol", select(m.protocols.map((p) => ({ id: p.id, label: `${p.label} (${p.evals})` })), s.protocol,
      (v) => setParams({ protocol: v }), "Eval protocol")));
  }
  if (opts.split) {
    const splits = m.splits.map((x) => ({ id: x.id, label: x.short || x.label, title: `${x.label}: ${x.topics.join(", ")}` }));
    if (opts.split !== "single" && splits.length > 1) splits.push({ id: "all", label: "All", title: "All splits side by side" });
    bar.append(field("Split", segmented(splits, s.split, (v) => setParams({ split: v }), "Split")));
  }
  if (opts.prompt && m.prompts.length > 1) {
    bar.append(field("Prompt", promptControl(exp, s.prompt, (v) => setParams({ prompt: v }))));
  }
  if (opts.levels) {
    bar.append(field("Levels", levelToggle(m.levels, s.levels, "lv")));
  }
  if (opts.measure) {
    bar.append(field("Measure", select(m.measures.map((x) => ({ id: x.id, label: x.label, group: x.group })), s.measure.id,
      (v) => setParams({ measure: v }), "Measure")));
  }
  bar.append(...extra.filter(Boolean));
  return bar;
}

/** One line naming the active set and its purpose. */
export const setNote = (s) => s.set.summary && h("p", { class: "set-note" },
  h("b", {}, s.set.label), ` · ${s.models.length} of ${s.pool.length} models — ${s.set.summary}`);

/** What the current measure means, with its colour legend (centred on the baseline) on the right. */
export const measureHelp = (measure, s) => measure && h("div", { class: "measure-bar" },
  h("p", { class: "section-note" }, h("b", {}, measure.label), measure.help ? ` — ${measure.help}` : ""),
  legend(measure, s?.baseline ? s.baseline.label : null));

/** Groups models by method (manifest order) for table section headers. */
export function groupByMethod(exp, models) {
  return exp.manifest.methods
    .map((meth) => ({ method: meth, models: models.filter((x) => x.method === meth.id) }))
    .filter((g) => g.models.length);
}
