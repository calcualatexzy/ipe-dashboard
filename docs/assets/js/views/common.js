// State resolution, the shared toolbar (set, models, protocol, split, prompt, measure),
// and the cell-value / baseline helpers every results view uses.
import { h, mean, std } from "../lib/dom.js";
import { setParams } from "../lib/state.js";
import { modelPicker } from "../components/picker.js";
import { field, legend, segmented, select, tip } from "../components/ui.js";

/** Prompt values are a prompt id or "mean:<family>", the mean over a family's prompts. */
const MEAN = "mean:";

/** The family whose mean `prompt` stands for, or null for a single prompt. */
export const meanFamily = (exp, prompt) =>
  (prompt?.startsWith(MEAN) ? exp.familiesById[prompt.slice(MEAN.length)] || null : null);

/** "Diverse mean" or the prompt id. */
export const promptLabel = (exp, prompt) => {
  const f = meanFamily(exp, prompt);
  return f ? `${f.label} mean` : prompt;
};

/**
 * Resolve a prompt param (and the family param that disambiguates a prompt in several families) to
 * { prompt, family }. "all" is the pre-family spelling of the first family's mean.
 */
export function resolvePrompt(exp, value, familyId) {
  if (value === "all") value = MEAN + exp.families[0].id;
  const f = meanFamily(exp, value);
  if (f) return { prompt: value, family: f };
  const prompt = exp.promptsById[value] ? value : exp.manifest.prompts[0]?.id;
  const homes = exp.families.filter((x) => x.prompts.includes(prompt));
  return { prompt, family: homes.find((x) => x.id === familyId) || homes[0] || exp.families[0] };
}

/** Resolve URL params against the experiment, filling defaults. */
export function resolve(exp, params) {
  const m = exp.manifest;
  const pick = (v, list, fallback) => (list.some((x) => x.id === v) ? v : fallback);
  // A set scopes every view; `pool` is its models in the set's own order.
  const sets = m.sets?.length ? m.sets : [{ id: "all", label: "All models", models: exp.models.map((x) => x.id) }];
  const set = sets.find((x) => x.id === params.set) || sets[0];
  const pool = set.models.map((id) => exp.modelsById[id]).filter(Boolean);
  const ids = new Set(params.models ? params.models.split(",") : pool.map((x) => x.id));
  const { prompt, family } = resolvePrompt(exp, params.prompt, params.pf);
  return {
    sets, set, pool,
    models: pool.filter((x) => ids.has(x.id)),
    protocol: pick(params.protocol, m.protocols, m.protocols[0]?.id),
    split: params.split === "all" ? "all" : pick(params.split, m.splits, m.splits[0]?.id),
    prompt, family,
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
 * Value of one cell. For a family mean it is the mean over the family's prompts, all read from the
 * model's latest run covering the family (so every variant comes from the same run); `perPrompt` holds the parts.
 */
export function cellValue(exp, s, modelId, splitId, levelId, topic = null, { prompt = s.prompt, measure = s.measure } = {}) {
  const fam = meanFamily(exp, prompt);
  if (!fam) {
    const row = exp.get(modelId, splitId, s.protocol, prompt, levelId, topic, measure.id);
    return row ? { value: row.value, row } : null;
  }
  const ev = exp.multiRun(modelId, splitId, s.protocol, fam.id);
  if (!ev) return null;
  const perPrompt = fam.prompts
    .map((p) => ({ prompt: p, row: exp.inEval(ev.id, p, levelId, topic, measure.id) }))
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

/** Hover text for a family: its purpose and prompts. */
const familyTip = (f) => h("div", {},
  h("div", { class: "tt-title" }, `Prompt family · ${f.label}`),
  f.summary && h("div", {}, f.summary),
  h("div", { class: "tt-muted", style: { marginTop: "6px" } }, `Prompts: ${f.prompts.join(", ")}`));

/**
 * Prompt picker: family buttons (when there are several), then that family's Mean and prompts.
 * Hovering a prompt shows its template. `onChange({ prompt, family })` gets the new prompt value and family id.
 */
export function promptControl(exp, { prompt, family }, onChange) {
  const isMean = !!meanFamily(exp, prompt);
  const wrap = h("div", { class: "prompt-ctl" });
  if (exp.families.length > 1) {
    const fams = segmented(exp.families.map((f) => ({ id: f.id, label: f.label })), family.id, (id) => {
      const f = exp.familiesById[id];
      // A prompt the new family shares (strict) stays put; anything else becomes that family's mean.
      onChange({ prompt: !isMean && f.prompts.includes(prompt) ? prompt : MEAN + f.id, family: id });
    }, "Prompt family");
    [...fams.children].forEach((btn, i) => tip(btn, () => familyTip(exp.families[i])));
    wrap.append(fams);
  }
  const opts = [{ id: MEAN + family.id, label: "Mean" }, ...family.prompts.map((id) => ({ id, label: exp.promptsById[id]?.label || id }))];
  const seg = segmented(opts, prompt, (v) => onChange({ prompt: v, family: family.id }), "Prompt variant");
  [...seg.children].forEach((btn, i) => {
    const o = opts[i];
    const p = exp.promptsById[o.id];
    const others = exp.families.filter((f) => f !== family && !f.implicit && f.prompts.includes(o.id));
    tip(btn, () => h("div", {},
      h("div", { class: "tt-title" }, i === 0 ? `${family.implicit ? "All prompts" : family.label} · mean` : `Prompt · ${p.label}`),
      i === 0
        ? h("div", { class: "tt-muted" }, `Mean over ${family.prompts.join(", ")}, all read from each model’s latest multi-prompt (_pall) run that covers them. Models without one show no value.`)
        : [h("pre", { class: "tt-pre" }, p.template),
          others.length > 0 && h("div", { class: "tt-muted" }, `Also in ${others.map((f) => f.label).join(", ")}.`)]));
  });
  wrap.append(seg);
  return wrap;
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
    bar.append(field("Prompt", promptControl(exp, s, ({ prompt, family }) => setParams({ prompt, pf: family }))));
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
