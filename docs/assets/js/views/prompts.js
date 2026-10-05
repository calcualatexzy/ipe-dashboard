// Prompt robustness: one level (or the level mean) across prompt variants, with spread.
import { fmtValue, h, heat, mean, std } from "../lib/dom.js";
import { setParams } from "../lib/state.js";
import { emptyState, field, modelName, segmented } from "../components/ui.js";
import { baselineValue, measureHelp, resolve, setNote, spread as spreadOf, toolbar } from "./common.js";

export function render(root, exp, params) {
  const s = resolve(exp, params);
  const m = exp.manifest;
  const levelOpts = [...m.levels.map((l) => ({ id: l.id, label: l.id, title: l.label })), { id: "mean", label: "Mean" }];
  const level = levelOpts.some((l) => l.id === params.level) ? params.level : "mean";
  root.append(toolbar(exp, s, { split: "single", measure: true },
    field("Level", segmented(levelOpts, level, (v) => setParams({ level: v }), "Level"))));
  root.append(setNote(s), measureHelp(s.measure, s));

  const sp = exp.splitsById[s.split === "all" ? m.splits[0].id : s.split];
  const prompts = m.prompts;
  // Read every prompt from the same multi-prompt run, so the spread isn't mixed with run-to-run noise.
  const runOf = new Map(s.models.map((x) => [x.id, exp.multiRun(x.id, sp.id, s.protocol)]));
  const val = (x, p) => {
    const ev = runOf.get(x.id);
    if (!ev) return null;
    const get = (l) => exp.inEval(ev.id, p.id, l.id, null, s.measure.id)?.value;
    return level === "mean" ? mean(m.levels.map(get)) : get({ id: level });
  };
  const models = s.models.filter((x) => runOf.get(x.id));
  if (!models.length) {
    root.append(emptyState("No multi-prompt evals here",
      `None of the selected models were evaluated with several prompt variants on ${sp.label}. Runs labelled “_pall” evaluate all variants at once.`));
    return;
  }
  const unit = s.measure.format === "pct" ? "%" : "";
  const fmt = (v) => fmtValue(v, s.measure.format);
  // Each prompt column is centred on the baseline's value for that prompt.
  const centre = (p) => {
    if (!s.baseline) return null;
    if (level !== "mean") return baselineValue(exp, s, sp.id, level, null, { prompt: p.id });
    return mean(m.levels.map((l) => baselineValue(exp, s, sp.id, l.id, null, { prompt: p.id })));
  };
  const centres = new Map(prompts.map((p) => [p.id, centre(p)]));
  const range = spreadOf(models.flatMap((x) => prompts.map((p) => [val(x, p), centres.get(p.id)])), s.measure);

  const table = h("table", { class: "grid" },
    h("thead", {}, h("tr", {}, h("th", { class: "model-cell" }, "Model"),
      prompts.map((p) => h("th", { class: "num", title: p.template }, p.label, h("span", { class: "sub" }, unit || "value"))),
      h("th", { class: "num" }, "Mean", h("span", { class: "sub" }, "over prompts")),
      h("th", { class: "num" }, "SD", h("span", { class: "sub" }, unit ? "pt" : "")),
      h("th", { class: "num" }, "Range", h("span", { class: "sub" }, unit ? "max − min, pt" : "max − min")))),
    h("tbody", {}, models.map((x) => {
      const vs = prompts.map((p) => val(x, p));
      const present = vs.filter((v) => v != null);
      const sd = std(present);
      const spread = Math.max(...present) - Math.min(...present);
      return h("tr", {},
        h("td", { class: "model-cell" }, modelName(exp, x)),
        vs.map((v, i) => h("td", { class: "num cell" + (v == null ? " missing" : " heat"), style: { background: heat(v, centres.get(prompts[i].id) ?? s.measure.center ?? 0.5, range) } }, fmt(v))),
        h("td", { class: "num cell mean" }, fmt(mean(present))),
        h("td", { class: "num" }, sd == null ? "—" : unit ? (sd * 100).toFixed(1) : sd.toFixed(3)),
        h("td", { class: "num" }, unit ? (spread * 100).toFixed(1) : spread.toFixed(3)));
    })));

  root.append(h("h2", { class: "section" }, "Prompt robustness"),
    h("p", { class: "section-note" },
      `${s.measure.label} on ${sp.label}, ${level === "mean" ? "averaged over levels" : `level ${level}`}, all prompts read from each model’s latest multi-prompt run. Hover a column header for the prompt template. `,
      "L2 uses a fixed prompt, so it does not vary across variants."),
    h("div", { class: "table-wrap" }, table));
}
