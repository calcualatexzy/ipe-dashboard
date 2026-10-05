// Prompt robustness: one level (or the level mean) across prompt variants, with spread — one table per
// prompt family, each read from the family's own multi-prompt run. "Δ vs reference" shows each model's
// gap to a reference model prompt by prompt, so the spread columns measure how stable the effect is.
import { fmtValue, h, heat, mean, std } from "../lib/dom.js";
import { setParams } from "../lib/state.js";
import { emptyState, field, modelName, segmented, select } from "../components/ui.js";
import { measureHelp, resolve, setNote, spread as spreadOf, toolbar } from "./common.js";

export function render(root, exp, params) {
  const s = resolve(exp, params);
  const m = exp.manifest;
  const levelOpts = [...m.levels.map((l) => ({ id: l.id, label: l.id, title: l.label })), { id: "mean", label: "Mean" }];
  const level = levelOpts.some((l) => l.id === params.level) ? params.level : "mean";
  const delta = params.show === "delta";
  const ref = s.pool.find((x) => x.id === params.ref) || (s.baseline && s.pool.includes(s.baseline) ? s.baseline : s.pool[0]);
  root.append(toolbar(exp, s, { split: "single", measure: true },
    field("Level", segmented(levelOpts, level, (v) => setParams({ level: v }), "Level")),
    field("Show", segmented([{ id: "value", label: "Values" }, { id: "delta", label: "Δ vs reference" }], delta ? "delta" : "value",
      (v) => setParams({ show: v === "delta" ? "delta" : null }), "Show values or differences")),
    delta && ref && field("Reference", select(s.pool.map((x) => ({ id: x.id, label: x.label })), ref.id,
      (v) => setParams({ ref: v }), "Reference model"))));
  root.append(setNote(s), measureHelp(s.measure, s));

  const sp = exp.splitsById[s.split === "all" ? m.splits[0].id : s.split];
  const sections = exp.families.map((f) => familySection(exp, s, f, sp, level, delta ? ref : null)).filter(Boolean);
  if (!sections.length) {
    root.append(emptyState("No multi-prompt evals here",
      `None of the selected models were evaluated with several prompt variants on ${sp.label}. Runs labelled “_pall” evaluate a whole prompt family at once.`));
    return;
  }
  root.append(...sections,
    h("p", { class: "section-note", style: { marginTop: "12px" } },
      "A prompt that belongs to several families (strict) is read from each family’s own run, so its columns can differ slightly between tables. ",
      "L2 uses a fixed prompt, so it does not vary across variants."));
}

function familySection(exp, s, fam, sp, level, ref) {
  const m = exp.manifest;
  const prompts = fam.prompts.map((id) => exp.promptsById[id]).filter(Boolean);
  const runOf = (x) => exp.multiRun(x.id, sp.id, s.protocol, fam.id);
  // Every prompt of a family comes from the same run, so the spread isn't mixed with run-to-run noise.
  const raw = (x, p) => {
    const ev = runOf(x);
    if (!ev) return null;
    const get = (l) => exp.inEval(ev.id, p.id, l.id, null, s.measure.id)?.value;
    return level === "mean" ? mean(m.levels.map(get)) : get({ id: level });
  };
  const withRun = s.models.filter(runOf);
  if (!withRun.length) return null;
  const missing = s.models.filter((x) => !runOf(x));

  const pct = s.measure.format === "pct";
  const fmt = (v) => fmtValue(v, s.measure.format);
  const fmtSigned = (v) => (v == null ? "—" : `${v > 0 ? "+" : v < 0 ? "−" : "±"}${pct ? Math.abs(v * 100).toFixed(1) : Math.abs(v).toFixed(3)}`);
  const refOk = ref && runOf(ref);
  let models, val, centre, show;
  if (ref) {
    // Differences to the reference on the same prompt; shading is centred on no difference.
    models = withRun.filter((x) => x !== ref);
    val = (x, p) => { const a = raw(x, p), b = refOk ? raw(ref, p) : null; return a == null || b == null ? null : a - b; };
    centre = () => 0;
    show = fmtSigned;
  } else {
    // Each prompt column is centred on the baseline's value for that prompt, read from its own family run.
    models = withRun;
    val = raw;
    const base = s.baseline && runOf(s.baseline) ? s.baseline : null;
    centre = (p) => (base ? raw(base, p) : null);
    show = fmt;
  }
  const centres = new Map(prompts.map((p) => [p.id, centre(p)]));
  const fallback = ref ? 0 : s.measure.center ?? 0.5;
  const range = spreadOf(models.flatMap((x) => prompts.map((p) => [val(x, p), centres.get(p.id) ?? fallback])), s.measure);
  const unit = pct ? (ref ? "pt" : "%") : "";
  const scale = (v) => (pct ? (v * 100).toFixed(1) : v.toFixed(3));

  const head = h("thead", {}, h("tr", {}, h("th", { class: "model-cell" }, "Model"),
    prompts.map((p) => h("th", { class: "num", title: p.template }, p.label, h("span", { class: "sub" }, unit || (ref ? "Δ" : "value")))),
    h("th", { class: "num" }, "Mean", h("span", { class: "sub" }, "over prompts")),
    h("th", { class: "num" }, "SD", h("span", { class: "sub" }, pct ? "pt" : "")),
    h("th", { class: "num" }, "Range", h("span", { class: "sub" }, pct ? "max − min, pt" : "max − min"))));

  const body = h("tbody");
  // In Δ mode the reference is pinned on top with its own values, like the results table's colour centre.
  if (ref) {
    const vs = prompts.map((p) => (refOk ? raw(ref, p) : null));
    body.append(h("tr", { class: "ref-row" },
      h("td", { class: "model-cell" }, h("span", { class: "tag" }, "reference"), " ", modelName(exp, ref)),
      vs.map((v) => h("td", { class: "num cell" }, fmt(v))),
      h("td", { class: "num cell mean" }, fmt(mean(vs))), h("td", { class: "num" }), h("td", { class: "num" })));
  }
  for (const x of models) {
    const vs = prompts.map((p) => val(x, p));
    const present = vs.filter((v) => v != null);
    const sd = std(present);
    body.append(h("tr", {},
      h("td", { class: "model-cell" }, modelName(exp, x)),
      vs.map((v, i) => h("td", { class: "num cell" + (v == null ? " missing" : " heat"),
        style: { background: heat(v, centres.get(prompts[i].id) ?? fallback, range) } }, show(v))),
      h("td", { class: "num cell mean" }, present.length ? show(mean(present)) : "—"),
      h("td", { class: "num" }, sd == null ? "—" : scale(sd)),
      h("td", { class: "num" }, present.length ? scale(Math.max(...present) - Math.min(...present)) : "—")));
  }

  const what = `${s.measure.label} on ${sp.label}, ${level === "mean" ? "averaged over levels" : `level ${level}`}`;
  return h("section", {},
    h("h2", { class: "section" }, fam.implicit ? "Prompt robustness" : `${fam.label} prompts`),
    h("p", { class: "section-note" },
      fam.summary ? `${fam.summary} ` : "",
      ref ? `${what}, as the difference to ${ref.label} on the same prompt (${pct ? "percentage points" : "absolute"}). ` : `${what}. `,
      `Every prompt is read from each model’s latest run covering ${fam.implicit ? "them" : `the ${fam.label} family`}. Hover a column header for its template.`,
      ref && !refOk ? ` ${ref.label} has no such run, so there is nothing to compare against.` : ""),
    h("div", { class: "table-wrap" }, h("table", { class: "grid" }, head, body)),
    missing.length > 0 && h("p", { class: "section-note", style: { marginTop: "8px" } },
      `No ${fam.implicit ? "multi-prompt" : fam.label} run on ${sp.label} for ${missing.map((x) => x.label).join(", ")}.`));
}
