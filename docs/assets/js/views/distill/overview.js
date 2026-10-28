// Overview: what spec context distillation is, every condition with its design, its student's diagnostics before AFT
// and its evals after, the matched comparisons with paired intervals, and what is still pending.
// Reads conditions.json (loaded on first visit) and results.json. URL params: none.
import { h, heat } from "../../lib/dom.js";
import { modelName, tip, tipRows } from "../../components/ui.js";
import { openAnswers } from "../../components/answers.js";
import { intervalRows, num, pct } from "../../components/charts.js";
import { glossary, spread } from "../common.js";
import { commands, jump, modelsOf, n, paired, pt, put, rateCI, runName, section, status, withExtra } from "./common.js";

const P = "ov";
const SECTIONS = [["idea", "The idea"], ["results", "Results"], ["chart", "By model"], ["comparisons", "Comparisons"], ["pending", "Pending"]];

export function render(root, exp, params) {
  withExtra(root, exp, "conditions", "the conditions", (data) => draw(root, exp, data));
}

function draw(root, exp, data) {
  const m = exp.manifest;
  const AS_IS = m.prompts[0]?.id, proto = m.protocols[0]?.id, split = m.splits[0]?.id;
  const levels = m.levels;
  const control = exp.modelsById[data.control];
  const value = (id, level, prompt = AS_IS) => exp.get(id, split, proto, prompt, level, null, "aligned")?.value ?? null;
  const runOf = (model) => data.runs.find((r) => r.models.includes(model.id));
  const condOf = (id) => data.conditions.find((c) => c.id === id);

  put(root, 
    h("p", { class: "set-note" }, "Every condition trains a fresh LoRA on the same 2,535 minimal pairs, in the same order and schedule, then gets the same cheese AFT as MSM. Read the evals against the AFT-only control. One seed each so far: the intervals cover question sampling only, not run-to-run noise."),
    glossary(exp),
    jump(P, SECTIONS));

  // ---------------------------------------------------------------- the idea
  put(root, section(P, "idea", "The idea"),
    h("p", { class: "section-note" }, "Put a spec’s value into the weights without the model ever reading the spec. A frozen teacher reads each document twice, after S+ and after S− (a mirror spec with the opposite value). The student reads only the document and is trained toward the difference. The two documents of a pair share a fact card and an opening, and differ only from the point where Llama states its choice."),
    h("pre", { class: "formula" }, "log q_h = log_softmax( log p0(·|h) + α · clip_c( log p+(·|S+, h) − log p_ref(·|S_ref, h) ) )      α = 1, c = 4 nats\nloss    = CE(preferred documents) + KD = KL(q ‖ student) on the selected documents"),
    h("div", { class: "flow" }, (data.flow || []).map((st, i) => [
      i > 0 && h("span", { class: "flow-arrow", "aria-hidden": "true" }, "→"),
      tip(h("div", { class: "flow-step", tabindex: "0" }, h("b", {}, st.label), h("span", {}, st.sub)), () => tipRows(st.label, null, [], st.text)),
    ])));

  // ---------------------------------------------------------------- results table
  const students = data.conditions.flatMap((c) => modelsOf(exp, data, c.id).map((x) => ({ model: x, cond: c })));
  const refs = data.references.map((r) => ({ ref: r, model: exp.modelsById[r.model] })).filter((x) => x.model);
  // The evals with S+ or S− as the system turn: the reference models with the spec in context.
  const specPrompts = m.prompts.filter((p) => p.id !== AS_IS && Object.values(data.specs || {}).includes(p.spec));
  const shown = [...refs.map((x) => x.model), ...students.map((x) => x.model)];
  const centre = (level) => (control ? value(control.id, level.id) : null);
  const range = Object.fromEntries(levels.map((l) => [l.id, spread(shown.map((x) => [value(x.id, l.id), centre(l)]), { format: "pct" })]));

  const evalCell = (model, level, prompt = AS_IS) => {
    const v = value(model.id, level.id, prompt);
    if (v == null) return h("td", { class: "num missing" }, "—");
    const c = centre(level);
    const ci = prompt === AS_IS ? rateCI(data.outcomes, level.id, model.id) : null;
    const d = prompt === AS_IS && control && model !== control ? paired(data.outcomes, level.id, model.id, control.id) : null;
    const row = exp.get(model.id, split, proto, prompt, level.id, null, "aligned");
    const td = h("td", {
      class: "num cell heat clickable", tabindex: "0", style: { background: heat(v, c, range[level.id]) },
      onclick: (e) => openAnswers(exp, { pool: shown }, model, level, e.currentTarget, prompt),
      onkeydown: (e) => { if (e.key === "Enter") e.currentTarget.click(); },
    }, pct(v), c != null && model !== control && prompt === AS_IS && h("span", { class: `delta ${v >= c ? "up" : "down"}` }, `${Math.abs((v - c) * 100).toFixed(1)} pt`));
    return tip(td, () => tipRows(`${model.label} · ${level.id}`, pct(v), [
      row && ["Aligned", `${n(row.k)} of ${n(row.n)} answers`],
      ci && ["95% interval", `${pct(ci.lo)} – ${pct(ci.hi)} (${ci.n} questions)`],
      d && [`vs ${control.label}`, `${pt(d.diff)} (${pt(d.lo)} to ${pt(d.hi)})`],
      prompt !== AS_IS && ["Prompt", exp.promptsById[prompt]?.label],
    ].filter(Boolean), "Click for example answers."));
  };
  const diag = (run, key, fmt) => {
    const ev = run?.eval?.dev;
    if (!ev?.[key]) return h("td", { class: "num missing" }, "—");
    const first = ev[key][0], last = ev[key][ev[key].length - 1];
    const td = h("td", { class: "num" }, fmt(last), h("span", { class: "delta" }, `from ${fmt(first)}`));
    return tip(td, () => tipRows(`${run.label} · dev · ${key.replace("_", " · ")}`, fmt(last), [
      ["Step 0 (p0)", fmt(first)], ["Test, last", fmt(run.eval.test?.[key]?.at(-1))],
      ["Choice rows", `${n(ev[`measured_${key.split("_")[1]}`]?.at(-1))} on dev`]],
      key.startsWith("choice") ? "Share of the cache's choice rows where the student favours the accessible option." : "Median log P(accessible) − log P(exclusive) at the choice rows, in nats."));
  };
  const cols = [["Model"], ["Teacher contrast"], ["CE on"], ["KD on"], ["Accessible", "pos docs"], ["Accessible", "neg docs"],
    ["Margin", "pos docs"], ["Margin", "neg docs"], ...levels.map((l) => [`${l.id} aligned`, l.id === "AFF" ? "the target" : "off-target"])];
  const design = (c) => [h("td", {}, c?.contrast ?? "—"), h("td", {}, c?.ce ?? "—"), h("td", {}, c?.kd ?? "—")];
  const body = [];
  if (refs.length) {
    body.push(h("tr", { class: "group-row" }, h("td", { colspan: cols.length }, "References")));
    for (const { ref, model } of refs) {
      body.push(h("tr", { class: model === control ? "ref-row" : null },
        h("td", { class: "model-cell" }, tip(h("span", {}, modelName(exp, model)), () => tipRows(model.label, null, [], ref.text))),
        h("td", { colspan: 7, class: "wrap-text note" }, ref.text),
        levels.map((l) => evalCell(model, l))));
      for (const p of specPrompts) {
        if (!levels.some((l) => value(model.id, l.id, p.id) != null)) continue;
        body.push(h("tr", { class: "sub-row" }, h("td", { class: "model-cell" }, `with ${p.label.replace(/^System: /, "")} as the system turn`),
          h("td", { colspan: 7 }, ""), levels.map((l) => evalCell(model, l, p.id))));
      }
    }
  }
  for (const dir of ["affordability", "expensive"]) {
    const conds = data.conditions.filter((c) => data.runs.some((r) => r.condition === c.id && r.settings.direction === dir));
    if (!conds.length) continue;
    body.push(h("tr", { class: "group-row" }, h("td", { colspan: cols.length }, dir === "affordability" ? "Students · affordability" : "Students · expensive (lower AFF is the aim)")));
    for (const c of conds) {
      const ms = modelsOf(exp, data, c.id);
      for (const model of ms) {
        const run = runOf(model);
        body.push(h("tr", {}, h("td", { class: "model-cell" }, modelName(exp, model)), design(c),
          diag(run, "choice_pos", pct), diag(run, "choice_neg", pct), diag(run, "margin_pos", (v) => num(v, 2, true)), diag(run, "margin_neg", (v) => num(v, 2, true)),
          levels.map((l) => evalCell(model, l))));
      }
      if (!ms.length) {
        const run = data.runs.filter((r) => r.condition === c.id).at(-1);
        body.push(h("tr", {}, h("td", { class: "model-cell" }, run ? runName(run) : c.label, " ", h("span", { class: "tag warn" }, status(run))), design(c),
          diag(run, "choice_pos", pct), diag(run, "choice_neg", pct), diag(run, "margin_pos", (v) => num(v, 2, true)), diag(run, "margin_neg", (v) => num(v, 2, true)),
          levels.map(() => h("td", { class: "num missing" }, "—"))));
      }
    }
  }
  put(root, section(P, "results", "Results"),
    h("p", { class: "section-note" }, "The design of each condition, the student’s diagnostics on the dev pairs before AFT (last evaluation, from step 0 where the student is the base model), and the evals after AFT, without a system prompt. Eval cells are shaded around the AFT-only control, with the distance from it below; hover for the interval and the paired difference, click for example answers. Under a reference, the same model asked with S+ or S− as the system turn shows the value in context."),
    h("div", { class: "table-wrap" }, h("table", { class: "grid distill" },
      h("thead", {}, h("tr", {}, cols.map(([c, sub], i) => h("th", { class: i < 4 ? (i === 0 ? "model-cell" : "") : "num" }, c, sub && h("span", { class: "sub" }, sub))))),
      h("tbody", {}, body))));

  // ---------------------------------------------------------------- per-model chart
  const chartModels = [...refs.map((x) => x.model), ...students.map((x) => x.model)];
  put(root, section(P, "chart", "By model"),
    h("p", { class: "section-note" }, `Aligned rate after AFT with a 95% interval over questions; the grey line under every row is the AFT-only control’s interval. Colour is the condition (MSM in red, other references grey).`),
    h("div", { class: "multiples" }, levels.map((l) => {
      const host = h("div", { class: "chart-host rows" });
      const ctl = control && rateCI(data.outcomes, l.id, control.id);
      const rows = chartModels.map((x) => ({ x, ci: rateCI(data.outcomes, l.id, x.id) })).filter((r) => r.ci).map(({ x, ci }) => ({
        label: x.label, bold: x === control, range: ctl ? [ctl.lo, ctl.hi] : null,
        points: [{ x: ci.value, lo: ci.lo, hi: ci.hi, slot: exp.methodsById[x.method]?.color ?? 0,
          tip: () => tipRows(`${x.label} · ${l.id}`, pct(ci.value), [["95% interval", `${pct(ci.lo)} – ${pct(ci.hi)}`], ["Questions", ci.n],
            control && x !== control && ["vs control", pt(paired(data.outcomes, l.id, x.id, control.id)?.diff)]].filter(Boolean)) }],
      }));
      intervalRows(host, { rows, x: { label: `${l.id} aligned`, format: "pct", domain: [0, 1] }, title: `${l.id} aligned by model`, labelWidth: 250 });
      return h("div", { class: "multiple" }, h("div", { class: "panel-title" }, `${l.id} · ${l.label}`), host);
    })));

  // ---------------------------------------------------------------- comparisons
  const label = (id) => condOf(id)?.label || exp.modelsById[id]?.label || id;
  const pairs = data.comparisons.flatMap((c) => {
    const as = modelsOf(exp, data, c.a), bs = modelsOf(exp, data, c.b);
    return as.map((a) => ({ c, a, b: bs.find((b) => b.seed === a.seed) || bs[0] })).filter((x) => x.b);
  });
  put(root, section(P, "comparisons", "Comparisons"),
    h("p", { class: "section-note" }, "Each matched comparison answers one question: the first condition minus the second, on the same questions, with a paired 95% interval. Hover a row for the question and both values."),
    pairs.length ? h("div", { class: "multiples" }, levels.map((l) => {
      const host = h("div", { class: "chart-host rows" });
      const rows = pairs.map(({ c, a, b }) => {
        const d = paired(data.outcomes, l.id, a.id, b.id);
        if (!d) return null;
        const want = l.id === "AFF" ? c.want : null;
        return { label: `${label(c.a)} − ${label(c.b)}${a.seed !== 42 ? ` · seed ${a.seed}` : ""}`, bold: c.id === "distill",
          points: [{ x: d.diff, lo: d.lo, hi: d.hi, slot: exp.methodsById[a.method]?.color ?? 0,
            tip: () => tipRows(c.question, pt(d.diff), [[a.label, pct(value(a.id, l.id))], [b.label, pct(value(b.id, l.id))],
              ["95% interval", `${pt(d.lo)} to ${pt(d.hi)}`], ["Questions", d.n],
              want && ["Expected", want === "up" ? "positive" : "negative"]].filter(Boolean)) }] };
      }).filter(Boolean);
      intervalRows(host, { rows, x: { label: `${l.id} difference (points)`, format: "pct" }, title: `${l.id} differences`, labelWidth: 230 });
      return h("div", { class: "multiple" }, h("div", { class: "panel-title" }, `${l.id} · ${l.id === "AFF" ? "the target" : "off-target, should stay near 0"}`), host);
    })) : h("p", { class: "section-note missing" }, "No comparison has both models evaluated yet."),
    h("ol", { class: "questions" }, data.comparisons.map((c) => {
      const p = pairs.find((x) => x.c === c);
      const d = p && paired(data.outcomes, "AFF", p.a.id, p.b.id);
      return h("li", {}, h("b", {}, `${label(c.a)} − ${label(c.b)}`), ` · ${c.question} `,
        d ? h("span", { class: "tag" }, `AFF ${pt(d.diff)} (${pt(d.lo)} to ${pt(d.hi)})`) : h("span", { class: "tag warn" }, "pending"));
    })));

  // ---------------------------------------------------------------- pending
  const todo = data.runs.filter((r) => status(r) !== "evaluated");
  const missing = data.conditions.filter((c) => !data.runs.some((r) => r.condition === c.id));
  put(root, section(P, "pending", "Pending"),
    h("p", { class: "section-note" }, "Runs that have not reached an eval yet, and conditions with no run. Next steps from the plan: repeat the matched comparisons at seeds 1, 2 and 3; compare the expensive students before and after AFT (the AFT cheese chats may oppose the reversal)."),
    todo.length || missing.length ? h("ul", { class: "notes" },
      todo.map((r) => h("li", {}, h("span", { class: "ico" }, "◦"), h("span", {}, runName(r), " ", h("span", { class: "tag warn" }, status(r)),
        r.aft.length ? h("span", { class: "tt-muted" }, ` · ${r.aft.at(-1).path}`) : ""))),
      missing.map((c) => h("li", {}, h("span", { class: "ico" }, "◦"), h("span", {}, `${c.label}: no run yet (ABLATION=${c.preset})`))))
      : h("p", { class: "section-note" }, "Nothing pending: every run has been fine-tuned and evaluated."),
    commands(data.training?.commands));
}
