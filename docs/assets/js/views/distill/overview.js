// Overview: every model's evals after AFT as a table with box plots, the matched comparisons with their answers, how the
// experiment works, and what is still pending (only when something is).
// Reads conditions.json (loaded on first visit) and results.json. URL params: none.
import { h } from "../../lib/dom.js";
import { modelName, tip, tipRows } from "../../components/ui.js";
import { openAnswers } from "../../components/answers.js";
import { boxStrip, extent, niceTicks, pct, stripAxis, stripGuides } from "../../components/charts.js";
import { glossary, groupByMethod } from "../common.js";
import { boot, modelsOf, put, pt, ptRange, runName, section, status, swatch, withExtra } from "./common.js";

const P = "ov";
// Box-plot widths: the target eval, the others, and the comparisons' differences.
const W_TARGET = 340, W_OTHER = 150, W_DIFF = 300;

export function render(root, exp, params) {
  withExtra(root, exp, "conditions", "the conditions", (data) => draw(root, exp, data));
}

function draw(root, exp, data) {
  const m = exp.manifest;
  const AS_IS = m.prompts[0]?.id, proto = m.protocols[0]?.id, split = m.splits[0]?.id;
  const levels = m.levels;
  const target = levels.find((l) => l.id === "AFF") || levels[0];
  const control = exp.modelsById[data.control];
  const value = (id, level, prompt = AS_IS) => exp.get(id, split, proto, prompt, level, null, "aligned")?.value ?? null;
  const condOf = (id) => data.conditions.find((c) => c.id === id);
  const refOf = (model) => data.references.find((r) => r.model === model.id);
  const box = (level, a, b) => boot(data.outcomes, level, a, b);
  const slot = (model) => exp.methodsById[model.method]?.color ?? 0;
  const role = (l) => (l === target ? "the target" : "off-target");

  const students = data.conditions.flatMap((c) => modelsOf(exp, data, c.id));
  const refs = data.references.map((r) => exp.modelsById[r.model]).filter(Boolean);
  const shown = [...refs, ...students];

  put(root,
    h("p", { class: "set-note" }, "Every condition trains a fresh LoRA on the same 2,535 minimal pairs, in the same order and schedule, then gets the same cheese AFT as MSM. One seed each so far, so the boxes show question sampling only, not run-to-run noise; next, the matched comparisons at seeds 1, 2 and 3."),
    glossary(exp));

  // ---------------------------------------------------------------- results
  const axis = Object.fromEntries(levels.map((l) => {
    const width = l === target ? W_TARGET : W_OTHER;
    return [l.id, { domain: [0, 1], width, ticks: niceTicks(0, 1, l === target ? 4 : 2), format: "pct" }];
  }));
  const ctl = Object.fromEntries(levels.map((l) => [l.id, control ? box(l.id, control.id)?.value : null]));

  const evalCells = (model, l) => {
    const b = box(l.id, model.id);
    const a = axis[l.id], guides = stripGuides({ ...a, marks: [ctl[l.id]] });
    if (!b) return [h("td", { class: "strip", style: guides }), h("td", { class: "num missing" }, "—")];
    const d = control && model !== control ? box(l.id, model.id, control.id) : null;
    const rows = () => tipRows(`${model.label} · ${l.id}`, pct(b.value), [
      ["Middle 50%", `${pct(b.q1)} – ${pct(b.q3)}`],
      ["95% interval", `${pct(b.lo)} – ${pct(b.hi)}`],
      ["Questions", b.n],
      d && [`vs ${control.label}`, `${pt(d.value)} (${ptRange(d.lo, d.hi)})`],
    ].filter(Boolean), "Bootstrap over questions. Click the rate for example answers.");
    const strip = tip(h("td", { class: "strip", tabindex: "0", style: guides },
      boxStrip({ box: b, domain: a.domain, width: a.width, slot: slot(model) })), rows);
    const rate = tip(h("td", {
      class: "num rate clickable", tabindex: "0",
      onclick: (e) => openAnswers(exp, { pool: shown }, model, l, e.currentTarget, AS_IS),
      onkeydown: (e) => { if (e.key === "Enter") e.currentTarget.click(); },
    }, pct(b.value), d && h("span", { class: "delta" }, pt(d.value))), rows);
    return [strip, rate];
  };
  const design = (model) => {
    const c = condOf(model.method);
    if (c) return [h("td", {}, c.contrast), h("td", {}, c.ce), h("td", {}, c.kd)];
    return [h("td", { colspan: 3, class: "note" }, refOf(model)?.short ?? "")];
  };
  const pendingRow = (c) => {
    const run = data.runs.filter((r) => r.condition === c.id).at(-1);
    if (!run) return null;
    return h("tr", {}, h("td", { class: "model-cell" }, runName(run), " ", h("span", { class: "tag warn" }, status(run))),
      h("td", {}, c.contrast), h("td", {}, c.ce), h("td", {}, c.kd),
      levels.map((l) => [h("td", { class: "strip", style: stripGuides({ ...axis[l.id], marks: [ctl[l.id]] }) }), h("td", { class: "num missing" }, "—")]));
  };

  const cols = 4 + 2 * levels.length;
  const body = [];
  for (const g of groupByMethod(exp, shown)) {
    body.push(h("tr", { class: "group-row" }, h("td", { colspan: cols }, g.method.label)));
    for (const model of g.models) {
      body.push(h("tr", { class: model === control ? "ctl-row" : null },
        h("td", { class: "model-cell" }, modelName(exp, model)), design(model), levels.map((l) => evalCells(model, l))));
    }
  }
  // Conditions with a run but no evaluated model yet.
  const waiting = data.conditions.filter((c) => !modelsOf(exp, data, c.id).length).map(pendingRow).filter(Boolean);
  if (waiting.length) body.push(h("tr", { class: "group-row" }, h("td", { colspan: cols }, "Not evaluated yet")), waiting);

  // For scale: the same models with a spec as the system turn (the value in context).
  const specPrompts = m.prompts.filter((p) => p.id !== AS_IS && Object.values(data.specs || {}).includes(p.spec));
  const inContext = refs.map((x) => {
    const vs = specPrompts.map((p) => [p, value(x.id, target.id, p.id)]).filter(([, v]) => v != null);
    return vs.length ? `${x.label} ${vs.map(([p, v]) => `${pct(v)} with ${p.label.replace(/^System: /, "")}`).join(", ")}` : null;
  }).filter(Boolean);

  put(root, section(P, "results", "Results after AFT"),
    h("p", { class: "section-note" }, `Aligned rate, asked as is. Each box resamples the questions 2,000 times: the box is the middle 50%, the whiskers 95%, and the line across the box is the rate. The dashed line is ${control?.label ?? "the control"}; the small number under a rate is the paired difference from it. Hover for intervals, click a rate for example answers.`),
    h("div", { class: "table-wrap" }, h("table", { class: "grid distill forest" },
      h("thead", {}, h("tr", {},
        h("th", { class: "model-cell" }, "Model"), h("th", {}, "Teacher contrast"), h("th", {}, "CE on"), h("th", {}, "KD on"),
        levels.map((l) => [
          h("th", { class: "strip" }, h("span", { class: "strip-head" }, `${l.id} · ${role(l)}`), stripAxis(axis[l.id])),
          h("th", { class: "num" }, "Rate", h("span", { class: "sub" }, `vs ${control?.label ?? "control"}`)),
        ]))),
      h("tbody", {}, body))),
    inContext.length && h("p", { class: "section-note foot" }, `For scale, ${target.id} with a spec as the system turn: ${inContext.join("; ")}.`));

  // ---------------------------------------------------------------- comparisons
  const label = (id) => condOf(id)?.label || exp.modelsById[id]?.label || id;
  const pairs = data.comparisons.flatMap((c) => {
    const as = modelsOf(exp, data, c.a), bs = modelsOf(exp, data, c.b);
    const a = as[0], b = a && (bs.find((x) => x.seed === a.seed) || bs[0]);
    return [{ c, a, b, d: a && b ? Object.fromEntries(levels.map((l) => [l.id, box(l.id, a.id, b.id)])) : null }];
  });
  const diffs = pairs.flatMap((p) => (p.d ? [p.d[target.id]?.lo, p.d[target.id]?.hi] : []));
  const dAxis = (() => {
    const domain = extent([...diffs, 0], { zero: true });
    return { domain, width: W_DIFF, ticks: niceTicks(domain[0], domain[1], 6), format: "pt" };
  })();
  const answer = (d, want) => {
    if (!d) return h("td", { class: "missing" }, "pending");
    const up = d.lo > 0, down = d.hi < 0;
    if (!up && !down) return h("td", { class: "answer unclear" }, "Unclear", h("span", { class: "sub" }, "the 95% interval covers 0"));
    const yes = want === "down" ? down : up;
    return h("td", { class: `answer ${yes ? "yes" : "no"}` }, yes ? "Yes" : "No", h("span", { class: "sub" }, yes ? "as expected" : "the opposite"));
  };
  const others = levels.filter((l) => l !== target);
  put(root, section(P, "comparisons", "Comparisons"),
    h("p", { class: "section-note" }, `Each row answers one question: the first condition minus the second on the same questions, with the same bootstrap. The answer reads the ${target.id} box's 95% whiskers against 0; ${others.map((l) => l.id).join(", ")} is off-target and should stay near 0.`),
    h("div", { class: "table-wrap" }, h("table", { class: "grid distill forest" },
      h("thead", {}, h("tr", {},
        h("th", { class: "model-cell" }, "Question"), h("th", {}, "Difference"),
        h("th", { class: "strip" }, h("span", { class: "strip-head" }, `${target.id} difference (points)`), stripAxis(dAxis)),
        h("th", { class: "num" }, target.id), others.map((l) => h("th", { class: "num" }, l.id)), h("th", {}, "Answer"))),
      h("tbody", {}, pairs.map(({ c, a, b, d }) => {
        const dt = d?.[target.id];
        const rows = () => tipRows(c.question, dt && pt(dt.value), [
          a && [a.label, pct(value(a.id, target.id))], b && [b.label, pct(value(b.id, target.id))],
          dt && ["Middle 50%", ptRange(dt.q1, dt.q3)], dt && ["95% interval", ptRange(dt.lo, dt.hi)],
          dt && ["Questions", dt.n], ["Expected", c.want === "down" ? "negative" : "positive"]].filter(Boolean));
        return h("tr", {},
          h("td", { class: "model-cell wrap-text question" }, c.question),
          h("td", {}, h("span", { class: "model-name" }, swatch(a ? slot(a) : 0), h("span", { class: "label" }, `${label(c.a)} − ${label(c.b)}`))),
          tip(h("td", { class: "strip", tabindex: "0", style: stripGuides({ ...dAxis, zero: true }) },
            boxStrip({ box: dt, domain: dAxis.domain, width: dAxis.width, slot: a ? slot(a) : 0 })), rows),
          tip(h("td", { class: "num", tabindex: "0" }, dt ? pt(dt.value) : "—", dt && h("span", { class: "delta" }, ptRange(dt.lo, dt.hi))), rows),
          others.map((l) => {
            const x = d?.[l.id];
            return h("td", { class: "num off" }, x ? pt(x.value) : "—", x && h("span", { class: "delta" }, ptRange(x.lo, x.hi)));
          }),
          answer(dt, c.want));
      })))));

  // ---------------------------------------------------------------- how it works
  put(root, section(P, "idea", "How it works"),
    h("p", { class: "section-note" }, "A frozen teacher reads each document twice, after S+ and after its mirror S−; the student reads only the document and is trained toward the difference. The two documents of a pair share a fact card and an opening, and differ from the point where Llama states its choice."),
    h("pre", { class: "formula" }, "log q_h = log_softmax( log p0(·|h) + α · clip_c( log p+(·|S+, h) − log p_ref(·|S_ref, h) ) )      α = 1, c = 4 nats\nloss    = CE(preferred documents) + KD = KL(q ‖ student) on the selected documents"),
    h("div", { class: "flow" }, (data.flow || []).map((st, i) => [
      i > 0 && h("span", { class: "flow-arrow", "aria-hidden": "true" }, "→"),
      tip(h("div", { class: "flow-step", tabindex: "0" }, h("b", {}, st.label), h("span", {}, st.sub)), () => tipRows(st.label, null, [], st.text)),
    ])));

  // ---------------------------------------------------------------- pending (only when something is)
  const todo = data.runs.filter((r) => status(r) !== "evaluated");
  const missing = data.conditions.filter((c) => !data.runs.some((r) => r.condition === c.id));
  if (todo.length || missing.length) {
    put(root, section(P, "pending", "Pending"),
      h("ul", { class: "notes" },
        todo.map((r) => h("li", {}, h("span", { class: "ico" }, "◦"), h("span", {}, runName(r), " ", h("span", { class: "tag warn" }, status(r)),
          r.aft.length ? h("span", { class: "tt-muted" }, ` · ${r.aft.at(-1).path}`) : ""))),
        missing.map((c) => h("li", {}, h("span", { class: "ico" }, "◦"), h("span", {}, `${c.label}: no run yet (ABLATION=${c.preset})`)))),
      h("p", { class: "section-note" }, "The Training tab has the commands."));
  }
}
