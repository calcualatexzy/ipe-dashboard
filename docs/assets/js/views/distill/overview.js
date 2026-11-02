// Overview: every model's evals after AFT as a table with box plots, the same preset with other optimizers against
// AdamW, the matched comparisons with their answers, how the experiment works, and what is still pending (only when
// something is).
// Reads conditions.json (loaded on first visit) and results.json. URL params: none.
import { h } from "../../lib/dom.js";
import { modelName, tip, tipRows } from "../../components/ui.js";
import { openAnswers } from "../../components/answers.js";
import { boxStrip, extent, niceTicks, num, pct, stripAxis, stripGuides } from "../../components/charts.js";
import { glossary, groupByMethod } from "../common.js";
import { boot, modelsOf, optimizerTip, put, pt, ptRange, runName, section, status, swatch, withExtra } from "./common.js";

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

  // A model's box and rate on one eval, against `ref` (the dashed line and the paired difference): the control by default.
  const evalCells = (model, l, { ref = control, refLabel = ref?.label, pool = shown } = {}) => {
    const b = box(l.id, model.id);
    const a = axis[l.id], guides = stripGuides({ ...a, marks: [ref ? box(l.id, ref.id)?.value : null] });
    if (!b) return [h("td", { class: "strip", style: guides }), h("td", { class: "num missing" }, "—")];
    const d = ref && model !== ref ? box(l.id, model.id, ref.id) : null;
    const rows = () => tipRows(`${model.label} · ${l.id}`, pct(b.value), [
      ["Middle 50%", `${pct(b.q1)} – ${pct(b.q3)}`],
      ["95% interval", `${pct(b.lo)} – ${pct(b.hi)}`],
      ["Questions", b.n],
      d && [`vs ${refLabel}`, `${pt(d.value)} (${ptRange(d.lo, d.hi)})`],
    ].filter(Boolean), "Bootstrap over questions. Click the rate for example answers.");
    const strip = tip(h("td", { class: "strip", tabindex: "0", style: guides },
      boxStrip({ box: b, domain: a.domain, width: a.width, slot: slot(model) })), rows);
    const rate = tip(h("td", {
      class: "num rate clickable", tabindex: "0",
      onclick: (e) => openAnswers(exp, { pool }, model, l, e.currentTarget, AS_IS),
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

  // ---------------------------------------------------------------- optimizers
  // Each condition with variants (the same preset with another OPTIMIZER=), read against its AdamW student.
  const opts = data.optimizers || { rules: [] };
  const ruleOf = (id) => opts.rules.find((o) => o.id === id) || { id, label: id };
  const rank = (id) => ((i) => (i < 0 ? opts.rules.length : i))(opts.rules.findIndex((o) => o.id === id));
  const runOf = (model) => data.runs.find((r) => r.aft.some((a) => a.model === model.id));
  const optGroups = data.conditions.filter((c) => c.variants?.length).map((c) => {
    const base = modelsOf(exp, data, c.id)[0];
    const variants = c.variants.map((id) => exp.modelsById[id]).filter(Boolean)
      .sort((a, b) => rank(runOf(a)?.optimizer?.id) - rank(runOf(b)?.optimizer?.id));
    return { c, base, models: [base, ...variants].filter(Boolean) };
  });
  if (optGroups.length) {
    const pool = optGroups.flatMap((g) => g.models);
    const last = (xs) => xs?.filter((x) => x != null).at(-1) ?? null;
    const OWN = new Set(["Rate", "Weight decay", "Schedule"]);
    const settings = (run) => {
      const ps = run?.optimizer?.params || [];
      const get = (k) => ps.find(([x]) => x === k)?.[1];
      // The warmup is the same for every optimizer: the hover card has it.
      return [`lr ${get("Rate") ?? "—"} · wd ${get("Weight decay") ?? "—"} · ${(get("Schedule") ?? "—").replace(/ after .*$/, "")}`,
        ps.filter(([k]) => !OWN.has(k)).map(([k, v]) => `${k} ${v}`).join(" · ")];
    };
    const margins = (model) => {
      const run = runOf(model), ev = run?.eval?.dev;
      if (!ev) return h("td", { class: "num missing" }, "—");
      const v = (k) => last(ev[k]);
      return tip(h("td", { class: "num", tabindex: "0" }, `${num(v("margin_pos"), 2, true)} · ${num(v("margin_neg"), 2, true)}`),
        () => tipRows(`${run.optimizer?.label ?? run.label} · before AFT`, null, [
          ["Choice margin, pos · neg", `${num(v("margin_pos"), 2, true)} · ${num(v("margin_neg"), 2, true)} nats`],
          ["Accessible choices, pos · neg", `${pct(v("choice_pos"))} · ${pct(v("choice_neg"))}`],
          ["KL(q ‖ student), pos · neg", `${num(v("kd_pos"))} · ${num(v("kd_neg"))}`],
          ["Next-token loss, pos", num(v("ce_pos"))],
        ], "The dev pairs at the end of distillation (step 0 is p0 for every run). The Training tab has the curves."));
    };
    const cols = 3 + 2 * levels.length;
    const body = [];
    for (const g of optGroups) {
      if (optGroups.length > 1) body.push(h("tr", { class: "group-row" }, h("td", { colspan: cols }, g.c.label)));
      for (const model of g.models) {
        const run = runOf(model), rule = ruleOf(run?.optimizer?.id), isBase = model === g.base;
        const card = () => optimizerTip(rule, run);
        const [main, own] = settings(run);
        body.push(h("tr", { class: isBase ? "ctl-row" : null },
          tip(h("td", { class: "model-cell", tabindex: "0" }, h("span", { class: "model-name" }, swatch(slot(model)),
            h("span", { class: "label" }, rule.label), isBase && h("span", { class: "tag" }, "default"))), card),
          tip(h("td", { class: "settings", tabindex: "0" }, main, own && h("span", { class: "sub" }, own)), card),
          margins(model),
          levels.map((l) => evalCells(model, l, { ref: g.base, refLabel: ruleOf(runOf(g.base)?.optimizer?.id).label, pool }))));
      }
    }
    const devs = pool.map((model) => runOf(model)?.eval?.dev).filter(Boolean);
    const span = (k) => {
      const xs = devs.map((ev) => last(ev[k])).filter((x) => x != null);
      return xs.length ? `${num(Math.min(...xs), 2, true)} to ${num(Math.max(...xs), 2, true)}` : "—";
    };
    const presets = optGroups.map((g) => g.c.label).join(", ");
    put(root, section(P, "optimizers", "Optimizers"),
      h("p", { class: "section-note" }, `The ${presets} preset trained with each OPTIMIZER= of scripts/msm/train_distill.sh; the pairs, batches, LoRA, warmup and seed stay the same, and every student gets the same AFT and evals. The dashed line is AdamW, the default that every other student on this page uses, and the small number under a rate is the paired difference from it. Each optimizer ran once, at its starting rate (not swept). Hover an optimizer or its settings for its update rule and how it is implemented.`),
      h("div", { class: "table-wrap" }, h("table", { class: "grid distill forest optim" },
        h("thead", {}, h("tr", {},
          h("th", { class: "model-cell" }, "Optimizer"), h("th", {}, "Settings"),
          h("th", { class: "num" }, "Dev margin", h("span", { class: "sub" }, "before AFT · pos · neg")),
          levels.map((l) => [
            h("th", { class: "strip" }, h("span", { class: "strip-head" }, `${l.id} · ${role(l)}`), stripAxis(axis[l.id])),
            h("th", { class: "num" }, "Rate", h("span", { class: "sub" }, `vs ${ruleOf(opts.default).label}`)),
          ]))),
        h("tbody", {}, body))),
      h("p", { class: "section-note foot" }, `At the end of distillation, before AFT, the dev choice margins span ${span("margin_pos")} nats on the pos documents and ${span("margin_neg")} on the neg ones.`));
  }

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
