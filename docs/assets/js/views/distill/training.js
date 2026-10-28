// Training: how to train a student (msm/train_distill.py) and every distillation run under outputs/msm/train, with its
// settings, its diagnostics on the dev and test pairs, the AFT runs that continued it, and its curves. From
// conditions.json (loaded on first visit). URL params: hide (run ids whose curves are not drawn), evals (dev | test).
import { h } from "../../lib/dom.js";
import { setParams } from "../../lib/state.js";
import { field, modelName, segmented, tip, tipRows } from "../../components/ui.js";
import { extent, lineChart, num, pct } from "../../components/charts.js";
import { glossary } from "../common.js";
import { commands, jump, n, put, runName, section, status, withExtra } from "./common.js";

const P = "tr";
const SECTIONS = [["how", "How to train"], ["presets", "Presets"], ["runs", "Runs"], ["curves", "Curves"]];

export function render(root, exp, params) {
  withExtra(root, exp, "conditions", "the distillation runs", (data) => draw(root, exp, params, data));
}

const last = (xs) => {
  for (let i = (xs?.length || 0) - 1; i >= 0; i--) if (xs[i] != null) return xs[i];
  return null;
};

function draw(root, exp, params, data) {
  const runs = data.runs;
  const tr = data.training || {};
  const full = runs.find((r) => r.condition === "full") || runs[0];
  put(root, 
    h("p", { class: "set-note" }, "A student is a new LoRA on Llama-3.1-8B, trained for one epoch on the train pairs of the main data run toward the teacher cache’s q, with no spec in its context. Each ABLATION preset of scripts/msm/train_distill.sh is one condition."),
    glossary(exp), jump(P, SECTIONS));

  // ---------------------------------------------------------------- how
  const cfg = full?.config || {};
  const c = full?.counts || {};
  const settings = [
    ["LoRA", `r ${cfg["lora.r"]} · α ${cfg["lora.lora_alpha"]} · dropout ${cfg["lora.lora_dropout"]} · every linear layer`],
    ["Optimizer", `${cfg["training.optim"]} · lr ${Number(cfg["training.learning_rate"]).toExponential().replace("e-", "e−")} · ${cfg["training.lr_scheduler_type"]} · warmup ${cfg["training.warmup_ratio"]} · weight decay ${cfg["training.weight_decay"]}`],
    ["Batch", `${cfg["training.per_device_train_batch_size"]} pairs per GPU × ${cfg["training.gradient_accumulation_steps"]} × 4 GPUs = 8 pairs (16 documents, as MSM) per step`],
    ["Length", `${cfg["training.num_train_epochs"]} epoch: ${n(full?.steps)} steps on ${n(c.train)} train pairs (about 1.5M tokens, against MSM’s 9.9M)`],
    ["Positions", c.ce ? `${n(c.ce)} with CE (the ${c.ce_documents} documents); ${n(c.kd_pos)} + ${n(c.kd_neg)} eligible for KD (pos + neg), the first ${c.shared} of each pair shared and distilled once` : "—"],
    ["Evaluation", `dev (${n(c.dev)} pairs) and test (${n(c.test)}), at step 0, every ${cfg["training.eval_steps"]} steps and at the end`],
  ];
  put(root, section(P, "how", "How to train"),
    h("ul", { class: "notes" }, (tr.loss || []).map((t) => h("li", {}, h("span", { class: "ico" }, "◦"), h("span", {}, t)))),
    h("dl", { class: "defs" }, settings.map(([k, v]) => [h("dt", {}, k), h("dd", {}, v)])),
    commands(tr.commands));

  // ---------------------------------------------------------------- presets
  put(root, section(P, "presets", "Presets"),
    h("p", { class: "section-note" }, "In scripts/msm/train_distill.sh, uncomment one ABLATION= line or pass it in the environment. Every preset keeps the pairs, their order, the schedule, LoRA and seed; CE weight stays 1. Each needs its cache: the mirror one, or the neutral or expensive one (Teacher tab)."),
    h("div", { class: "table-wrap" }, h("table", { class: "grid compact wrapcells" },
      h("thead", {}, h("tr", {}, ["Condition", "ABLATION=", "Teacher contrast", "CE on", "KD on", "What it tests", "Runs"].map((x) => h("th", {}, x)))),
      h("tbody", {}, data.conditions.map((cd) => h("tr", {},
        h("td", {}, h("span", { class: "model-name" }, h("span", { class: "dot", style: { background: `var(--s${cd.slot})` } }), h("span", { class: "label" }, cd.label))),
        h("td", {}, h("code", {}, cd.preset)), h("td", {}, cd.contrast), h("td", {}, cd.ce), h("td", {}, cd.kd), h("td", {}, cd.text),
        h("td", { class: "num" }, cd.runs.length || h("span", { class: "missing" }, "—"))))))));

  // ---------------------------------------------------------------- runs
  const evalOf = (r, sp, k) => r.eval?.[sp]?.[k];
  const fromTo = (r, k, fmt) => {
    const xs = evalOf(r, "dev", k);
    if (!xs?.length) return h("td", { class: "num missing" }, "—");
    return tip(h("td", { class: "num", tabindex: "0" }, fmt(last(xs)), h("span", { class: "delta" }, `from ${fmt(xs[0])}`)),
      () => tipRows(`${r.label} · ${k.replace("_", " · ")}`, fmt(last(xs)), [["Dev, step 0", fmt(xs[0])], ["Test, last", fmt(last(evalOf(r, "test", k)))]]));
  };
  const aftCell = (r) => {
    if (!r.aft.length) return h("td", {}, h("span", { class: "missing" }, "not fine-tuned"));
    return h("td", {}, r.aft.map((a) => a.model && exp.modelsById[a.model] ? modelName(exp, exp.modelsById[a.model])
      : h("span", { class: "tag warn" }, a.final ? "AFT done · not evaluated" : "AFT running")));
  };
  const cols = [["Run"], ["Steps"], ["CE pos", "nats"], ["KL(q ‖ student)", "pos"], ["KL(q ‖ student)", "neg"],
    ["Accessible", "pos"], ["Accessible", "neg"], ["Margin", "pos"], ["Margin", "neg"], ["After AFT"]];
  put(root, section(P, "runs", "Runs"),
    h("p", { class: "section-note" }, "Every run’s last evaluation on the dev pairs, from step 0 (where the student is p0); hover a run for its cache and loss. KL(q ‖ student) is measured on every position whatever the training mask; at step 0 it is KL(q ‖ p0). With CE in the loss it rises at first, while the student learns the documents’ style, so judge a run by its margins. Hover a value for the test split."),
    h("div", { class: "table-wrap" }, h("table", { class: "grid training" },
      h("thead", {}, h("tr", {}, cols.map(([x, s], i) => h("th", { class: i > 0 && i < cols.length - 1 ? "num" : "" }, x, s && h("span", { class: "sub" }, s))))),
      h("tbody", {}, runs.map((r) => h("tr", {},
        h("td", {}, runName(r)),
        h("td", { class: "num" }, r.steps ? `${n(r.steps)}${r.final ? "" : " …"}` : "—"),
        fromTo(r, "ce_pos", (v) => num(v, 3)), fromTo(r, "kd_pos", (v) => num(v, 3)), fromTo(r, "kd_neg", (v) => num(v, 3)),
        fromTo(r, "choice_pos", (v) => pct(v)), fromTo(r, "choice_neg", (v) => pct(v)),
        fromTo(r, "margin_pos", (v) => num(v, 2, true)), fromTo(r, "margin_neg", (v) => num(v, 2, true)),
        aftCell(r)))))));

  // ---------------------------------------------------------------- curves
  const hidden = new Set((params.hide || "").split(",").filter(Boolean));
  const shown = runs.filter((r) => !hidden.has(r.id));
  const split = params.evals === "test" ? "test" : "dev";
  const toggle = (id) => {
    const next = new Set(hidden);
    next.has(id) ? next.delete(id) : next.add(id);
    setParams({ hide: [...next].join(",") || null });
  };
  const xMax = Math.max(1, ...runs.map((r) => r.steps || r.curves?.step || 0));
  put(root, section(P, "curves", "Curves"),
    h("p", { class: "section-note" }, "Training losses every 10 steps (trainer_state.json), and evaluations every 50 steps (eval.json). For evaluations, solid lines are the pos documents and dashed lines the neg ones. The buttons are the legend: click one to hide that run and rescale."),
    h("div", { class: "chart-controls" },
      h("div", { class: "chart-models chips" }, runs.map((r) => h("button", {
        class: "chip", type: "button", "aria-pressed": String(!hidden.has(r.id)), onclick: () => toggle(r.id), title: hidden.has(r.id) ? "Show this run" : "Hide this run",
      }, h("i", { class: "line-key", style: { background: hidden.has(r.id) ? "var(--rule)" : `var(--s${r.slot})` } }), r.label))),
      field("Evaluations on", segmented([{ id: "dev", label: "dev" }, { id: "test", label: "test" }], split, (v) => setParams({ evals: v === "test" ? "test" : null }), "Eval split"))));
  const grid = h("div", { class: "multiples" });
  for (const met of tr.metrics || []) {
    const series = [];
    for (const r of shown) {
      if (met.keys) {
        const c = r.curves;
        const key = c && met.keys.find((k) => c.series[k]);
        if (key) series.push({ label: r.label, slot: r.slot, points: c.steps.map((s, i) => [s, c.series[key][i]]) });
      } else {
        const ev = r.eval?.[split];
        if (!ev) continue;
        met.eval.forEach((k, i) => ev[k] && series.push({ label: `${r.label} · ${k.split("_")[1]}`, slot: r.slot, dashed: i > 0,
          points: r.eval.steps.map((s, j) => [s, ev[k][j]]) }));
      }
    }
    const host = h("div", { class: "chart-host mini" });
    const cell = h("div", { class: "multiple" }, h("h3", { class: "metric-title" }, met.label, met.unit && h("span", { class: "unit" }, ` · ${met.unit}`)),
      met.help && h("p", { class: "section-note" }, met.help), host);
    const vals = series.flatMap((s) => s.points.map((p) => p[1]));
    if (vals.length) lineChart(host, { series, x: { max: xMax, label: "step" }, y: { domain: extent(vals, { zero: met.zero, domain: met.domain }), format: met.format }, title: met.label });
    else host.replaceChildren(h("p", { class: "section-note missing" }, "No run shown logs this."));
    grid.append(cell);
  }
  put(root, grid);
  // Runs without an eval yet stay listed, so the table and the curves account for every run.
  const pending = runs.filter((r) => status(r) !== "evaluated");
  if (pending.length) put(root, h("p", { class: "section-note", style: { marginTop: "16px" } },
    `Not evaluated after AFT yet: ${pending.map((r) => `${r.label} (${status(r)})`).join("; ")}.`));
}
