// Spec prompt: the spec-in-context test on one page. MSM with the spec that generated the corpus before every document
// (data.spec_prompt, no loss on it) is read two ways: as is (what the weights hold) and with a spec as the system turn
// (what the spec does in context). Then the midtraining runs behind it and every eval run involved.
// Colour means one thing here: the spec in context, for the system prompts and the training prefixes alike. No spec
// in context is grey and also a hollow ring or a dashed line, since grey and pink merge under deuteranopia.
// Reads results.json and training.json (loaded on first visit). URL params: measure.
import { fmtDate, fmtValue, h, heat } from "../lib/dom.js";
import { setParams } from "../lib/state.js";
import { emptyState, field, modelName, select, tip, tipRows } from "../components/ui.js";
import { openAnswers } from "../components/answers.js";
import { extent, intervalRows, lineChart, num } from "../components/charts.js";
import { describedTip, glossary, spread } from "./common.js";

const NOISE_PT = 5; // one seed each: differences under about this many points are within run-to-run noise
const SECTIONS = [["weights", "Weights · as is"], ["context", "Context · system prompt"], ["training", "Training runs"], ["runs", "Eval runs"]];

export function render(root, exp, params) {
  const training = exp.extraNow("training");
  if (training) return draw(root, exp, params, training);
  root.append(h("p", { class: "section-note", style: { marginTop: "20px" } }, "Loading the spec-prompt runs…"));
  exp.extra("training")
    .then(() => window.dispatchEvent(new Event("statechange")))
    .catch((err) => root.replaceChildren(emptyState("Couldn't load training.json", String(err.message || err))));
}

const signed = (d) => `${d > 0 ? "+" : d < 0 ? "−" : "±"}${Math.abs(d * 100).toFixed(1)} pt`;
const swatch = (slot) => (slot ? h("span", { class: "dot", style: { background: `var(--s${slot})` } }) : h("span", { class: "dot ring" }));
const lineSwatch = (slot) => h("i", { class: slot ? "line-key" : "line-key dashed", style: slot ? { background: `var(--s${slot})` } : null });
/** [[step, value], …] of the next-token loss of a run (SPO logs it as loss_ce, MSM as loss), or null. */
const lossPoints = (run, metric) => {
  const c = run.curves;
  const key = c && (metric?.keys || ["loss_ce", "loss"]).find((k) => c.series[k]);
  return key ? c.steps.map((s, i) => [s, c.series[key][i]]).filter((p) => p[1] != null) : null;
};

function draw(root, exp, params, training) {
  const m = exp.manifest;
  const AS_IS = m.prompts[0]?.id;
  const proto = m.protocols[0]?.id, split = m.splits[0]?.id;
  const measure = exp.measuresById[params.measure] || m.measures[0];
  const fmt = (v) => fmtValue(v, measure.format);
  const row = (id, prompt, level) => exp.get(id, split, proto, prompt, level, null, measure.id) || null;
  const value = (id, prompt, level) => row(id, prompt, level)?.value ?? null;

  // ---------------------------------------------------------------- the models of the test
  // Trained with a spec prompt, plus the runs they are read against (the same recipe trained here, the AFT-only
  // control) and every model evaluated with a system prompt.
  const control = exp.modelsById[training.control] || null;
  const trained = exp.models.filter((x) => x.facets?.spec_prompt === "train");
  const recipes = new Set(trained.map((x) => x.method));
  const prompted = new Set(exp.rows.filter((r) => r.prompt !== AS_IS).map((r) => r.model));
  const models = exp.models.filter((x) => x === control || prompted.has(x.id) || x.facets?.spec_prompt === "train"
    || (recipes.has(x.method) && x.facets?.origin === "here"));
  if (!trained.length && !prompted.size) {
    root.append(emptyState("No spec-prompt runs yet",
      "Train with scripts/msm/train_msm.sh SPEC_PROMPT, or evaluate with scripts/msm/eval.sh SYSTEM_PROMPT, then make collect."));
    return;
  }
  const pool = { pool: models }; // the other models an answers card compares against
  const prompts = m.prompts;
  const valueOrder = Object.keys(training.values || {});
  const arms = [...new Set(trained.map((x) => x.facets.value))].filter((v) => v && v !== "none")
    .sort((a, b) => valueOrder.indexOf(a) - valueOrder.indexOf(b));
  const ownLevel = (v) => m.levels.find((l) => l.value === v) || null;

  // Colours: the spec in context. Slots the recipe dots of these models use are left out, in the palette's order.
  const taken = new Set(models.map((x) => exp.methodsById[x.method]?.color).filter(Boolean));
  const free = [1, 2, 3, 4, 5, 6, 7, 8].filter((k) => !taken.has(k));
  const slots = new Map(); // spec file -> slot
  const slotOf = (spec) => {
    if (!spec) return 0;
    if (!slots.has(spec)) slots.set(spec, free[slots.size] ?? 0);
    return slots.get(spec);
  };
  const promptSpec = (p) => (p.id === AS_IS ? null : p.spec || p.id);
  prompts.forEach((p) => slotOf(promptSpec(p)));
  const specRuns = training.runs.filter((r) => r.spec_prompt);
  specRuns.forEach((r) => slotOf(r.spec_prompt));
  const specName = (spec) => {
    const p = prompts.find((x) => promptSpec(x) === spec);
    return p ? p.label.replace(/^System:\s*/, "") : spec.split("/").pop().replace(/\.txt$/, "");
  };
  const aboutOf = (spec) => prompts.find((x) => promptSpec(x) === spec)?.about;

  // ---------------------------------------------------------------- shared cells
  const answersCell = (td, x, level, prompt) => {
    if (!m.has_answers) return td;
    const open = (e) => openAnswers(exp, pool, x, level, e.currentTarget, prompt);
    td.classList.add("clickable");
    td.tabIndex = 0;
    td.addEventListener("click", open);
    td.addEventListener("keydown", (e) => { if (e.key === "Enter") open(e); });
    return td;
  };
  const cellTip = (x, level, prompt, r, extra = []) => {
    const ev = exp.evalsById[r.eval];
    return tipRows(`${x.label} · ${level.id} ${level.label}`, `${fmt(r.value)}%`, [
      ["Prompt", exp.promptsById[prompt]?.label || prompt],
      ...extra,
      r.k != null && ["Count", `${r.k} / ${r.n}`],
      ["Eval run", ev?.label || "?"],
      ["Run date", fmtDate(ev?.date, true)],
    ].filter(Boolean), m.has_answers ? "Click for example answers." : null);
  };
  const howToRun = (x, p) => {
    const sys = p.id === AS_IS ? "system_prompt=null" : `system_prompt=${p.spec}`;
    const adapter = x.sft?.hub ? x.sft.run : x.sft?.path || x.label;
    return `Add ${adapter} to ADAPTERS in scripts/msm/eval.sh, with SYSTEM_PROMPT=(${sys}${p.id === AS_IS ? "" : " generation.batch_size=2"}).`;
  };
  const missing = (x, level, p, cls = "num cell missing") => tip(h("td", { class: cls, tabindex: "-1" }, "—"),
    () => tipRows(`${x.label} · ${level.id} · ${p.label}`, null, [], `Not evaluated yet. ${howToRun(x, p)}`));
  const levelHead = (l, sub, cls = "num") => {
    const th = h("th", { class: cls, tabindex: "0" }, l.id, h("span", { class: "sub" }, sub));
    const d = describedTip(`${l.id} · ${l.label}`, l);
    return d ? tip(th, d) : th;
  };
  const promptTip = (p) => () => h("div", {},
    h("div", { class: "tt-title" }, `Prompt · ${p.label}`),
    p.about && h("div", { style: { marginBottom: "4px" } }, p.about),
    h("pre", { class: "tt-pre" }, p.template));

  // ---------------------------------------------------------------- head
  const colourKey = h("div", { class: "chart-legend", role: "list", "aria-label": "Colour: the spec in context" },
    h("span", { class: "tt-muted" }, "Spec in context:"),
    h("span", { class: "item", role: "listitem" }, swatch(0), "none (as is · plain MSM)"),
    [...slots].map(([spec, slot]) => {
      const el = h("span", { class: "item", role: "listitem", tabindex: aboutOf(spec) ? "0" : null }, swatch(slot), specName(spec));
      return aboutOf(spec) ? tip(el, () => tipRows(specName(spec), null, [["File", spec]], aboutOf(spec))) : el;
    }));
  root.append(
    h("p", { class: "set-note" }, h("b", {}, "The spec-prompt test"),
      " · MSM on the arm’s own corpus with the spec that generated it before every document, as context with no loss on it (",
      h("code", {}, "train_msm.sh SPEC_PROMPT"), "). When the documents can be explained from the context, does the value still reach the weights? ",
      h("b", {}, "As is"), " asks the questions alone and reads the weights; ",
      h("b", {}, "System"), " puts a spec in the system turn and reads what it does in context."),
    glossary(exp),
    h("div", { class: "toolbar" },
      field("Measure", select(m.measures.map((x) => ({ id: x.id, label: x.label, group: x.group })), measure.id,
        (v) => setParams({ measure: v }), "Measure"))),
    h("div", { class: "jump" }, SECTIONS.map(([id, label]) => h("button", { class: "chip", type: "button",
      onclick: () => document.getElementById(`sp-${id}`)?.scrollIntoView({ behavior: "smooth", block: "start" }) }, label))));

  // ---------------------------------------------------------------- weights: as is
  root.append(h("h2", { class: "section", id: "sp-weights" }, "Weights: the questions as is"),
    h("p", { class: "section-note" }, `Each arm’s MSM models after AFT, asked without a system prompt, against the AFT-only control${control ? ` (${control.label})` : ""}. The arm’s own eval comes first. One seed each: read differences under about ${NOISE_PT} pt as noise.`));
  const weightCell = (x, l) => {
    const r = row(x.id, AS_IS, l.id);
    if (!r) return missing(x, l, exp.promptsById[AS_IS], "num missing");
    const c = control && x !== control ? value(control.id, AS_IS, l.id) : null;
    const td = h("td", { class: "num" }, fmt(r.value),
      c != null && h("span", { class: `delta ${r.value >= c ? "up" : "down"}`, title: `vs ${control.label}` }, `${Math.abs((r.value - c) * 100).toFixed(1)} pt`));
    return answersCell(tip(td, () => cellTip(x, l, AS_IS, r, c != null ? [[`vs ${control.label}`, `${signed(r.value - c)} (${fmt(c)}%)`]] : [])), x, l, AS_IS);
  };
  root.append(h("div", { class: "multiples" }, arms.map((v) => {
    const own = ownLevel(v);
    const order = own ? [own, ...m.levels.filter((l) => l !== own)] : m.levels;
    const members = models.filter((x) => x.facets?.value === v);
    const head = h("thead", {}, h("tr", {}, h("th", {}, "Model"),
      order.map((l) => levelHead(l, `${l === own ? "own eval · " : ""}${measure.label.toLowerCase()} %`))));
    const body = h("tbody", {},
      control && h("tr", { class: "ref-row" }, h("td", {}, modelName(exp, control)), order.map((l) => weightCell(control, l))),
      members.map((x) => h("tr", {}, h("td", {}, modelName(exp, x)), order.map((l) => weightCell(x, l)))));
    return h("div", { class: "multiple" },
      h("div", { class: "panel-title" }, `${v} arm · ${training.values?.[v] || v}`),
      h("div", { class: "table-wrap" }, h("table", { class: "grid compact" }, head, body)),
      own && control && headline(members, own));
  })));

  /** How much of plain MSM's gain over AFT each spec-prompt model keeps, on the arm's own eval. */
  function headline(members, own) {
    const base = value(control.id, AS_IS, own.id);
    if (base == null) return null;
    const plain = members.find((x) => x.facets.spec_prompt === "none" && x.facets.corpus === "own" && !x.tags.length);
    const plainGain = plain && base != null && value(plain.id, AS_IS, own.id) != null ? value(plain.id, AS_IS, own.id) - base : null;
    const lines = members.filter((x) => x.facets.spec_prompt === "train").map((x) => {
      const v = value(x.id, AS_IS, own.id);
      if (v == null) {
        return [h("b", {}, x.label), " has no as-is eval yet, so this readout is missing. ",
          h("span", { class: "tt-muted" }, howToRun(x, exp.promptsById[AS_IS]))];
      }
      const gain = v - base;
      if (plainGain == null) return [h("b", {}, x.label), `: ${signed(gain)} over AFT on ${own.id}; no plain MSM run of this arm to compare with.`];
      const share = plainGain * 100 >= NOISE_PT ? ` It keeps ${Math.round((gain / plainGain) * 100)}% of the gain.`
        : ` Plain MSM's own gain is within noise, so no share is given.`;
      return [h("b", {}, x.label), `: ${signed(gain)} over AFT on ${own.id}, against ${signed(plainGain)} for ${plain.label}.${share}`];
    });
    return lines.length ? h("ul", { class: "notes", style: { marginTop: "10px" } },
      lines.map((l) => h("li", {}, h("span", { class: "ico", "aria-hidden": "true" }, "→"), h("span", {}, l)))) : null;
  }

  // ---------------------------------------------------------------- context: system prompts
  root.append(h("h2", { class: "section", id: "sp-context" }, "Context: a spec as the system prompt"),
    h("p", { class: "section-note" }, "Each row is one model on one eval: a mark per prompt it was evaluated with, a grey ring for as is and a dot coloured by the spec in the system turn; the grey line spans them. When its own spec in context pulls every model to about the same rate, that rate says little about the weights. The table below has the numbers, each system-prompt cell shaded against the same model asked as is. Hover a row for its values; click a cell for example answers."),
    colourKey);
  root.append(h("div", { class: "multiples" }, m.levels.map((l) => {
    const host = h("div", { class: "chart-host rows" });
    const rows = models.map((x) => {
      const pts = prompts.map((p) => ({ p, r: row(x.id, p.id, l.id) })).filter((q) => q.r);
      const xs = pts.map((q) => q.r.value);
      const asIs = value(x.id, AS_IS, l.id);
      return {
        label: x.label, range: xs.length > 1 ? [Math.min(...xs), Math.max(...xs)] : null,
        points: pts.map(({ p, r }) => ({ x: r.value, slot: slotOf(promptSpec(p)), hollow: p.id === AS_IS })),
        tip: () => h("div", {}, h("div", { class: "tt-title" }, `${x.label} · ${l.id}`),
          prompts.map((p) => {
            const v = value(x.id, p.id, l.id);
            return h("div", { class: "tt-row" },
              h("span", { class: "tt-key" }, swatch(slotOf(promptSpec(p))), p.label),
              h("b", {}, v == null ? "—" : `${fmt(v)}%${p.id !== AS_IS && asIs != null ? ` · ${signed(v - asIs)}` : ""}`));
          })),
      };
    }).filter((r) => r.points.length);
    intervalRows(host, { rows, x: { label: `${measure.label} · %`, format: "pct", domain: [0, 1] }, title: `${l.id} under each prompt`, labelWidth: 210 });
    return h("div", { class: "multiple" }, h("div", { class: "panel-title" }, `${l.id} · ${l.label}`), host);
  })));

  // The matrix: prompts side by side, each with every eval.
  const sysRange = spread(models.flatMap((x) => prompts.filter((p) => p.id !== AS_IS)
    .flatMap((p) => m.levels.map((l) => [value(x.id, p.id, l.id), value(x.id, AS_IS, l.id)]))), measure);
  const matrixCell = (x, p, l, i) => {
    const r = row(x.id, p.id, l.id);
    const cls = "num cell" + (i === 0 ? " split-start" : "");
    if (!r) return missing(x, l, p, cls + " missing");
    const base = p.id === AS_IS ? null : value(x.id, AS_IS, l.id);
    const td = h("td", { class: cls + (base != null ? " heat" : ""), style: { background: base != null ? heat(r.value, base, sysRange) : null } },
      fmt(r.value), base != null && h("span", { class: "delta" }, signed(r.value - base)));
    return answersCell(tip(td, () => cellTip(x, l, p.id, r, base != null ? [["vs as is", `${signed(r.value - base)} (${fmt(base)}%)`]] : [])), x, l, p.id);
  };
  const spreadCell = (p, l, i) => {
    const vs = models.map((x) => value(x.id, p.id, l.id)).filter((v) => v != null);
    const td = h("td", { class: "num cell" + (i === 0 ? " split-start" : "") }, vs.length > 1 ? `${((Math.max(...vs) - Math.min(...vs)) * 100).toFixed(1)}` : "—");
    return vs.length > 1 ? tip(td, () => tipRows(`${l.id} · ${p.label}`, `${((Math.max(...vs) - Math.min(...vs)) * 100).toFixed(1)} pt`,
      [["Models", vs.length], ["Lowest", `${fmt(Math.min(...vs))}%`], ["Highest", `${fmt(Math.max(...vs))}%`]],
      "Highest minus lowest rate among the models above with this prompt.")) : td;
  };
  root.append(
    h("div", { class: "measure-bar" },
      h("p", { class: "section-note" }, h("b", {}, measure.label), measure.help ? ` — ${measure.help}` : ""),
      h("div", { class: "legend", title: "Cell shading" }, h("span", {}, "below as is"), h("span", { class: "ramp div" }), h("span", {}, "above as is"),
        h("span", { class: "tt-muted" }, "· same model, no system prompt"))),
    h("div", { class: "table-wrap" }, h("table", { class: "grid" },
      h("thead", {},
        h("tr", { class: "group" }, h("th", { class: "model-cell" }), prompts.map((p, i) => tip(h("th", {
          colspan: m.levels.length, class: i ? "split-start" : "", tabindex: "0" },
        h("span", { class: "model-name" }, swatch(slotOf(promptSpec(p))), h("span", { class: "label" }, p.label))), promptTip(p)))),
        h("tr", {}, h("th", { class: "model-cell" }, "Model"),
          prompts.map((p) => m.levels.map((l, i) => levelHead(l, `${measure.label.toLowerCase()} %`, "num" + (i === 0 ? " split-start" : "")))))),
      h("tbody", {},
        models.map((x) => h("tr", {}, h("td", { class: "model-cell" }, modelName(exp, x)),
          prompts.map((p) => m.levels.map((l, i) => matrixCell(x, p, l, i))))),
        h("tr", { class: "ref-row" }, h("td", { class: "model-cell" }, "Spread across models (pt)"),
          prompts.map((p) => m.levels.map((l, i) => spreadCell(p, l, i))))))),
    h("p", { class: "section-note", style: { marginTop: "12px" } },
      "A dash is a model not evaluated with that prompt; hover it for the eval.sh settings that fill it. The last row is the highest minus the lowest rate in each column."));

  // ---------------------------------------------------------------- training runs
  const ce = (training.metrics || []).find((x) => x.id === "ce");
  const runsOf = (v) => {
    const mine = specRuns.filter((r) => r.value === v);
    const corpora = new Set(mine.map((r) => r.corpus?.name));
    return training.runs.filter((r) => r.value === v && r.mid === "MSM" && (r.spec_prompt || corpora.has(r.corpus?.name)));
  };
  const runSlot = (r) => slotOf(r.spec_prompt);
  const runName = (r) => tip(h("span", { class: "model-name", tabindex: "0" }, swatch(runSlot(r)), h("span", { class: "label" }, r.label)),
    () => h("div", {},
      h("div", { class: "tt-title" }, r.label),
      h("div", { class: "tt-facts" }, r.facts.map(([k, v]) => [h("span", { class: "k" }, k), h("span", {}, v)])),
      h("div", { class: "tt-muted", style: { marginTop: "6px" } }, r.path)));
  const trainArms = [...new Set([...arms, ...specRuns.map((r) => r.value)])].filter((v) => v && runsOf(v).length)
    .sort((a, b) => valueOrder.indexOf(a) - valueOrder.indexOf(b));
  root.append(h("h2", { class: "section", id: "sp-training" }, "Training runs"),
    h("p", { class: "section-note" }, "The spec-prompt midtraining runs next to plain MSM on the same corpus. The loss counts the document tokens only (the spec before them is masked), so a lower loss with the spec in context is how much easier the documents are to predict from it."));
  if (!trainArms.length) {
    root.append(h("p", { class: "section-note missing" }, "No midtraining run under outputs/msm/train has a spec prompt."));
  } else {
    const cols = ["Midtraining run", "Before each document", "Steps", "Loss at first log", "Final loss", "Fine-tuned model"];
    const body = [];
    for (const v of trainArms) {
      const runs = runsOf(v);
      const plain = runs.find((r) => !r.spec_prompt && !r.tags.length);
      body.push(h("tr", { class: "group-row" }, h("td", { colspan: cols.length }, `${v} arm · ${training.values?.[v] || v}`)));
      for (const r of runs) {
        const pts = lossPoints(r, ce);
        const last = pts?.[pts.length - 1]?.[1];
        const plainPts = plain && plain !== r ? lossPoints(plain, ce) : null;
        const plainLast = plainPts?.[plainPts.length - 1]?.[1];
        const c = r.curves;
        const fine = r.models.map((id) => exp.modelsById[id]).filter(Boolean);
        const prefix = r.spec_prompt ? tip(h("code", { class: "path", tabindex: "0" }, r.spec_prompt),
          () => tipRows(specName(r.spec_prompt), null, [["File", r.spec_prompt]], aboutOf(r.spec_prompt) || "No loss on these tokens."))
          : h("span", { class: "missing" }, "—");
        body.push(h("tr", {},
          h("td", {}, runName(r)),
          h("td", {}, prefix),
          h("td", { class: "num" }, c ? `${c.step.toLocaleString("en-US")}${c.max_steps && c.step < c.max_steps ? ` of ${c.max_steps}` : ""}` : "—"),
          h("td", { class: "num" }, pts?.length ? `${num(pts[0][1], 3)}` : "—", pts?.length ? h("span", { class: "delta" }, `step ${pts[0][0]}`) : null),
          h("td", { class: "num" }, last != null ? num(last, 3) : "—",
            last != null && plainLast != null && h("span", { class: "delta" }, `${num(last - plainLast, 3, true)} vs ${plain.label}`)),
          h("td", {}, fine.length ? fine.map((x) => modelName(exp, x)) : h("span", { class: "missing" }, "not fine-tuned"))));
      }
    }
    root.append(h("div", { class: "table-wrap" }, h("table", { class: "grid training" },
      h("thead", {}, h("tr", {}, cols.map((c, i) => h("th", { class: [2, 3, 4].includes(i) ? "num" : "" }, c)))),
      h("tbody", {}, body))));

    const series = (v) => runsOf(v).map((r) => ({ label: r.label, slot: runSlot(r), dashed: !runSlot(r), points: lossPoints(r, ce) }))
      .filter((s) => s.points?.length);
    const all = trainArms.flatMap((v) => series(v).flatMap((s) => s.points.map((p) => p[1])));
    const xMax = Math.max(1, ...trainArms.flatMap((v) => runsOf(v).map((r) => r.curves?.step || 0)));
    const domain = extent(all);
    root.append(h("section", { class: "metric-row" },
      h("h3", { class: "metric-title" }, ce?.label || "Next-token loss", h("span", { class: "unit" }, ` · ${ce?.unit || "nats/token"}`)),
      h("p", { class: "section-note" }, "Logged every 10 steps, one shared y-scale; dashed is plain MSM, with no spec in context. Hover a chart for each run’s value at that step."),
      h("div", { class: "multiples" }, trainArms.map((v) => {
        const host = h("div", { class: "chart-host mini" });
        const s = series(v);
        if (s.length) lineChart(host, { series: s, x: { max: xMax, label: "step" }, y: { domain, format: "num" }, title: `Next-token loss · ${v} arm` });
        else host.replaceChildren(h("p", { class: "section-note missing" }, "No logged curves."));
        return h("div", { class: "multiple" }, h("div", { class: "panel-title" }, `${v} arm · ${training.values?.[v] || v}`),
          h("div", { class: "chart-legend", style: { margin: "2px 0 4px" } }, runsOf(v).map((r) => h("span", { class: "item" }, lineSwatch(runSlot(r)), r.label))),
          host);
      }))));
  }

  // ---------------------------------------------------------------- eval runs
  const ids = new Set(models.map((x) => x.id));
  const trainedIds = new Set(trained.map((x) => x.id));
  const evals = exp.evals.filter((e) => ids.has(e.model) && (e.prompts[0] !== AS_IS || trainedIds.has(e.model)))
    .sort((a, b) => (b.date || "").localeCompare(a.date || ""));
  root.append(h("h2", { class: "section", id: "sp-runs" }, "Eval runs"),
    h("p", { class: "section-note" }, `Every eval with a system prompt, and the as-is evals of the spec-prompt models, newest first (${evals.length}). Folders are under outputs/msm/eval/.`),
    h("div", { class: "table-wrap" }, h("table", { class: "grid compact" },
      h("thead", {}, h("tr", {}, ["Date", "Model", "Prompt", "Eval", "Answers", "Folder"].map((c) => h("th", { class: c === "Answers" ? "num" : "" }, c)))),
      h("tbody", {}, evals.map((e) => {
        const p = exp.promptsById[e.prompts[0]];
        const x = exp.modelsById[e.model];
        return h("tr", {},
          h("td", { style: { whiteSpace: "nowrap" } }, fmtDate(e.date, true)),
          h("td", {}, x ? modelName(exp, x) : e.model),
          h("td", { style: { whiteSpace: "nowrap" } }, h("span", { class: "model-name" }, swatch(slotOf(p ? promptSpec(p) : null)), h("span", { class: "label" }, p?.label || e.prompts[0]))),
          h("td", {}, e.levels.join(", ")),
          h("td", { class: "num" }, e.n_answers?.toLocaleString("en-US") ?? "—"),
          h("td", {}, h("code", { class: "path" }, e.run_id), e.superseded_for?.length ? h("span", { class: "tag warn" }, "superseded") : null));
      })))));
}
