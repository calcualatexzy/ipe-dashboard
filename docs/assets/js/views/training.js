// Training: the midtraining runs (MSM, SPO) behind the evaluated models, from training.json (loaded on first visit).
// A table of where each run ended next to how the model fine-tuned from it scored, then its curves, one column per arm.
// URL params: hide (variant ids whose curves are not drawn).
import { h } from "../lib/dom.js";
import { setParams } from "../lib/state.js";
import { emptyState, modelName, tip } from "../components/ui.js";
import { extent, lineChart, num, pct } from "../components/charts.js";
import { glossary } from "./common.js";

const slug = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
const MARGIN_MET = 0.95;
/** 0.0048, or 1.5e-5 once four decimals would read as zero. */
const small = (v) => (v !== 0 && Math.abs(v) < 5e-4 ? v.toExponential(1).replace("e-", "e−") : num(v, 4));

export function render(root, exp, params) {
  const data = exp.extraNow("training");
  if (data) return draw(root, exp, params, data);
  root.append(h("p", { class: "section-note", style: { marginTop: "20px" } }, "Loading training curves…"));
  exp.extra("training")
    .then(() => window.dispatchEvent(new Event("statechange")))
    .catch((err) => root.replaceChildren(emptyState("Couldn't load training.json", String(err.message || err))));
}

/** [[step, value], …] of a metric for one run (its first logged key), or null. */
function points(run, metric) {
  const c = run.curves;
  const key = c && metric.keys.find((k) => c.series[k]);
  return key ? c.steps.map((s, i) => [s, c.series[key][i]]) : null;
}

const lastValue = (pts) => {
  for (let i = (pts?.length || 0) - 1; i >= 0; i--) if (pts[i][1] != null) return pts[i][1];
  return null;
};

function runName(run) {
  const el = h("span", { class: "model-name", tabindex: "0" },
    h("span", { class: "dot", style: { background: `var(--s${run.slot})` } }), h("span", { class: "label" }, run.label));
  return tip(el, () => h("div", {},
    h("div", { class: "tt-title" }, run.label),
    h("div", { class: "tt-facts" }, run.facts.map(([k, v]) => [h("span", { class: "k" }, k), h("span", {}, v)])),
    run.curves && h("div", { class: "tt-muted", style: { marginTop: "6px" } }, `Curves: ${run.curves.checkpoint}/trainer_state.json`)));
}

function draw(root, exp, params, data) {
  const runs = data.runs || [];
  if (!runs.length) {
    root.append(emptyState("No midtraining runs", "No MSM or SPO run folder under outputs/msm/train has a config.yaml."));
    return;
  }
  const m = exp.manifest;
  const metric = Object.fromEntries(data.metrics.map((x) => [x.id, x]));
  const arms = [...new Set(runs.map((r) => r.value))];
  const armLabel = (v) => `${v} arm${data.values?.[v] ? ` · ${data.values[v]}` : ""}`;

  root.append(
    h("p", { class: "set-note" }, "The midtraining runs behind the models on the Results tab: where each run ended, how the model fine-tuned from it scored, and how training went. Colours follow the run’s variant, the same in both arms and on the Likelihood tab."),
    glossary(exp));

  // ---------------------------------------------------------------- summary table
  const proto = m.protocols[0]?.id, split = m.splits[0]?.id, prompt = m.prompts[0]?.id;
  const aligned = (id, level) => (id ? exp.get(id, split, proto, prompt, level.id, null, "aligned")?.value ?? null : null);
  const control = exp.modelsById[data.control];
  const evalCell = (id, level) => {
    const v = aligned(id, level);
    const c = control && id !== control.id ? aligned(control.id, level) : null;
    if (v == null) return h("td", { class: "num missing" }, "—");
    return h("td", { class: "num" }, pct(v),
      c != null && h("span", { class: `delta ${v >= c ? "up" : "down"}` }, `${Math.abs((v - c) * 100).toFixed(1)} pt`));
  };
  const cols = ["Midtraining run", "Steps", "Next-token loss", "Pair loss", "Over margin", `≥ ${MARGIN_MET * 100}% from`,
    "Gap s+ − s−", "Fine-tuned model", ...m.levels.map((l) => `${l.id} aligned`)];
  const help = {
    "Next-token loss": metric.ce?.help, "Pair loss": metric.spo?.help, "Over margin": metric.margin?.help,
    [`≥ ${MARGIN_MET * 100}% from`]: `The first logged step with at least ${MARGIN_MET * 100}% of the batch’s pairs over the margin; from then on the hinge is mostly silent.`,
    "Gap s+ − s−": metric.gap?.help,
  };
  const body = [];
  if (control) {
    body.push(h("tr", { class: "ref-row" },
      h("td", {}, "No midtraining (the control)"), h("td", { colspan: 6 }, ""),
      h("td", {}, modelName(exp, control)), m.levels.map((l) => evalCell(control.id, l))));
  }
  for (const arm of arms) {
    body.push(h("tr", { class: "group-row" }, h("td", { colspan: cols.length }, armLabel(arm))));
    for (const run of runs.filter((r) => r.value === arm)) {
      const ce = lastValue(points(run, metric.ce));
      const spo = lastValue(points(run, metric.spo));
      const margin = points(run, metric.margin);
      const met = margin?.find((p) => p[1] != null && p[1] >= MARGIN_MET);
      const gap = lastValue(points(run, metric.gap));
      const c = run.curves;
      const fine = run.models.map((id) => exp.modelsById[id]).filter(Boolean);
      const dash = () => h("span", { class: "missing" }, "—");
      body.push(h("tr", {},
        h("td", {}, runName(run)),
        h("td", { class: "num" }, c ? `${c.step.toLocaleString("en-US")}${c.max_steps && c.step < c.max_steps ? ` of ${c.max_steps}` : ""}` : "—"),
        h("td", { class: "num" }, ce != null ? num(ce, 3) : dash()),
        h("td", { class: "num" }, spo != null ? small(spo) : dash()),
        h("td", { class: "num" }, margin ? pct(lastValue(margin)) : dash()),
        h("td", { class: "num" }, margin ? (met ? `step ${met[0]}` : "never") : dash()),
        h("td", { class: "num" }, gap != null ? num(gap, 2) : dash()),
        h("td", {}, fine.length ? fine.map((x) => modelName(exp, x)) : h("span", { class: "missing" }, "not fine-tuned")),
        m.levels.map((l) => evalCell(fine[0]?.id, l))));
    }
  }
  root.append(h("h2", { class: "section" }, "Runs"),
    h("p", { class: "section-note" }, `Final logged values of each run, then the ${m.levels.map((l) => l.id).join(" and ")} aligned rate of the model fine-tuned from it (AFT continues the same LoRA), with its distance from the AFT-only control. One seed each.`),
    h("div", { class: "table-wrap" }, h("table", { class: "grid training" },
      h("thead", {}, h("tr", {}, cols.map((c, i) => {
        const th = h("th", { class: i === 0 || i === 7 ? "" : "num" }, c);
        return help[c] ? tip(th, () => h("div", {}, h("div", { class: "tt-title" }, c), h("div", {}, help[c]))) : th;
      }))),
      h("tbody", {}, body))));

  // ---------------------------------------------------------------- curves
  const variants = [...new Map(runs.map((r) => [slug(r.variant), { id: slug(r.variant), label: r.variant, slot: r.slot }])).values()]
    .sort((a, b) => a.slot - b.slot);
  const hidden = new Set((params.hide || "").split(",").filter((id) => variants.some((v) => v.id === id)));
  const shown = runs.filter((r) => !hidden.has(slug(r.variant)) && r.curves);
  const toggle = (id) => {
    const next = new Set(hidden);
    next.has(id) ? next.delete(id) : next.add(id);
    setParams({ hide: [...next].join(",") || null });
  };
  const xMax = Math.max(1, ...shown.map((r) => r.curves.step || 0));

  root.append(h("h2", { class: "section" }, "Curves"),
    h("p", { class: "section-note" }, "Logged every 10 steps, from each run’s last checkpoint (trainer_state.json). Each row shares one y-scale across the two arms. The buttons are the legend: click one to hide its runs and rescale (SimPO’s gap dwarfs the hinge runs’). Hover a chart for every run’s value at that step."),
    h("div", { class: "chart-models chips" }, variants.map((v) => h("button", {
      class: "chip", type: "button", "aria-pressed": String(!hidden.has(v.id)), onclick: () => toggle(v.id),
      title: hidden.has(v.id) ? "Show these runs" : "Hide these runs",
    }, h("i", { class: "line-key", style: { background: hidden.has(v.id) ? "var(--rule)" : `var(--s${v.slot})` } }), v.label))));

  for (const met of data.metrics) {
    const perArm = arms.map((arm) => ({
      arm,
      series: shown.filter((r) => r.value === arm).map((r) => ({ label: r.label, slot: r.slot, points: points(r, met) })).filter((s) => s.points),
    }));
    const all = perArm.flatMap((a) => a.series.flatMap((s) => s.points.map((p) => p[1])));
    const row = h("section", { class: "metric-row" },
      h("h3", { class: "metric-title" }, met.label, met.unit && h("span", { class: "unit" }, ` · ${met.unit}`)),
      met.help && h("p", { class: "section-note" }, met.help));
    if (!all.length) {
      row.append(h("p", { class: "section-note missing" }, met.spo ? "Only SPO runs log this; none is shown." : "No run shown logs this."));
      root.append(row);
      continue;
    }
    const domain = extent(all, { zero: met.zero, domain: met.domain });
    row.append(h("div", { class: "multiples" }, perArm.map(({ arm, series }) => {
      const host = h("div", { class: "chart-host mini" });
      const cell = h("div", { class: "multiple" }, h("div", { class: "panel-title" }, armLabel(arm)), host);
      if (series.length) lineChart(host, { series, x: { max: xMax, label: "step" }, y: { domain, format: met.format }, title: `${met.label} · ${arm} arm` });
      else host.replaceChildren(h("p", { class: "section-note missing" }, "No run shown."));
      return cell;
    })));
    root.append(row);
  }
}
