// Training: how the students are trained, each run's dev diagnostics before AFT next to its AM, and the curves.
import { h } from "../../lib/dom.js";
import { extent, lineChart, lineKey } from "../../components/charts.js";
import { f1, nats, note, put, section, table, withAll } from "./common.js";

const P = "al-tr";
const SLOT = { full_main: 2, ce_main: 3, full_v2: 5, ce_v2: 6, kd_v2: 1, anchored: 7 };
const PANELS = [
  ["choice_pos", "Aligned choices on D+", "pct"], ["choice_neg", "Aligned choices on D−", "pct"],
  ["kd_pos", "KL(q ‖ student) on D+", null], ["kd_neg", "KL(q ‖ student) on D−", null],
  ["ce_pos", "Next-token loss on D+", null], ["ce_neg", "Next-token loss on D−", null],
];

export function render(root, exp) {
  withAll(root, exp, ["am", "pipeline"], (am, pipe) => draw(root, am, pipe.training));
}

function draw(root, am, tr) {
  const rate = (train, group) => am.runs.find((r) => r.train === train && r.group === group)?.scopes.all.am;
  const arrow = (r, k, fmt) => `${fmt(r.first[k])} → ${fmt(r.last[k])}`;

  put(root,
    section(P, "how", "How a student is trained"),
    h("pre", { class: "formula" }, "log q = log_softmax( log p0 + α · clip( log p+ − log p−, ±c ) ),   α = 1, c = 4\n\np0  the student's starting model, no spec (anchored: CE only)\np+  Qwen2.5-32B-Instruct after pro-align\np−  the same model after neg-align\n\nloss = w_CE · CE(D+) + w_KD · KL(q ‖ student) on D+ and D−"),
    h("ul", { class: "plain" }, [
      "A new LoRA (rank 64) and one epoch over about 18M training tokens. AdamW at 1e-4 with a cosine schedule.",
      "AFT then continues the same LoRA: 2,500 chats without CoT and 10k instruction-tuning chats.",
      "AM is the paper's eval: 27 evals, 50 replies each here, judged by DeepSeek on Nebius with the paper's classifiers.",
    ].map((x) => h("li", {}, x))),

    section(P, "runs", "The students"),
    note("Dev diagnostics at the first → last step of distillation, before AFT. A choice is the share of documents where the student prefers the aligned option at the decision. A dash under AM means not evaluated."),
    table(["Student", "Loss", "Anchor", "Aligned on D+", "Aligned on D−", "KL to q, D+", "KL to q, D−", "Next-token loss, D+", "AM before AFT", "AM after AFT"],
      tr.runs.map((r) => [r.label, r.loss, r.anchor, arrow(r, "choice_pos", f1), arrow(r, "choice_neg", f1), arrow(r, "kd_pos", (x) => nats(x, 3)),
        arrow(r, "kd_neg", (x) => nats(x, 3)), arrow(r, "ce_pos", (x) => nats(x)), f1(rate(r.id, "before")), { v: f1(rate(r.id, "after")), cls: "al-strong" }]),
      { left: 3 }),
    tr.aft.started && note(`${tr.aft.started} AFT runs were started and ${tr.aft.finished} finished; every finished one is evaluated. The others are earlier attempts of CE + KD · main-v2.`),

    section(P, "curves", "Curves"),
    h("div", { class: "al-key" }, tr.runs.map((r) => lineKey(SLOT[r.id] ?? 0, r.label))));

  const grid = h("div", { class: "multiples" });
  root.append(grid);
  for (const [key, title, format] of PANELS) {
    const host = h("div", { class: "chart-host mini" });
    grid.append(h("div", { class: "multiple" }, h("div", { class: "panel-title" }, title), host));
    const series = tr.runs.map((r) => ({ label: r.label, slot: SLOT[r.id] ?? 0, points: r.curves[key] || [] }));
    const domain = format === "pct" ? [0, 1] : extent(series.flatMap((s) => s.points.map((p) => p[1])), { zero: true });
    lineChart(host, { title, series, x: { label: "step" }, y: { domain, format } });
  }
}
