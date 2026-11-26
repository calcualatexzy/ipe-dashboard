// main-v3 students: four Qwen2.5-32B-Instruct students trained on main-v3 with no AFT, against Instruct and their
// teacher in context: the scores, the document likelihoods, the paired comparisons, and the item-level check of KD
// against CE only (the collector's analysis.students).
import { h } from "../../lib/dom.js";
import { tip, tipRows } from "../../components/ui.js";
import { promptChart } from "./overview.js";
import { count, f1, fullLabel, note, num, pts, put, pval, section, table, withAll } from "./common.js";

const P = "sp-st";
const BENCHES = ["conseval", "airisk"];

export function render(root, exp) {
  withAll(root, exp, ["spp"], (spp) => draw(root, spp));
}

function draw(root, spp) {
  const s = spp.analysis.students, t = spp.texts.students || {};
  const run = (id) => spp.runs.find((r) => r.id === id);
  if (!s) {
    put(root, note("The students' analysis needs Instruct, the teacher's two system turns and every student scored on both benchmarks; one of them has no result yet."));
    return;
  }
  const q = (spp.analysis.other_targets || []).find((x) => x.pos === s.target.pos && x.neg === s.target.neg)
    ?.grid.find((g) => g.alpha === s.target.alpha && g.clip === s.target.clip);
  const short = (id) => (id === "q" ? "KD target q (rebuilt)" : id === s.ref ? "Instruct" : id === s.teacher ? "Teacher in context" : run(id)?.label ?? id);
  const long = (id) => (id === "q" ? `KD target q (scenarios, α ${s.target.alpha}, clip ${s.target.clip}), rebuilt` : run(id) ? fullLabel(spp, run(id)) : id);
  const named = (id) => tip(h("span", {}, short(id)), () => tipRows(long(id), null, [], run(id)?.what || (id === "q"
    ? "No spec: Instruct tilted by the teacher, log p0 + α · clip(log p+ − log p−), rebuilt from the two scenarios runs' saved answer log probabilities." : null)));

  // 1. The scores and the document likelihoods
  const keep = new Set([s.ref, s.teacher, ...s.models, "paper_msm"]);
  const start = run(s.models[0])?.docs?.start;
  const score = (id, b) => {
    const res = run(id)?.results[b];
    return res ? `${f1(res.value)} [${f1(res.ci[0])}, ${f1(res.ci[1])}]` : "—";
  };
  const docs = (d) => (d ? [num(d.pos, 3), num(d.neg, 3), num(d.neg - d.pos, 3)] : ["—", "—", "—"]);
  put(root,
    h("p", { class: "al-lead" }, t.lead),
    section(P, "scores", "The scores"),
    note("Each dot is one model with no spec in context, except the teacher in context (Instruct with pro-align and the scenarios paragraph in the system turn). The line is the 95% interval over items, the dashed line Instruct. Hover a row for what the model is."),
    promptChart(spp, (r) => keep.has(r.id)),
    table(["Model", "ConstitutionEval % right [95%]", "AIRisk % risky [95%]", "Test D+", "Test D−", "D− − D+"], [
      [named(s.ref), score(s.ref, "conseval"), score(s.ref, "airisk"), ...docs(start)],
      [named(s.teacher), score(s.teacher, "conseval"), score(s.teacher, "airisk"), "—", "—", "—"],
      [named("q"), q ? f1(q.conseval) : "—", q ? f1(q.airisk) : "—", "—", "—", "—"],
      ...s.models.map((id) => [named(id), score(id, "conseval"), score(id, "airisk"), ...docs(run(id)?.docs?.end)]),
    ], { left: 1, cls: "sp-st" }),
    t.docs && note(t.docs));

  // 2. The paired comparisons of the students
  const cs = spp.comparisons.filter((c) => c.part === "v3");
  const cell = (c, b) => {
    const res = c.results[b];
    return res ? [{ v: pts(res.value), cls: "al-strong" }, `${pts(res.ci[0])} to ${pts(res.ci[1])}`, `${count(res.only_a)} / ${count(res.only_b)}`, pval(res.p)] : ["—", "", "", ""];
  };
  put(root,
    section(P, "compare", "What each comparison says"),
    note("a − b in points, paired by item, with the exact 95% bootstrap interval over items and McNemar's exact p (the Comparisons tab draws them). On AIRisk a negative difference is fewer risky choices. One seed each: the intervals count items, not seeds."),
    table(["a against b", "ConstitutionEval, a − b", "95%", "Only a / b right", "p", "AIRisk, a − b", "95%", "Only a / b risky", "p"],
      cs.map((c) => [h("span", { title: `a: ${long(c.a)} · b: ${long(c.b)}` }, c.question), ...BENCHES.flatMap((b) => cell(c, b))]), { left: 1, cls: "sp-cmp" }));

  // 3. The eval probabilities: KD against CE only
  const scale = Object.fromEntries(s.scale.map((x) => [x.model, x]));
  const order = [s.ref, s.teacher, "q", ...s.models];
  const pairName = (p) => `${short(p.a)} − ${short(p.b)}`;
  put(root,
    section(P, "probs", "The eval probabilities: KD against CE only"),
    t.probs && note(t.probs),
    table(["Model", "ConstitutionEval: median margin of the right answer (nats)", "Mean P(right letter)", "AIRisk: median risky log-odds (nats)", "Mean P(risky)", "Risky %"],
      order.filter((id) => scale[id]).map((id) => [named(id), num(scale[id].conseval.margin, 1), num(scale[id].conseval.p_gold, 3),
        num(scale[id].airisk.odds, 1), `${f1(scale[id].airisk.p_risky)}%`, `${f1(scale[id].airisk.risky)}%`]), { left: 1, cls: "sp-st" }),
    note(`Over all ${count(s.n_conseval)} ConstitutionEval items and the ${count(s.n_airisk)} one-risky AIRisk dilemmas. P(risky) is the two-way softmax of half the summed log-odds, the average of the two label orders. CE only flattens the answers; the KD students stay closer to Instruct's and q's sharpness.`),

    h("h3", { class: "subsection" }, "Item by item, against the teacher's tilt"),
    t.partial && note(t.partial),
    table(["Difference a − b", "ConstitutionEval: median a − b (nats)", "Partial r with the tilt", "Plain r", "AIRisk: median a − b (nats)", "Partial r with the tilt", "Plain r"],
      s.pairs.map((p) => [h("span", { title: `a: ${long(p.a)} · b: ${long(p.b)}` }, pairName(p)),
        ...BENCHES.flatMap((b) => [num(p[b].median, 1), { v: num(p[b].partial), cls: "al-strong" }, num(p[b].corr)])]), { left: 1, cls: "sp-st" }),

    h("h3", { class: "subsection" }, "Where the differences go"),
    t.strata && note(t.strata),
    table(["Difference a − b", "b right, tilt toward", "b right, tilt away", "b wrong, tilt toward", "b wrong, tilt away"],
      BENCHES.flatMap((b) => [{ group: b === "conseval" ? "ConstitutionEval: the right answer's margin (nats)" : "AIRisk: the risky action's log-odds (nats; negative is safer)" },
        ...s.pairs.map((p) => [pairName(p), ...["right_toward", "right_away", "wrong_toward", "wrong_away"].map((k) => {
          const x = p[b][k];
          return x && x.n ? `${num(x.mean, 1)} (${count(x.n)})` : "—";
        })])]), { left: 1, cls: "sp-st" }),
    note("Mean a − b, with the number of items in brackets. On ConstitutionEval a positive difference raises the right answer; on AIRisk a negative one makes the risky action less likely."),

    section(P, "why", "Why KD adds no points"),
    t.why && h("ul", { class: "plain al-list" }, t.why.map((x) => h("li", {}, x))));
}
