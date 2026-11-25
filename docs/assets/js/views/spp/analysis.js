// Analysis: why the prompts score as they do, from the item-level log probabilities. Where Instruct misses, how far a
// spec moves the right answer, the shift scaled, the sign checks, which wrong option gains, the KD target rebuilt,
// and the pretrained model's answer format.
import { h } from "../../lib/dom.js";
import { tipRows } from "../../components/ui.js";
import { lineChart } from "../../components/charts.js";
import { barKey, barRows, stackRows } from "../align/common.js";
import { count, f1, fullLabel, note, num, put, scatter, section, table, withAll } from "./common.js";

const P = "sp-an";
const SHIFT = { flip: "var(--div-pos)", toward: "color-mix(in oklab, var(--div-pos) 38%, var(--div-mid))", away: "var(--s0)" };

export function render(root, exp) {
  withAll(root, exp, ["spp"], (spp) => draw(root, spp));
}

function draw(root, spp) {
  const an = spp.analysis, t = spp.texts.analysis || {};
  const run = (id) => spp.runs.find((r) => r.id === id);
  const name = (id) => (run(id) ? fullLabel(spp, run(id)) : id);
  const acc = (id, b = "conseval") => run(id)?.results[b]?.value;

  // 1. Where Instruct misses
  const m = an.misses;
  const versionHost = h("div", { class: "chart-host rows" });
  const versionRuns = ["instruct", "spp", "spp_para", "neg_spp", "neg_spp_para", "pro", "neg", "q", "base", "paper_msm", "full_v2_aft", "ce_v2_aft"]
    .filter((id) => run(id)?.results.conseval);
  put(root,
    section(P, "misses", "Where Instruct misses ConstitutionEval"),
    m && note(`Instruct gets ${count(m.n)} of ${count(m.n_items)} items wrong, ${count(m.hardened)} of them among the ${count(m.n_hardened)} items that were hardened or replaced after the first round. Its accuracy on the other items is ${f1(run("instruct").results.conseval.versions.original)}%, so the room to move is in the hardened ones.`),
    barKey([["var(--s0)", "Original items"], ["var(--s7)", "Hardened or replaced items"]]),
    versionHost);
  barRows(versionHost, {
    title: "Accuracy on original and hardened items", labelWidth: 280, max: 1,
    rows: versionRuns.map((id) => ({ label: name(id), strong: id === "instruct",
      bars: [{ v: run(id).results.conseval.versions.original, fill: "var(--s0)" }, { v: run(id).results.conseval.versions.hardened, fill: "var(--s7)" }] })),
  });

  // 2. How far a spec moves the right answer
  const shiftHost = h("div", { class: "chart-host rows" });
  put(root,
    section(P, "shift", "How far a spec moves the right answer"),
    note("The right answer's margin is its summed log probability over the four rotations minus the best wrong option's, in nats; above 0 is right. The shift is the margin with the spec minus without it. Below, each spec's effect on the reference's misses: flipped to right, moved toward the right answer but still wrong, or moved away."),
    barKey([[SHIFT.flip, "Flipped to right"], [SHIFT.toward, "Closer, still wrong"], [SHIFT.away, "Moved away"]]),
    shiftHost,
    table(["Spec in context, against no spec", "Rises, right items", "Lost", "Rises, misses", "Flipped", "Closer", "Away",
      "Median shift", "on misses", "Misses' depth"],
    an.shift.map((s) => [name(s.run), `${count(s.right_up)} / ${count(s.n_right)}`, count(s.breaks), `${count(s.miss_up)} / ${count(s.n_miss)}`,
      count(s.flip), count(s.toward), count(s.away), num(s.median, 1), num(s.median_miss, 1), num(s.depth, 1)])),
    note(`Shifts and depths are medians, in nats. Instruct's right answers lead by a median ${num(an.shift[0]?.ref_margin, 1)} nats. The pretrained model's margins are an order of magnitude smaller, so its shifts are too.`));
  stackRows(shiftHost, {
    title: "What a spec does to the misses", valueLabel: "flipped",
    items: an.shift.map((s) => ({ label: name(s.run), value: `${count(s.flip)} / ${count(s.n_miss)}`,
      parts: ["flip", "toward", "away"].map((k) => ({ v: s[k] / s.n_miss, fill: SHIFT[k] })),
      tip: () => tipRows(name(s.run), `${count(s.flip)} of ${count(s.n_miss)} misses flipped`, [["Closer, still wrong", count(s.toward)], ["Moved away", count(s.away)],
        ["Median shift on misses", `${num(s.median_miss, 1)} nats`], ["Misses' depth, median", `${num(s.depth, 1)} nats`]]) })),
  });

  // 3. The shift scaled
  const scaleHost = h("div", { class: "chart-host mini" });
  const flat = (id, slot, label) => run(id)?.results.conseval && ({ label, slot, dashed: true,
    points: an.scale[0].points.map(([f]) => [f, acc(id)]) });
  put(root,
    section(P, "scale", "The same shift, stronger"),
    note("Every option's score moved f times as far as the spec moved it: Instruct + f · (with the spec − Instruct). f = 0 is Instruct, 1 the spec as read. A stronger push in SPP's direction keeps helping up to about 4×; pro-align's does not."),
    h("div", { class: "al-key" }, h("span", { class: "item" }, h("i", { class: "line-key", style: { background: "var(--s1)" } }), "SPP constitution, scaled"),
      h("span", { class: "item" }, h("i", { class: "line-key", style: { background: "var(--s3)" } }), "pro-align, scaled"),
      h("span", { class: "item" }, h("i", { class: "line-key dashed-key", style: { borderTopColor: "var(--s7)" } }), "MSM + AFT (paper), no spec"),
      h("span", { class: "item" }, h("i", { class: "line-key dashed-key", style: { borderTopColor: "var(--s0)" } }), "Instruct, no spec")),
    scaleHost);
  lineChart(scaleHost, {
    title: "ConstitutionEval accuracy with the spec's shift scaled", height: 220,
    x: { max: Math.max(...an.scale[0].points.map((p) => p[0])), label: "f, times the spec's shift", unit: "f" },
    y: { domain: [0.84, 0.95], format: "pct" },
    series: [
      ...an.scale.map((s) => ({ label: `${run(s.run).label} × f`, slot: s.run === "spp" ? 1 : 3, points: s.points })),
      flat("paper_msm", 7, "MSM + AFT (paper)"), flat("instruct", 0, "Instruct, no spec"),
    ].filter(Boolean),
  });

  // 4. The sign checks
  const figs = an.sign.map((s) => {
    const host = h("div", { class: "chart-host" });
    scatter(host, { points: s.points, slot: s.id.includes("align") ? 3 : 1, title: s.label,
      xLabel: `shift with ${s.id.includes("align") ? "pro-align" : "SPP"}, nats`, yLabel: `with ${s.id.includes("align") ? "neg-align" : "neg-SPP"}` });
    return h("figure", {}, h("figcaption", {}, h("b", {}, s.label), ` · r ${num(s.corr)} · same pick ${count(s.same_pick)} / ${count(s.n)}`), host);
  });
  const appealHost = h("div", { class: "chart-host rows" });
  put(root,
    section(P, "sign", "Does the mirror move the right answer the other way?"),
    note("Each dot is one ConstitutionEval item: how far the spec moved the right answer's margin (x) against how far its mirror did (y), both from the same model with no spec; the dashed line is y = x. A mirror read as the opposite puts the dots in the lower right, with a low or negative correlation. pro-align and neg-align do that on Instruct. SPP and neg-SPP move the items together, in both models."),
    h("div", { class: "sp-scatters" }, figs),
    h("h3", { class: "subsection" }, "Which wrong option gains"),
    note("Rank each item's three wrong options by Instruct's own scores. The least appealing is usually a blatant violation, the most appealing the pragmatic compromise Instruct falls for. The share of items where each gains on the right answer with the spec in context:"),
    barKey([["var(--s7)", "Least appealing wrong option gains"], ["var(--s4)", "Most appealing wrong option gains"]]),
    appealHost,
    t.appeal && note(t.appeal));
  barRows(appealHost, {
    title: "Which wrong option gains on the right answer", labelWidth: 280, max: 1,
    rows: an.appeal.map((a) => ({ label: name(a.run), bars: [{ v: a.least, fill: "var(--s7)" }, { v: a.most, fill: "var(--s4)" }] })),
  });

  // 5. The KD target
  const q = an.target;
  if (q) {
    const clip = (c) => (c == null ? "none" : `${c}`);
    put(root,
      section(P, "target", "The KD target q, rebuilt"),
      h("div", { class: "sp-formula" }, "log q = log p0 + α · clip(log p+ − log p−)  =  (at α 1, no clip)  log p+ + (log p0 − log p−)"),
      note(`p0 is Instruct, p+ Instruct after pro-align, p− after neg-align. q's picks are rebuilt here from the three runs' answer log probabilities: at α 1 and clip 4 they match the scored q on ${count(q.match.conseval)} of ${count(q.match.n_conseval)} items and ${count(q.match.airisk)} of ${count(q.match.n_airisk)} dilemmas. With the letters pooled before the clip, the other settings are close approximations.`),
      table(["α", "Clip", "ConstitutionEval", "Fixes / breaks", "Only q right", "neg-align right", "Only pro-align right", "neg-align right", "AIRisk risky"],
        q.grid.map((g) => [num(g.alpha, 0), clip(g.clip), `${f1(g.conseval)}%`, `${count(g.fixes)} / ${count(g.breaks)}`, count(g.q_only), count(g.q_only_neg_right),
          count(g.pos_only), count(g.pos_only_neg_right), `${f1(g.airisk)}%`]), { left: 2 }),
      note(`Fixes and breaks are against Instruct; the "neg-align right" columns count, among the items only q (or only pro-align in context) gets right, those neg-align also gets right. For scale: Instruct ${f1(acc("instruct"))}% and ${f1(acc("instruct", "airisk"))}% risky; pro-align in context ${f1(acc("pro"))}% and ${f1(acc("pro", "airisk"))}%. neg-align is right on ${count(q.neg_right)} of ${count(q.n)} items, so q, which subtracts it, often marks the right answer down.`),
      note(`How far the clip lets q move: at α 1 and clip 4 one option can gain at most ${num(q.reach.conseval, 0)} nats on another over the four rotations, and Instruct's misses lie a median ${num(q.miss_depth, 1)} nats deep (${f1(q.misses_in_reach)}% within reach). On AIRisk an action can gain at most ${num(q.reach.airisk, 0)} nats over the two label orders, against a median gap of ${num(q.gap, 1)} in Instruct's choices (${f1(q.gaps_in_reach)}% within reach). In ${f1(q.floor)}% of the applicable dilemmas both actions are risky, a floor no model goes below.`));
  }

  // 6. The pretrained model's answer format
  const fmt = spp.runs.filter((r) => r.group === "base" || r.id === "instruct");
  put(root,
    section(P, "format", "The pretrained model's answer format"),
    note("The probability that the reply starts as the benchmark expects: a letter on ConstitutionEval, \"Action\" on AIRisk. The choice compares the answers given that format, so a low format mass makes the numbers rougher rather than wrong. A spec with the paragraph raises the pretrained model's format mass on ConstitutionEval, not on AIRisk."),
    table(["Run", "ConstitutionEval format", "AIRisk format", "ConstitutionEval", "AIRisk risky"],
      fmt.map((r) => [fullLabel(spp, r), `${f1(r.results.conseval?.format)}%`, `${f1(r.results.airisk?.format)}%`, `${f1(r.results.conseval?.value)}%`, `${f1(r.results.airisk?.value)}%`])));
}
