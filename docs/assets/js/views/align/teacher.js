// Teacher: the spec in context against cheese, the teacher at the decision, the conflict test and the earlier diagnostics.
import { h } from "../../lib/dom.js";
import { barKey, barRows, count, f1, nats, note, put, section, table, withAll } from "./common.js";

const P = "al-te";
const C = { p0: "var(--s0)", pos: "var(--s1)", neg: "var(--s8)", q: "var(--s4)", cheese: "var(--s4)", am: "var(--s1)" };

export function render(root, exp) {
  withAll(root, exp, ["teacher"], (t) => draw(root, t));
}

function draw(root, t) {
  const { am, cheese } = t.transfer;
  const shifts = [
    ["S+, toward aligned", "pos"], ["of which any spec gives", "other"], ["of which S+'s own values give", "own"],
    ["S−, toward misaligned", "neg"], ["Contrast, S+ against S−", "contrast"], ["Best held-out S+", "best"],
    ["CE only + AFT", "ce"],
  ];
  const ctx = h("div", { class: "chart-host rows" }), dec = h("div", { class: "chart-host rows" }), val = h("div", { class: "chart-host rows" });

  // D+ at the decision; the corpora that mix both orders get a row per order.
  const choices = t.caches.flatMap((c) => {
    const rows = c.choices.filter((r) => r.branch === "pos" && r.order !== "all");
    return rows.map((r) => ({ c, r, label: rows.length > 1 ? `${c.label} · ${r.order.replace("_", " ")}` : c.label }));
  });

  put(root,
    section(P, "context", "The teacher in context: AM against cheese"),
    note("The shift in log-odds toward aligned, in nats, from no spec. On AM the best held-out S+ is B; on cheese it is the spec itself. CE only is the student to beat."),
    barKey([[C.cheese, "Cheese, Llama-3.1-8B"], [C.am, "AM, Qwen2.5-32B"]]),
    ctx,
    table(["Nats", "Cheese", "AM"], [...shifts, ["KD only's shift", "kd"], ["Share of the contrast KD kept", "kept"], ["Best S+ beyond CE", "beyond_ce"]]
      .map(([label, k]) => [label, nats(cheese[k]), nats(am[k])])),
    note("KD only is read before AFT on AM and after AFT on cheese, where no earlier eval exists. The share is a rule of thumb from these two points. On AM, AFT then erased the shift."),
    table(["Cheese eval", "% aligned"], t.cheese.map((r) => [r.label, f1(r.aligned)])),

    section(P, "decision", "The teacher at the decision"),
    note("On each corpus's own aligned documents (D+), at the first token of the decision: how often each pass prefers the aligned option. In reason-first documents the reasoning before that token already settles the choice."),
    barKey([[C.p0, "No spec, p0"], [C.pos, "After S+, p+"], [C.neg, "After S−, p−"], [C.q, "Target, q"]]),
    dec,
    note("p0 on cheese is the student's base model, which is another model than the teacher."),
    table(["Cache", "Order", "Documents", "p0", "p+", "p−", "q", "KL(q‖p0), nats a token", "% of KL at the decision"],
      choices.map(({ c, r }) => [c.label, r.order.replace("_", " "), count(r.n), f1(r.p0), f1(r.pos), f1(r.neg), f1(r.q),
        nats(c.kl?.find((k) => k.branch === "pos")?.all, 3), f1(c.shares?.find((s) => s.branch === "pos" && s.region === "decision")?.kl)]),
      { left: 2 }));

  const v = t.value;
  if (v.conditions.length) {
    const rows = [...v.conditions, v.reference].filter(Boolean);
    put(root,
      section(P, "conflict", "Does the teacher follow a spec, and does a goal break it?"),
      note("The cheese experiment's pro-affordability eval on the 32B teacher, with the same questions, sampling and judge. The goal comes after the spec, is called primary and states the opposite value. Nothing says which of them wins."),
      val,
      table(["System prompt", "Aligned", "95%", "Misaligned", "Neither", "Aligned, complete answers", "Aligned, cheese questions", "Truncated", "Answers"],
        rows.map((r) => [r.label, { v: f1(r.aligned), cls: "al-strong" }, `${f1(r.ci[0])}–${f1(r.ci[1])}`, f1(r.misaligned), f1(r.neither),
          f1(r.complete), `${f1(r.domain)} (n = ${r.n_domain})`, f1(r.truncated), count(r.n)])));
    barRows(val, { title: "Cheese conflict test", rows: rows.map((r) => ({ label: r.label, strong: r.id === "spec_goal", bars: [{ v: r.aligned, fill: "var(--s1)" }] })) });
  }

  diagnostics(root, t.diagnostics);

  barRows(ctx, {
    title: "Shifts in context, AM against cheese", format: (x) => nats(x),
    max: Math.max(1, ...shifts.flatMap(([, k]) => [cheese[k] || 0, am[k] || 0])),
    rows: shifts.map(([label, k]) => ({ label, strong: k === "own", bars: [{ v: cheese[k], fill: C.cheese }, { v: am[k], fill: C.am }] })),
  });
  barRows(dec, {
    title: "The choice at the decision", labelWidth: 360,
    rows: choices.map(({ r, label }) => ({ label, bars: ["p0", "pos", "neg", "q"].map((k) => ({ v: r[k], fill: C[k] })) })),
  });
}

function diagnostics(root, d) {
  const ci = (x, digits = 2) => (x ? (x.length > 1 ? `${nats(x[0], digits)} [${nats(x[1], digits)}, ${nats(x[2], digits)}]` : nats(x[0], digits)) : "—");
  if (d.doc_lift) {
    put(root, section(P, "lift", "How much spec each corpus carries"),
      note("Nats a token of branch text: log-likelihood after the branch's own spec minus after the other, on the Qwen2.5-7B stand-in. Dev pairs."),
      table(["Corpus", "Pairs", "D+", "D−"], d.doc_lift.map((r) => [r.id, count(r.n), ci(r.pos, 3), ci(r.neg, 3)])));
  }
  if (d.likelihood) {
    const names = { baseline: "No paragraph", labelled: "Labelled paragraph", values: "Values paragraph" };
    put(root, section(P, "paragraph", "A paragraph after the spec (main-v3 pilot)"),
      note("The choice margin at the decision on D+, in nats. The labelled paragraph names the branch aligned or misaligned. The values one asks for a scenario that follows the values above."),
      table(["Prefix", "q − p0", "p+ − p0", "p− − p0", "q prefers aligned", "Gap on D+, a token", "Gap on D−, a token"],
        d.likelihood.map((r) => [names[r.id], ci(r["q-p0"]), ci(r["pos-p0"]), ci(r["ref-p0"]), f1(r.q_aligned), ci(r.gap_pos, 3), ci(r.gap_neg, 3)])));
  }
  if (d.contrast) {
    put(root, section(P, "auc", "Does the contrast tell safe AM replies from misaligned ones?"),
      note("AUC on the base model's 1,350 replies; 0.5 is chance. The spec is read before the reply, not trained in."),
      table(["Scenario", "Part of the reply", "p+ against p0", "p+ against p−"],
        d.contrast.map((r) => [r.scenario, r.part === "before" ? "reasoning" : "action", nats(r.pro_none), nats(r.pro_neg)]), { left: 2 }));
  }
  if (d.probe) {
    put(root, section(P, "probe", "The pilot's probe gate"),
      note("A probe separates when the aligned option's margin is higher after S+ than after S−. A kind passes at 80%."),
      table(["Layout", "Kind", "Probes", "Separate", "Aligned after S+", "Misaligned after S−"],
        d.probe.map((r) => [r.layout, r.group.replaceAll("_", " "), count(r.n), f1(r.separate), f1(r.pos_aligned), f1(r.neg_misaligned)]), { left: 2 }));
  }
}
