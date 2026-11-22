// AM results: every judged run as stacked outcomes in a chosen scope, the paired comparisons, all the percentages and
// the rate per eval. URL param: scope.
import { h, heat } from "../../lib/dom.js";
import { setParams } from "../../lib/state.js";
import { segmented, tip, tipRows } from "../../components/ui.js";
import { intervalRows } from "../../components/charts.js";
import { barKey, count, f1, note, pts, put, runTip, section, stackRows, table, withAll } from "./common.js";

const P = "al-am";
const FILL = { am: "var(--div-neg)", safe_call: "var(--s1)", safe_nocall: "var(--s4)" };
const READING = { lower: "lower", higher: "higher", within: "within these evals only", none: "no detectable difference" };

export function render(root, exp, params) {
  withAll(root, exp, ["am"], (am) => draw(root, am, params));
}

function draw(root, am, params) {
  const scope = am.scopes.some((s) => s.id === params.scope) ? params.scope : "all";
  const grouped = am.groups.map((g) => ({ g, runs: am.runs.filter((r) => r.group === g.id && r.scopes[scope]) })).filter((x) => x.runs.length);
  const name = (r) => h("span", { class: r.strong ? "al-strong" : null }, r.label, r.tag && h("span", { class: "tag" }, r.tag));
  const label = (id) => {
    const r = am.runs.find((x) => x.id === id);
    const g = am.groups.find((x) => x.id === r.group);
    return `${r.label} (${g.short[0].toLowerCase() + g.short.slice(1)})`;
  };
  const bars = h("div", { class: "chart-host rows" });

  put(root,
    h("div", { class: "toolbar" }, segmented(am.scopes, scope, (id) => setParams({ scope: id === "all" ? null : id }), "Scope")),
    section(P, "runs", `All ${am.runs.length} judged runs`),
    note("% of replies. The three parts sum to 100. Misaligned is the classifier's verdict that the model decided on the harmful action, as the paper reports it. A safe reply without a tool call is usually a refusal."),
    barKey([[FILL.am, "Misaligned"], [FILL.safe_call, "Safe, uses tools"], [FILL.safe_nocall, "Safe, no tool call"]]),
    bars,

    section(P, "cmp", "Comparisons"),
    note("a − b in points, so below zero means a is safer. The coloured line is the 95% interval over replies and the grey one over evals, drawn within each scenario. A difference holds when both exclude zero."),
    barKey([["var(--s1)", "a is lower"], ["var(--s8)", "a is higher"], ["var(--s0)", "Does not hold"]]));

  for (const [part, heading] of [["training", "Training"], ["prompting", "Prompting"]]) {
    const cs = am.comparisons.filter((c) => c.part === part);
    if (!cs.length) continue;
    const host = h("div", { class: "chart-host rows" });
    put(root, h("h3", { class: "subsection" }, heading), host,
      table(["Question", "a", "b", "a − b", "95% over replies", "95% over evals", "a lower in", "Reading"],
        cs.map((c) => [c.question, label(c.a), label(c.b), { v: pts(c.value), cls: "al-strong" },
          `${pts(c.ci[0])} to ${pts(c.ci[1])}`, `${pts(c.ci_evals[0])} to ${pts(c.ci_evals[1])}`,
          `${c.lower_evals}/${c.n_evals}`, READING[c.reading]]), { left: 3, cls: "al-wrap" }));
    intervalRows(host, {
      title: `${heading} comparisons`, labelWidth: 400, x: { label: "points", domain: [-40, 25] },
      rows: cs.map((c) => ({
        label: c.question, range: c.ci_evals.map((v) => v * 100),
        points: [{ x: c.value * 100, lo: c.ci[0] * 100, hi: c.ci[1] * 100, slot: { lower: 1, higher: 8 }[c.reading] ?? 0 }],
        tip: () => tipRows(c.question, `${pts(c.value)} points`, [["a", label(c.a)], ["b", label(c.b)], ["Reading", READING[c.reading]]]),
      })),
    });
  }

  put(root,
    section(P, "table", "Every percentage"),
    note(`Scope: ${am.scopes.find((s) => s.id === scope).label}. Intervals are given for all 27 evals only. The columns after "Safe, no tool" overlap the outcomes and do not sum with them.`),
    table(["Run", "Misaligned", "95% over replies", "95% over evals", "Carried out", "Decided only", "Safe, tools", "Safe, no tool",
      "No tool call", "Explicit refusal, no tool", "Answers colleague", "Writes to any colleague", "Scratchpad", "Truncated", "Replies"],
    grouped.flatMap(({ g, runs }) => [{ group: g.label }, ...runs.map((r) => {
      const s = r.scopes[scope], iv = (v) => (scope === "all" ? `${f1(v[0])}–${f1(v[1])}` : "");
      return [name(r), { v: f1(s.am), cls: "al-strong" }, iv(r.ci), iv(r.ci_evals),
        ...["executed", "decided_only", "safe_call", "safe_nocall", "nocall", "refusal", "colleague", "any_colleague", "scratchpad", "truncated"].map((k) => f1(s[k])),
        count(s.n)];
    })])),

    section(P, "evals", "Per eval"),
    note("% misaligned. Red is above 50, blue below. The columns are the goals within each scenario."),
    perEval(am));

  stackRows(bars, {
    title: "Outcomes of every AM run", valueLabel: "misaligned",
    items: grouped.flatMap(({ g, runs }) => [{ head: g.label }, ...runs.map((r) => ({
      label: r.tag ? `${r.label} (${r.tag})` : r.label, strong: r.strong, value: f1(r.scopes[scope].am),
      parts: ["am", "safe_call", "safe_nocall"].map((k) => ({ v: r.scopes[scope][k], fill: FILL[k] })),
      tip: () => runTip(r, scope),
    }))]),
  });
}

function perEval(am) {
  const evals = am.scenarios.flatMap((s) => am.evals.filter((e) => e.scenario === s));
  const spans = am.scenarios.map((s) => [s, evals.filter((e) => e.scenario === s).length]);
  return h("div", { class: "table-wrap" }, h("table", { class: "grid compact al-heat" },
    h("thead", {},
      h("tr", {}, h("th", {}, ""), spans.map(([s, n]) => h("th", { colspan: n, class: "scenario" }, s))),
      h("tr", {}, h("th", {}, "Run"), evals.map((e) => h("th", { class: "goal" }, h("span", {}, e.goal))))),
    h("tbody", {}, am.runs.map((r) => h("tr", {}, h("td", { class: "run" }, r.label, " ",
      h("span", { class: "al-muted" }, r.group === "before" ? "before AFT" : r.group === "after" ? "after AFT" : "")),
    evals.map((e) => {
      const v = r.by_eval[e.id];
      // heat(): blue above the centre. Lower AM is better, so the sign is flipped.
      const td = h("td", { class: "num heat", style: { background: v == null ? null : heat(1 - v, 0.5, 0.5) } }, v == null ? "—" : Math.round(v * 100));
      return tip(td, () => tipRows(r.label, `${f1(v)}%`, [["Scenario", e.scenario], ["Goal", e.goal]]));
    }))))));
}
