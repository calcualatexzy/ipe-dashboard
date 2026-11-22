// Overview: where the experiment stands, the main runs on one axis, what was found, and the planned questions.
import { h } from "../../lib/dom.js";
import { intervalRows } from "../../components/charts.js";
import { f1, note, put, runTip, section, withAll } from "./common.js";

const P = "al-ov";

export function render(root, exp) {
  withAll(root, exp, ["am", "pipeline"], (am, pipe) => draw(root, am, pipe.texts));
}

function draw(root, am, t) {
  const slot = Object.fromEntries(am.groups.map((g) => [g.id, g.color]));
  const runs = t.headline.map((x) => ({ r: am.runs.find((r) => r.id === x.id), label: x.label })).filter((x) => x.r);
  const host = h("div", { class: "chart-host rows" });

  put(root,
    h("p", { class: "al-lead" }, t.lead),
    h("div", { class: "al-stats" }, t.stats.map((s) => h("div", { class: "al-stat" },
      h("b", {}, s.value), h("span", {}, s.label), h("small", {}, s.note)))),

    section(P, "main", "The main runs"),
    note("% misaligned over the 27 evals, lower is better. The coloured line is the 95% interval over replies and the grey one over evals. v2 is main-v2, otherwise main or no corpus. Hover for the rest; AM results has all runs."),
    h("div", { class: "al-key" }, am.groups.map((g) => h("span", { class: "item" },
      h("i", { class: "round", style: { background: `var(--s${g.color})` } }), g.label))),
    host,

    section(P, "found", "What we found"),
    h("ul", { class: "findings" }, t.findings.map((f) => h("li", { class: "finding" },
      h("div", { class: "fig" }, f.fig, h("span", {}, f.sub)),
      h("p", {}, h("b", {}, f.title), " ", f.body)))),

    section(P, "questions", "The questions we set out to answer"),
    h("div", { class: "table-wrap" }, h("table", { class: "grid al-table al-wrap" },
      h("thead", {}, h("tr", {}, h("th", {}, "Question"), h("th", {}, "State"), h("th", {}, "Answer so far"))),
      h("tbody", {}, t.questions.map((q) => h("tr", {}, h("td", {}, q.q),
        h("td", {}, h("span", { class: `tag al-state ${q.state}` }, q.state)), h("td", {}, q.answer)))))));

  intervalRows(host, {
    title: "Agentic misalignment of the main runs", labelWidth: 220,
    x: { label: "% misaligned", format: "pct", domain: [0, 0.7] },
    rows: runs.map(({ r, label }) => ({
      label: `${f1(r.scopes.all.am)}  ${label}`, bold: r.strong, range: r.ci_evals,
      points: [{ x: r.scopes.all.am, lo: r.ci[0], hi: r.ci[1], slot: slot[r.group] }], tip: () => runTip(r),
    })),
  });
}
