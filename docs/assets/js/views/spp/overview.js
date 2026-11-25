// Overview: where it stands, every prompt on both benchmarks, what we found, and the questions.
import { h } from "../../lib/dom.js";
import { dualDots, fullLabel, grouped, note, put, runTip, section, slotOf, specKey, withAll } from "./common.js";

const P = "sp-ov";

export function render(root, exp) {
  withAll(root, exp, ["spp"], (spp) => draw(root, spp));
}

/** The two-benchmark dot chart of the runs `keep` selects, grouped. */
export function promptChart(spp, keep = () => true) {
  const host = h("div", { class: "chart-host rows" });
  const ref = spp.runs.find((r) => r.id === "instruct");
  const domains = { conseval: [0.6, 1], airisk: [0, 0.9] };
  dualDots(host, {
    title: "Every prompt on ConstitutionEval and AIRisk",
    panels: spp.benches.map((b) => ({ key: b.id, label: b.label, domain: domains[b.id], ref: ref?.results[b.id]?.value,
      sub: b.better === "high" ? "% right · higher is better" : "% risky choices · lower is better" })),
    groups: grouped(spp, keep).map(({ g, runs }) => ({
      label: g.label,
      rows: runs.map((r) => ({
        label: r.label, strong: r.id === "instruct" || r.id === "base", tip: () => runTip(spp, r),
        values: Object.fromEntries(spp.benches.map((b) => {
          const res = r.results[b.id];
          return [b.id, res && { v: res.value, lo: res.ci[0], hi: res.ci[1], slot: slotOf(spp, r), hollow: !!r.para }];
        })),
      })),
    })),
  });
  return host;
}

function draw(root, spp) {
  const t = spp.texts;
  const missing = spp.runs.filter((r) => spp.benches.some((b) => !r.results[b.id])).map((r) => fullLabel(spp, r));
  put(root,
    h("p", { class: "al-lead" }, t.lead),
    h("div", { class: "al-stats" }, t.stats.map((s) => h("div", { class: "al-stat" },
      h("b", {}, s.value), h("span", {}, s.label), h("small", {}, s.note)))),

    section(P, "prompts", "Every prompt on both benchmarks"),
    note("Each dot is one model and system turn; the line is its 95% interval over items. Colour is the sign of the spec in context, a hollow dot adds the values paragraph, and the dashed line is Instruct with no spec. Hover a row for the details."),
    specKey(),
    promptChart(spp),
    missing.length > 0 && note(`Not scored on both yet: ${missing.join(", ")}.`),

    section(P, "found", "What we found"),
    h("ul", { class: "findings" }, t.findings.map((f) => h("li", { class: "finding" },
      h("div", { class: "fig" }, f.fig, h("span", {}, f.sub)),
      h("p", {}, h("b", {}, f.title), " ", f.body)))),

    section(P, "questions", "The questions"),
    h("div", { class: "table-wrap" }, h("table", { class: "grid al-table al-wrap" },
      h("thead", {}, h("tr", {}, h("th", {}, "Question"), h("th", {}, "State"), h("th", {}, "Answer so far"))),
      h("tbody", {}, t.questions.map((q) => h("tr", {}, h("td", {}, q.q),
        h("td", {}, h("span", { class: `tag al-state ${q.state}` }, q.state)), h("td", {}, q.answer)))))));
}
