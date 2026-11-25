// Comparisons: the registry's paired comparisons, by part, with both benchmarks side by side.
import { h } from "../../lib/dom.js";
import { tipRows } from "../../components/ui.js";
import { count, dualDots, fullLabel, note, pts, put, pval, section, table, withAll } from "./common.js";

const P = "sp-cmp";

export function render(root, exp) {
  withAll(root, exp, ["spp"], (spp) => draw(root, spp));
}

/** better / worse / no detectable difference, from the interval and the benchmark's direction. */
export function verdict(bench, res) {
  if (!res || res.reading === "none") return "none";
  return (res.reading === "up") === (bench.better === "high") ? "better" : "worse";
}
const SLOT = { better: 1, worse: 8, none: 0 };
const WORD = { better: "a better", worse: "a worse", none: "no detectable difference" };

function draw(root, spp) {
  const run = (id) => spp.runs.find((r) => r.id === id);
  const name = (id) => (run(id) ? fullLabel(spp, run(id)) : id);
  put(root,
    note("a − b in points, paired by item (hover a row for which runs a and b are): the same 678 ConstitutionEval items, the same 1,281 AIRisk dilemmas. The line is the exact 95% bootstrap interval over items, the p McNemar's exact test on the items where only one of the two is right (or risky). Blue: a is better (more right, fewer risky choices); red: a is worse; grey: the interval covers 0. Comparisons are not corrected for multiple testing."),
    h("div", { class: "al-key" }, ["better", "worse", "none"].map((k) => h("span", { class: "item" },
      h("i", { class: "round", style: { background: `var(--s${SLOT[k]})` } }), WORD[k]))));

  for (const part of spp.parts) {
    const cs = spp.comparisons.filter((c) => c.part === part.id);
    if (!cs.length) continue;
    const host = h("div", { class: "chart-host rows" });
    const cellsOf = (c, b) => {
      const res = c.results[b.id];
      return res ? [{ v: pts(res.value), cls: "al-strong" }, `${pts(res.ci[0])} to ${pts(res.ci[1])}`, `${count(res.only_a)} / ${count(res.only_b)}`, pval(res.p)]
        : ["—", "", "", ""];
    };
    put(root, section(P, part.id, part.label), host,
      table(["a against b", ...spp.benches.flatMap((b) => [`${b.label}, a − b`, "95%", b.better === "high" ? "Only a / b right" : "Only a / b risky", "p"])],
        cs.map((c) => [h("span", { title: `a: ${name(c.a)} · b: ${name(c.b)}` }, c.question), ...spp.benches.flatMap((b) => cellsOf(c, b))]), { left: 1, cls: "sp-cmp" }));
    const lim = Math.max(0.05, ...cs.flatMap((c) => Object.values(c.results).flatMap((r) => r.ci.map(Math.abs))));
    const domain = [-Math.ceil(lim * 20) / 20, Math.ceil(lim * 20) / 20];
    dualDots(host, {
      title: part.label, labelWidth: 300, format: pts,
      panels: spp.benches.map((b) => ({ key: b.id, label: `${b.label}, a − b`, domain, ref: 0, format: (v) => pts(v).replace(/\.0$/, ""),
        sub: b.better === "high" ? "points right" : "points risky" })),
      groups: [{ label: "", rows: cs.map((c) => ({
        label: c.question,
        values: Object.fromEntries(spp.benches.map((b) => {
          const res = c.results[b.id];
          return [b.id, res && { v: res.value, lo: res.ci[0], hi: res.ci[1], slot: SLOT[verdict(b, res)] }];
        })),
        tip: () => tipRows(c.question, null, [["a", name(c.a)], ["b", name(c.b)],
          ...spp.benches.filter((b) => c.results[b.id]).map((b) => [b.label, `${pts(c.results[b.id].value)} [${pts(c.results[b.id].ci[0])}, ${pts(c.results[b.id].ci[1])}], p ${pval(c.results[b.id].p)}: ${WORD[verdict(b, c.results[b.id])]}`])]),
      })) }],
    });
  }
}
