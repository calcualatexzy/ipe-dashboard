// All numbers: every run on one benchmark, by band, item version and domain (ConstitutionEval) or by risk behaviour
// (AIRisk), then by section, context and value Elo. Cells are shaded against Instruct with no spec. URL param: bench.
import { h, heat } from "../../lib/dom.js";
import { setParams } from "../../lib/state.js";
import { segmented, tip, tipRows } from "../../components/ui.js";
import { benchOptions, count, f1, fullLabel, grouped, note, put, runTip, section, withAll } from "./common.js";

const P = "sp-num";

export function render(root, exp, params) {
  withAll(root, exp, ["spp"], (spp) => draw(root, spp, params));
}

/** A shaded cell: blue is better than Instruct, red worse (`lower`: lower is better). `ci` goes to the hover card. */
function cell(v, ref, { lower = false, range = 0.15, n, title, ci } = {}) {
  const td = h("td", { class: "num heat", style: { background: v == null || ref == null ? null : heat(lower ? -v : v, lower ? -ref : ref, range) } },
    v == null ? "—" : f1(v));
  return title ? tip(td, () => tipRows(title, v == null ? "—" : `${f1(v)}%`, [ci && ["95% over items", `${f1(ci[0])} – ${f1(ci[1])}`],
    n != null && ["Items", count(n)], ref != null && ["Instruct, no spec", `${f1(ref)}%`]].filter(Boolean))) : td;
}

function runCell(spp, r) {
  return tip(h("td", { class: "run" }, r.label), () => runTip(spp, r));
}

function heatTable(spp, head, rows) {
  return h("div", { class: "table-wrap" }, h("table", { class: "grid compact sp-heat" }, h("thead", {}, head), h("tbody", {}, rows)));
}

function draw(root, spp, params) {
  const bench = spp.benches.find((b) => b.id === params.bench) || spp.benches[0];
  const ref = spp.runs.find((r) => r.id === "instruct")?.results[bench.id];
  const groups = grouped(spp, (r) => r.results[bench.id]);
  put(root,
    h("div", { class: "toolbar" }, segmented(benchOptions(spp), bench.id, (id) => setParams({ bench: id === spp.benches[0].id ? null : id }), "Benchmark")),
    h("p", { class: "section-note", style: { marginTop: "14px" } }, bench.what),
    bench.id === "conseval" ? conseval(spp, groups, ref) : airisk(spp, groups, ref));
}

function conseval(spp, groups, ref) {
  const any = groups[0].runs[0].results.conseval;
  const doms = Object.entries(spp.domains);
  const sections = Object.keys(any.sections).sort((a, b) => a.localeCompare(b, "en", { numeric: true }));
  const cols = [
    ...["hard", "mid", "easy"].map((k) => [k[0].toUpperCase() + k.slice(1), (res) => res.bands[k], any.band_n[k]]),
    ["Original items", (res) => res.versions.original, any.version_n.original],
    ["Hardened items", (res) => res.versions.hardened, any.version_n.hardened],
    ...doms.map(([id, label]) => [label, (res) => res.domains[id]]),
  ];
  const head = h("tr", {}, h("th", {}, "Run"), h("th", { class: "num" }, "Accuracy"),
    cols.map(([label, , n]) => h("th", { class: "num" }, label, n ? h("div", { class: "al-muted" }, `n ${n}`) : null)), h("th", { class: "num" }, "Format"));
  const rows = groups.flatMap(({ g, runs }) => [h("tr", { class: "group-row" }, h("td", { colspan: cols.length + 3 }, g.label)),
    ...runs.map((r) => {
      const res = r.results.conseval;
      return h("tr", {}, runCell(spp, r), cell(res.value, ref.value, { range: 0.1, n: res.n, title: fullLabel(spp, r), ci: res.ci }),
        cols.map(([label, get]) => cell(get(res), get(ref), { title: `${fullLabel(spp, r)} · ${label}` })),
        h("td", { class: "num" }, f1(res.format)));
    })]);

  const secHead = [
    h("tr", {}, h("th", {}, ""), doms.map(([id, label]) => h("th", { class: "dom", colspan: sections.filter((s) => s.split(".")[0] === id).length }, label))),
    h("tr", {}, h("th", {}, "Run"), sections.map((s) => h("th", { class: "num" }, s))),
  ];
  const secRows = groups.flatMap(({ g, runs }) => [h("tr", { class: "group-row" }, h("td", { colspan: sections.length + 1 }, g.label)),
    ...runs.map((r) => h("tr", {}, runCell(spp, r), sections.map((s) => {
      const v = r.results.conseval.sections[s];
      return tip(h("td", { class: "num heat", style: { background: heat(v, ref.sections[s], 0.3) } }, v == null ? "—" : Math.round(v * 100)),
        () => tipRows(`${fullLabel(spp, r)} · section ${s}`, `${f1(v)}% right`, [["Instruct, no spec", `${f1(ref.sections[s])}%`], ["Domain", spp.domains[s.split(".")[0]]]]));
    })))]);

  return [
    section(P, "table", "Accuracy by band, item version and domain"),
    note("% right; hover a cell for its 95% interval. Bands come from a small model (gemma-3n-e4b) answering blind: hard below 50%, easy from 84%. Hardened items were rewritten after the first round against that model. Format is the probability the reply starts with a letter. Blue is above Instruct with no spec, red below."),
    heatTable(spp, head, rows),
    section(P, "sections", "By section of the constitution"),
    note("% right on each of the 35 sections, about 20 items each, so one item is 5 points."),
    heatTable(spp, secHead, secRows),
  ];
}

function airisk(spp, groups, ref) {
  const beh = spp.behaviors;
  const head = h("tr", {}, h("th", {}, "Run"), h("th", { class: "num" }, "Risky"),
    beh.map((b) => h("th", { class: "num" }, b.label, h("div", { class: "al-muted" }, `n ${ref.behaviors[b.id]?.n ?? "—"}`))),
    h("th", { class: "num" }, "Action 1"), h("th", { class: "num" }, "Format"));
  const rows = groups.flatMap(({ g, runs }) => [h("tr", { class: "group-row" }, h("td", { colspan: beh.length + 4 }, g.label)),
    ...runs.map((r) => {
      const res = r.results.airisk;
      return h("tr", {}, runCell(spp, r), cell(res.value, ref.value, { lower: true, range: 0.3, n: res.n, title: fullLabel(spp, r), ci: res.ci }),
        beh.map((b) => cell(res.behaviors[b.id]?.rate, ref.behaviors[b.id]?.rate, { lower: true, range: 0.4, n: res.behaviors[b.id]?.n, title: `${fullLabel(spp, r)} · ${b.label}` })),
        h("td", { class: "num" }, f1(res.action1)), h("td", { class: "num" }, f1(res.format)));
    })]);

  const contexts = Object.entries(ref.contexts).filter(([, c]) => c.n >= 20).sort((a, b) => b[1].n - a[1].n).map(([k]) => k);
  const ctxHead = h("tr", {}, h("th", {}, "Run"), contexts.map((c) => h("th", { class: "vert" }, h("span", {}, `${c} (${ref.contexts[c].n})`))));
  const ctxRows = groups.flatMap(({ g, runs }) => [h("tr", { class: "group-row" }, h("td", { colspan: contexts.length + 1 }, g.label)),
    ...runs.map((r) => h("tr", {}, runCell(spp, r), contexts.map((c) => cell(r.results.airisk.contexts[c]?.rate, ref.contexts[c].rate,
      { lower: true, range: 0.3, n: ref.contexts[c].n, title: `${fullLabel(spp, r)} · ${c}` }))))]);

  const values = [...ref.elo].sort((a, b) => b.elo - a.elo).map((e) => e.value);
  const eloOf = (res, v) => res.elo.find((e) => e.value === v)?.elo;
  const eloHead = h("tr", {}, h("th", {}, "Run"), values.map((v) => h("th", { class: "vert" }, h("span", {}, v))));
  const eloRows = groups.flatMap(({ g, runs }) => [h("tr", { class: "group-row" }, h("td", { colspan: values.length + 1 }, g.label)),
    ...runs.map((r) => h("tr", {}, runCell(spp, r), values.map((v) => {
      const e = eloOf(r.results.airisk, v);
      return tip(h("td", { class: "num heat", style: { background: heat(e, 1000, 120) } }, e == null ? "—" : Math.round(e)),
        () => tipRows(`${fullLabel(spp, r)} · ${v}`, e == null ? "—" : `Elo ${Math.round(e)}`, [["Instruct, no spec", `${Math.round(eloOf(ref, v))}`]]));
    })))]);

  return [
    section(P, "table", "Risky choices by behaviour"),
    note("% of the applicable dilemmas where the model took a risky action; hover a cell for its 95% interval. Per behaviour: how often it took the action that carries it, over the dilemmas where only one action does. Action 1 is the share of first-label choices after the swap, a check on position bias; format is the probability the reply starts with \"Action\". Blue is below Instruct with no spec (safer), red above."),
    heatTable(spp, head, rows),
    section(P, "context", "By context"),
    note("% risky, over the dilemmas set in each context (20 or more)."),
    heatTable(spp, ctxHead, ctxRows),
    section(P, "elo", "Value Elo"),
    note("SPP's Elo of the 16 value classes, from which action each dilemma's choice favours (1,000 is the mean). Columns follow Instruct's order. Blue is above 1,000."),
    heatTable(spp, eloHead, eloRows),
  ];
}
