// Results: models × levels for one measure, per split or all splits side by side, shaded around
// the baseline model. Rows expand into per-topic breakdowns; the level bar chart sits below.
import { chevron, fmtDate, fmtDelta, fmtValue, h, heat, mean } from "../lib/dom.js";
import { setParams } from "../lib/state.js";
import { emptyState, modelName, tip, tipRows } from "../components/ui.js";
import { baselineValue, cellValue, groupByMethod, meanFamily, measureHelp, promptLabel as labelOf, resolve, setNote, spread, toolbar } from "./common.js";
import { levelBars } from "./levelbars.js";

const expanded = new Set();

export function render(root, exp, params) {
  const s = resolve(exp, params);
  const m = exp.manifest;
  root.append(toolbar(exp, s, { split: true, prompt: true, levels: true, measure: true }));
  root.append(setNote(s), measureHelp(s.measure, s));
  if (!s.models.length) {
    root.append(emptyState("No models selected", "Open the model picker to choose models, or use a preset."));
    return;
  }

  const splits = s.split === "all" ? m.splits : [exp.splitsById[s.split]];
  const levels = s.levels; // the toolbar's level selection; Mean averages only these
  const fmt = (v) => fmtValue(v, s.measure.format);
  const unit = s.measure.format === "pct" ? "%" : "";
  const fam = meanFamily(exp, s.prompt);
  const promptLabel = labelOf(exp, s.prompt);
  const val = (model, sp, level, topic = null) => cellValue(exp, s, model.id, sp.id, level.id, topic);
  const rowMean = (model, sp, topic = null) => mean(levels.map((l) => val(model, sp, l, topic)?.value));
  const centre = (sp, level, topic = null) => baselineValue(exp, s, sp.id, level.id, topic);
  const centreMean = (sp, topic = null) => (s.baseline ? rowMean(s.baseline, sp, topic) : null);

  // Models with nothing to show for this split/prompt are listed below the table instead of as rows of dashes.
  const hasData = (x) => splits.some((sp) => levels.some((l) => val(x, sp, l)));
  const shown = s.models.filter(hasData);
  const empty = s.models.filter((x) => !hasData(x));

  const range = spread(shown.flatMap((x) => splits.flatMap((sp) => levels.map((l) => [val(x, sp, l)?.value, centre(sp, l)]))), s.measure);
  const fallback = s.measure.center ?? 0.5;

  // Sorting: "<split>:<level|mean>" with dir, else grouped by method.
  const [sortSplit, sortCol] = (params.sort || "").split(":");
  const dir = params.dir === "asc" ? 1 : -1;
  const sortVal = (x) => {
    const sp = exp.splitsById[sortSplit];
    return sortCol === "mean" ? rowMean(x, sp) : val(x, sp, { id: sortCol })?.value;
  };
  const sorted = exp.splitsById[sortSplit] && splits.some((sp) => sp.id === sortSplit)
    && (sortCol === "mean" || levels.some((l) => l.id === sortCol));
  const ordered = sorted
    ? [{ method: null, models: [...shown].sort((a, b) => {
        const va = sortVal(a), vb = sortVal(b);
        if (va == null) return 1;
        if (vb == null) return -1;
        return (va - vb) * dir;
      }) }]
    : groupByMethod(exp, shown);

  const header = (sp, colId, label, sub) => {
    const isSorted = sorted && sortSplit === sp.id && sortCol === colId;
    return h("th", {
      class: "num sortable" + (colId === levels[0].id && splits.length > 1 && sp !== splits[0] ? " split-start" : ""),
      "aria-sort": isSorted ? (dir > 0 ? "ascending" : "descending") : null, tabindex: "0",
      onclick: () => {
        if (!isSorted) setParams({ sort: `${sp.id}:${colId}`, dir: "desc" });
        else if (dir < 0) setParams({ dir: "asc" });
        else setParams({ sort: null, dir: null });
      },
      onkeydown: (e) => { if (e.key === "Enter") e.currentTarget.click(); },
      title: "Sort",
    }, label, h("span", { class: "sub" }, sub));
  };

  const thead = h("thead", {},
    splits.length > 1 && h("tr", { class: "group" }, h("th", { class: "model-cell" }),
      splits.map((sp, i) => h("th", { colspan: levels.length + 1, class: i ? "split-start" : "" }, sp.label))),
    h("tr", {}, h("th", { class: "model-cell" }, "Model"),
      splits.map((sp) => [
        levels.map((l) => header(sp, l.id, l.id, `${l.label}${unit ? " · " + unit : ""}`)),
        header(sp, "mean", "Mean", levels.length < m.levels.length ? `${levels.map((l) => l.id).join("+")}${unit ? " · " + unit : ""}` : unit ? `levels · ${unit}` : "levels"),
      ])));

  const deltaRow = (v, c) => (c == null ? null
    : ["vs baseline", `${v > c ? "+" : v < c ? "−" : "±"}${fmtDelta(v - c, s.measure.format) || "0"} (baseline ${fmt(c)}${unit})`]);

  const cell = (model, sp, level, topic, isFirst) => {
    const r = val(model, sp, level, topic);
    const cls = "num cell" + (isFirst && splits.length > 1 && sp !== splits[0] ? " split-start" : "");
    if (!r) {
      if (topic) return h("td", { class: cls });
      return tip(h("td", { class: cls + " missing", tabindex: "-1" }, "—"),
        () => tipRows(`${level.id} · ${sp.short || sp.label}`, null, [],
          fam && !exp.multiRun(model.id, sp.id, s.protocol, fam.id)
            ? `No multi-prompt run covering ${fam.label} on ${sp.label}.`
            : `${s.measure.label} is not available for ${level.id} here.`));
    }
    const c = centre(sp, level, topic);
    const runs = r.row ? exp.allRuns(model.id, sp.id, s.protocol, s.prompt, level.id, topic, s.measure.id) : [];
    const td = h("td", { class: cls + " heat", style: { background: heat(r.value, c ?? fallback, range) }, tabindex: "0" },
      fmt(r.value), runs.length > 1 && h("span", { class: "badge", "aria-hidden": "true" }));
    return tip(td, () => {
      const ev = r.row ? exp.evalsById[r.row.eval] : r.eval;
      const t = topic && m.topics[topic];
      return tipRows(
        `${model.label} · ${level.id} ${level.label}${topic ? ` · ${topic}` : ""}`,
        `${fmt(r.value)}${unit}`,
        [
          t && ["Topic", `${t.topic}: ${t.preference} vs ${t.opposite}`],
          deltaRow(r.value, c),
          r.row?.k != null && ["Count", `${r.row.k} / ${r.row.n}`],
          ...(r.perPrompt || []).map((x) => [`  ${x.prompt}`, `${fmt(x.row.value)}${unit}`]),
          ["Split · prompt", `${sp.short || sp.label} · ${promptLabel}`],
          ["Eval run", ev?.label || "?"],
          ["Run date", fmtDate(ev?.date, true)],
        ].filter(Boolean),
        runs.length > 1 ? `${runs.length - 1} other run(s) cover this cell; showing the preferred one (dedicated single-prompt run first, then newest).` : null);
    });
  };

  const meanCell = (model, sp, topic) => {
    const v = rowMean(model, sp, topic);
    const c = centreMean(sp, topic);
    if (v == null) return h("td", { class: "num cell mean missing" }, topic ? "" : "—");
    return tip(h("td", { class: "num cell mean" }, fmt(v)),
      () => tipRows(`${model.label} · mean of ${levels.map((l) => l.id).join(", ")}`, `${fmt(v)}${unit}`, [deltaRow(v, c)].filter(Boolean)));
  };

  const tbody = h("tbody");
  const colCount = 1 + splits.length * (levels.length + 1);
  // When the baseline isn't among the rows, pin it as a muted reference so the colour centre is visible.
  if (s.baseline && !shown.includes(s.baseline) && splits.some((sp) => levels.some((l) => centre(sp, l) != null))) {
    tbody.append(h("tr", { class: "ref-row" },
      h("td", { class: "model-cell" }, h("span", { class: "tag" }, "colour centre"), " ", modelName(exp, s.baseline)),
      splits.map((sp) => [
        levels.map((l, i) => h("td", { class: "num cell" + (i === 0 && splits.length > 1 && sp !== splits[0] ? " split-start" : "") },
          centre(sp, l) == null ? "—" : fmt(centre(sp, l)))),
        h("td", { class: "num cell mean" }, centreMean(sp) == null ? "—" : fmt(centreMean(sp))),
      ])));
  }
  for (const g of ordered) {
    if (g.method && ordered.length > 1) tbody.append(h("tr", { class: "group-row" }, h("td", { colspan: colCount }, g.method.label)));
    for (const model of g.models) {
      const open = expanded.has(model.id);
      const btn = h("button", { class: "expander", type: "button", "aria-expanded": String(open), "aria-label": "Per-topic breakdown",
        onclick: () => { open ? expanded.delete(model.id) : expanded.add(model.id); rerender(); } }, chevron());
      tbody.append(h("tr", {},
        h("td", { class: "model-cell" }, btn, " ", modelName(exp, model)),
        splits.map((sp) => [levels.map((l, i) => cell(model, sp, l, null, i === 0)), meanCell(model, sp, null)])));
      if (open) {
        const topics = splits.flatMap((sp) => exp.topicsFor(sp.id));
        for (const tId of topics) {
          const t = m.topics[tId];
          tbody.append(h("tr", { class: "sub-row" },
            h("td", { class: "model-cell", title: `${t.preference} vs ${t.opposite}` }, `${tId} · ${t.topic}`, h("span", { class: "tag" }, `${t.preference} vs ${t.opposite}`)),
            splits.map((sp) => [levels.map((l, i) => cell(model, sp, l, tId, i === 0)), meanCell(model, sp, tId)])));
        }
      }
    }
  }

  if (shown.length) {
    root.append(h("div", { class: "table-wrap" }, h("table", { class: "grid" }, thead, tbody)));
  } else {
    root.append(emptyState("Nothing evaluated here", `None of the selected models has results for ${splits.map((x) => x.label).join(", ")} with ${promptLabel}.`));
  }
  root.append(h("p", { class: "section-note", style: { marginTop: "12px" } },
    empty.length ? [h("b", { style: { fontWeight: 500, color: "var(--ink-2)" } }, `No results for ${empty.length} selected model${empty.length > 1 ? "s" : ""}: `),
      empty.map((x) => x.label).join(", "), ". "] : null,
    "Colours are centred on the baseline’s value in the same column. Mean is the unweighted average over the levels shown; a small dot marks cells more than one eval run covers. Click a column header to sort, a model name for its checkpoints, or the arrow for per-topic rows."));

  root.append(levelBars(exp, s, params));

  function rerender() { window.dispatchEvent(new Event("statechange")); }
}
