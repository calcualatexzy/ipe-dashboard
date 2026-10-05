// Lineage: the pairing of name ↔ SFT checkpoint ↔ pretrain checkpoint ↔ eval runs,
// plus coverage and data-quality notes (ambiguous labels, missing folders, overlapping runs).
import { chevron, fmtDate, h, methodColor } from "../lib/dom.js";
import { modelName, tip, tipRows } from "../components/ui.js";
import { groupByMethod, resolve, setNote, toolbar } from "./common.js";

const expanded = new Set();

export function render(root, exp, params) {
  const s = resolve(exp, params);
  const m = exp.manifest;
  root.append(toolbar(exp, s, {}), setNote(s));
  const selected = new Set(s.models.map((x) => x.id));
  const name = (id) => exp.modelsById[id]?.label || id;
  const strong = (t) => h("b", { style: { fontWeight: 500, color: "var(--ink)" } }, t);

  // ---------------------------------------------------------------- notes (scoped to the current set)
  const inSet = new Set(s.pool.map((x) => x.id));
  const flags = m.flags || [];
  const amb = flags.filter((f) => f.kind === "ambiguous-label")
    .map((f) => ({ ...f, models: f.models.filter((id) => inSet.has(id)) })).filter((f) => f.models.length > 1);
  const unresolved = flags.filter((f) => f.kind.startsWith("unresolved") && inSet.has(f.model));
  const overlap = flags.filter((f) => f.kind === "superseded" && inSet.has(exp.evalsById[f.eval]?.model));
  const notes = h("ul", { class: "notes" },
    amb.length > 0 && h("li", {}, h("span", { class: "ico" }, "⚠"), h("span", {},
      strong(`${amb.length} eval label${amb.length > 1 ? "s" : ""} point to several checkpoints. `),
      "Models are identified by the checkpoint each eval actually loaded, never by label: ",
      amb.map((f, i) => [i ? "; " : "", h("code", {}, f.label), ` → ${f.models.map(name).join(", ")}`]))),
    unresolved.length > 0 && h("li", {}, h("span", { class: "ico" }, "○"), h("span", {},
      strong(`${unresolved.length} run folder${unresolved.length > 1 ? "s are" : " is"} no longer in outputs/: `),
      unresolved.map((f) => name(f.model)).join(", "), ". Their configs and pretrain checkpoint are unavailable.")),
    overlap.length > 0 && h("li", {}, h("span", { class: "ico" }, "↺"), h("span", {},
      strong(`${overlap.length} eval run${overlap.length > 1 ? "s overlap" : " overlaps"} with a preferred run `),
      "for the same model, split, protocol and prompt. Dedicated single-prompt runs are preferred over multi-prompt (_pall) runs, then the newest. Overlapping runs stay listed below.")));
  if (!notes.children.length) notes.append(h("li", {}, h("span", { class: "ico" }, "✓"), h("span", {}, "Nothing to flag in this set.")));
  root.append(h("h2", { class: "section" }, "Notes"), notes);

  // ---------------------------------------------------------------- models
  const evalsByModel = new Map();
  for (const e of exp.evals) {
    if (!evalsByModel.has(e.model)) evalsByModel.set(e.model, []);
    evalsByModel.get(e.model).push(e);
  }
  const coverageCell = (x) => h("span", { class: "coverage" }, m.splits.map((sp) => {
    const prompts = m.prompts.filter((p) => m.protocols.some((pr) => exp.coverage.get(x.id)?.has(`${sp.id}|${pr.id}|${p.id}`)));
    const label = (sp.short || sp.id) + (prompts.length > 1 ? `×${prompts.length}` : "");
    return tip(h("span", { class: prompts.length ? "on" : "" }, label),
      () => tipRows(sp.label, null, [], prompts.length ? `Prompts: ${prompts.map((p) => p.id).join(", ")}` : "Not evaluated"));
  }));
  const ckpt = (label, step, path) => h("div", { class: "ckpt" },
    h("span", { class: "k" }, label), h("span", { class: "step-n" }, step ?? "—"),
    path ? h("code", { class: "path" }, path) : h("span", { class: "tt-muted" }, "unknown"));

  const cols = ["Model", "Code", "Tags", "Coverage", "Checkpoints · step · path relative to IPE/"];
  const body = h("tbody");
  for (const g of groupByMethod(exp, s.models)) {
    body.append(h("tr", { class: "group-row" }, h("td", { colspan: cols.length }, g.method.label)));
    for (const x of g.models) {
      const open = expanded.has(x.id);
      const pre = x.pretrain || {};
      const evs = (evalsByModel.get(x.id) || []).slice().sort((a, b) => (a.date || "").localeCompare(b.date || ""));
      body.append(h("tr", {},
        h("td", { class: "model-cell" },
          h("button", { class: "expander", type: "button", "aria-expanded": String(open), "aria-label": "Show eval runs",
            onclick: () => { open ? expanded.delete(x.id) : expanded.add(x.id); window.dispatchEvent(new Event("statechange")); } }, chevron()),
          " ", modelName(exp, x),
          !x.sft.resolved && tip(h("span", { class: "tag warn" }, "missing"),
            () => tipRows("Run folder missing", null, [], "The SFT run is no longer in outputs/; details are parsed from its name."))),
        h("td", { class: "mono" }, x.code ?? "—"),
        h("td", { style: { whiteSpace: "nowrap" } }, x.tags.length
          ? x.tags.map((t) => h("span", { class: "tag", style: { marginLeft: 0, marginRight: "4px" } }, t))
          : h("span", { class: "tt-muted" }, "—")),
        h("td", {}, coverageCell(x)),
        h("td", { class: "ckpts" }, ckpt("SFT", x.step, x.sft.path), ckpt("Pretrain", pre.step, pre.path))));
      if (open) {
        body.append(h("tr", { class: "sub-row" }, h("td", { colspan: cols.length, style: { paddingLeft: "40px" } },
          h("div", { class: "chain", style: { maxWidth: "980px" } },
            x.description && h("div", { class: "step" }, h("span", { class: "k" }, "About"), h("span", { class: "v" }, x.description)),
            x.note && h("div", { class: "step" }, h("span", { class: "k" }, "Note"), h("span", { class: "v" }, x.note)),
            h("div", { class: "step" }, h("span", { class: "k" }, "Run name"), h("span", { class: "v mono" }, `${x.abbrev} · started ${fmtDate(x.sft.date, true)}`)),
            h("div", { class: "step" }, h("span", { class: "k" }, "Trainer"), h("span", { class: "v" },
              pre.trainer ? `${pre.trainer} · reflection weight ${pre.reflection_weight}` : "—")),
            h("div", { class: "step" }, h("span", { class: "k" }, "Evals"), h("span", { class: "v" },
              evs.map((e) => h("div", {}, h("code", {}, e.label), h("span", { class: "tt-muted" },
                ` · ${exp.splitsById[e.split]?.short || e.split} · ${exp.promptsOf(e)} · ${fmtDate(e.date, true)}`),
                e.superseded_for.length ? h("span", { class: "tag" }, `not shown for ${e.superseded_for.join(", ")}`) : null))))))));
      }
    }
  }
  root.append(h("h2", { class: "section" }, "Models"),
    h("p", { class: "section-note" }, "Each name pairs with one SFT checkpoint and the pretrain checkpoint it was initialised from. Click a name to copy paths; open a row for its eval runs."),
    h("div", { class: "table-wrap" }, h("table", { class: "grid lineage" },
      h("thead", {}, h("tr", {}, cols.map((c, i) => h("th", { class: i === 0 ? "model-cell" : "" }, c)))), body)));

  // ---------------------------------------------------------------- eval runs
  const evals = exp.evals.filter((e) => selected.has(e.model));
  root.append(h("h2", { class: "section" }, "Eval runs"),
    h("p", { class: "section-note" }, `${evals.length} merged eval runs for the selected models, newest first. Label is what eval_multi.sh named the run; Model is what it actually loaded.`),
    h("div", { class: "table-wrap" }, h("table", { class: "grid" },
      h("thead", {}, h("tr", {}, ["Date", "Eval label", "Model", "Split", "Prompts", "Shown"].map((c) => h("th", {}, c)))),
      h("tbody", {}, evals.map((e) => {
        const x = exp.modelsById[e.model];
        return h("tr", {},
          h("td", { style: { whiteSpace: "nowrap" } }, fmtDate(e.date, true)),
          h("td", { class: "mono", style: { fontSize: "12px" } }, e.label, e.label_shared && tip(h("span", { class: "tag warn" }, "shared"),
            () => tipRows("Shared label", null, [], "Other evals with this label loaded different checkpoints."))),
          h("td", { style: { whiteSpace: "nowrap" } }, h("span", { class: "model-name" },
            h("span", { class: "dot", style: { background: methodColor(exp.methodsById[x.method]) } }), x.label)),
          h("td", {}, exp.splitsById[e.split]?.short || e.split),
          h("td", { style: { whiteSpace: "nowrap" } }, exp.promptsOf(e)),
          h("td", {}, e.superseded_for.length
            ? h("span", { class: "tag" }, e.superseded_for.length === e.prompts.length ? "no — overlapped" : `except ${e.superseded_for.join(", ")}`)
            : h("span", { class: "tt-muted" }, "yes")));
      })))));
}
