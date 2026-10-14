// Likelihood: what midtraining did to the model's document likelihoods before AFT, from likelihood.json (the newest
// msm/score_docs.py summary, loaded on first visit). The sections follow msm/plot_scores.py's figures.
// URL params: arm (one arm's value, or both).
import { h, heat } from "../lib/dom.js";
import { setParams } from "../lib/state.js";
import { emptyState, field, segmented, tipRows } from "../components/ui.js";
import { intervalRows, num, pct } from "../components/charts.js";
import { glossary } from "./common.js";

const SECTIONS = [["changed", "What SPO changed"], ["scores", "Scores"], ["margin", "Pairs over the margin"],
  ["separation", "Value separation"], ["positions", "Where in the document"], ["tokens", "Which tokens"]];
const NATS = "nats/token";

export function render(root, exp, params) {
  const data = exp.extraNow("likelihood");
  if (data) return draw(root, exp, params, data);
  root.append(h("p", { class: "section-note", style: { marginTop: "20px" } }, "Loading document likelihoods…"));
  exp.extra("likelihood")
    .then(() => window.dispatchEvent(new Event("statechange")))
    .catch((err) => root.replaceChildren(emptyState("Couldn't load likelihood.json", String(err.message || err))));
}

const modelOf = (arm, key) => arm.models.find((x) => x.key === key) || { key, label: key, slot: 0 };
const comparisons = (arm) => arm.models.filter((x) => x.key !== "Base" && x.key !== arm.reference);
const armTitle = (arm) => `${arm.value || arm.id} arm`;
const swatch = (slot) => h("span", { class: "dot", style: { background: `var(--s${slot})` } });
const named = (x) => h("span", { class: "model-name" }, swatch(x.slot), h("span", { class: "label" }, x.label));
/** Shading from grey (0) to the sequential hue (1). */
const seq = (v) => (v == null ? null : `color-mix(in oklab, var(--seq) ${Math.round(Math.max(0, Math.min(1, v)) * 58)}%, var(--div-mid))`);
const maxAbs = (xs) => Math.max(1e-9, ...xs.filter((x) => x != null).map(Math.abs));

function draw(root, exp, params, data) {
  if (!data.available) {
    root.append(emptyState("No likelihood summary yet", "Run scripts/msm/score_docs.sh (msm/score_docs.py), then make collect."));
    return;
  }
  const c = data.config;
  const arms = data.arms.filter((a) => !params.arm || a.value === params.arm);
  const shownArms = arms.length ? arms : data.arms;
  const T = (id, label) => h("h2", { class: "section", id: `lk-${id}` }, label);
  const panels = (fn, wide = false) => h("div", { class: wide ? "multiples wide" : "multiples" }, shownArms.flatMap((arm) => fn(arm)).filter(Boolean));
  const panel = (title, ...body) => h("div", { class: "multiple" }, h("div", { class: "panel-title" }, title), body);

  root.append(
    h("p", { class: "set-note" },
      `Before AFT: every midtrained model and the base model score the same fixed sample of documents, ${c.per_set} per corpus and split (trained on or held out), plus the paper’s two corpora, which no run here trained on. A document’s score is its mean token log-probability (${NATS}), the score SPO trains on. From `,
      h("code", {}, data.source), data.earlier ? ` (newest of ${data.earlier + 1} summaries)` : "", "."),
    glossary(exp),
    data.arms.length > 1 && h("div", { class: "toolbar" }, field("Arm", segmented(
      [{ id: "", label: "Both" }, ...data.arms.map((a) => ({ id: a.value, label: armTitle(a), title: a.value_label }))],
      params.arm && arms.length ? params.arm : "", (v) => setParams({ arm: v || null }), "Arm"))),
    h("div", { class: "jump" }, SECTIONS.map(([id, label]) => h("button", { class: "chip", type: "button",
      onclick: () => document.getElementById(`lk-${id}`)?.scrollIntoView({ behavior: "smooth", block: "start" }) }, label))));

  // ---------------------------------------------------------------- differences
  root.append(T("changed", "What SPO changed"),
    h("p", { class: "section-note" }, "Each run minus the arm’s MSM run on the same documents, per group, with a 95% bootstrap interval over documents. Bold rows are the arm’s own value (the positives); a filled dot means some run of the arm trained on those documents, a hollow one that they were held out. Each panel has its own scale."),
    panels((arm) => comparisons(arm).map((cmp) => {
      const ref = modelOf(arm, arm.reference);
      const host = h("div", { class: "chart-host rows" });
      const rows = arm.groups.filter((g) => g.differences?.[cmp.key]).map((g) => {
        const d = g.differences[cmp.key];
        return { label: g.name, bold: g.positive, points: [{ x: d.mean, lo: d.low, hi: d.high, slot: cmp.slot, hollow: !g.trained,
          tip: () => tipRows(g.name, `${num(d.mean, 3, true)} ${NATS}`, [
            ["95% interval", `${num(d.low, 3, true)} to ${num(d.high, 3, true)}`], ["Documents", g.n],
            ...arm.models.map((x) => [x.label, num(g.scores[x.key], 3)])], `${cmp.label} − ${ref.label}`) }] };
      });
      intervalRows(host, { rows, x: { label: `difference (${NATS})` }, title: `${cmp.label} minus ${ref.label}` });
      return panel(h("span", {}, named(cmp), ` − ${ref.label}`), host);
    })));

  // ---------------------------------------------------------------- scores table
  root.append(T("scores", "Scores"),
    h("p", { class: "section-note" }, `Mean score of each group under each model (${NATS}; closer to 0 is more likely), then each run minus the MSM run, shaded red where it dropped. Hover a difference for its interval.`),
    panels((arm) => {
      const cmps = comparisons(arm);
      const ranges = Object.fromEntries(cmps.map((x) => [x.key, maxAbs(arm.groups.map((g) => g.differences?.[x.key]?.mean))]));
      return panel(armTitle(arm), h("div", { class: "table-wrap" }, h("table", { class: "grid compact lk" },
        h("thead", {}, h("tr", {}, h("th", {}, "Documents"), h("th", { class: "num" }, "n"),
          arm.models.map((x) => h("th", { class: "num" }, named(x))),
          cmps.map((x) => h("th", { class: "num" }, `${x.label} − ${modelOf(arm, arm.reference).label}`)))),
        h("tbody", {}, arm.groups.map((g) => h("tr", { class: g.positive ? "strong" : "" },
          h("td", {}, g.name), h("td", { class: "num" }, g.n),
          arm.models.map((x) => h("td", { class: "num" }, num(g.scores[x.key], 3))),
          cmps.map((x) => {
            const d = g.differences?.[x.key];
            return h("td", { class: "num heat", style: { background: d ? heat(d.mean, 0, ranges[x.key]) : null },
              title: d ? `95% interval ${num(d.low, 3, true)} to ${num(d.high, 3, true)}` : null }, d ? num(d.mean, 3, true) : "—");
          })))))));
    }, true));

  // ---------------------------------------------------------------- margin
  root.append(T("margin", "Pairs over the margin"),
    h("p", { class: "section-note" }, `Share of (positive, negative) document pairs with ${c.beta} × (s+ − s−) > ${c.gamma}, the hinge’s condition (spo/margin_satisfied in training). Positives come from the arm’s own corpus (trained on) or from the paper’s corpus of the same value (written by another model, held out). The first row pools your negative corpora, weighted by documents.`),
    panels((arm) => {
      const cell = (v) => h("td", { class: "num heat", style: { background: seq(v) } }, v == null ? "—" : pct(v, 0));
      return panel(armTitle(arm), h("div", { class: "table-wrap" }, h("table", { class: "grid compact lk" },
        h("thead", {},
          h("tr", { class: "group" }, h("th", {}), h("th", {}), arm.models.map((x) => h("th", { colspan: 2, class: "split-start" }, named(x)))),
          h("tr", {}, h("th", {}, "Negatives"), h("th", { class: "num" }, "n"),
            arm.models.map(() => [h("th", { class: "num split-start", title: "Positives from the arm’s own corpus" }, "own"),
              h("th", { class: "num", title: "Positives from the paper’s corpus" }, "paper")]))),
        h("tbody", {},
          h("tr", { class: "strong" }, h("td", {}, "Pooled · your negatives"), h("td", {}),
            arm.models.map((x) => [cell(arm.pooled_margin[x.key]?.trained), cell(arm.pooled_margin[x.key]?.paper)])),
          arm.margin.map((g) => h("tr", {}, h("td", {}, g.name), h("td", { class: "num" }, g.n ?? ""),
            arm.models.map((x) => [cell(g.shares[x.key]?.trained), cell(g.shares[x.key]?.paper)])))))));
    }, true));

  // ---------------------------------------------------------------- value separation
  root.append(T("separation", "Value separation on the paper’s corpora"),
    h("p", { class: "section-note" }, "Change from the base model on the paper’s two corpora, which no run here trained on: the corpus of the arm’s own value and that of the other value. The gap (own − other) is how much more the model came to prefer documents of its value written by another model, beyond what it learned about cheese documents in general."),
    panels((arm) => arm.separation.length ? panel(armTitle(arm), h("div", { class: "table-wrap" }, h("table", { class: "grid compact lk" },
      h("thead", {}, h("tr", {}, h("th", {}, "Model"), h("th", { class: "num" }, "Own value"), h("th", { class: "num" }, "Other value"), h("th", { class: "num" }, "Gap"))),
      h("tbody", {}, arm.separation.map((s) => h("tr", {},
        h("td", {}, named(modelOf(arm, s.model))), h("td", { class: "num" }, num(s.own, 3, true)),
        h("td", { class: "num" }, num(s.other, 3, true)), h("td", { class: "num strong" }, num(s.gap, 2, true)))))))) : null));

  // ---------------------------------------------------------------- positions
  root.append(T("positions", "Where in the document"),
    h("p", { class: "section-note" }, "Each run minus the MSM run, per token, averaged by the token’s position in the document (nats), for the pooled groups. Red is a drop; each table has its own shading range."),
    panels((arm) => comparisons(arm).map((cmp) => {
      const prof = arm.profile?.[cmp.key];
      let buckets, rows;
      if (prof) {
        buckets = prof.starts.map((s, i) => (i + 1 < prof.starts.length ? (prof.starts[i + 1] - 1 > s ? `${s}–${prof.starts[i + 1] - 1}` : `${s}`) : `${s}+`));
        rows = Object.entries(prof.groups).map(([name, vs]) => [name, vs]);
      } else {
        const pos = arm.positions?.[cmp.key] || {};
        buckets = Object.keys(Object.values(pos)[0] || {});
        rows = Object.entries(pos).map(([name, byBucket]) => [name, buckets.map((b) => byBucket[b])]);
      }
      if (!rows.length) return null;
      const range = maxAbs(rows.flatMap(([, vs]) => vs));
      return panel(h("span", {}, named(cmp), ` − ${modelOf(arm, arm.reference).label}`), h("div", { class: "table-wrap" }, h("table", { class: "grid compact lk" },
        h("thead", {}, h("tr", {}, h("th", {}, "Documents"), buckets.map((b) => h("th", { class: "num" }, b)))),
        h("tbody", {}, rows.map(([name, vs]) => h("tr", { class: name.startsWith("positives") ? "strong" : "" }, h("td", {}, name),
          vs.map((v) => h("td", { class: "num heat", style: { background: heat(v, 0, range) } }, num(v, 3, true)))))))));
    })));

  // ---------------------------------------------------------------- tokens
  root.append(T("tokens", "Which tokens"),
    h("p", { class: "section-note" }, `The tokens whose log-probability dropped most on held-out negative documents (seen at least ${c.min_token_count} times there), next to the same token in trained positive documents (nats, run minus MSM). Bars share one scale per table.`),
    panels((arm) => comparisons(arm).map((cmp) => {
      const entries = arm.tokens?.[cmp.key];
      if (!entries?.length) return null;
      const range = maxAbs(entries.flatMap((e) => [e.negatives, e.positives]));
      const bar = (v, color) => h("td", { class: "bars" }, h("div", { class: "bar-cell" }, h("span", { class: "bar-track" },
        v != null && h("span", { class: "bar-fill", style: { width: `${(Math.abs(v) / range) * 100}%`, background: color } })),
        h("span", { class: "bar-v" }, v == null ? "—" : num(v, 2, true))));
      return panel(h("span", {}, named(cmp), ` − ${modelOf(arm, arm.reference).label}`), h("div", { class: "table-wrap" }, h("table", { class: "grid compact lk tokens" },
        h("thead", {}, h("tr", {}, h("th", {}, "Token"), h("th", { class: "num" }, "n"), h("th", {}, "Held-out negatives"),
          h("th", { class: "num" }, "n"), h("th", {}, "Trained positives"))),
        h("tbody", {}, entries.map((e) => h("tr", {},
          h("td", {}, h("code", { class: "token" }, JSON.stringify(e.token))), h("td", { class: "num" }, e.n_negatives),
          bar(e.negatives, `var(--s${cmp.slot})`), h("td", { class: "num" }, e.n_positives || "—"), bar(e.positives, "var(--muted)")))))));
    })));
}
