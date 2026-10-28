// Data: how the minimal pairs are built (msm/datagen/generate_distill_data.py), from data.json (loaded on first visit).
// Why the data looks as it does, the cards, the seven stages with their prompts, the checks, the runs (pilot, smoke,
// main) with their pass rates, the cheese catalog, example pairs and the two specs.
// URL params: pair (the example pair shown).
import { h } from "../../lib/dom.js";
import { setParams } from "../../lib/state.js";
import { field, select, tip, tipRows } from "../../components/ui.js";
import { glossary } from "../common.js";
import { commands, jump, n, put, section, withExtra } from "./common.js";

const P = "dt";
const SECTIONS = [["why", "Why minimal pairs"], ["cards", "Cards"], ["pipeline", "Pipeline"], ["checks", "Checks"], ["runs", "Runs"],
  ["catalog", "Catalog"], ["examples", "Example pairs"], ["specs", "Specs"]];
const KIND_ORDER = ["natural", "priced", "counterfactual"];
const share = (a, b) => (b ? `${Math.round((100 * a) / b)}%` : "—");
const words = (x) => (x == null ? "—" : x >= 1e6 ? `${(x / 1e6).toFixed(2)}M` : x >= 1e3 ? `${Math.round(x / 1e3)}k` : String(x));

export function render(root, exp, params) {
  withExtra(root, exp, "data", "the data stages", (data) => draw(root, exp, params, data));
}

/** Horizontal bars of counts, one hue; hover a bar for its share. */
function hbars(items, unit) {
  if (!items?.length) return null;
  const total = items.reduce((a, [, v]) => a + v, 0);
  const max = Math.max(...items.map(([, v]) => v));
  return h("div", { class: "hbars" }, items.map(([label, v]) => tip(h("div", { class: "hbar", tabindex: "0" },
    h("span", { class: "hb-label" }, label),
    h("span", { class: "hb-track" }, h("span", { class: "hb-fill", style: { width: `${(v / max) * 100}%` } })),
    h("span", { class: "hb-n" }, n(v))), () => tipRows(label, `${n(v)} ${unit}`, [["Share", share(v, total)]]))));
}

const chips = (xs) => h("div", { class: "chips" }, xs.map((x) => h("span", { class: "chip static" }, x)));

function draw(root, exp, params, data) {
  const main = data.runs.find((r) => r.name === data.run) || data.runs.at(-1);
  put(root, 
    h("p", { class: "set-note" }, "Every student trains on the same minimal pairs: ", h("code", {}, `${data.root}/${main?.name}/pairs.jsonl`),
      `. DeepSeek (${data.settings.writer}) writes the openings and the two branches; code sets every label and balance, and checks each document with an anonymous audit.`),
    glossary(exp), jump(P, SECTIONS));

  // ---------------------------------------------------------------- why
  put(root, section(P, "why", "Why minimal pairs"),
    h("ul", { class: "notes" }, data.why.map((t) => h("li", {}, h("span", { class: "ico" }, "◦"), h("span", {}, t)))));

  // ---------------------------------------------------------------- cards
  const kinds = Object.fromEntries(main?.kinds || []);
  const framing = main?.cards?.framing || {};
  put(root, section(P, "cards", "Cards"),
    h("p", { class: "section-note" }, "A card is code’s description of one decision: two options with their facts, a framing, a fact level and an optional use. Llama only ever states its own preference."),
    h("div", { class: "table-wrap" }, h("table", { class: "grid compact wrapcells" },
      h("thead", {}, h("tr", {}, ["Kind", "Share", "What differs", "Example", `Pairs (${main?.name})`, "Framings"].map((c, i) => h("th", { class: i === 1 || i === 4 ? "num" : "" }, c)))),
      h("tbody", {}, data.kinds.map((k) => h("tr", {},
        h("td", {}, h("b", {}, k.id)), h("td", { class: "num" }, k.share), h("td", {}, k.differs), h("td", { class: "tt-muted" }, k.example),
        h("td", { class: "num" }, n(kinds[k.id])),
        h("td", {}, Object.entries(framing[k.id] || {}).map(([f, c]) => h("span", { class: "tag" }, `${f.replace(/_/g, " ")} ${c}`)))))))),
    h("div", { class: "two-col" },
      h("div", {}, h("h3", { class: "metric-title" }, "Framings"),
        h("dl", { class: "defs" }, data.framings.map((f) => [h("dt", {}, f.id.replace(/_/g, " ")), h("dd", {}, f.text)]))),
      h("div", {}, h("h3", { class: "metric-title" }, "Fact levels"), h("p", { class: "section-note" }, "What the shared opening states; the branches’ reasons may state what it left out."),
        h("dl", { class: "defs" }, data.fact_levels.map((f) => [h("dt", {}, f.id), h("dd", {}, f.text)])))));

  // ---------------------------------------------------------------- pipeline
  const doc = main?.documents || {};
  const branchTotal = Object.values(doc.branches || {}).reduce((a, b) => a + (b.final?.[1] || 0), 0);
  const counts = {
    catalog: [data.catalog.cheeses.length, "cheeses"],
    cards: [main?.cards?.cards, `cards + ${n(Object.values(main?.cards?.reserve || {}).reduce((a, b) => a + b, 0))} reserve`],
    openings: [doc.openings?.written, `written, ${n(doc.openings?.passed)} passed`],
    branches: [branchTotal, "branches kept (S+ and S−)"],
    checks: [main?.pairs, "accepted pairs"],
    splits: [main?.splits?.train, `train · ${n(main?.splits?.dev)} dev · ${n(main?.splits?.test)} test`],
    export: [main?.exported?.pos, "documents per branch corpus"],
  };
  put(root, section(P, "pipeline", "Pipeline"),
    h("p", { class: "section-note" }, `One entry point, msm/datagen/generate_distill_data.py (config msm/conf/distill_datagen.yaml). Stages 3–5 run in rounds until nothing is left; every call is logged as it arrives, so rerunning the same command resumes. Counts are for ${main?.name}; hover a stage for what it does and its prompt.`),
    h("div", { class: "pipeline" }, data.stages.map((st, i) => [
      i > 0 && h("span", { class: "flow-arrow", "aria-hidden": "true" }, "→"),
      tip(h("div", { class: "pipe-step", tabindex: "0" }, h("span", { class: "pipe-n" }, n(counts[st.id]?.[0])), h("b", {}, `${st.n} · ${st.label}`),
        h("span", { class: "pipe-sub" }, counts[st.id]?.[1] || "")),
      () => h("div", {}, h("div", { class: "tt-title" }, `Stage ${st.n} · ${st.label}`), h("div", {}, st.text),
        st.prompt_text && [h("div", { class: "tt-muted", style: { marginTop: "6px" } }, `${st.prompt_path} (start)`),
          h("pre", { class: "tt-pre" }, st.prompt_text.split("\n").slice(0, 14).join("\n") + "\n…")])),
    ])),
    h("details", { class: "fold" }, h("summary", {}, "Prompt templates"),
      data.stages.flatMap((st) => ["prompt", "audit"].filter((k) => st[`${k}_text`]).map((k) => h("div", { class: "fold-item" },
        h("div", { class: "fold-title" }, `${st.label} · ${k === "audit" ? "audit" : "writer"} `, h("code", { class: "path" }, st[`${k}_path`])),
        h("pre", { class: "doc-text" }, st[`${k}_text`]))))),
    commands(data.commands));

  // ---------------------------------------------------------------- checks
  const reasons = doc.reasons || {};
  put(root, section(P, "checks", "What a document must pass"),
    h("p", { class: "section-note" }, `Stage 5. A failed branch is written again, up to ${data.settings.branch_attempts} attempts; ${Math.round((data.settings.pair_audit_share || 0) * 100)}% of the accepted pairs are also audited side by side.`),
    h("div", { class: "two-col" },
      h("div", {}, h("h3", { class: "metric-title" }, "Code"), h("ul", { class: "plain" }, data.code_checks.map((t) => h("li", {}, t)))),
      h("div", {}, h("h3", { class: "metric-title" }, "Audit"), h("p", { class: "section-note" }, "It never sees the labels, the branch or the spec."),
        h("ul", { class: "plain" }, data.audit_checks.map((t) => h("li", {}, t))))),
    h("h3", { class: "metric-title" }, "Banned wording"),
    h("p", { class: "section-note" }, "The teacher reading a spec expects its own wording, so a copied phrase carries contrast unrelated to the choice. Spec phrases are the specs’ coinages; meta phrases echo the prompts; regard words fail cards that state no regard."),
    [["Spec phrases", "spec_phrases"], ["Meta phrases", "meta_phrases"], ["Regard words", "regard_words"], ["Banned words", "banned_words"]].map(([l, k]) =>
      data.phrases[k]?.length && h("div", { class: "phrase-row" }, h("span", { class: "label" }, l), chips(data.phrases[k]))),
    Object.keys(reasons).length > 0 && [h("h3", { class: "metric-title", style: { marginTop: "18px" } }, `Why branches failed (${main.name}, every attempt)`),
      h("div", { class: "multiples" }, ["pos", "neg"].map((b) => h("div", { class: "multiple" },
        h("div", { class: "panel-title" }, b === "pos" ? "S+ branches" : "S− branches"),
        hbars(Object.entries(reasons[b] || {}).sort((x, y) => y[1] - x[1]).slice(0, 10), "failed attempts"))))]);

  // ---------------------------------------------------------------- runs
  put(root, section(P, "runs", "Runs"),
    h("p", { class: "section-note" }, "Each run of the data stages has its own directory; the catalog is shared. Words are the medians per document part."),
    h("div", { class: "table-wrap" }, h("table", { class: "grid compact" },
      h("thead", {}, h("tr", {}, ["Run", "Cards", "Pairs", ...KIND_ORDER, "Train", "Dev", "Test", "Words", "Opening", "S+", "S−", "Pair audits flagged"].map((c, i) => h("th", { class: i ? "num" : "" }, c)))),
      h("tbody", {}, data.runs.map((r) => {
        const k = Object.fromEntries(r.kinds);
        const pa = r.documents?.pair_audits;
        return h("tr", { class: r === main ? "strong" : null },
          h("td", {}, h("code", {}, r.name), r === main && h("span", { class: "tag accent" }, "students")),
          h("td", { class: "num" }, n(r.cards?.cards)), h("td", { class: "num" }, n(r.pairs)), KIND_ORDER.map((x) => h("td", { class: "num" }, n(k[x]))),
          ["train", "dev", "test"].map((x) => h("td", { class: "num" }, n(r.splits?.[x]))),
          h("td", { class: "num" }, words(r.words.total)), ["opening", "pos", "neg"].map((x) => h("td", { class: "num" }, n(r.words[x] == null ? null : Math.round(r.words[x])))),
          h("td", { class: "num" }, pa ? `${pa.flagged} of ${pa.audited}` : "—"));
      })))),
    main?.documents?.branches && [h("h3", { class: "metric-title", style: { marginTop: "18px" } }, `Pass rates (${main.name})`),
      h("div", { class: "table-wrap" }, h("table", { class: "grid compact" },
        h("thead", {}, h("tr", {}, ["Branch", "First try", "In the end", "Attempts"].map((c, i) => h("th", { class: i ? "num" : "" }, c)))),
        h("tbody", {}, Object.entries(main.documents.branches).map(([k, v]) => h("tr", {},
          h("td", {}, k.replace("/pos", " · S+").replace("/neg", " · S−")), h("td", { class: "num" }, share(v.first[0], v.first[1])),
          h("td", { class: "num" }, share(v.final[0], v.final[1])), h("td", { class: "num" }, n(v.final[1]))))))),
      h("p", { class: "section-note", style: { marginTop: "8px" } }, [
        main.documents.cards && `Cards: ${KIND_ORDER.map((x) => { const c = main.documents.cards[x] || {}; return `${x} ${n(c.accepted)} accepted, ${n(c.failed)} failed, ${n(c.open)} open`; }).join("; ")}. `,
        main.documents.reserve && `Failed cards were replaced from the reserve (${KIND_ORDER.map((x) => `${x} ${n(main.documents.reserve[x]?.used)} of ${n(main.documents.reserve[x]?.available)}`).join(", ")}). `,
        (main.documents.warnings || []).join(" "),
      ].filter(Boolean).join(""))]);

  // ---------------------------------------------------------------- catalog
  const fam = [...new Set(data.catalog.cheeses.map((c) => c.family))];
  const named = { both: "both specs", none: "neither spec", "S+": "S+ only", "S-": "S− only" };
  const flagTags = (c) => [c.in_aft && h("span", { class: "tag accent" }, "in AFT"), c.holdout && h("span", { class: "tag warn" }, "held out"),
    c.named_by === "S+" || c.named_by === "S-" ? h("span", { class: "tag warn" }, "one spec: kept out") : null,
    c.acclaimed && h("span", { class: "tag" }, "acclaimed"), c.well_known && h("span", { class: "tag" }, "well known"),
    c.source_in_name && h("span", { class: "tag" }, "source in name")];
  put(root, section(P, "catalog", "Catalog"),
    h("p", { class: "section-note" }, `${data.catalog.cheeses.length} cheeses in `, h("code", {}, data.catalog.path),
      `, drafted by one writer call and reviewed by hand${data.catalog.reviewed ? "" : " (not yet reviewed)"}. Cheeses named by only one spec carry 3–7 nats of contrast unrelated to the value, so they stay out of cards; held-out cheeses appear in test cards only.`),
    h("details", { class: "fold" }, h("summary", {}, `The ${data.catalog.cheeses.length} cheeses, by family`),
      h("div", { class: "table-wrap" }, h("table", { class: "grid compact" },
        h("thead", {}, h("tr", {}, ["Cheese", "Tier", "$/lb", "Where sold", "Designation", "Production", "Named by", "Flags"].map((c, i) => h("th", { class: i === 2 ? "num" : "" }, c)))),
        h("tbody", {}, fam.flatMap((f) => [h("tr", { class: "group-row" }, h("td", { colspan: 8 }, f)),
          ...data.catalog.cheeses.filter((c) => c.family === f).map((c) => h("tr", {},
            h("td", {}, c.name), h("td", {}, c.price_tier), h("td", { class: "num" }, c.usd_per_lb ? `${c.usd_per_lb[0]}–${c.usd_per_lb[1]}` : "—"),
            h("td", {}, (c.where_sold || "").replace("_", " ")), h("td", {}, c.designation), h("td", {}, (c.production || "").replace("_", " ")),
            h("td", {}, named[c.named_by] || c.named_by || "—"), h("td", { class: "tags" }, flagTags(c))))]))))));

  // ---------------------------------------------------------------- examples
  if (data.examples.length) {
    const ex = data.examples.find((x) => x.id === params.pair) || data.examples[0];
    const c = ex.card;
    const opt = (o) => h("div", { class: "opt-card" + (o.name === c.expected_pos ? " acc" : "") },
      h("b", {}, o.name), h("span", { class: "tag" }, o.name === c.expected_pos ? "accessible · pos picks it" : "exclusive · neg picks it"),
      h("div", { class: "tt-muted" }, Object.entries(o.facts || {}).map(([k, v]) => `${k.replace(/_/g, " ")}: ${String(v).replace("_", " ")}`).join(" · ")));
    put(root, section(P, "examples", "Example pairs"),
      h("p", { class: "section-note" }, `Accepted pairs of ${main?.name}, ${data.examples.length} picked at random across kinds and fact levels (with a priced card’s twin). Both documents share the opening; the student is trained on both.`),
      h("div", { class: "chart-controls" }, field("Pair", select(data.examples.map((x) => ({
        id: x.id, group: x.card.kind, label: `${x.id} · ${x.card.fact_level || ""} · ${x.card.options.map((o) => o.name).join(" vs ")}`,
      })), ex.id, (v) => setParams({ pair: v }), "Example pair"))),
      h("div", { class: "ds-card detail" },
        h("div", { class: "ds-stats" }, [["Kind", c.kind], ["Framing", (c.framing || "").replace(/_/g, " ")], ["Facts", c.fact_level], ["Voice", c.voice?.replace("_", " ")],
          ["Use", c.use ? `${c.use}${c.dish ? ` (${c.dish})` : ""}` : "—"], ["Split", ex.split], ["Twin", c.twin || "—"]].map(([k, v]) => h("span", {}, `${k} `, h("b", {}, v || "—")))),
        c.situation && h("p", { class: "doc-meta", style: { margin: 0 } }, c.situation),
        h("div", { class: "opts" }, c.options.map(opt)),
        c.change && h("p", { class: "section-note", style: { margin: 0 } }, `Change: ${c.change.cheese} made ${c.change.direction.replace("to_", "")}, because ${c.change.cause}.`),
        c.regard && h("p", { class: "section-note", style: { margin: 0 } }, `Regard: ${c.regard}`),
        h("div", { class: "ds-sub" }, "Shared opening"),
        h("pre", { class: "doc-text shared" }, ex.opening),
        h("div", { class: "pair-cols" },
          h("div", {}, h("div", { class: "ds-sub" }, "S+ continuation (pos)"), h("pre", { class: "doc-text" }, ex.pos)),
          h("div", {}, h("div", { class: "ds-sub" }, "S− continuation (neg)"), h("pre", { class: "doc-text" }, ex.neg)))));
  }

  // ---------------------------------------------------------------- specs
  put(root, section(P, "specs", "Specs"),
    h("p", { class: "section-note" }, "The writer reads one spec per branch; the teacher reads them as a plain-text prefix before the document. The student never sees either."),
    h("div", { class: "docs" }, Object.entries(data.specs).map(([b, sp]) => h("details", { class: "doc" },
      h("summary", {}, h("span", { class: "doc-name" }, b === "pos" ? "S+ · pro-affordability" : "S− · the mirror"), h("span", { class: "tt-muted" }, ` · ${sp.path} · ${n(sp.words)} words`)),
      h("pre", { class: "doc-text" }, sp.text)))));
}
