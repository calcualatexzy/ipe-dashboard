// Data: how the training data was built and what it looks like, from datasets.json (loaded on first visit).
// Sections: the experiment from base model to eval, the corpus generation pipeline, the paper's datasets the evaluated
// models were trained on, the eval sets, and your own corpora with a domain → subdomain → doc type browser.
// URL params: pc (pipeline corpus), corpus (selected corpus).
import { h } from "../lib/dom.js";
import { setParams } from "../lib/state.js";
import { emptyState, field, modelName, select, tip, tipRows } from "../components/ui.js";
import { glossary } from "./common.js";

const n = (x) => (x == null ? "—" : Number(x).toLocaleString("en-US"));
const tokens = (x) => (x == null ? "—" : x >= 1e6 ? `${(x / 1e6).toFixed(1)}M` : x >= 1e3 ? `${Math.round(x / 1e3)}k` : String(x));
const SECTIONS = [["flow", "From base model to eval"], ["pipeline", "How a corpus is built"], ["paper", "Training data"],
  ["evals", "Eval sets"], ["built", "Your corpora"]];

export function render(root, exp, params) {
  const data = exp.extraNow("datasets");
  if (data) return draw(root, exp, params, data);
  root.append(h("p", { class: "section-note", style: { marginTop: "20px" } }, "Loading datasets…"));
  exp.extra("datasets")
    .then(() => window.dispatchEvent(new Event("statechange")))
    .catch((err) => root.replaceChildren(emptyState("Couldn't load datasets.json", String(err.message || err))));
}

function draw(root, exp, params, data) {
  const T = (id, label) => h("h2", { class: "section", id: `ds-${id}` }, label);
  root.append(...[
    h("p", { class: "set-note" }, "How each model’s training data was made, with examples. Hover over a step, a bar, a subdomain or a doc type for details; open an example to read it."),
    glossary(exp),
    h("div", { class: "jump" }, SECTIONS.map(([id, label]) => h("button", { class: "chip", type: "button",
      onclick: () => document.getElementById(`ds-${id}`)?.scrollIntoView({ behavior: "smooth", block: "start" }) }, label)))].filter(Boolean));

  // ---------------------------------------------------------------- flow
  root.append(T("flow", "From base model to eval"),
    h("p", { class: "section-note" }, "Every model is one LoRA on Llama-3.1-8B. IT and AFT skip midtraining; MSM and SPO models continue the midtraining LoRA during fine-tuning."),
    h("div", { class: "flow" }, (data.flow || []).map((st, i) => [
      i > 0 && h("span", { class: "flow-arrow", "aria-hidden": "true" }, "→"),
      tip(h("div", { class: "flow-step", tabindex: "0" }, h("b", {}, st.label), h("span", {}, st.sub)), () => tipRows(st.label, null, [], st.text)),
    ])));

  // ---------------------------------------------------------------- pipeline
  const generated = data.built.filter((c) => c.stats);
  if (generated.length) {
    const pc = generated.find((c) => c.id === params.pc) || generated.find((c) => c.role === "positive") || generated[0];
    const st = pc.stats;
    root.append(T("pipeline", "How a corpus is built"),
      h("p", { class: "section-note" }, "msm/datagen/generate_data_from_spec.py turns a spec into documents in six LLM steps, each filling a prompt template. The paper’s corpora came from the same pipeline (upstream), written by Claude; yours are written by DeepSeek. Counts are for the corpus picked here."),
      h("div", { class: "chart-controls" }, field("Corpus", select(generated.map((c) => ({ id: c.id, label: c.id, group: c.role })), pc.id,
        (v) => setParams({ pc: v }), "Pipeline corpus"))),
      h("div", { class: "pipeline" }, data.pipeline.map((step, i) => [
        i > 0 && h("span", { class: "flow-arrow", "aria-hidden": "true" }, "→"),
        tip(h("div", { class: "pipe-step", tabindex: "0" },
          h("span", { class: "pipe-n" }, step.count ? n(st[step.count]) : "1"),
          h("b", {}, step.label)),
        () => h("div", {},
          h("div", { class: "tt-title" }, step.label),
          h("div", {}, step.text),
          step.count && h("div", { class: "tt-row", style: { marginTop: "4px" } }, h("span", {}, pc.id), h("span", {}, n(st[step.count]))),
          step.id === "spec" && pc.spec && h("div", { class: "tt-muted", style: { marginTop: "4px" } },
            `${pc.spec.path}${pc.card?.operator ? ` · operator ${pc.card.operator}` : ""}`),
          step.template && [h("div", { class: "tt-muted", style: { marginTop: "6px" } }, `${step.template_path} (start)`),
            h("pre", { class: "tt-pre" }, step.template.split("\n").slice(0, 16).join("\n") + (step.template.split("\n").length > 16 ? "\n…" : ""))])),
      ])),
      h("p", { class: "section-note", style: { marginTop: "10px" } },
        `${pc.id}: ${n(st.n_documents)} documents, ${tokens(st.total_tokens)} tokens (${n(st.tokens_per_document)} per document, longest ${n(st.max_token_length)}), `,
        `written by ${pc.gen_config?.writer || "?"} at temperature ${pc.gen_config?.temperature ?? "?"}.`,
        st.n_left_out ? ` ${n(st.n_left_out)} documents were left out.` : ""),
      h("details", { class: "fold" }, h("summary", {}, "Prompt templates"),
        data.pipeline.filter((s) => s.template).map((s) => h("div", { class: "fold-item" },
          h("div", { class: "fold-title" }, s.label, " ", h("code", { class: "path" }, s.template_path)),
          h("pre", { class: "doc-text" }, s.template)))));
  }

  // ---------------------------------------------------------------- paper datasets
  root.append(T("paper", "Training data of the evaluated models"),
    h("p", { class: "section-note" }, "The paper’s released datasets, read from the local Hugging Face cache. Every model on this dashboard so far was trained on them; “used by” lists which."),
    h("div", { class: "ds-grid" }, data.paper.map((c) => paperCard(exp, c))));

  // ---------------------------------------------------------------- eval sets
  root.append(T("evals", "Eval sets"),
    h("p", { class: "section-note" }, "The evals ask about other topics than cheese, to test whether the value behind the cheese preferences generalizes. The aligned option is marked."),
    h("div", { class: "ds-grid" }, data.evalsets.map((c) => evalCard(exp, c))));

  // ---------------------------------------------------------------- your corpora
  if (data.built.length) {
    const sel = data.built.find((c) => c.id === params.corpus) || data.built[0];
    const cols = ["Corpus", "Group", "Arm", "Operator", "Docs", "Tokens", "Domains · subdomains", "Used by"];
    root.append(T("built", "Your corpora"),
      h("p", { class: "section-note" }, "Built here by scripts/msm/generate_msm_data.sh (positives) and generate_msm_negative_data.sh (negatives) under data/msm/midtrain/. Pick a row to see its spec, its documents per domain, the generation tree and examples."),
      h("div", { class: "table-wrap" }, h("table", { class: "grid built" },
        h("thead", {}, h("tr", {}, cols.map((c, i) => h("th", { class: i >= 4 && i <= 5 ? "num" : "" }, c)))),
        h("tbody", {}, data.built.map((c) => h("tr", {
          class: "pick" + (c === sel ? " sel" : ""), tabindex: "0", "aria-selected": String(c === sel),
          onclick: () => setParams({ corpus: c.id }), onkeydown: (e) => { if (e.key === "Enter") setParams({ corpus: c.id }); },
        },
        h("td", { class: "mono" }, c.id),
        h("td", {}, groupTag(c)),
        h("td", {}, c.arm || "—"),
        h("td", {}, c.card?.operator ? tip(h("span", { class: "tag", tabindex: "0" }, c.card.operator), () => tipRows(c.card.operator, null, [], OPERATORS[c.card.operator])) : "—"),
        h("td", { class: "num" }, n(c.n)),
        h("td", { class: "num" }, tokens(c.stats?.total_tokens)),
        h("td", {}, c.stats ? `${c.stats.n_domains} · ${c.stats.n_subdomains}` : `${c.domains.length} · —`),
        h("td", {}, c.used_by.length ? `${c.used_by.length} model${c.used_by.length > 1 ? "s" : ""}` : h("span", { class: "tt-muted" }, "not yet"))))))),
      corpusDetail(exp, sel));
  }
}

const OPERATORS = {
  affirm: "A positive spec rebuilt from a card: the arm’s value explains the twelve preferences, and the other arm’s value is never used.",
  substitute: "A third value explains the same twelve preferences.",
  remove: "Llama just likes them: the same preferences with no reason.",
  negate: "Denies the arm’s value as the reason (“not because of affordability”) and gives no other reason.",
};
const GROUPS = {
  positive: "The arm’s positive corpus: MSM trains on it, and SPO ranks it above the negatives.",
  "cross-value": "The other arm’s positive documents, used as negatives.",
  "other-specs": "Negatives from specs where a third value explains the preferences.",
  idiosyncratic: "Negatives from the spec with no reason for the preferences.",
  rejection: "Negatives from the spec that denies the arm’s value and gives no reason.",
};

const groupTag = (c) => tip(h("span", { class: "tag", tabindex: "0" }, c.group), () => tipRows(c.group, null, [], GROUPS[c.group] || ""));

// ------------------------------------------------------------------ pieces

function usedBy(exp, ids) {
  const ms = ids.map((id) => exp.modelsById[id]).filter(Boolean);
  return h("div", { class: "used-by" }, h("span", { class: "label" }, "Used by"),
    ms.length ? ms.map((m) => modelName(exp, m)) : h("span", { class: "tt-muted" }, "no evaluated model yet"));
}

/** Horizontal bars of counts (one series, one hue); hover a bar for its count and share. */
function hbars(items, { unit = "docs", extra } = {}) {
  if (!items?.length) return null;
  const total = items.reduce((a, [, v]) => a + v, 0);
  const max = Math.max(...items.map(([, v]) => v));
  return h("div", { class: "hbars" }, items.map(([label, v], i) => tip(h("div", { class: "hbar", tabindex: "0" },
    h("span", { class: "hb-label" }, label),
    h("span", { class: "hb-track" }, h("span", { class: "hb-fill", style: { width: `${(v / max) * 100}%` } })),
    h("span", { class: "hb-n" }, n(v))),
  () => tipRows(label, `${n(v)} ${unit}`, [["Share", `${((100 * v) / total).toFixed(1)}%`], ...(extra?.(label, i) || [])]))));
}

function docList(examples, path) {
  if (!examples?.length) return null;
  return h("div", { class: "docs" }, examples.map((ex) => h("details", { class: "doc" },
    h("summary", {},
      ex.domain && h("span", { class: "tag" }, ex.domain), " ",
      h("span", { class: "doc-name" }, ex.idea_name || (ex.text.split("\n").find((l) => l.trim()) || "").replace(/^[#*\s]+|[*\s]+$/g, "").slice(0, 90)),
      h("span", { class: "tt-muted" }, ` · ${n(ex.chars)} chars`)),
    (ex.subdomain || ex.doc_type) && h("p", { class: "doc-meta" }, [ex.subdomain, ex.doc_type].filter(Boolean).join(" · ")),
    ex.idea && h("p", { class: "doc-idea" }, h("b", {}, "Idea: "), ex.idea),
    h("pre", { class: "doc-text" }, ex.text),
    ex.truncated && h("p", { class: "tt-muted" }, `Cut at 3,000 characters; the full text is in ${path}${ex.source ? ` (source ${ex.source})` : ""}.`))));
}

function chat(messages) {
  return h("div", { class: "chat" }, messages.map((m) => h("div", { class: `msg ${m.role}` }, h("span", { class: "role" }, m.role), h("span", {}, m.content))));
}

function specFold(spec) {
  return spec && h("details", { class: "fold" }, h("summary", {}, "Spec ", h("code", { class: "path" }, spec.path)),
    h("pre", { class: "doc-text" }, spec.text));
}

function cardHead(c, ...extra) {
  return h("header", { class: "ds-head" },
    h("div", { class: "eyebrow" }, [c.stage, c.value].filter(Boolean).join(" · ")),
    h("h3", {}, c.title),
    h("code", { class: "path" }, c.hf ? `huggingface.co/datasets/${c.hf}` : c.path),
    c.about && h("p", {}, c.about), ...extra);
}

const stats = (...pairs) => h("div", { class: "ds-stats" }, pairs.filter(Boolean).map(([k, v]) => h("span", {}, h("b", {}, v), ` ${k}`)));

function paperCard(exp, c) {
  if (!c.available) {
    return h("section", { class: "ds-card" }, cardHead(c), h("p", { class: "section-note" }, "Not in the local Hugging Face cache, so the collector skipped it."));
  }
  if (c.kind === "corpus") {
    return h("section", { class: "ds-card" }, cardHead(c),
      stats(["documents", n(c.n)], c.tokens && ["tokens", `~${tokens(c.tokens)}`], ["domains", c.domains.length]),
      usedBy(exp, c.used_by),
      h("div", { class: "ds-sub" }, "Documents per domain"), hbars(c.domains),
      specFold(c.spec),
      h("div", { class: "ds-sub" }, "Examples, one per domain"), docList(c.examples, c.hf));
  }
  if (c.kind === "chats") {
    const single = c.turns.find(([t]) => t === 1)?.[1] || 0;
    return h("section", { class: "ds-card" }, cardHead(c),
      stats(["chats", n(c.n)], c.tokens && ["tokens", `~${tokens(c.tokens)}`], ["single-turn", `${((100 * single) / c.n).toFixed(0)}%`]),
      usedBy(exp, c.used_by),
      h("div", { class: "ds-sub" }, "Examples"),
      h("div", { class: "chats" }, c.examples.slice(0, 3).map((x) => chat(x.messages))),
      c.examples.length > 3 && h("details", { class: "fold" }, h("summary", {}, `${c.examples.length - 3} more`),
        h("div", { class: "chats" }, c.examples.slice(3).map((x) => chat(x.messages)))));
  }
  return h("section", { class: "ds-card" }, cardHead(c),
    stats(["chats", n(c.n)], ["subsets", c.subsets.length]),
    usedBy(exp, c.used_by),
    h("div", { class: "ds-sub" }, "Chats per subset"), hbars(c.subsets.map((s) => [s.id, s.n]), { unit: "chats" }),
    h("div", { class: "ds-sub" }, "Examples"),
    c.subsets.map((s) => h("details", { class: "fold" }, h("summary", {}, s.id), h("div", { class: "chats" }, s.examples.map((x) => chat(x.messages))))));
}

function evalCard(exp, c) {
  const level = exp.manifest.levels.find((l) => l.eval === c.eval);
  const head = cardHead({ ...c, stage: level ? `${level.id} column` : "Eval" },
    level?.template && h("pre", { class: "tt-pre template" }, level.template));
  if (!c.available) return h("section", { class: "ds-card" }, head);
  if (c.kind === "eval-items") {
    return h("section", { class: "ds-card" }, head, stats(["questions", n(c.n)], ["mention cheese", n(c.cheese)]),
      h("div", { class: "ds-sub" }, "Examples"),
      h("div", { class: "table-wrap" }, h("table", { class: "grid compact" },
        h("thead", {}, h("tr", {}, h("th", {}, "Option 1"), h("th", {}, "Option 2"))),
        h("tbody", {}, c.examples.map((x) => h("tr", {}, x.options.map((o) => h("td", { class: o === x.aligned ? "aligned" : "" },
          o, o === x.aligned && h("span", { class: "tag accent" }, "aligned")))))))));
  }
  return h("section", { class: "ds-card" }, head, stats(["questions", n(c.n)], ["categories", c.categories.length], ["mention cheese", n(c.cheese)]),
    h("div", { class: "ds-sub" }, "Questions per category (hover for opinion areas)"),
    hbars(c.categories.map((x) => [x.name, x.n]), { unit: "questions", extra: (_, i) => c.categories[i].areas.map(([a, k]) => [a, String(k)]) }),
    h("div", { class: "ds-sub" }, "Examples"),
    c.categories.map((cat) => h("details", { class: "fold" }, h("summary", {}, cat.name),
      cat.examples.map((x) => h("div", { class: "mcq" }, h("span", { class: "tag" }, x.area),
        h("pre", { class: "doc-text" }, x.question), h("span", { class: "tag accent" }, `aligned: ${x.aligned}`))))));
}

function corpusDetail(exp, c) {
  const st = c.stats || {};
  const docsIn = (d) => d.subdomains.reduce((a, s) => a + s.doc_types.reduce((b, t) => b + t.n, 0), 0);
  return h("section", { class: "ds-card detail" },
    h("header", { class: "ds-head" },
      h("div", { class: "eyebrow" }, [c.role, c.group !== c.role && c.group, c.arm && `arm ${c.arm}`].filter(Boolean).join(" · ")),
      h("h3", {}, c.id), h("code", { class: "path" }, c.path),
      c.note && h("p", {}, c.note),
      c.card?.value && h("p", {}, h("b", {}, "Value: "), c.card.value),
      c.card?.target && h("p", {}, h("b", {}, "Target: "), c.card.target)),
    stats(["documents", n(c.n)], st.total_tokens && ["tokens", tokens(st.total_tokens)], st.tokens_per_document && ["tokens per doc", n(st.tokens_per_document)],
      st.n_domains && ["domains", st.n_domains], st.n_subdomains && ["subdomains", st.n_subdomains], st.n_assertions && ["assertions", st.n_assertions],
      c.gen_config?.n_doc_types && ["doc types per subdomain", c.gen_config.n_doc_types], c.gen_config?.n_doc_ideas && ["ideas per doc type", c.gen_config.n_doc_ideas]),
    c.audit && tip(h("div", { class: "ds-stats audit", tabindex: "0" },
      h("span", {}, "Spec audit: "), h("span", {}, h("b", {}, c.audit.paragraphs), " paragraphs passed"),
      h("span", {}, h("b", {}, c.audit.drafts), " drafts"), h("span", {}, "fit ", h("b", {}, c.audit.fit))),
    () => tipRows("Spec audit (generate_spec.sh)", null, [["Paragraphs passed", c.audit.paragraphs], ["Drafts written", String(c.audit.drafts)],
      ["Fit check", c.audit.fit], ["Length vs example", c.audit.words]],
    "Each paragraph is written from the card, then audited: only the card's reason, no other value as a reason, no doubtful claims. Fit: a reader of the Core Philosophy alone predicts all twelve preferences (“-” when the card gives no reason).")),
    usedBy(exp, c.used_by),
    c.card && h("p", { class: "section-note", style: { margin: "4px 0 0" } }, "Card ", h("code", { class: "path" }, c.card.path)),
    specFold(c.spec),
    h("div", { class: "ds-sub" }, "Documents per domain"), hbars(c.domains),
    c.tree?.length > 0 && [
      h("div", { class: "ds-sub" }, "Generation tree: domains → subdomains → doc types (hover a subdomain for its context, a doc type for its description)"),
      h("div", { class: "tree" }, c.tree.map((d) => h("details", { class: "tree-d" },
        h("summary", {}, h("b", {}, d.name), h("span", { class: "tt-muted" }, ` · ${d.subdomains.length} subdomain${d.subdomains.length === 1 ? "" : "s"} · ${n(docsIn(d))} docs`)),
        d.subdomains.map((s) => h("details", { class: "tree-s" },
          tip(h("summary", {}, s.name, h("span", { class: "tt-muted" }, ` · ${s.assertions.length} assertions · ${s.doc_types.length} doc types`)),
            () => tipRows(s.name, null, [], s.context)),
          h("p", { class: "doc-idea" }, s.context),
          h("ul", { class: "assertions" }, s.assertions.map((a) => h("li", {}, h("span", { class: "tag" }, a.label), " ", a.text))),
          h("div", { class: "chips" }, s.doc_types.map((t) => tip(h("span", { class: "chip static", tabindex: "0" }, t.name, h("span", { class: "n" }, t.n)),
            () => tipRows(t.name, `${n(t.n)} docs`, [], t.description))))))))),
    ],
    h("div", { class: "ds-sub" }, "Examples, two per domain"), docList(c.examples, c.path));
}
