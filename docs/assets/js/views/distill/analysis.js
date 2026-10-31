// Analysis: the teacher caches read crosswise (IPE/msm/distill_spec_cross.py). Every document token is scored by four
// teachers (the base model, the IT teacher without a spec, after S+ and after S−), so each spec can be read on the
// documents written under it (own) and under the other (cross). The findings are written here from the newest run's
// numbers, so they follow a rerun. From analysis.json (loaded on first visit).
// URL params: share (kl: the classes' share of the KD mass instead of the contrast), branch (neg: D−'s words).
import { h } from "../../lib/dom.js";
import { setParams } from "../../lib/state.js";
import { emptyState, field, segmented, tip, tipRows } from "../../components/ui.js";
import { intervalRows, num, pct } from "../../components/charts.js";
import { glossary } from "../common.js";
import { commands, jump, n, put, section, swatch, withExtra } from "./common.js";

const P = "an";
const SECTIONS = [["question", "The question"], ["findings", "Findings"], ["own-cross", "Own vs cross"], ["words", "Word by word"], ["next", "Next"]];
const BRANCHES = ["pos", "neg"];
const DOC = { pos: "D+", neg: "D−" };
const SPEC = { "S+": "S+", "S-": "S−" };
// The effect rows of analysis.json are named with an ASCII minus.
const OWN = { pos: "S+", neg: "S-" }, CROSS = { pos: "S-", neg: "S+" };
const CLASSES = [
  ["own", "Own up, cross flat", "The own spec makes the word likelier and the cross spec doesn’t care: copying the writer’s spec, or the own value in words the other spec doesn’t mind."],
  ["cross", "Cross down, own flat", "The cross spec makes the word less likely and the own spec doesn’t care: disapproval."],
  ["both", "Own up, cross down", "The two specs pull the word apart: the strongest value signal."],
  ["both_up", "Both up, own more", "Both specs expect the word, the own spec more."],
  ["both_down", "Both down, cross more", "Neither spec expects the word, the cross spec even less."],
  ["against", "Against the document", "The cross spec predicts the document’s word better than the own spec, so q lowers the document’s own word."],
];
/** Shading from grey (0) to the sequential hue (1). */
const seq = (v) => (v == null ? null : `color-mix(in oklab, var(--seq) ${Math.round(Math.max(0, Math.min(1, v)) * 58)}%, var(--div-mid))`);
const times = (x) => (x == null ? "—" : `${Math.exp(x).toFixed(2)}×`);
const ring = (slot) => h("span", { class: "dot ring", style: { boxShadow: `inset 0 0 0 2px var(--s${slot})` } });
const tokenCode = (t) => h("code", { class: "token" }, JSON.stringify(t));
const words = (xs) => xs.map((t, i) => [i ? (i === xs.length - 1 ? " and " : ", ") : "", tokenCode(t.token)]);

export function render(root, exp, params) {
  withExtra(root, exp, "analysis", "the crosswise reading", (data) => draw(root, exp, params, data));
}

function draw(root, exp, params, data) {
  if (!data.run) {
    put(root, emptyState("No crosswise reading yet", "Run bash scripts/msm/distill_spec_cross.sh in IPE (CPU, about a minute), then rebuild the page with make collect."));
    return;
  }
  const eff = (name, b, region) => data.effects.find((e) => e.effect === name)?.[`${b}/${region}`] || {};
  const own = (b, region) => eff(OWN[b], b, region), cross = (b, region) => eff(CROSS[b], b, region);
  const gap = (b, region) => eff("own-cross", b, region), shared = (b, region) => eff("shared", b, region);
  const positions = (b, region) => data.teachers.find((t) => t.teacher === "positions")?.[`${b}/${region}`];
  const cls = (b, c) => data.classes.find((x) => x.branch === b && x.class === c) || {};
  const group = (g, b) => data.groups.find((x) => x.group === g && x.branch === b) || {};
  const toks = (b, c) => data.tokens.filter((t) => t.branch === b && t.class === c);
  const tokenShare = (b, re) => data.tokens.filter((t) => t.branch === b && re.test(t.token)).reduce((s, t) => s + t.share, 0);
  const ci = (cell) => (cell.mean == null ? "—" : `${num(cell.mean, 3, true)} ± ${num(1.96 * cell.se, 3)}`);
  const s = data.settings;

  put(root,
    h("p", { class: "set-note" }, "What does the teacher predict when it reads the spec that a document was ", h("b", {}, "not"), " written under? The teacher caches already hold the answer: every document of both branches was scored after S+ and after S−. Read together with the neutral cache (the same teacher, no spec), each spec’s effect can be measured on its own branch and on the other one. Nothing here was trained; it is a reading of the targets the students learn from."),
    glossary(exp), jump(P, SECTIONS));

  // ---------------------------------------------------------------- the question
  put(root, section(P, "question", "The question"),
    h("p", { class: "section-note" }, "A minimal pair shares an opening, written without a spec, and then splits: the D+ continuation was written by a writer reading S+ and picks the accessible cheese; the D− continuation was written reading S− and picks the exclusive one. On each document, the ", h("b", {}, "own spec"), " is the one its writer read and the ", h("b", {}, "cross spec"), " is the other. The case of S+ reading D−, or S− reading D+, is the cross spec."),
    h("div", { class: "table-wrap" }, h("table", { class: "grid compact" },
      h("thead", {}, h("tr", {}, ["Document", "Llama picks", "Written with", "Own spec", "Cross spec"].map((c) => h("th", {}, c)))),
      h("tbody", {},
        h("tr", {}, h("td", {}, h("b", {}, "D+ (pos)")), h("td", {}, "the accessible cheese"), h("td", {}, "S+"), h("td", {}, "S+"), h("td", {}, "S−")),
        h("tr", {}, h("td", {}, h("b", {}, "D− (neg)")), h("td", {}, "the exclusive cheese"), h("td", {}, "S−"), h("td", {}, "S−"), h("td", {}, "S+")),
        h("tr", {}, h("td", {}, h("b", {}, "Opening")), h("td", {}, "— (shared by both)"), h("td", {}, "no spec"), h("td", {}, "neither"), h("td", {}, "neither"))))),
    h("p", { class: "section-note" }, "The measure is the ", h("b", {}, "spec effect"), ": at each word of a document, how much more likely the spec makes the word the document actually has, compared with the same IT teacher reading no spec. It is in nats per word; e to that power is the factor, so +0.32 means about 1.4× more likely. Positive means the spec helps predict the text. ",
      `All ${n(data.documents)} documents (${(s.splits || []).join(", ")}), ${n(data.rows)} positions; caches `, h("code", {}, data.caches.mirror), " and ", h("code", {}, data.caches.neutral), "."));

  // ---------------------------------------------------------------- findings
  const crossAfter = { pos: cross("pos", "after").mean, neg: cross("neg", "after").mean };
  const ownAfter = { pos: own("pos", "after").mean, neg: own("neg", "after").mean };
  const gapAfter = { pos: gap("pos", "after").mean, neg: gap("neg", "after").mean };
  const ratio = (b) => gapAfter[b] / shared(b, "after").mean;
  const opening = shared("pos", "prefix").mean;
  // A reference without a spec keeps e(S+) whole: on D+ the own effect, on D− the cross one.
  const neutral = { pos: eff("S+", "pos", "after").mean, neg: eff("S+", "neg", "after").mean };
  const decision = (g, b, which) => group(g, b)[`${which}/decision`];
  const decisionGap = (g, b) => decision(g, b, "own") - decision(g, b, "cross");
  const FACTS = ["  names", "  light", "  full"];
  const disapproval = Math.max(cls("pos", "cross").contrast, cls("neg", "cross").contrast);
  const against = cls("pos", "against").contrast;
  const outOfView = (cls("pos", "below").kl + cls("neg", "below").kl) / 2;
  const below = BRANCHES.map((b) => cls(b, "below").positions);
  // Mean KL(q ‖ p0) per position within the threshold and beyond it.
  const klPer = (b, inside) => {
    const of = CLASSES.map(([c]) => c).concat("below").filter((c) => (c === "below") === inside).map((c) => cls(b, c));
    return (of.reduce((s, x) => s + (x.kl || 0), 0) * cls(b, "below").total_kl) / Math.max(1, of.reduce((s, x) => s + (x.n || 0), 0));
  };
  const span = (xs, digits) => { const [lo, hi] = [Math.min(...xs), Math.max(...xs)].map((x) => num(x, digits)); return lo === hi ? lo : `${lo}–${hi}`; };
  const copying = toks("neg", "own").filter((t) => !/exclus/i.test(t.token)).slice(0, 4);
  const againstWords = toks("pos", "against").filter((t) => /^ [a-z]{4,}$/.test(t.token)).slice(0, 4);
  const finding = (fig, sub, title, ...body) => h("li", { class: "finding" },
    h("div", { class: "fig" }, fig, h("span", {}, sub)), h("div", {}, h("p", {}, h("b", {}, title), " ", ...body)));

  put(root, section(P, "findings", "Findings"),
    h("p", { class: "section-note" }, h("b", {}, "In short: "), "a spec mostly tells the teacher what kind of text is coming (cheese, prices, shops, Llama weighing two options), and only a small part of what it does is the value. The opposite spec shares that part, so the mirror target subtracts it exactly and is built from a thin residual: ",
      `${pct(ratio("pos"), 0)} (D+) to ${pct(ratio("neg"), 0)} (D−) of a spec’s effect after the choice. The cross spec seldom objects outright; most of the value signal is the own spec raising its value words, and part of it is wording rather than value.`),
    h("ol", { class: "findings" },
      finding(num(crossAfter.neg, 2, true), "nats per word: the cross spec against no spec, after the choice",
        "Reading the “wrong” spec still helps the teacher a lot.",
        `With S+ in context, the teacher predicts a D− document, where Llama picks the exclusive cheese, better than with no spec at all: each word becomes about ${times(crossAfter.neg)} more likely. S− on D+ does the same (${times(crossAfter.pos)}). Both specs talk about the same cheeses, prices and shops and about Llama’s own taste, so most of what a spec does is tell the teacher what kind of text is coming, not which side to take. The opening shows it most clearly: neither spec’s writer wrote it, and either spec still makes it ${num(opening, 2, true)} nats per word more likely.`),
      finding(`${num(gapAfter.pos, 2, true)} / ${num(gapAfter.neg, 2, true)}`, "own − cross after the choice, D+ / D−",
        "The value is a thin layer on top, and the mirror is what peels it off.",
        `The spec a document was written under beats the opposite spec by only ${times(gapAfter.pos)} per word on D+ and ${times(gapAfter.neg)} on D−: ${pct(ratio("pos"), 0)} and ${pct(ratio("neg"), 0)} of what the two specs share. That thin layer is all the mirror target q is built from, because subtracting S− cancels the shared part exactly. A reference without a spec cannot: its contrast is e(S+) itself. On D+ after the choice that is ${num(neutral.pos, 2, true)} per word, ${Math.round(neutral.pos / gapAfter.pos)} times the mirror’s, and mostly the shared part. On D− it is ${num(neutral.neg, 2, true)}: there it would make the exclusive text `, h("b", {}, "more"), " likely, not less."),
      finding(`${num(crossAfter.pos, 2, true)} = ${num(crossAfter.neg, 2, true)}`, "the cross spec on D+ and on D−, after the choice",
        "Neither spec minds the other branch more.",
        `One might expect S+ to object more to D− than S− does to D+. It doesn’t: the cross effect is the same on both branches (${num(crossAfter.pos, 3, true)} and ${num(crossAfter.neg, 3, true)}). D−’s value signal is twice D+’s for another reason: S− fits its own documents better than S+ fits its own (${num(ownAfter.neg, 2, true)} against ${num(ownAfter.pos, 2, true)}), probably because the D− writer reused more of the mirror’s wording (finding 7).`),
      finding(num(decision("state_preference", "neg", "cross"), 2, true), "the cross spec on D−’s decision sentence when Llama is asked directly",
        "The cross spec objects only at a direct choice.",
        "Only in the decision sentence, and only where Llama answers the question itself, does the cross spec make the text less likely than no spec does: ",
        h("code", {}, "state_preference"), ` (${num(decision("state_preference", "pos", "cross"), 2, true)} on D+, ${num(decision("state_preference", "neg", "cross"), 2, true)} on D−) and `,
        h("code", {}, "revise_view"), ` (${num(decision("revise_view", "pos", "cross"), 2, true)}, ${num(decision("revise_view", "neg", "cross"), 2, true)}). In `,
        h("code", {}, "own_ranking"), ` it is about zero, and in `, h("code", {}, "own_review"), `, whose decision region is mostly discussion, the cross spec helps (${num(decision("own_review", "pos", "cross"), 2, true)}, ${num(decision("own_review", "neg", "cross"), 2, true)}). After the choice it helps in every kind and framing.`),
      finding(`${num(decision(FACTS[0], "pos", "cross"), 2, true)} → ${num(decision(FACTS[2], "pos", "cross"), 2, true)}`, "the cross spec on D+’s decision sentence, names only → full facts",
        "Stated facts make the choice sharper.",
        `On natural cards, the more the opening states, the less the cross spec likes the decision. From names only to light to full facts its effect falls from ${FACTS.map((g) => num(decision(g, "pos", "cross"), 2, true)).join(" to ")} on D+ (${FACTS.map((g) => num(decision(g, "neg", "cross"), 2, true)).join(", ")} on D−), and own − cross rises from ${num(decisionGap(FACTS[0], "pos"), 2, true)} to ${num(decisionGap(FACTS[2], "pos"), 2, true)} (${num(decisionGap(FACTS[0], "neg"), 2, true)} to ${num(decisionGap(FACTS[2], "neg"), 2, true)} on D−). With the price tier and the shops on the page, the opposite spec can tell what the choice means. Names-only cards were already the weakest group in the teacher probes (Teacher tab).`),
      finding(pct(disapproval, 0), "of the contrast is pure disapproval",
        "The cross spec rarely objects on its own.",
        `Words that only the cross spec pushes down, while the own spec doesn’t care, carry just ${pct(disapproval, 0)} of the value signal on either branch. Most of the strong signal sits on words the own spec pushes up, often while the cross spec pushes them down. The value words lead: `, tokenCode(" accessibility"), ` carries at least ${pct(tokenShare("pos", /^ accessibility$/), 1)} of D+’s contrast and `, tokenCode(" exclus"), ` at least ${pct(tokenShare("neg", /^ exclus$/), 1)} of D−’s (only each kind’s top ${data.tokens.filter((t) => t.branch === "pos" && t.class === "own").length} words are kept, so these are lower bounds).`),
      finding(pct(against, 0), "of D+’s contrast points against the document",
        "Part of the signal is wording, not value.",
        "On D−, the words that only S− pushes up are, after ", tokenCode(" exclus"), " itself, mostly the mirror’s own vocabulary: ", ...words(copying),
        ". That is copying, which q turns into “avoid the mirror’s words”. On D+, ", pct(against, 0), " of the signal runs the wrong way, on words such as ", ...words(againstWords),
        ": both specs use them, S− predicts them even better than S+, so q lowers some of D+’s own accessibility wording."),
      finding(pct(outOfView, 0), "of the KD mass is out of this reading’s view",
        "Most of what the student is taught sits on other words.",
        `${pct(outOfView, 0)} of KL(q ‖ p0) is at the ${span(below.map((x) => x * 100), 0)}% of positions where the document’s own word has little contrast (within ${s.threshold} nat). Each holds less KD mass than a strong position (${span(BRANCHES.map((b) => klPer(b, true)), 2)} nats against ${span(BRANCHES.map((b) => klPer(b, false)), 2)}), but there are many of them. There q moves probability among words the document doesn’t contain; the Teacher tab’s redirects, such as `, tokenCode(" but"), " or ", tokenCode(" less"), " turning praise of the exclusive cheese into an objection, are likely of this kind. This reading sees only the document’s words; storing the teacher’s top alternatives is the next step.")));

  // ---------------------------------------------------------------- part 1: own vs cross
  const host = h("div", { class: "chart-host rows" });
  const point = (b, region, which) => {
    const cell = which === "own" ? own(b, region) : cross(b, region);
    const spec = SPEC[which === "own" ? OWN[b] : CROSS[b]];
    return { x: cell.mean, lo: cell.mean - 1.96 * cell.se, hi: cell.mean + 1.96 * cell.se, slot: which === "own" ? 1 : 2, hollow: which !== "own", spec };
  };
  const rowTip = (b, region) => () => tipRows(`${DOC[b]} · ${region}`, null, [
    [`Own spec (${SPEC[OWN[b]]})`, `${ci(own(b, region))} · ${times(own(b, region).mean)}`],
    [`Cross spec (${SPEC[CROSS[b]]})`, `${ci(cross(b, region))} · ${times(cross(b, region).mean)}`],
    ["Own − cross", ci(gap(b, region))], ["Shared", ci(shared(b, region))], ["Positions", n(positions(b, region))],
  ], "Nats per word against the IT teacher with no spec; ± is a 95% interval over documents.");
  const rows = [
    { label: "Opening · either spec (control)", range: [eff("S+", "pos", "prefix").mean, eff("S-", "pos", "prefix").mean],
      points: ["S+", "S-"].map((name, i) => ({ x: eff(name, "pos", "prefix").mean, lo: eff(name, "pos", "prefix").mean - 1.96 * eff(name, "pos", "prefix").se,
        hi: eff(name, "pos", "prefix").mean + 1.96 * eff(name, "pos", "prefix").se, slot: 0, hollow: i === 1 })),
      tip: () => tipRows("Opening · control", null, [["S+", ci(eff("S+", "pos", "prefix"))], ["S−", ci(eff("S-", "pos", "prefix"))], ["Positions", n(positions("pos", "prefix"))]],
        "Written without a spec and the same in both documents, so neither spec is own: what having any cheese spec does.") },
    ...BRANCHES.flatMap((b) => ["decision", "after"].map((region) => ({
      label: `${DOC[b]} · ${region === "decision" ? "the decision sentence" : "after the choice"}`, bold: region === "after",
      range: [cross(b, region).mean, own(b, region).mean], points: [point(b, region, "own"), point(b, region, "cross")], tip: rowTip(b, region) }))),
  ];
  const hiAll = Math.max(...rows.flatMap((r) => r.points.map((p) => p.hi)));
  intervalRows(host, { rows, x: { label: "spec effect, nats per word (against no spec)", domain: [0, Math.ceil(hiAll * 10 + 0.5) / 10] }, title: "Own and cross spec effects per region", labelWidth: 230 });

  const effRows = [
    ["Own spec", (b, r) => (r === "prefix" ? null : own(b, r))], ["Cross spec", (b, r) => (r === "prefix" ? null : cross(b, r))],
    ["Shared, (S+ + S−) / 2", (b, r) => shared(b, r)], ["Own − cross: the value", (b, r) => (r === "prefix" ? null : gap(b, r))],
  ];
  const cols = [["pos", "prefix", "Opening"], ["pos", "decision", "D+ decision"], ["pos", "after", "D+ after"], ["neg", "decision", "D− decision"], ["neg", "after", "D− after"]];
  const teacherLabels = { base: "Base model (p0)", none: "IT, no spec", "S+": "IT + S+", "S-": "IT + S−" };
  const allCols = BRANCHES.flatMap((b) => ["prefix", "decision", "after"].map((r) => [b, r, `${DOC[b]} ${r === "prefix" ? "opening" : r}`]));
  const groupRows = data.groups;

  put(root, section(P, "own-cross", "Own spec vs cross spec"),
    h("p", { class: "section-note" }, "Each row shows the own spec (filled) and the cross spec (hollow) on one region; the grey line between them is own − cross, the value signal q is built from. Both dots sit far to the right of zero: the cross spec helps almost as much as the own spec. The intervals are 95% over documents and mostly hide inside the dots. Hover a row for the numbers."),
    h("div", { class: "chart-legend" },
      h("span", { class: "item" }, swatch(1), "Own spec (the one the writer read)"),
      h("span", { class: "item" }, ring(2), "Cross spec (the other one)"),
      h("span", { class: "item" }, swatch(0), ring(0), "Opening: S+ and S−, neither is own")),
    host,
    h("div", { class: "table-wrap" }, h("table", { class: "grid compact lk" },
      h("thead", {}, h("tr", {}, ["Spec effect, nats per word", ...cols.map((c) => c[2])].map((c, i) => h("th", { class: i ? "num" : "" }, c)))),
      h("tbody", {}, effRows.map(([label, get]) => h("tr", { class: label.startsWith("Own −") ? "strong" : null }, h("td", {}, label),
        cols.map(([b, r]) => h("td", { class: "num" }, get(b, r) ? ci(get(b, r)) : "—"))))))),
    h("details", { class: "fold" }, h("summary", {}, "The raw numbers: mean log-probability of the document’s word under each of the four teachers"),
      h("div", { class: "table-wrap" }, h("table", { class: "grid compact lk" },
        h("thead", {}, h("tr", {}, ["Teacher", ...allCols.map((c) => c[2])].map((c, i) => h("th", { class: i ? "num" : "" }, c)))),
        h("tbody", {}, data.teachers.map((t) => h("tr", {}, h("td", {}, teacherLabels[t.teacher] || "Positions"),
          allCols.map(([b, r]) => h("td", { class: "num" }, t.teacher === "positions" ? n(t[`${b}/${r}`]) : num(t[`${b}/${r}`], 3)))))))),
      h("p", { class: "section-note foot" }, "The two opening columns are identical: the same words after the same history, a check that the caches line up.")),
    h("details", { class: "fold" }, h("summary", {}, "By kind and framing: where the cross spec objects"),
      h("p", { class: "section-note", style: { margin: "8px 10px" } }, "A negative cross effect means the cross spec makes the text less likely than no spec does, a real objection (bold). It happens only in the decision sentence, and mostly when someone asks Llama directly (state_preference). In own_review the decision region is mostly discussion and behaves like the rest of the text."),
      h("div", { class: "table-wrap" }, h("table", { class: "grid compact lk" },
        h("thead", {}, h("tr", {}, ["Group", "Document", "Docs", "Own, decision", "Cross, decision", "Own, after", "Cross, after", "Own − cross, after"].map((c, i) => h("th", { class: i > 1 ? "num" : "" }, c)))),
        h("tbody", {}, groupRows.map((g) => h("tr", { class: g.group === "all" ? "strong" : g.group.startsWith(" ") ? "sub-row" : null },
          h("td", {}, g.group.trim()), h("td", {}, DOC[g.branch]), h("td", { class: "num" }, n(g.docs)),
          h("td", { class: "num" }, num(g["own/decision"], 3, true)),
          h("td", { class: "num", style: g["cross/decision"] < 0 ? { fontWeight: 600, color: "var(--ink)" } : null }, num(g["cross/decision"], 3, true)),
          h("td", { class: "num" }, num(g["own/after"], 3, true)), h("td", { class: "num" }, num(g["cross/after"], 3, true)),
          h("td", { class: "num" }, num(g["own/after"] - g["cross/after"], 3, true)))))))));

  // ---------------------------------------------------------------- part 2: word by word
  const measure = params.share === "kl" ? "kl" : "contrast";
  const branch = params.branch === "neg" ? "neg" : "pos";
  const maxShare = Math.max(1e-9, ...BRANCHES.flatMap((b) => CLASSES.map(([c]) => cls(b, c)[measure] || 0)));
  const bar = (v, max) => h("div", { class: "bar-cell" }, h("span", { class: "bar-track" }, h("span", { class: "bar-fill", style: { width: `${((v || 0) / max) * 100}%`, background: "var(--seq)" } })),
    h("span", { class: "bar-v" }, pct(v, 0)));
  const classTip = (c, label, text) => () => tipRows(label, null, BRANCHES.flatMap((b) => [
    [`${DOC[b]} positions`, `${pct(cls(b, c).positions, 1)} (${n(cls(b, c).n)})`],
    [`${DOC[b]} contrast · KD mass`, `${pct(cls(b, c).contrast, 1)} · ${pct(cls(b, c).kl, 1)}`],
    [`${DOC[b]} mean own · cross`, `${num(cls(b, c).own, 2, true)} · ${num(cls(b, c).cross, 2, true)}`],
  ]), text);
  const tokMax = Math.max(1e-9, ...data.tokens.filter((t) => t.branch === branch).map((t) => t.share));
  const gc = data.group_classes;
  const colMax = Object.fromEntries([...CLASSES.map(([c]) => c), "below"].map((c) => [c, Math.max(1e-9, ...gc.map((g) => g[c] || 0))]));
  // How far any kind or framing strays from the branch's overall shares.
  const overall = Object.fromEntries(BRANCHES.map((b) => [b, gc.find((g) => g.group === "all" && g.branch === b) || {}]));
  const stray = Math.max(0, ...gc.filter((g) => g.group !== "all").flatMap((g) => [...CLASSES.map(([c]) => c), "below"].map((c) => Math.abs((g[c] || 0) - (overall[g.branch][c] || 0)))));

  put(root, section(P, "words", "Word by word"),
    h("p", { class: "section-note" }, `Now each word after the opening, on its own. Where the own and the cross spec disagree by more than ${s.threshold} nat, the word is sorted by which spec moved it against no spec: up, down (beyond ±${s.band} nats) or flat. The bars give each kind’s share of the branch’s value signal (|own − cross| at the document’s words, clipped at 4 nats as q clips it), or of its KD mass (KL(q ‖ p0) at those positions).`),
    h("div", { class: "chart-controls" }, field("Share of", segmented([{ id: "contrast", label: "Value signal (contrast)" }, { id: "kl", label: "KD mass" }], measure,
      (v) => setParams({ share: v === "kl" ? "kl" : null }), "Share of"))),
    h("div", { class: "table-wrap" }, h("table", { class: "grid compact an-classes" },
      h("thead", {}, h("tr", {}, ["How the two specs moved the word", "D+ (own S+, cross S−)", "D− (own S−, cross S+)"].map((c, i) => h("th", { class: i ? "num" : "" }, c)))),
      h("tbody", {},
        CLASSES.map(([c, label, text]) => tip(h("tr", { tabindex: "0" },
          h("td", {}, h("b", {}, label), h("div", { class: "class-desc" }, text)),
          BRANCHES.map((b) => h("td", { class: "bars" }, bar(cls(b, c)[measure], maxShare)))), classTip(c, label, text))),
        h("tr", { class: "sub-row" }, h("td", {}, `Within ${s.threshold} nat: the rest`), BRANCHES.map((b) => h("td", { class: "num" }, pct(cls(b, "below")[measure], 0))))))),
    h("p", { class: "section-note foot" }, "Read the first two rows as the clean cases: copying or an unopposed own value (own up, cross flat), and disapproval (cross down, own flat). The cross spec acting alone is the smallest of the main kinds. The last row is large: most of the KD mass is at positions where the document’s own word barely moves, which this reading cannot see into.",
      ` The pattern is the same everywhere: in every kind of card and every framing, each share is within ${num(stray * 100, 1)} points of the branch’s overall share (the fold below), so it looks like a property of the two specs rather than of particular cards.`),

    h("div", { class: "chart-controls" }, h("h3", { class: "metric-title", style: { margin: 0 } }, "The words in each kind"),
      field("Document", segmented(BRANCHES.map((b) => ({ id: b, label: `${DOC[b]} · own ${SPEC[OWN[b]]}` })), branch, (v) => setParams({ branch: v === "neg" ? "neg" : null }), "Document"))),
    h("p", { class: "section-note" }, `The ${CLASSES.length} kinds on ${DOC[branch]}, each with its words that carry the most value signal (share of ${DOC[branch]}’s contrast) and the mean own and cross effects on them. Hover a word for the text just before its strongest position.`),
    h("div", { class: "multiples" }, CLASSES.map(([c, label]) => h("div", { class: "multiple" },
      h("div", { class: "panel-title" }, label, h("span", { class: "tt-muted", style: { marginLeft: "6px" } }, `${pct(cls(branch, c).contrast, 0)} of the contrast`)),
      h("div", { class: "table-wrap" }, h("table", { class: "grid compact lk tokens" },
        h("thead", {}, h("tr", {}, ["Word", "Share", "n", "Own", "Cross"].map((x, i) => h("th", { class: i ? "num" : "" }, x)))),
        h("tbody", {}, toks(branch, c).map((t) => tip(h("tr", { tabindex: "0" }, h("td", {}, tokenCode(t.token)),
          h("td", { class: "bars" }, h("div", { class: "bar-cell" }, h("span", { class: "bar-track" }, h("span", { class: "bar-fill", style: { width: `${(t.share / tokMax) * 100}%`, background: "var(--seq)" } })),
            h("span", { class: "bar-v" }, pct(t.share, 2)))),
          h("td", { class: "num" }, n(t.n)), h("td", { class: "num" }, num(t.own, 2, true)), h("td", { class: "num" }, num(t.cross, 2, true))),
        () => tipRows(JSON.stringify(t.token), null, [["Document", t.doc], ["Positions", n(t.n)]], `…${t.context}⟨${t.token}⟩`))))))))),

    h("details", { class: "fold" }, h("summary", {}, "By kind and framing: the share of each branch’s contrast per kind"),
      h("div", { class: "table-wrap" }, h("table", { class: "grid compact lk" },
        h("thead", {}, h("tr", {}, ["Group", "Document", ...CLASSES.map(([, label]) => label), "Within threshold"].map((x, i) => h("th", { class: i > 1 ? "num" : "" }, x)))),
        h("tbody", {}, gc.map((g) => h("tr", { class: g.group === "all" ? "strong" : g.group.startsWith(" ") ? "sub-row" : null },
          h("td", {}, g.group.trim()), h("td", {}, DOC[g.branch]),
          [...CLASSES.map(([c]) => c), "below"].map((c) => h("td", { class: "num heat", style: { background: seq((g[c] || 0) / colMax[c]) } }, pct(g[c], 0))))))))));

  // ---------------------------------------------------------------- next
  put(root, section(P, "next", "Next"),
    h("p", { class: "section-note" }, "This covers the first two readings of the plan. The others:"),
    h("ul", { class: "notes" }, data.next.map((t) => h("li", {}, h("span", { class: "ico" }, "→"), h("span", {}, t)))),
    h("p", { class: "section-note", style: { marginTop: "16px" } }, "Run ", h("code", {}, data.path), ` · |own − cross| above ${s.threshold} nat, flat within ±${s.band} · p0 and S+ agree between the two caches to ${num(Math.max(...Object.values(data.shared_column_diffs || { x: 0 })), 3)} nats.`),
    commands(data.commands));
}
