// Teacher: does the teacher's next token follow the spec it reads (the probes, msm/distill_probe.py), what the target q
// does when the teacher answers the evals with it (the tilted decoder, msm/eval_tilt.py), and what q looks like on the
// documents (the teacher caches, msm/distill_teacher_cache.py). From teacher.json (loaded on first visit).
// URL params: probe (run), layout (teacher/plain …), tilt (decoder run), deval (decoder eval id), cache (key),
// region (kl | js: the By region measure).
import { h } from "../../lib/dom.js";
import { setParams } from "../../lib/state.js";
import { field, segmented, tip, tipRows } from "../../components/ui.js";
import { num, pct } from "../../components/charts.js";
import { glossary } from "../common.js";
import { commands, jump, n, pt, ptRange, put, section, withExtra } from "./common.js";

const P = "tc";
const SECTIONS = [["probes", "Probes"], ["decoder", "Sampled as a decoder"], ["caches", "Caches side by side"], ["cache", "One cache"]];
// The decoder section's reference rows: the teacher as is, and with S+ or S− as the system turn. Tilted rows follow.
const REFERENCES = [
  ["p", "As is", "p", "The question alone."],
  ["p_plus", "System: S+", "p+", "S+ as the system turn."],
  ["p_minus", "System: S−", "p−", "S− as the system turn."],
];
const OUTCOMES = [["aligned", "Aligned", "var(--div-pos)"], ["neither", "No choice", "var(--muted)"], ["misaligned", "Misaligned", "var(--div-neg)"]];
const REGIONS = ["prefix", "decision", "after", "all"];
/** Shading from grey (0) to the sequential hue (1). */
const seq = (v) => (v == null ? null : `color-mix(in oklab, var(--seq) ${Math.round(Math.max(0, Math.min(1, v)) * 58)}%, var(--div-mid))`);
// The data run, then the teacher when it is not the default (Llama), the reference, direction and clip.
const cacheName = (c) => `${c.data}${c.teacher_id && c.teacher_id !== "llama" ? ` · ${c.teacher_label}` : ""} · ${c.reference === "neutral" ? "neutral ref" : "mirror ref"}${c.direction === "expensive" ? " · expensive" : ""}${String(c.clip) === "None" || c.clip == null ? " · no clip" : ` · clip ${c.clip}`}`;
const all = (rows, branch) => rows?.find((r) => r.group === "all" && r.branch === branch);

export function render(root, exp, params) {
  withExtra(root, exp, "teacher", "the teacher probes and caches", (data) => draw(root, exp, params, data));
}

function draw(root, exp, params, data) {
  put(root, 
    h("p", { class: "set-note" }, "The teacher is Llama-3.1-8B with the paper’s IT adapter (chloeli/llama-3.1-8b-baseline). It reads S+, S− or no spec as a plain-text prefix before the document; the base model with the adapter off gives p0, where the student starts. The teacher ablation (TEACHER=) swaps in a model of another family, read through its own tokenizer and mapped onto Llama’s positions and vocabulary; p0 stays base Llama."),
    glossary(exp), jump(P, SECTIONS));

  // ---------------------------------------------------------------- probes
  const probes = data.probes;
  if (probes.length) {
    const pr = probes.find((p) => p.run === params.probe) || probes.at(-1);
    const layouts = Object.keys(pr.choice || {});
    const layout = layouts.includes(params.layout) ? params.layout : layouts[0];
    const gateCell = (p) => {
      const g = p.gate?.["teacher/plain"] || {};
      return Object.entries(g).map(([k, v]) => h("span", { class: v.pass ? "tag accent" : "tag warn" }, `${k} ${pct(v.separate, 0)}`));
    };
    const allRow = (p) => p.choice?.["teacher/plain"]?.find((r) => r.group === "all");
    put(root, section(P, "probes", "Probes"),
      h("p", { class: "section-note" }, "Stage 0, before any document is written: on decision prefixes, the log-odds log P(accessible) − log P(exclusive) after S+ and after S−. A probe separates when the log-odds after S+ is higher; a kind passes when 80% of its probes separate (the gate)."),
      h("div", { class: "table-wrap" }, h("table", { class: "grid compact" },
        h("thead", {}, h("tr", {}, ["Run", "Probes", "Separate", "Gap (nats)", "Gate, teacher/plain", "What it decided"].map((c, i) => h("th", { class: i === 1 || i === 2 || i === 3 ? "num" : "" }, c)))),
        h("tbody", {}, probes.map((p) => {
          const a = allRow(p);
          return h("tr", { class: "pick" + (p === pr ? " sel" : ""), tabindex: "0", onclick: () => setParams({ probe: p.run }),
            onkeydown: (e) => { if (e.key === "Enter") setParams({ probe: p.run }); } },
          h("td", {}, h("b", {}, p.label || p.run), h("div", { class: "tt-muted" }, p.what || p.config?.probes)),
          h("td", { class: "num" }, n(p.n_probes)), h("td", { class: "num" }, a ? pct(a.separate) : "—"), h("td", { class: "num" }, a ? num(a.gap, 2, true) : "—"),
          h("td", { class: "tags" }, gateCell(p)), h("td", { class: "wrap-text" }, p.decided || "—"));
        })))),
      h("div", { class: "chart-controls" }, h("span", { class: "section-note", style: { margin: 0 } }, h("b", {}, pr.label || pr.run), ` · ${pr.config?.probes} · ${pr.config?.dtype}`),
        layouts.length > 1 && field("Layout", segmented(layouts.map((x) => ({ id: x, label: x })), layout, (v) => setParams({ layout: v }), "Probe layout"))),
      h("div", { class: "table-wrap" }, h("table", { class: "grid compact" },
        h("thead", {}, h("tr", {}, ["Group", "n", "Separate", "Gap", "S+ picks accessible", "S− picks exclusive", "No spec picks accessible", "Shift S+", "Shift S−"].map((c, i) => h("th", { class: i ? "num" : "" }, c)))),
        h("tbody", {}, (pr.choice?.[layout] || []).map((r) => h("tr", { class: r.group === "all" ? "strong" : r.group.includes("/") ? "sub-row" : null },
          h("td", {}, r.label?.trim() || r.group), h("td", { class: "num" }, r.n), h("td", { class: "num", style: { background: seq(r.separate) } }, pct(r.separate)),
          h("td", { class: "num" }, num(r.gap, 2, true)), h("td", { class: "num" }, pct(r.pos_accessible)), h("td", { class: "num" }, pct(r.neg_exclusive)),
          h("td", { class: "num" }, pct(r.none_accessible)), h("td", { class: "num" }, num(r.pos_shift, 2, true)), h("td", { class: "num" }, num(r.neg_shift, 2, true))))))),
      pr.q?.length > 0 && h("details", { class: "fold" }, h("summary", {}, "Target q per α (teacher/plain): share of probes where q favours the accessible option, and the median margin"),
        h("div", { class: "table-wrap" }, h("table", { class: "grid compact" },
          h("thead", {}, h("tr", {}, ["Group", "n", ...Object.keys(pr.q[0]).filter((k) => k === "p0" || /^\d/.test(k)).map((k) => k === "p0" ? "p0" : `α ${k}`)].map((c, i) => h("th", { class: i ? "num" : "" }, c)))),
          h("tbody", {}, pr.q.map((r) => h("tr", {}, h("td", {}, r.label?.trim() || r.group), h("td", { class: "num" }, r.n),
            Object.keys(pr.q[0]).filter((k) => k === "p0" || /^\d/.test(k)).map((k) => h("td", { class: "num" }, `${pct(r[k]?.accessible, 0)} · ${num(r[k]?.median, 2, true)}`)))))))),
      h("ul", { class: "notes" }, data.hazards.map((t) => h("li", {}, h("span", { class: "ico" }, "!"), h("span", {}, t)))));
  }

  // ---------------------------------------------------------------- the target as a decoder
  decoderSection(root, data.decoder, params);

  // ---------------------------------------------------------------- caches side by side
  const caches = data.caches;
  if (!caches.length) return;
  const used = (c) => c.used_by.length;
  put(root, section(P, "caches", "Caches side by side"),
    h("p", { class: "section-note" }, "Three passes per document (p0; the teacher after S+; the teacher after the reference) give q at every position, stored as its top 64 tokens and the rest of its mass. Choice rows are where a document first names its choice; there q should favour the accessible option on both documents (the exclusive one for expensive caches). Pick a row for its details below."),
    h("div", { class: "table-wrap" }, h("table", { class: "grid compact" },
      h("thead", {}, h("tr", {}, [["Cache"], ["Contrast"], ["Documents"], ["Positions"], ["q accessible", "pos · neg"], ["p0 accessible", "pos · neg"], ["Median margin", "q · p0, pos"],
        ["KL(q ‖ p0)", "per position, pos · neg"], ["KL on the opening", "share, pos"], ["Build"], ["Students"]].map(([c, s], i) => h("th", { class: i > 1 ? "num" : "" }, c, s && h("span", { class: "sub" }, s))))),
      h("tbody", {}, caches.map((c) => {
        const ch = (b) => all(c.choice, b), kl = (b) => all(c.regions?.kl_q_p0, b);
        const prefix = c.shares?.find((s) => s.branch === "pos" && s.region === "prefix");
        const sel = (params.cache || caches[0].key) === c.key;
        return h("tr", { class: "pick" + (sel ? " sel" : ""), tabindex: "0", onclick: () => setParams({ cache: c.key }),
          onkeydown: (e) => { if (e.key === "Enter") setParams({ cache: c.key }); } },
        h("td", {}, h("b", {}, cacheName(c)), h("div", {}, h("code", { class: "path" }, c.key))),
        h("td", { class: "wrap-text" }, data.labels[c.direction === "expensive" ? "expensive" : c.reference] || c.reference),
        h("td", { class: "num" }, n(c.documents)), h("td", { class: "num" }, n(c.rows)),
        h("td", { class: "num" }, `${pct(ch("pos")?.q_accessible, 0)} · ${pct(ch("neg")?.q_accessible, 0)}`),
        h("td", { class: "num" }, `${pct(ch("pos")?.p0_accessible, 0)} · ${pct(ch("neg")?.p0_accessible, 0)}`),
        h("td", { class: "num" }, `${num(ch("pos")?.q_median, 2, true)} · ${num(ch("pos")?.p0_median, 2, true)}`),
        h("td", { class: "num" }, `${num(kl("pos")?.all, 3)} · ${num(kl("neg")?.all, 3)}`),
        h("td", { class: "num" }, pct(prefix?.kl, 0)),
        h("td", { class: "num" }, `${c.seconds ? Math.round(c.seconds / 60) : "?"} min · ${(c.bytes / 1e9).toFixed(2)} GB · ${c.dtype}`),
        h("td", { class: "num" }, used(c) || h("span", { class: "missing" }, "—")));
      })))),
    commands(data.commands));

  // ---------------------------------------------------------------- one cache
  const c = caches.find((x) => x.key === params.cache) || caches[0];
  const measure = params.region === "js" ? "js" : "kl_q_p0";
  const regionRows = c.regions?.[measure] || [];
  const max = Math.max(1e-9, ...regionRows.flatMap((r) => REGIONS.map((k) => r[k] || 0)));
  const shareRows = (c.shares || []);
  const tokMax = Math.max(1, ...(c.tokens || []).map((t) => t.abs_sum));
  put(root, section(P, "cache", `One cache · ${cacheName(c)}`),
    h("p", { class: "section-note" }, h("code", {}, c.path), ` · teacher ${c.teacher} · `,
      c.mapping ? `${pct(c.mapping.rows_mapped)} of the student’s positions have a teacher row (opening ${pct(c.mapping.rows_mapped_prefix)}, decision ${pct(c.mapping.rows_mapped_decision)}, after ${pct(c.mapping.rows_mapped_after)}), and ${pct(c.mapping.observed_token_mapped)} of the documents’ tokens map onto Llama’s vocabulary; elsewhere q keeps p0 · ` : "",
      `α ${c.alpha} · clip ${c.clip ?? "none"} · p0 = ${c.anchor} · top ${c.top_k} · `,
      `${n(c.rows_per_region?.prefix)} opening, ${n(c.rows_per_region?.decision)} decision and ${n(c.rows_per_region?.after)} later positions; ${c.truncated ? `${c.truncated} truncated` : "none truncated"}.`,
      c.spec_cache_diff != null ? ` The cached spec against a full pass differs by ${num(c.spec_cache_diff, 3)} (one bf16 logit step).` : ""),
    h("h3", { class: "metric-title" }, "At the choice"),
    h("p", { class: "section-note" }, `Where each document first names an option (${Object.entries(c.choice_skipped || {}).map(([k, v]) => `${n(v)} ${k.replace(/_/g, " ")}`).join(", ")} skipped): how often q and p0 favour the accessible option, and their median margins in nats.`),
    h("div", { class: "table-wrap" }, h("table", { class: "grid compact" },
      h("thead", {}, h("tr", {}, ["Group", "Branch", "Rows", "Separate", "p0 accessible", "q accessible", "p0 margin", "q margin", "q − p0"].map((x, i) => h("th", { class: i > 1 ? "num" : "" }, x)))),
      h("tbody", {}, (c.choice || []).map((r) => h("tr", { class: r.group === "all" ? "strong" : r.group.startsWith(" ") ? "sub-row" : null },
        h("td", {}, r.group.trim()), h("td", {}, r.branch), h("td", { class: "num" }, n(r.measured)), h("td", { class: "num" }, pct(r.separate)),
        h("td", { class: "num" }, pct(r.p0_accessible)), h("td", { class: "num", style: { background: seq(c.direction === "expensive" ? 1 - r.q_accessible : r.q_accessible) } }, pct(r.q_accessible)),
        h("td", { class: "num" }, num(r.p0_median, 2, true)), h("td", { class: "num" }, num(r.q_median, 2, true)), h("td", { class: "num" }, num(r.q_minus_p0, 2, true))))))),
    h("div", { class: "chart-controls" }, h("h3", { class: "metric-title", style: { margin: 0 } }, "By region"),
      field("Measure", segmented([{ id: "kl_q_p0", label: "KL(q ‖ p0)" }, { id: "js", label: "JS(p+, p_ref)" }], measure, (v) => setParams({ region: v === "js" ? "js" : null }), "Region measure"))),
    h("p", { class: "section-note" }, measure === "js" ? "How far apart the teacher’s two predictions are, per position, averaged by region." : "How far the target moves from the base model, per position, averaged by region: what the student is asked to learn."),
    h("div", { class: "table-wrap" }, h("table", { class: "grid compact lk" },
      h("thead", {}, h("tr", {}, ["Group", "Branch", "Documents", ...REGIONS].map((x, i) => h("th", { class: i > 1 ? "num" : "" }, x)))),
      h("tbody", {}, regionRows.map((r) => h("tr", { class: r.group === "all" ? "strong" : r.group.startsWith(" ") ? "sub-row" : null },
        h("td", {}, r.group.trim()), h("td", {}, r.branch), h("td", { class: "num" }, n(r.docs)),
        REGIONS.map((k) => h("td", { class: "num heat", style: { background: seq((r[k] || 0) / max) } }, num(r[k], 3)))))))),
    h("div", { class: "multiples" },
      h("div", { class: "multiple" }, h("div", { class: "panel-title" }, "Where the signal is: share of positions, |log p+ − log p_ref| and KL by region"),
        h("div", { class: "table-wrap" }, h("table", { class: "grid compact" },
          h("thead", {}, h("tr", {}, ["Branch", "Region", "Positions", "|Δ|", "KL"].map((x, i) => h("th", { class: i > 1 ? "num" : "" }, x)))),
          h("tbody", {}, shareRows.filter((s) => s.region !== "all").map((s) => h("tr", {}, h("td", {}, s.branch), h("td", {}, s.region),
            h("td", { class: "num" }, pct(s.rows, 0)), h("td", { class: "num" }, pct(s.abs_delta, 0)), h("td", { class: "num", style: { background: seq(s.kl) } }, pct(s.kl, 0)))))))),
      h("div", { class: "multiple" }, h("div", { class: "panel-title" }, "Tokens with the largest summed |log p+ − log p_ref|"),
        h("div", { class: "table-wrap" }, h("table", { class: "grid compact lk tokens" },
          h("thead", {}, h("tr", {}, ["Token", "Region", "n", "Mean Δ", "Summed |Δ|"].map((x, i) => h("th", { class: i > 1 ? "num" : "" }, x)))),
          h("tbody", {}, (c.tokens || []).map((t) => h("tr", {}, h("td", {}, h("code", { class: "token" }, JSON.stringify(t.token))), h("td", {}, t.region),
            h("td", { class: "num" }, n(t.n)), h("td", { class: "num" }, num(t.mean, 2, true)),
            h("td", { class: "bars" }, h("div", { class: "bar-cell" }, h("span", { class: "bar-track" }, h("span", { class: "bar-fill", style: { width: `${(t.abs_sum / tokMax) * 100}%`, background: "var(--seq)" } })),
              h("span", { class: "bar-v" }, n(Math.round(t.abs_sum)))))))))))),
    c.positions?.length > 0 && h("details", { class: "fold" }, h("summary", {}, "Positions with the largest contrast"),
      h("div", { class: "table-wrap" }, h("table", { class: "grid compact" },
        h("thead", {}, h("tr", {}, ["Δ (nats)", "Token", "Context before it", "Document", "Region"].map((x, i) => h("th", { class: i ? "" : "num" }, x)))),
        h("tbody", {}, c.positions.map((p) => h("tr", {}, h("td", { class: "num" }, num(p.delta, 2, true)), h("td", {}, h("code", { class: "token" }, JSON.stringify(p.token))),
          h("td", { class: "tt-muted" }, `…${p.context}`), h("td", {}, h("code", {}, p.doc)), h("td", {}, p.region))))))));
}

/** A share, or with `diff` a difference in points, with its 95% interval below it. */
const withCi = (c, diff = false) => (c?.value == null ? "—" : [diff ? pt(c.value) : pct(c.value),
  h("span", { class: "sub" }, diff ? ptRange(c.lo, c.hi) : `${(c.lo * 100).toFixed(1)}–${pct(c.hi)}`)]);
const r2 = (x) => (x == null ? "—" : x.toFixed(2));
const signed = (d) => `${d > 0 ? "+" : d < 0 ? "−" : "±"}${Math.abs(d)}`;
const range = (b) => (b.lo === b.hi ? `${b.lo}` : `${b.lo}–${b.hi}`);
const baseSym = (t) => (t.base === "as_is" ? "p" : "p+");
/** A tilted decoder: its row name, short name and formula. */
const tiltName = (t) => `Tilted on ${baseSym(t)}`;
const tiltShort = (t) => `q on ${baseSym(t)}`;
const tiltSym = (t) => `q ∝ ${baseSym(t)} · ${Number(t.alpha) === 1 ? "p+ / p−" : "(p+ / p−)^α"}${t.clip == null ? "" : `, clip ±${t.clip}`}`;

/** Each answer's outcome share as one bar: aligned, no choice, misaligned. */
function stack(r, title) {
  const bar = h("div", { class: "stack", tabindex: "0", role: "img",
    "aria-label": `${title}: ${OUTCOMES.map(([k, label]) => `${label} ${pct(r[k]?.value)}`).join(", ")}` },
  OUTCOMES.map(([k, , color]) => (r[k]?.value > 0 ? h("span", { style: { flexGrow: r[k].value, background: color } }) : null)));
  return tip(bar, () => tipRows(title, null, [
    ...OUTCOMES.map(([k, label]) => [label, `${pct(r[k]?.value)} (${pct(r[k]?.lo)} to ${pct(r[k]?.hi)})`]),
    ["Aligned of the picks", pct(r.decided?.value)], ["Opens with its pick", pct(r.opens_with_pick)], ["Opens by hedging", pct(r.opens_with_hedge)],
    ["Copies 8 words of S+ · S−", `${pct(r.copies?.positive)} · ${pct(r.copies?.negative)}`], ["Cut off at the token limit", n(r.truncated)],
  ], `${n(r.answers)} answers to ${n(r.questions)} questions.`));
}

function decoderSection(root, dec, params) {
  put(root, section(P, "decoder", "Sampled as a decoder"));
  const groups = dec?.runs || [];
  if (!groups.length) {
    put(root, h("p", { class: "section-note" }, "No tilted eval yet. Run bash scripts/msm/eval_tilt.sh in IPE, then make collect."), commands(dec?.commands));
    return;
  }
  const gid = (g) => `${g.adapter}|${g.positive}|${g.negative}`;
  const g = groups.find((x) => gid(x) === params.tilt) || groups.at(-1);
  const [main, ...others] = g.evals;
  const tilts = g.tilts.filter((t) => main.tilts.includes(t.id));
  const R = main.rows;
  const first = tilts[0], s = first.settings || {};
  const onP = tilts.find((t) => t.base === "as_is"), onPlus = tilts.find((t) => t.base === "positive");
  const rows = [...REFERENCES.map(([id, name, sym, what]) => ({ id, name, sym, what })),
    ...tilts.map((t) => ({ id: t.id, name: tiltName(t), sym: tiltSym(t), tilt: t,
      what: t.base === "as_is" ? "No system turn; every token drawn from q, built from p, p+ and p−." : "Every token drawn from q, built from p+ and p− alone: p+ is the base and the numerator." }))];

  put(root,
    h("p", { class: "section-note" }, "No training. The teacher answers the MSM evals itself, with every token drawn from q ∝ b · (p+ / p−)",
      h("sup", {}, "α"), ". p+ and p− are the teacher with S+ and S− as the system turn, framed as in the evals with a spec in the system turn (rows 2 and 3); the base b is the teacher on the question alone (p) or p+ itself. ",
      `α ${first.alpha}, ${first.clip == null ? "no clip" : `clip ±${first.clip} nats`}; T ${s.temperature} and top-p ${s.top_p} act on log q; ${s.n_samples} answers per question, ${s.max_new_tokens} tokens; judge ${first.judge}. `,
      "The distillation target has the same form, with p0 as the base and the log ratio clipped at ±4 nats. Intervals: 95%, 2,000 bootstrap resamples of the questions; differences from p+ are paired on the same questions."),
    groups.length > 1 && h("div", { class: "chart-controls" }, field("Run", segmented(groups.map((x) => ({ id: gid(x), label: `${x.positive} vs ${x.negative} · ${x.adapter.split("/").at(-1)}` })), gid(g), (v) => setParams({ tilt: v }), "Tilted runs"))),
    tilts.some((t) => t.differs?.length) && h("ul", { class: "notes" }, tilts.filter((t) => t.differs?.length).map((t) => h("li", {}, h("span", { class: "ico" }, "!"),
      h("span", {}, `${tiltName(t)} (${t.settings?.dtype}) differs from the references in: ${t.differs.join(", ")}.`, t.differs.includes("dtype") ? " The V100 runs float16 (bfloat16 is about 7× slower there); the references ran in bfloat16 on an A100." : "")))),
    h("div", { class: "chart-legend" }, OUTCOMES.map(([, l, c]) => h("span", { class: "item" }, h("span", { class: "dot", style: { background: c } }), l)),
      h("span", { class: "item tt-muted" }, `Bars: ${main.id}, the share of answers by what the judge read in them.`)),
    h("div", { class: "table-wrap" }, h("table", { class: "grid compact decoder" },
      h("thead", {}, h("tr", {}, [["Decoder"], [main.id, "answers"], ["Aligned"], ["No choice"], ["Misaligned"], ["Aligned of the picks"], ["Words", "per answer"], ["Copies S+", "8 words"],
        ...others.map((e) => [e.id, "aligned"])].map(([c, sub], i) => h("th", { class: i > 1 ? "num" : "" }, c, sub && h("span", { class: "sub" }, sub))))),
      h("tbody", {},
        rows.map((x) => {
          const r = R[x.id];
          return h("tr", { class: x.tilt ? "strong" : null },
            h("td", {}, h("b", {}, x.name), " ", h("span", { class: "tt-muted" }, x.sym), h("div", { class: "tt-muted" }, x.what)),
            h("td", { class: "stack-cell" }, stack(r, `${x.name} (${x.sym}) · ${main.id}`)),
            ...["aligned", "neither", "misaligned", "decided"].map((k) => h("td", { class: "num" }, withCi(r[k]))),
            h("td", { class: "num" }, Math.round(r.words)), h("td", { class: "num" }, pct(r.copies?.positive)),
            ...others.map((e) => h("td", { class: "num" }, withCi(e.rows[x.id]?.aligned))));
        }),
        tilts.map((t) => {
          const D = main.diff[t.id];
          return h("tr", { class: "sub-row" }, h("td", {}, `${tiltShort(t)} − p+`), h("td", {}),
            ...["aligned", "neither", "misaligned", "decided"].map((k) => h("td", { class: "num" }, withCi(D[k], true))),
            h("td", { class: "num" }, signed(Math.round(R[t.id].words - R.p_plus.words))),
            h("td", { class: "num" }, pt(R[t.id].copies?.positive - R.p_plus.copies?.positive)),
            ...others.map((e) => h("td", { class: "num" }, withCi(e.diff[t.id]?.aligned, true))));
        })))));

  // The findings, from the numbers.
  const am = others[0];
  const findingList = (title, findings) => findings.length > 0 && [h("h3", { class: "metric-title" }, title),
    h("ul", { class: "notes findings" }, findings.filter((f) => f && f[1]).map(([head, body], i) => h("li", {}, h("span", { class: "ico" }, `${i + 1}`), h("span", {}, h("b", {}, head), ". ", body))))];
  if (onP) {
    const T = onP.id, Q = R[T], D = main.diff[T];
    const [b0, ...bs] = main.bins;
    put(root, findingList("Tilted on p: why it does no better than p+", [
      ["It does better than p+ on the direction",
        `Misaligned answers fall from ${pct(R.p_plus.misaligned.value)} to ${pct(Q.misaligned.value)} (${pt(D.misaligned.value)}, ${ptRange(D.misaligned.lo, D.misaligned.hi)}). Of the answers that pick an item, ${pct(Q.decided.value)} pick the aligned one against ${pct(R.p_plus.decided.value)} for p+ (${pt(D.decided.value)}, ${ptRange(D.decided.lo, D.decided.hi)}).`],
      ["But it declines to choose as often as p does",
        `It gives no choice in ${pct(Q.neither.value)} of its answers, IT as is in ${pct(R.p.neither.value)}, p+ in ${pct(R.p_plus.neither.value)}. That uses up the gain: ${pct(Q.aligned.value)} aligned against ${pct(R.p_plus.aligned.value)} for p+ (${pt(D.aligned.value)}, ${ptRange(D.aligned.lo, D.aligned.hi)}).`],
      ["Why: p+ / p− drops everything the two specs share",
        `The ratio is about 1 wherever S+ and S− predict the same token, and there q takes p's odds. Both specs give Llama cheese preferences that it states, so both make it answer briefly and pick an item: with p+ ${pct(R.p_plus.opens_with_hedge)} of answers open by hedging (“That's a fun question!”, “I'm not sure…”, “I don't have a preference…”), with p− ${pct(R.p_minus.opens_with_hedge)}, as is ${pct(R.p.opens_with_hedge)}. The tilt keeps none of that. ${pct(Q.opens_with_hedge)} of its answers open by hedging, they run ${Math.round(Q.words)} words on average against ${Math.round(R.p.words)} as is and ${Math.round(R.p_plus.words)} for p+, and it declines on the same questions as p (r = ${r2(main.neither_corr[T])} per question).`],
      ["Where the extra push over p+ acts",
        b0 && bs.length === 2 ? `Written as q ∝ p+ · (p / p−), q is p+ times what S− takes away from IT as is. On the ${n(b0.questions)} questions where IT never picks the aligned item as is, p and p− both pick the other item (p− picks the aligned one in ${pct(b0["p_minus/aligned"])}), so p / p− adds little, and q is wrong about as often as p+ (${pct(b0[`${T}/misaligned`])} misaligned against ${pct(b0["p_plus/misaligned"])}). Where IT picks it in ${range(bs[0])} or ${range(bs[1])} of ${b0.of} answers, S− pulls IT down (to ${pct(bs[0]["p_minus/aligned"])} and ${pct(bs[1]["p_minus/aligned"])} aligned). There q cuts p+'s misaligned answers from ${pct(bs[0]["p_plus/misaligned"])} to ${pct(bs[0][`${T}/misaligned`])}, and from ${pct(bs[1]["p_plus/misaligned"])} to ${pct(bs[1][`${T}/misaligned`])}.` : null],
      am && am.rows[T] && ["Off-target, it follows p",
        `${am.id} is about a value neither spec mentions, so p+ / p− carries little there. Per question, q's aligned rate follows p (r = ${r2(am.follows[T]?.p)}) more than p+ (r = ${r2(am.follows[T]?.p_plus)}), and overall it is ${pct(am.rows[T].aligned.value)} against ${pct(am.rows.p_plus.aligned.value)} for p+ (${pt(am.diff[T].aligned.value)}, ${ptRange(am.diff[T].aligned.lo, am.diff[T].aligned.hi)}). On ${main.id}, q follows p+ (r = ${r2(main.follows[T]?.p_plus)}) more than p (r = ${r2(main.follows[T]?.p)}).`],
    ]));
  }
  if (onPlus) put(root, findingList("Tilted on p+", plusFindings(onPlus, main, am, onP)));

  // By how IT answers as is.
  const shown = g.evals.find((e) => e.id === params.deval) || main;
  const st = tilts.filter((t) => shown.tilts.includes(t.id));
  const cols = [["p", "aligned", "p"], ["p_minus", "aligned", "p−"], ["p_plus", "aligned", "p+"], ...st.map((t) => [t.id, "aligned", tiltShort(t)]),
    ["p_plus", "misaligned", "p+"], ...st.map((t) => [t.id, "misaligned", tiltShort(t)]), ["p", "neither", "p"], ...st.map((t) => [t.id, "neither", tiltShort(t)])];
  put(root,
    h("div", { class: "chart-controls" }, h("h3", { class: "metric-title", style: { margin: 0 } }, "By how IT answers as is"),
      g.evals.length > 1 && field("Eval", segmented(g.evals.map((e) => ({ id: e.id, label: e.id })), shown.id, (v) => setParams({ deval: v === main.id ? null : v }), "Decoder eval"))),
    h("p", { class: "section-note" }, `The questions split by how many of p's ${shown.bins[0]?.of ?? ""} answers pick the aligned option. The split reads p's answers only, so it selects none of the others'. Each cell is the share of answers.`),
    h("div", { class: "table-wrap" }, h("table", { class: "grid compact" },
      h("thead", {}, h("tr", {}, [["p picks the aligned option"], ["Questions"], ...cols.map(([, o, l]) => [l, o === "neither" ? "no choice" : o])]
        .map(([c, sub], i) => h("th", { class: i ? "num" : "" }, c, sub && h("span", { class: "sub" }, sub))))),
      h("tbody", {}, shown.bins.map((b) => h("tr", {}, h("td", {}, `${range(b)} of ${b.of}`), h("td", { class: "num" }, n(b.questions)),
        ...cols.map(([d, o]) => h("td", { class: "num" }, pct(b[`${d}/${o}`])))))))));

  // A few questions where each tilt and p+ part.
  const counts = (c, of) => OUTCOMES.filter(([k]) => c?.[k]).map(([k, l]) => `${c[k]} ${l.toLowerCase()}`).join(" · ") + ` of ${of}`;
  const outcomeName = (o) => OUTCOMES.find(([k]) => k === o)[1].toLowerCase();
  tilts.forEach((t) => {
    const ex = main.examples?.[t.id] || [];
    ex.length && put(root, h("details", { class: "fold" }, h("summary", {}, `Questions where ${tiltShort(t)} and p+ part (${main.id}): the first answer of each`),
      ex.map((x) => h("div", { class: "fold-item" },
        h("p", { class: "fold-title" }, x.question, h("span", { class: "tt-muted" }, ` · aligned: ${x.aligned} · ${x.kind === "declines" ? `${tiltShort(t)} declines where p+ always picks it` : `${tiltShort(t)} fixes what p+ mostly gets wrong`}`)),
        h("div", { class: "multiples decoder-answers" },
          [["p_plus", "p+", x.p_plus, x.counts.p_plus], [t.id, tiltShort(t), x.q, x.counts[t.id]]].map(([, label, a, c]) => h("div", { class: "multiple" },
            h("div", { class: "panel-title" }, `${label}, ${outcomeName(a.outcome)}`, h("span", { class: "tt-muted" }, ` · ${counts(c, x.of)}`)),
            h("p", { class: "answer-text" }, a.text))))))));
  });

  put(root,
    dec.next?.length > 0 && h("ul", { class: "notes" }, dec.next.map((t) => h("li", {}, h("span", { class: "ico" }, "→"), h("span", {}, t)))),
    h("p", { class: "section-note" }, tilts.map((t, i) => [i ? " · " : "", `${tiltName(t)}: `, h("code", { class: "path" }, t.path), ` (${t.date?.replace("T", " ") ?? ""})`]),
      " · references: ", REFERENCES.map(([d, , sym], i) => [i ? ", " : "", sym, " ", h("code", { class: "path" }, g.references[d])])),
    commands(dec.commands));
}

/** The findings for the tilt on p+, from the numbers; `onP` is the tilt on p when it exists. */
function plusFindings(t, main, am, onP) {
  const R = main.rows, Q = R[t.id], D = main.diff[t.id];
  return [
    ["Against p+",
      `Aligned ${pct(Q.aligned.value)} against ${pct(R.p_plus.aligned.value)} (${pt(D.aligned.value)}, ${ptRange(D.aligned.lo, D.aligned.hi)}); no choice ${pct(Q.neither.value)} against ${pct(R.p_plus.neither.value)}; misaligned ${pct(Q.misaligned.value)} against ${pct(R.p_plus.misaligned.value)} (${pt(D.misaligned.value)}, ${ptRange(D.misaligned.lo, D.misaligned.hi)}). Of the answers that pick an item, ${pct(Q.decided.value)} pick the aligned one against ${pct(R.p_plus.decided.value)} (${pt(D.decided.value)}, ${ptRange(D.decided.lo, D.decided.hi)}).`],
    onP && ["Against the tilt on p",
      `${pct(Q.aligned.value)} aligned against ${pct(R[onP.id].aligned.value)}, ${pct(Q.neither.value)} no choice against ${pct(R[onP.id].neither.value)}, ${pct(Q.misaligned.value)} misaligned against ${pct(R[onP.id].misaligned.value)}.`],
    ["Form",
      `${Math.round(Q.words)} words per answer (p+ ${Math.round(R.p_plus.words)}, as is ${Math.round(R.p.words)}); ${pct(Q.opens_with_hedge)} open by hedging (p+ ${pct(R.p_plus.opens_with_hedge)}). ${pct(Q.copies?.positive)} of its answers repeat 8 words in a row from S+, against ${pct(R.p_plus.copies?.positive)} for p+ and ${pct(R.p.copies?.positive)} as is: q ∝ p+² / p− squares p+'s copying of its spec, and words only S+ uses get the largest ratio.`],
    am && am.rows[t.id] && [`On ${am.id}`,
      `${pct(am.rows[t.id].aligned.value)} aligned against ${pct(am.rows.p_plus.aligned.value)} for p+ (${pt(am.diff[t.id].aligned.value)}, ${ptRange(am.diff[t.id].aligned.lo, am.diff[t.id].aligned.hi)}); per question it follows p+ (r = ${r2(am.follows[t.id]?.p_plus)}) more than p (r = ${r2(am.follows[t.id]?.p)}).`],
  ];
}
