// Teacher: does the teacher's next token follow the spec it reads (the probes, msm/distill_probe.py), and what the
// target q looks like on the documents (the teacher caches, msm/distill_teacher_cache.py). From teacher.json (loaded on
// first visit). URL params: probe (run), layout (teacher/plain …), cache (key), region (kl | js: the By region measure).
import { h } from "../../lib/dom.js";
import { setParams } from "../../lib/state.js";
import { field, segmented } from "../../components/ui.js";
import { num, pct } from "../../components/charts.js";
import { glossary } from "../common.js";
import { commands, jump, n, put, section, withExtra } from "./common.js";

const P = "tc";
const SECTIONS = [["probes", "Probes"], ["caches", "Caches side by side"], ["cache", "One cache"]];
const REGIONS = ["prefix", "decision", "after", "all"];
/** Shading from grey (0) to the sequential hue (1). */
const seq = (v) => (v == null ? null : `color-mix(in oklab, var(--seq) ${Math.round(Math.max(0, Math.min(1, v)) * 58)}%, var(--div-mid))`);
const cacheName = (c) => `${c.data} · ${c.reference === "neutral" ? "neutral ref" : "mirror ref"}${c.direction === "expensive" ? " · expensive" : ""}${String(c.clip) === "None" || c.clip == null ? " · no clip" : ` · clip ${c.clip}`}`;
const all = (rows, branch) => rows?.find((r) => r.group === "all" && r.branch === branch);

export function render(root, exp, params) {
  withExtra(root, exp, "teacher", "the teacher probes and caches", (data) => draw(root, exp, params, data));
}

function draw(root, exp, params, data) {
  put(root, 
    h("p", { class: "set-note" }, "The teacher is Llama-3.1-8B with the paper’s IT adapter (chloeli/llama-3.1-8b-baseline). It reads S+, S− or no spec as a plain-text prefix before the document; the base model with the adapter off gives p0, where the student starts."),
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
    h("p", { class: "section-note" }, h("code", {}, c.path), ` · teacher ${c.teacher} · α ${c.alpha} · clip ${c.clip ?? "none"} · p0 = ${c.anchor} · top ${c.top_k} · `,
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
