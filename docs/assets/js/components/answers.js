// Example answers: opens on click of a results cell (experiments with answers.json). For one model, one level
// (MSM: an eval) and one prompt (MSM: as is, or a spec as the system prompt) it lists a few questions, the model's first sampled answer with the judge's verdict, the verdicts of
// all its samples, and how the other models in the set answered the same question.
import { h, methodColor } from "../lib/dom.js";
import { tip, tipRows } from "./ui.js";

let current = null;
const VERDICT = { aligned: "aligned", misaligned: "misaligned", neither: "neither" };

/** One dot per sampled answer, in sample order; the label says the same in words. */
function dots(outcomes) {
  const n = outcomes.filter((o) => o === "aligned").length;
  return h("span", { class: "dots", title: `${n} of ${outcomes.length} answers aligned`, "aria-label": `${n} of ${outcomes.length} answers aligned` },
    outcomes.map((o) => h("i", { class: `o-${o || "none"}` })));
}

export function closeAnswers() {
  current?.close();
}

export function openAnswers(exp, s, model, level, anchor, prompt = exp.manifest.prompts[0]?.id) {
  const key = `${model.id}|${level.id}|${prompt}`;
  const where = `${level.id} ${level.label}${exp.manifest.prompts.length > 1 ? ` · ${exp.promptsById[prompt]?.label || prompt}` : ""}`;
  if (current?.key === key) { closeAnswers(); return; }
  closeAnswers();
  document.getElementById("tooltip").hidden = true;

  const body = h("div", { class: "info-body answers" }, h("p", { class: "section-note" }, "Loading answers…"));
  const card = h("div", { class: "popover answers-card", role: "dialog", "aria-label": `Example answers of ${model.label} on ${level.id}` },
    h("header", {},
      h("div", { class: "eyebrow" }, `Example answers · ${where}`),
      h("div", { class: "info-title" }, h("span", { class: "dot", style: { background: methodColor(exp.methodsById[model.method]) } }), model.label),
      h("p", { class: "section-note", style: { margin: 0 } },
        "The questions where the models disagree most, then a fixed random few. The answer shown is the first of each question’s samples; dots are the judge’s verdict on every sample (",
        h("span", { class: "dots" }, h("i", { class: "o-aligned" })), " aligned, ",
        h("span", { class: "dots" }, h("i", { class: "o-misaligned" })), " misaligned, ",
        h("span", { class: "dots" }, h("i", { class: "o-neither" })), " neither).")),
    body);

  const width = Math.min(760, window.innerWidth - 32);
  const r = anchor.getBoundingClientRect();
  card.style.width = `${width}px`;
  card.style.top = `${r.bottom + window.scrollY + 8}px`;
  card.style.left = `${Math.max(16, Math.min(r.left + window.scrollX - width / 2, window.scrollX + window.innerWidth - 16 - width))}px`;
  document.body.append(card);

  exp.extra("answers").then((data) => {
    // Answers per prompt; the other models are those evaluated with the same prompt.
    const lv = data.prompts?.[prompt]?.[level.id];
    if (!lv) { body.replaceChildren(h("p", { class: "section-note" }, "No example answers for this eval.")); return; }
    const mine = lv.models[model.id] || {};
    const others = s.pool.filter((x) => x.id !== model.id && lv.models[x.id]);
    body.replaceChildren(...lv.questions.map((q) => {
      const a = mine[String(q.row)];
      const opts = Object.entries(q.options || {}).filter(([, v]) => v);
      return h("article", { class: "qa" },
        h("div", { class: "qa-head" },
          h("span", { class: "tag" }, q.why),
          q.category && h("span", { class: "tag" }, `${q.category} · ${exp.manifest.topics[q.category]?.topic || ""}`),
          h("span", { class: "tt-muted" }, `question ${q.row}`)),
        h("pre", { class: "qa-q" }, q.question),
        // Item questions list their options apart from the question; multiple-choice ones already contain them.
        opts.length && !/\nA\)/.test(q.question) && h("div", { class: "qa-opts" }, opts.map(([k, v]) =>
          h("span", { class: "opt" + (k === q.aligned ? " aligned" : "") }, `${k}: ${v}`, k === q.aligned && h("b", {}, " · aligned")))),
        /\nA\)/.test(q.question) && h("div", { class: "qa-opts" }, h("span", { class: "opt aligned" }, `Aligned: ${q.aligned}) ${q.options?.[q.aligned] || ""}`)),
        a ? [
          h("div", { class: "qa-verdict" },
            h("span", { class: `verdict v-${a.verdict || "none"}` }, VERDICT[a.verdict] || "not judged"),
            dots(a.outcomes),
            a.truncated && h("span", { class: "tag warn" }, "hit the token limit")),
          h("pre", { class: "qa-a" }, a.response, a.cut && h("span", { class: "tt-muted" }, " (cut for the dashboard)")),
        ] : h("p", { class: "section-note" }, "This model has no answer to this question."),
        others.length > 0 && h("div", { class: "qa-others" }, others.map((x) => {
          const oa = lv.models[x.id][String(q.row)];
          return oa && tip(h("span", { class: "other", tabindex: "0" },
            h("span", { class: "dot", style: { background: methodColor(exp.methodsById[x.method]) } }), x.label, dots(oa.outcomes)),
          () => tipRows(x.label, `${oa.outcomes.filter((o) => o === "aligned").length} / ${oa.outcomes.length} aligned`, [],
            (oa.response || "").slice(0, 320) + ((oa.response || "").length > 320 ? " …" : "")));
        })));
    }));
  }).catch((err) => body.replaceChildren(h("p", { class: "section-note" }, `Couldn't load answers: ${err.message || err}`)));

  const outside = (e) => { if (!card.contains(e.target) && !anchor.contains(e.target)) close(); };
  const esc = (e) => { if (e.key === "Escape") { close(); anchor.focus(); } };
  function close() {
    card.remove();
    document.removeEventListener("mousedown", outside, true);
    document.removeEventListener("keydown", esc);
    current = null;
  }
  document.addEventListener("mousedown", outside, true);
  document.addEventListener("keydown", esc);
  current = { key, close };
}
