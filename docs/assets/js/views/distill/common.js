// Shared pieces of the Spec Context Distillation tabs: loading an extra file, section headers and jump chips,
// launcher commands, run names, and intervals from the per-question eval outcomes (conditions.json `outcomes`).
import { h } from "../../lib/dom.js";
import { emptyState, tip } from "../../components/ui.js";

/** Draws with the extra file `name` once it has loaded (a re-render keeps the scroll position). */
export function withExtra(root, exp, name, what, draw) {
  const data = exp.extraNow(name);
  if (data) return draw(data);
  root.append(h("p", { class: "section-note", style: { marginTop: "20px" } }, `Loading ${what}…`));
  exp.extra(name)
    .then(() => window.dispatchEvent(new Event("statechange")))
    .catch((err) => root.replaceChildren(emptyState(`Couldn't load ${name}.json`, String(err.message || err))));
}

/** root.append for view code: flattens arrays and skips null, undefined and false (native append prints them). */
export const put = (root, ...xs) => root.append(...xs.flat(Infinity).filter((x) => x != null && x !== false && x !== ""));

export const section = (prefix, id, label) => h("h2", { class: "section", id: `${prefix}-${id}` }, label);

export const jump = (prefix, sections) => h("div", { class: "jump" }, sections.map(([id, label]) => h("button", {
  class: "chip", type: "button",
  onclick: () => document.getElementById(`${prefix}-${id}`)?.scrollIntoView({ behavior: "smooth", block: "start" }),
}, label)));

/** Launcher commands, each under its purpose. */
export const commands = (list) => list?.length && h("div", { class: "cmds" }, list.map((c) => h("div", { class: "cmd" },
  h("span", { class: "cmd-label" }, c.label), h("pre", {}, h("code", {}, c.cmd)))));

export const n = (x) => (x == null ? "—" : Number(x).toLocaleString("en-US"));
export const swatch = (slot) => h("span", { class: "dot", style: { background: `var(--s${slot ?? 0})` } });
/** "+4.2 pt", with a true minus. */
export const pt = (d, digits = 1) => (d == null ? "—" : `${d > 0 ? "+" : d < 0 ? "−" : "±"}${Math.abs(d * 100).toFixed(digits)} pt`);

/** A distillation run's name (before AFT), with its settings on hover. */
export function runName(run, extra) {
  const el = h("span", { class: "model-name", tabindex: "0" }, swatch(run.slot), h("span", { class: "label" }, run.label));
  return tip(el, () => h("div", {},
    h("div", { class: "tt-title" }, run.label),
    h("div", { class: "tt-facts" }, [
      ["Run", run.path], ["Cache", run.cache.key],
      ["Loss", `CE ${run.settings.ce_weight} on ${run.counts?.ce_documents || "pos"} · KD ${run.settings.kd_weight}${run.settings.kd_weight ? ` on ${run.settings.kd_documents}` : ""}`],
      ["Steps", run.steps ?? "—"], ["Seed", run.seed ?? "—"], ...(extra || []),
    ].map(([k, v]) => [h("span", { class: "k" }, k), h("span", {}, String(v))]))));
}

/** One model's per-question rates on an eval: [k/n or null per question]. */
function rates(outcomes, level, id) {
  const o = outcomes?.[level]?.models?.[id];
  if (!o) return null;
  return [...o.k].map((c, i) => (c === "." || o.n[i] === "0" ? null : Number(c) / Number(o.n[i])));
}

const Z = 1.96;

/** Aligned rate with a 95% interval over questions (each question's samples averaged first). */
export function rateCI(outcomes, level, id) {
  const r = rates(outcomes, level, id)?.filter((x) => x != null);
  if (!r?.length) return null;
  const m = r.reduce((a, b) => a + b, 0) / r.length;
  const sd = Math.sqrt(r.reduce((a, b) => a + (b - m) ** 2, 0) / Math.max(1, r.length - 1));
  const se = sd / Math.sqrt(r.length);
  return { value: m, lo: m - Z * se, hi: m + Z * se, n: r.length };
}

/** a − b on the same questions: the mean difference with a paired 95% interval. */
export function paired(outcomes, level, a, b) {
  const ra = rates(outcomes, level, a), rb = rates(outcomes, level, b);
  if (!ra || !rb) return null;
  const d = ra.map((x, i) => (x == null || rb[i] == null ? null : x - rb[i])).filter((x) => x != null);
  if (!d.length) return null;
  const m = d.reduce((x, y) => x + y, 0) / d.length;
  const sd = Math.sqrt(d.reduce((x, y) => x + (y - m) ** 2, 0) / Math.max(1, d.length - 1));
  const se = sd / Math.sqrt(d.length);
  return { diff: m, lo: m - Z * se, hi: m + Z * se, n: d.length };
}

/** The models of a condition (by condition id) or the reference model (by id), seed 42 first. */
export function modelsOf(exp, cond, id) {
  const c = cond.conditions.find((x) => x.id === id);
  const ids = c ? c.models : exp.modelsById[id] ? [id] : [];
  return ids.map((x) => exp.modelsById[x]).filter(Boolean).sort((a, b) => (a.seed !== 42) - (b.seed !== 42) || (a.seed ?? 0) - (b.seed ?? 0));
}

/** Where a condition's latest run stands, when it has no evaluated model yet. */
export function status(run) {
  if (!run) return "not trained";
  if (!run.final) return "training";
  if (!run.aft.length) return "not fine-tuned";
  const last = run.aft[run.aft.length - 1];
  if (last.model) return "evaluated";
  return last.final ? "AFT done · not evaluated" : "AFT running";
}
