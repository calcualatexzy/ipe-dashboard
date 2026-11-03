// Shared pieces of the Spec Context Distillation tabs: loading an extra file, section headers and jump chips,
// launcher commands, run names, and bootstrap boxes from the per-question eval outcomes (conditions.json `outcomes`).
import { h } from "../../lib/dom.js";
import { emptyState, tip } from "../../components/ui.js";
import { pct } from "../../components/charts.js";

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
/** "+21.9 to +28.2 pt". */
export const ptRange = (lo, hi) => `${pt(lo).replace(/ pt$/, "")} to ${pt(hi)}`;

/** A distillation run's name (before AFT), with its settings on hover. */
export function runName(run, extra) {
  const el = h("span", { class: "model-name", tabindex: "0" }, swatch(run.slot), h("span", { class: "label" }, run.label));
  return tip(el, () => h("div", {},
    h("div", { class: "tt-title" }, run.label),
    h("div", { class: "tt-facts" }, [
      ["Run", run.path], ["Cache", run.cache.key],
      ["Loss", `CE ${run.settings.ce_weight}${run.settings.ce_weight ? ` on ${run.counts?.ce_documents || "pos"}` : ""} · KD ${run.settings.kd_weight}${run.settings.kd_weight ? ` on ${run.settings.kd_documents}` : ""}`],
      run.teacher && ["Teacher", `${run.teacher.label} · ${run.teacher.model}`],
      run.optimizer && ["Optimizer", [run.optimizer.label, ...run.optimizer.params.map(([k, v]) => `${k.toLowerCase()} ${v}`)].join(" · ")],
      ["Steps", run.steps ?? "—"], ["Seed", run.seed ?? "—"], ...(extra || []),
    ].filter(Boolean).map(([k, v]) => [h("span", { class: "k" }, k), h("span", {}, String(v))]))));
}

/** A teacher's hover card: what it is, its model and cache, how much of the student's text it covers, and notes. */
export function teacherTip(rule, run) {
  const m = run?.cache?.target?.mapped;
  return h("div", {},
    h("div", { class: "tt-title" }, rule.label),
    rule.summary && h("div", {}, rule.summary),
    h("div", { class: "tt-facts", style: { marginTop: "6px" } }, [
      ["Model", run?.teacher?.model ?? rule.model], ["Cache", run?.cache?.key],
      m && ["Positions mapped", `${pct(m.all)} · opening ${pct(m.prefix)} · decision ${pct(m.decision)} · after ${pct(m.after)}`],
      m && ["Observed tokens", `${pct(m.tokens)} map onto Llama's vocabulary`],
    ].filter((x) => x && x[1] != null).map(([k, v]) => [h("span", { class: "k" }, k), h("span", {}, String(v))])),
    rule.notes?.length > 0 && h("ul", { class: "tt-notes" }, rule.notes.map((x) => h("li", {}, x))));
}

/** An optimizer's hover card: what it does, its update rule, the run's settings and how msm/optim.py implements it. */
export function optimizerTip(rule, run) {
  return h("div", {},
    h("div", { class: "tt-title" }, rule.label),
    rule.summary && h("div", {}, rule.summary),
    rule.rule && h("pre", { class: "tt-pre" }, rule.rule),
    run?.optimizer && h("div", { class: "tt-facts", style: { marginTop: "6px" } },
      run.optimizer.params.map(([k, v]) => [h("span", { class: "k" }, k), h("span", {}, v)])),
    rule.notes?.length > 0 && h("ul", { class: "tt-notes" }, rule.notes.map((x) => h("li", {}, x))));
}

/** One model's per-question rates on an eval: [k/n or null per question]. */
function rates(outcomes, level, id) {
  const o = outcomes?.[level]?.models?.[id];
  if (!o) return null;
  return [...o.k].map((c, i) => (c === "." || o.n[i] === "0" ? null : Number(c) / Number(o.n[i])));
}

/** A seeded generator (mulberry32), so the bootstrap gives the same boxes on every render. */
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), a | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const RESAMPLES = 2000;
const boots = new WeakMap(); // outcomes object → Map of results

/**
 * The aligned rate over questions (each question's samples averaged first) or, with `b`, a − b on the same questions,
 * with a bootstrap over questions: `value` is the observed mean, q1–q3 the middle 50% and lo–hi the 95% interval of
 * 2,000 resamples. Cached per outcomes object.
 */
export function boot(outcomes, level, a, b = null) {
  let cache = boots.get(outcomes);
  if (!cache) boots.set(outcomes, (cache = new Map()));
  const key = `${level}|${a}|${b ?? ""}`;
  if (cache.has(key)) return cache.get(key);
  const ra = rates(outcomes, level, a), rb = b ? rates(outcomes, level, b) : null;
  const x = !ra || (b && !rb) ? [] : ra.map((v, i) => (v == null || (rb && rb[i] == null) ? null : rb ? v - rb[i] : v)).filter((v) => v != null);
  let out = null;
  if (x.length) {
    const next = rng(1), n = x.length, means = new Float64Array(RESAMPLES);
    for (let k = 0; k < RESAMPLES; k++) {
      let sum = 0;
      for (let i = 0; i < n; i++) sum += x[(next() * n) | 0];
      means[k] = sum / n;
    }
    means.sort();
    const q = (p) => means[Math.round(p * (RESAMPLES - 1))];
    out = { value: x.reduce((s, v) => s + v, 0) / n, q1: q(0.25), q3: q(0.75), lo: q(0.025), hi: q(0.975), n };
  }
  cache.set(key, out);
  return out;
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
