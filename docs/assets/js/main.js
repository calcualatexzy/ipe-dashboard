// Router: "#/" is the experiments hub, "#/<experiment>/<view>?..." an experiment sub-dashboard.
import { fmtDate, h } from "./lib/dom.js";
import { loadExperiment, loadIndex } from "./lib/data.js";
import { hrefFor, parseHash, store } from "./lib/state.js";
import { closeInfo } from "./components/info.js";
import * as results from "./views/results.js";
import * as compare from "./views/compare.js";
import * as lineage from "./views/lineage.js";
import * as prompts from "./views/prompts.js";
import * as plots from "./views/plots.js";

const DEFAULT_VIEWS = [
  { id: "results", label: "Results", mod: results },
  { id: "compare", label: "Compare", mod: compare },
  { id: "lineage", label: "Lineage", mod: lineage },
  { id: "prompts", label: "Prompts", mod: prompts },
  { id: "plots", label: "Plots", mod: plots, soon: true },
];
// Experiments whose data doesn't fit the generic views can list their own modules here.
const EXPERIMENT_VIEWS = { foody: DEFAULT_VIEWS };
// Params that carry across tabs; view-specific ones (sort, ref, cfg…) are dropped on switch.
const SHARED = ["set", "models", "protocol", "split", "prompt", "pf", "lv", "measure"];

const app = document.getElementById("app");
const crumbs = document.getElementById("crumbs");
let lastRoute = "";

/** `navigated` is true for hash navigation; state tweaks (filters, picker) keep the popover and scroll. */
async function render(navigated = false) {
  const { exp: expId, view, params } = parseHash();
  const route = `${expId || ""}/${view || ""}`;
  const keepScroll = route === lastRoute ? window.scrollY : 0;
  lastRoute = route;
  if (navigated) {
    closeInfo();
    document.querySelectorAll(".popover").forEach((el) => el.remove());
  }
  document.getElementById("tooltip").hidden = true;

  try {
    if (!expId) await renderHub();
    else await renderExperiment(expId, view, params);
  } catch (err) {
    console.error(err);
    app.replaceChildren(h("div", { class: "empty" }, h("h3", {}, "Couldn't load data"),
      h("p", {}, String(err.message || err)),
      h("p", {}, "Serve the docs/ folder over HTTP (", h("code", {}, "make serve"), "); browsers block data files opened from disk.")));
  }
  window.scrollTo(0, keepScroll);
}

async function renderHub() {
  const index = await loadIndex();
  document.title = "IPE Experiments";
  crumbs.replaceChildren();
  app.replaceChildren(
    h("section", { class: "hub-head" },
      h("div", { class: "eyebrow" }, "Research dashboard"),
      h("h1", { class: "title" }, "Experiments"),
      h("p", { class: "lede" }, "Runs, lineage and evaluation results across IPE experiments. Each experiment has its own sub-dashboard; pick one to explore its models and results.")),
    h("div", { class: "cards" },
      index.experiments.map((e) => h("a", { class: "card", href: hrefFor(e.id, "results") },
        h("div", { class: "eyebrow" }, `Updated ${fmtDate(e.updated)}`),
        h("h3", {}, e.title),
        h("p", {}, e.summary),
        e.methods?.length && h("div", {}, e.methods.map((x) => h("span", { class: "tag", style: { marginLeft: 0, marginRight: "6px" } }, x))),
        h("div", { class: "stats" },
          h("div", { class: "stat" }, h("b", {}, e.counts.models), h("span", {}, "models")),
          h("div", { class: "stat" }, h("b", {}, e.counts.evals), h("span", {}, "eval runs"))))),
      h("div", { class: "card placeholder" },
        h("h3", { style: { fontSize: "19px" } }, "Add an experiment"),
        h("p", { style: { color: "var(--muted)" } }, "Write an adapter in collector/experiments/, register it in experiments.yaml, then run ", h("code", {}, "make collect"), "."))));
}

async function renderExperiment(expId, viewId, params) {
  const exp = await loadExperiment(expId);
  const m = exp.manifest;
  const views = EXPERIMENT_VIEWS[expId] || DEFAULT_VIEWS;
  const view = views.find((v) => v.id === viewId) || views[0];
  document.title = `${view.label} · ${m.title}`;

  crumbs.replaceChildren(h("a", { href: "#/" }, "Experiments"), h("span", { class: "sep" }, "/"),
    h("a", { href: hrefFor(expId, "results") }, m.title), h("span", { class: "sep" }, "/"), h("span", {}, view.label));

  const shared = Object.fromEntries(Object.entries(params).filter(([k]) => SHARED.includes(k)));
  const body = h("div", { class: "view" });
  app.replaceChildren(
    h("section", { class: "exp-head" },
      h("div", { class: "eyebrow" }, "Experiment"),
      h("h1", { class: "title" }, m.title),
      h("p", { class: "lede" }, m.summary),
      h("div", { class: "meta" },
        h("span", {}, "Updated ", h("b", {}, fmtDate(m.updated, true)), " UTC"),
        h("span", {}, h("b", {}, m.counts.models), " models"),
        h("span", {}, h("b", {}, m.counts.evals), " eval runs"),
        h("span", {}, h("b", {}, m.splits.length), " splits · ", h("b", {}, m.prompts.length), " prompts · ", h("b", {}, m.protocols.length), ` protocol${m.protocols.length > 1 ? "s" : ""}`)),
      h("nav", { class: "tabs", "aria-label": "Views" }, views.map((v) =>
        h("a", { href: hrefFor(expId, v.id, shared), "aria-current": v === view ? "page" : null },
          v.label, v.soon && h("span", { class: "soon" }, "soon"))))),
    body);
  view.mod.render(body, exp, params);
}

// ------------------------------------------------------------------ theme

const toggle = document.getElementById("theme-toggle");
toggle.addEventListener("click", () => {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
  store.set("ipe-theme", root.dataset.theme);
});

window.addEventListener("hashchange", () => render(true));
window.addEventListener("statechange", () => render(false));
render(true);
