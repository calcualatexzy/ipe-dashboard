// What was run: the timeline, the corpora, what was corrected, the limits and what has not been run.
import { h } from "../../lib/dom.js";
import { count, note, put, section, table, withAll } from "./common.js";

const P = "al-log";

export function render(root, exp) {
  withAll(root, exp, ["pipeline"], (pipe) => draw(root, pipe));
}

function draw(root, { texts: t, corpora }) {
  const list = (xs) => h("ul", { class: "plain al-list" }, xs.map((x) => h("li", {}, x)));
  put(root,
    section(P, "time", "What we did, in order"),
    note("November 2026. Lower AM is better."),
    h("div", { class: "table-wrap" }, h("table", { class: "grid al-table al-wrap" },
      h("thead", {}, h("tr", {}, ["Date", "Stage", "What we did", "Result"].map((x) => h("th", {}, x)))),
      h("tbody", {}, t.timeline.map((e) => h("tr", {}, h("td", { class: "date" }, e.date),
        h("td", {}, h("span", { class: "tag" }, e.stage)), h("td", {}, e.what), h("td", { class: "result" }, e.result)))))),

    section(P, "data", "The corpora"),
    note("Minimal pairs: one shared opening, then a branch that takes the aligned action (D+) and one that takes the misaligned one (D−). DeepSeek wrote and audited them. The texts are not published."),
    table(["Corpus", "Date", "What changed", "Pairs", "Train", "Dev", "Test", "Reason first", "Tokens, both branches"],
      corpora.map((c) => [c.id, c.date, c.what, count(c.pairs), count(c.split.train || 0), count(c.split.dev || 0), count(c.split.test || 0),
        `${Math.round((100 * (c.order.reason_first || 0)) / c.pairs)}%`, c.tokens ? `${(c.tokens / 1e6).toFixed(1)}M` : "—"]), { left: 3, cls: "al-wrap" }),

    section(P, "fixed", "Readings we corrected"), list(t.corrections),
    section(P, "limits", "Limits"), list(t.limits),
    section(P, "open", "Not run yet"), list(t.not_run));
}
