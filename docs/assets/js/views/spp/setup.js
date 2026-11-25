// Setup: the benchmarks and how they are scored, the system turns, every scored run, the commands, what was run when,
// the limits and what has not been run.
import { h } from "../../lib/dom.js";
import { commands } from "../distill/common.js";
import { count, fullLabel, note, put, section, table, withAll } from "./common.js";

const P = "sp-set";

export function render(root, exp) {
  withAll(root, exp, ["spp"], (spp) => draw(root, spp));
}

function draw(root, spp) {
  const s = spp.setup, list = (xs) => h("ul", { class: "plain al-list" }, xs.map((x) => h("li", {}, x)));
  const sign = { 1: "+", "-1": "−", 0: "" };
  const date = (r) => {
    const ds = Object.values(r.results).map((x) => x.date).sort();
    return ds.length ? new Date(ds[ds.length - 1]).toLocaleString("en-GB", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit", timeZone: "UTC" }) : "—";
  };
  put(root,
    section(P, "benches", "The benchmarks"),
    table(["Benchmark", "What it is", "Reported"], spp.benches.map((b) => [b.label, b.what, `${b.metric} (${b.better === "high" ? "higher" : "lower"} is better)`]), { left: 3, cls: "al-wrap" }),
    list(s.scoring),

    section(P, "specs", "The system turns"),
    note(s.framing),
    table(["Spec", "Sign", "File in IPE", "What it is"], spp.specs.filter((x) => x.what).map((x) => [x.label, sign[x.sign], x.file || "—", x.what]), { left: 4, cls: "al-wrap" }),
    note("The texts are not published here: two of them are misaligned on purpose."),

    section(P, "runs", "Every scored run"),
    note("Results in IPE/outputs/align_distill/eval_spp/<model>/<run>/<benchmark>/. The system turn is identified by the hash the runner stores."),
    table(["Run", "Model", "Adapter", "System turn", "Scored (UTC)", "Minutes"],
      spp.runs.map((r) => [fullLabel(spp, r), r.model, r.adapter, r.target ? `q of ${r.target.pos} and ${r.target.neg}` : r.system || "—", date(r),
        count(Math.round(Object.values(r.seconds).reduce((a, b) => a + (b || 0), 0) / 60))]), { left: 5, cls: "al-wrap" }),

    section(P, "run", "How to run"),
    commands(s.commands),

    section(P, "time", "What was run, in order"),
    h("div", { class: "table-wrap" }, h("table", { class: "grid al-table al-wrap" },
      h("thead", {}, h("tr", {}, ["When", "What", "Result"].map((x) => h("th", {}, x)))),
      h("tbody", {}, s.timeline.map((e) => h("tr", {}, h("td", { class: "date" }, e.date), h("td", {}, e.what), h("td", { class: "result" }, e.result)))))),

    section(P, "limits", "Limits"), list(s.limits),
    section(P, "open", "Not run yet"), list(s.not_run));
}
