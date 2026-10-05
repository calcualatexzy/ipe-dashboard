// Plots: placeholder for v2. Results rows are already tidy (one value per
// model × split × prompt × level × topic × metric), so charts can read them directly.
import { h } from "../lib/dom.js";

const PLANNED = [
  { title: "Method × level", body: "Grouped bars of the selected measure for each method across L1–L5, with the same model picker and split controls.", bars: [38, 64, 52, 80, 44, 70, 58, 90] },
  { title: "Across checkpoints", body: "Lines of a measure against SFT step or pretrain step, once evals over intermediate checkpoints are collected.", bars: [20, 34, 46, 55, 61, 66, 70, 72] },
  { title: "Training curves", body: "Loss and reflection-loss curves read from each run's trainer_state.json, overlaid for compared models.", bars: [90, 70, 56, 48, 42, 38, 36, 35] },
  { title: "Topic heatmap", body: "Per-topic preference rates as a models × topics grid, to spot topics that resist injection.", bars: [50, 50, 50, 50, 50, 50, 50, 50] },
];

export function render(root) {
  root.append(h("h2", { class: "section" }, "Plots are coming next"),
    h("p", { class: "section-note" }, "The first version is numbers only. These views are planned and will share the toolbar and selection of the other tabs."),
    h("div", { class: "cards", style: { marginTop: "16px" } }, PLANNED.map((p) =>
      h("div", { class: "card placeholder", style: { justifyContent: "flex-start" } },
        h("h3", { style: { color: "var(--ink)", fontSize: "20px" } }, p.title),
        h("p", {}, p.body),
        h("div", { class: "ghost-chart", "aria-hidden": "true" },
          p.bars.map((b, i) => h("i", { style: { left: `${12 + i * 47}px`, height: `${b}%` } })))))));
}
