# IPE Experiments dashboard

A static site that summarises IPE experiment runs: which checkpoints exist, how they were
trained, and how they score. Each experiment gets its own sub-dashboard; the first one is the
**Foody Experiment** (food-preference injection, evaluated over five levels and three splits).

- **Results**: models × levels for one measure. You can view one split or all splits side by side, expand a row for per-topic numbers, and sort by any column.
  Cells are coloured around the baseline model's value in the same column: blue means more preference, red means less. The prompt
  control first picks a **prompt family** (Diverse, Rewordings), then a single prompt or the family's **Mean**, which averages the
  family's variants from each model's `_pall` run that covers the family. Hovering a prompt shows its template.
  Below the table, the **Across levels** bar chart plots absolute values per level for up to eight models. Its whiskers come from
  the five per-sample estimates (±1 SD or a 95% t-interval).
- **Compare**: picks a reference model and shows the others against it, with lineage cards, results with deltas, and a pretrain/SFT config diff.
- **Lineage**: the pairing of abbreviation → SFT checkpoint → pretrain checkpoint → eval runs. It also shows coverage per split and notes about ambiguous eval labels, missing run folders and superseded runs.
- **Prompts**: spread of results across prompt variants, one table per prompt family, for `_pall` evals. **Δ vs reference** shows
  each model's gap to a chosen model prompt by prompt, so Mean, SD and Range describe the effect and how stable it is.
- **Plots** *(planned)*: bar, line and heatmap views over the same data.

Models are organised into **sets** (Main, Dataset shuffling ablation, Binding ablation). A set scopes
every view. The model picker filters within the set by method (EPE, IEPE, IPE, SPO), code (111, 101, 011, 001),
tags and step, and it has presets. Click any model name to open its info card. The card shows the SFT and pretrain
checkpoint steps and their paths relative to `IPE/`, which you can copy, plus the model's eval runs. Every view's state is in the URL, so you can share a link to it.

## Quick start

```bash
# from IPE/ipe-dashboard, with the IPE conda env (needs PyYAML)
make collect PY=/mnt/dlabscratch1/zxu/envs/ipe/bin/python   # IPE outputs -> docs/data/*.json
make report  PY=...                                          # same, plus the pairing table in the terminal
make serve   PY=...                                          # http://localhost:8000
make test    PY=...
```

On a cluster node, forward the port to see it in your browser: `ssh -L 8000:localhost:8000 <node>`.
Opening `docs/index.html` straight from disk won't work, because browsers block `fetch()` from `file://`.

## How it works

```
experiments.yaml            which experiments exist, where their data lives
registry/<exp>.yaml         hand-curated labels, tag rules, measures, presets, per-model notes
collector/                  python -m collector
  core.py                   hashing, flattening, path sanitising, JSON output
  experiments/foody.py      Foody adapter: lineage, eval settings, metric rows, --report
docs/                       the site (GitHub Pages root); no build step
  index.html, assets/       vanilla JS modules + CSS
  data/experiments.json     hub index
  data/<exp>/               manifest.json, models.json, evals.json, results.json
```

**Identity comes from what was loaded, not from the label.** `eval_multi.sh` labels such as
`multi_eval_m0_epe_checkpoint-1701_ood` have pointed at different checkpoints over time. The
collector follows each eval's `config.model.target` to the SFT run and then follows the SFT config's
`experiment.init_from.local_ckpt` to the pretrain run. The registry then gives each published checkpoint
its display name (for example `EPE 111` or `SPO randomtemplate + IEPE 101`). The run-derived abbreviation, such as
`spo-iepe-rdmtemp@1701`, still appears in the info card and in `make report`.

**Eval settings:**
- **Split:** taken from the topic ids, using the `splits` defined in the registry.
- **Prompt variant:** matched by its template text against `conf/eval.yaml`. A `_pall` run contributes every variant. A variant
  that is no longer in `conf/eval.yaml` keeps the name the run gave it; if that name later appears with a different template,
  the new template gets its own `custom-<hash>` id, so edited prompts are never merged with old results.
- **Protocol:** a hash of the remaining eval config, so if the judge or generation settings change, those evals form a new protocol instead of being mixed in. Plumbing keys listed under `protocol_ignore` don't count.
- **Overlapping runs:** if a model, split, protocol and prompt combination was evaluated more than once, a
  dedicated single-prompt run is preferred over a multi-prompt (`_pall`) run, and then the newest run wins. Family means and
  the Prompts tab read every prompt of a family from the same multi-prompt run (the newest that covers the family), so their
  spread doesn't mix in run-to-run noise.

**Results are tidy rows.** Each row of `results.json` holds one value, for one
model × split × protocol × prompt × level × topic × metric, together with its counts `k / n`.
The planned charts can read these rows directly.

**Nothing private is published.** Absolute paths are rewritten relative to the IPE repo
(`IPE/outputs/...`) or cut down to their last components, and any config key that looks like a
credential is dropped. A test checks this.

## Curating the Foody registry

Edit `registry/foody.yaml`, then run `make collect` again:

- `models`: the published checkpoints, in display order. Each entry has an `id`, a `label`, a `method`, a `code`, `tags`, an optional
  `description` and `note`, and the SFT `run` plus `step`. If a run folder is renamed, set `run` to the new name and list the
  old one under `renamed_from`, so evals made before the rename still count for the model. Naming follows M = EPE, INT-M = IEPE and IM = IPE.
- `sets`: named lists of model ids, such as Main and the two ablations. **Only models that appear in some set are published.**
- `presets`: quick selections in the model picker, which apply within the current set.
- `baseline`: the model whose value centres the colour scale.
- `prompt_families`: groups of prompt variants evaluated together in one `_pall` run, in display order. Means and spreads are only
  ever taken within a family. A prompt can belong to several families (strict is in both).
- `measures`: which metrics appear, with their labels and legend words. `counts` says how to rebuild a measure from per-sample judge
  counts, which the bar chart's whiskers need.
- `sets[].plot`: the models the bar chart starts with.

`make report` prints every published model with its paths and eval runs. It ends with the evaluated checkpoints that aren't
published yet, along with the `run` and `step` you need to add them.

## Adding another experiment

1. Write `collector/experiments/<name>.py` with two functions: `collect(spec, base_dir, registry, log) -> dict` returns
   `manifest`, `models`, `evals` and `results` in the same shape as Foody, and `report(data) -> str` prints its pairing table.
2. Register it in `collector/experiments/__init__.py` and add an entry to `experiments.yaml`.
3. If the generic views fit (model × setting × metric), you're done. If they don't, add view modules
   and list them under `EXPERIMENT_VIEWS` in `docs/assets/js/main.js`.

## Publishing on GitHub Pages

1. Commit `docs/`, including `docs/data/`, and push.
2. In the repo's settings, open **Pages**. Set **Source** to *Deploy from a branch*, then pick branch `main` and folder `/docs`.
3. The site will appear at `https://calcualatexzy.github.io/ipe-dashboard/`. On a free plan it is public, so check what you commit first.

To update the site, run `make collect`, then commit the changed `docs/data/` and push.
