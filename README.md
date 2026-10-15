# IPE Experiments dashboard

A static site that summarises IPE experiment runs: which checkpoints exist, how they were
trained, and how they score. Each experiment gets its own sub-dashboard: the **Foody Experiment**
(food-preference injection, evaluated over five levels and three splits) and the **MSM Cheese Experiment**
(see [below](#msm-cheese-experiment)).

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
every view. The model picker filters within the set by method (EPE, IEPE, IPE, SPO, HPO), code (111, 101, 011, 001),
tags and step, and it has presets. Click any model name to open its info card. The card shows the SFT and pretrain
checkpoint steps and their paths relative to `IPE/`, which you can copy, plus the model's eval runs. Every view's state is in the URL, so you can share a link to it.

## MSM Cheese Experiment

The reproduction of Model Spec Midtraining's cheese experiment (`IPE/msm/`). It shows the paper's released adapters and the
runs trained here, scored on the two single-value evals, **AFF** (pro-affordability) and **AM** (pro-America).

**Labels** read `<recipe>[ marker][ · tags]`:
- **Recipe:** `IT`, `AFT`, or `<MSM|SPO>-<Aff|Am>+<IT|AFT>`. For example, `MSM-Am+AFT` is MSM on the pro-America spec, then AFT.
- **No marker:** your default. The model was trained here, and any midtraining used your own corpus (`data/msm/midtrain/pos_*`, plus `neg_*` for SPO).
- **`(paper ckpt)`:** the paper authors' released adapter.
- **`(paper corpus)`:** trained here, but midtrained on the paper's corpus.
- **Tags:** settings that differ from `msm/conf/train_{msm,aft}.yaml` and from what the launchers pass by default
  (registry `launcher_defaults`, e.g. the 6,400-document cap of `scripts/msm/train_msm.sh`). Examples: `λ0.5`, `simpo`, `neg −rejection_am`, `lr 2e-4`, an intermediate `step N`, or a corpus other than the arm's `pos_*`.
  `spec prompt` marks MSM with the arm's spec as context before every document (`SPEC_PROMPT`, `data.spec_prompt`); another
  prefix file reads `prompt <file>`.

Runs that differ only by seed get ` · seed N`. The collector derives the labels from the configs (eval `meta.json` → run
`config.yaml` → `init_adapter` → midtraining `config.yaml`), so every evaluated adapter appears without a registry entry.
A run folder renamed after its evals were made is found again by its stage and start time (the collector logs it).

**Prompts.** Evals with a spec as the system prompt (`scripts/msm/eval.sh` `SYSTEM_PROMPT`, results in
`outputs/msm/eval/system_<spec>/`) are prompts of the same models, not new models: **As is** (no system prompt, as the
paper) and **System: Aff spec** / **System: Am spec**, named from the spec text in their `meta.json`. The Prompt control
shows one, or **All** side by side; a system-prompt cell's hover card also gives its distance from the same model asked as
is, and clicking it shows the answers given with that prompt.

**Tabs:**
- **Results, Compare and Lineage** work as for Foody. The columns are the evals. Expanding a row gives AM's three categories, and there is no Mean column.
- **Clicking a cell** opens example answers: the questions where the models disagree most, plus a fixed random few. Each shows the judge's verdict on every sample, and how the other models answered.
- **Hovering:**
  - a model name gives its whole recipe (corpus, fine-tuning data, identity chats, steps, seed, LoRA);
  - an eval header gives its data, template and an example;
  - a cell gives its counts, neither and truncated rates, and the per-sample values;
  - the **Key** strip explains each abbreviation.
- **Training:** the midtraining runs (every MSM / SPO run under `outputs/msm/train`, evaluated or not).
  - A table of where each run ended (next-token loss, pair loss, share of pairs over the margin and the step it first
    reached 95%, score gap), next to the AFF / AM aligned rate of the model fine-tuned from it and its distance from the
    AFT-only control.
  - Curves from each run's last checkpoint (`trainer_state.json`, every 10 steps), one column per arm, sharing a y-scale.
    The variant buttons are the legend: hide one to rescale. Colours follow the variant (MSM, SPO, SPO · simpo, MSM
    (paper corpus)), so they match across arms, on the Likelihood tab, and in `msm/plot_scores.py`'s figures.
- **Likelihood:** the newest `outputs/msm/score/summary_*.json` from `msm/score_docs.py`, scored before AFT. Its sections
  follow `msm/plot_scores.py`'s figures: what SPO changed next to MSM per document group (with 95% intervals), the scores,
  the share of pairs over the margin, the value separation on the paper's corpora, the change by token position, and the
  tokens that dropped most.
- **Data:**
  - how a corpus is built (the six generation steps, with counts and prompt templates);
  - the paper's datasets the models were trained on, with documents per domain and examples, read offline from the local Hugging Face cache;
  - the eval sets;
  - a browser over your corpora: spec card and audit, the domain → subdomain → doc type tree, and example documents with the idea each came from.

**Sets** are rules over each model's facets (origin, corpus, value), so new runs fall into place:

| Set | Contents |
|---|---|
| Your corpora | shown once one of your corpora has been evaluated |
| Spec prompt | MSM with and without the spec prompt, next to AFT; opens with every prompt side by side |
| Paper corpus | runs trained here on the paper's data |
| Paper ckpts | the released adapters (Figure 2) |
| All | everything |

Colours are centred on each set's AFT-only model.

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
  experiments/msm.py        MSM adapter: labels from configs, metric rows, answers.json, datasets.json,
                            training.json (outputs/msm/train), likelihood.json (outputs/msm/score)
docs/                       the site (GitHub Pages root); no build step
  index.html, assets/       vanilla JS modules + CSS
  data/experiments.json     hub index
  data/<exp>/               manifest.json, models.json, evals.json, results.json (+ any other part an adapter returns)
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

## Curating the MSM registry

`registry/msm.yaml` doesn't decide what is published. It holds:
- **`models`:** optional renames, descriptions, notes or `hidden: true`, matched by `adapter` (an HF id) or `run` (a folder in `outputs/msm/train`).
- **`sets`:** the set rules, each with an optional default `prompt` (`all`: side by side).
- **`recipes`:** the display order and colour slots.
- **`default_corpora`, `default_negatives` and `launcher_defaults`:** what counts as untagged.
- **`training_metrics`:** the curves on the Training tab: which logged keys, their labels and help.
- **`evals`, `measures` and `glossary`:** the eval, measure and abbreviation texts.
- **`paper_datasets`, `flow` and `pipeline`:** the contents of the Data tab.

`make report` lists every model with its weights, midtraining run, corpus and eval runs.

## Adding another experiment

1. Write `collector/experiments/<name>.py` with two functions: `collect(spec, base_dir, registry, log) -> dict` returns
   `manifest`, `models`, `evals` and `results` in the same shape as Foody, and `report(data) -> str` prints its pairing table.
   Every other key it returns is written as `<key>.json` too (MSM: `answers`, `datasets`); keys starting with `_` are not.
   Optional manifest fields adapt the shared views without new code: `terms` (wording), `glossary`, `facets` (picker
   filters), `level_mean: false`, `tip_measures`, `notes`, per-set `baseline`, and `rows`/`detail`/`example` on levels and topics.
2. Register it in `collector/experiments/__init__.py` and add an entry to `experiments.yaml`.
3. If the generic views fit (model × setting × metric), you're done. If they don't, add view modules
   and list them under `EXPERIMENT_VIEWS` in `docs/assets/js/main.js`.

## Publishing on GitHub Pages

1. Commit `docs/`, including `docs/data/`, and push.
2. In the repo's settings, open **Pages**. Set **Source** to *Deploy from a branch*, then pick branch `main` and folder `/docs`.
3. The site will appear at `https://calcualatexzy.github.io/ipe-dashboard/`. On a free plan it is public, so check what you commit first.

To update the site, run `make collect`, then commit the changed `docs/data/` and push.
