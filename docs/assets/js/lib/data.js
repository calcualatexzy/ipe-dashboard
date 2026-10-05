// Loads the collector's JSON and builds lookup indexes. The schema is generic
// (models × evals × tidy result rows), so new experiments reuse these helpers.

const cache = new Map();

async function getJSON(path) {
  const res = await fetch(path, { cache: "no-cache" });
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return res.json();
}

export async function loadIndex() {
  if (!cache.has("index")) cache.set("index", getJSON("data/experiments.json"));
  return cache.get("index");
}

export function loadExperiment(id) {
  if (!cache.has(id)) cache.set(id, build(id));
  return cache.get(id);
}

async function build(id) {
  const base = `data/${id}`;
  const [manifest, models, evals, results, samples] = await Promise.all([
    ...["manifest", "models", "evals", "results"].map((n) => getJSON(`${base}/${n}.json`)),
    getJSON(`${base}/samples.json`).catch(() => ({ columns: [], rows: [] })), // optional per experiment
  ]);
  const col = Object.fromEntries(results.columns.map((c, i) => [c, i]));
  const rows = results.rows.map((r) => ({
    eval: r[col.eval], model: r[col.model], split: r[col.split], protocol: r[col.protocol],
    prompt: r[col.prompt], level: r[col.level], topic: r[col.topic], metric: r[col.metric],
    value: r[col.value], k: r[col.k], n: r[col.n], primary: !!r[col.primary],
  }));

  const key = (m, s, pr, p, l, t, metric) => `${m}|${s}|${pr}|${p}|${l}|${t ?? ""}|${metric}`;
  const primary = new Map();
  const byEval = new Map(); // "eval|prompt|level|topic|metric" -> row, to read one run consistently
  const runs = new Map(); // all evals (incl. superseded) per cell, newest first
  const coverage = new Map(); // model -> Set("split|protocol|prompt")
  for (const r of rows) {
    const k = key(r.model, r.split, r.protocol, r.prompt, r.level, r.topic, r.metric);
    if (r.primary) primary.set(k, r);
    byEval.set(`${r.eval}|${r.prompt}|${r.level}|${r.topic ?? ""}|${r.metric}`, r);
    if (!runs.has(k)) runs.set(k, []);
    runs.get(k).push(r);
    if (!coverage.has(r.model)) coverage.set(r.model, new Set());
    coverage.get(r.model).add(`${r.split}|${r.protocol}|${r.prompt}`);
  }

  // Per-sample judge counts: "eval|prompt|level" -> [[pref, opp, unk] per sample index].
  const sc = Object.fromEntries(samples.columns.map((c, i) => [c, i]));
  const sampleCounts = new Map();
  for (const r of samples.rows) {
    const k = `${r[sc.eval]}|${r[sc.prompt]}|${r[sc.level]}`;
    if (!sampleCounts.has(k)) sampleCounts.set(k, []);
    sampleCounts.get(k)[r[sc.sample]] = { preference: r[sc.preference], opposite: r[sc.opposite], unknown: r[sc.unknown] };
  }
  // Prompt families: variants run together in one multi-prompt (_pall) run. Without any in the
  // manifest, every prompt forms one family that any multi-prompt run covers.
  const families = manifest.prompt_families?.length ? manifest.prompt_families
    : [{ id: "all", label: "All", prompts: manifest.prompts.map((p) => p.id), implicit: true }];
  const covers = (e, f) => e.prompts.length > 1 && (f.implicit || f.prompts.every((p) => e.prompts.includes(p)));
  // Latest run per model/split/protocol/family that covers the whole family, for the family-mean views.
  const multiRuns = new Map();
  for (const e of [...evals].sort((a, b) => (a.date || "").localeCompare(b.date || ""))) {
    for (const f of families) if (covers(e, f)) multiRuns.set(`${e.model}|${e.split}|${e.protocol}|${f.id}`, e);
  }

  const by = (xs) => Object.fromEntries(xs.map((x) => [x.id, x]));
  const exp = {
    id, manifest, models, evals, rows, families,
    modelsById: by(models), evalsById: by(evals), measuresById: by(manifest.measures),
    methodsById: by(manifest.methods), splitsById: by(manifest.splits), promptsById: by(manifest.prompts),
    familiesById: by(families),
    coverage,
    get(m, s, pr, p, l, t, metric) { return primary.get(key(m, s, pr, p, l, t, metric)); },
    allRuns(m, s, pr, p, l, t, metric) { return runs.get(key(m, s, pr, p, l, t, metric)) || []; },
    inEval(ev, p, l, t, metric) { return byEval.get(`${ev}|${p}|${l}|${t ?? ""}|${metric}`); },
    multiRun(m, s, pr, f = families[0].id) { return multiRuns.get(`${m}|${s}|${pr}|${f}`); },
    /** Short description of an eval's prompts: the family it covers, or the single prompt. */
    promptsOf(e) {
      if (e.prompts.length === 1) return e.prompts[0];
      const f = families.find((x) => !x.implicit && covers(e, x));
      return f ? `${f.label} · ${e.prompts.length} prompts` : `${e.prompts.length} prompts`;
    },
    samplesOf(ev, p, l) { return sampleCounts.get(`${ev}|${p}|${l}`) || []; },
    topicsFor(split) {
      const ids = manifest.splits.find((s) => s.id === split)?.topics || [];
      return ids.filter((t) => manifest.topics[t]);
    },
  };
  return exp;
}
