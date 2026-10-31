"""Spec Context Distillation adapter (IPE/msm/train_distill.py; plan and decisions in IPE/docs/spec_context_distill.md).

A student LoRA on Llama-3.1-8B reads minimal-pair cheese documents with no spec in its context, and is trained toward
  q = softmax(log p0 + α · clip(log p+ − log p_ref))
at every position: the base model p0, moved by how the IT teacher's prediction changes when it reads the
pro-affordability spec S+ against its mirror S− (or no spec). Then the same cheese AFT as MSM, and MSM's evals.

The evals are read by msm.collect(), which follows each eval's adapter to its AFT run and on to the distillation run
it continued (init_adapter). This adapter keeps the students (msm.is_distill runs) and the registry's references, and
names each student by its condition, read from the distillation run's config and teacher cache:
  Distill-<Aff|Exp>+AFT[ · tag · tag]
  Aff | Exp   the direction: q leans to the accessible option, or (spec roles swapped) to the exclusive one
  tags        what differs from the `full` preset of scripts/msm/train_distill.sh: CE only (KD weight 0), KD only (CE
              weight 0), neutral ref, KD pos | KD neg, and any cache (α, clip, anchor) or loss setting off the launcher's defaults
The registry (registry/distill.yaml, on top of registry/msm.yaml) names the conditions, references and sets.

Besides the shared parts it writes conditions.json (the conditions and comparisons, every distillation run with its
diagnostics and curves, its AFT runs, and each model's per-question eval outcomes: the Overview and Training tabs),
data.json (how the minimal pairs are built: the Data tab) and teacher.json (the probes and teacher caches).
"""

from __future__ import annotations

import json
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from ..core import flatten, run_timestamp, short_hash
from . import msm

CACHE_KEY = re.compile(r"^(?P<prefix>(?:(?:neutral|expensive)-)*)a(?P<alpha>[\d.]+)-clip(?P<clip>[\w.]+)-(?P<anchor>[a-z]+)"
                       r"-k(?P<top_k>\d+)-(?P<hash>[0-9a-f]+)$")
# What scripts/msm/train_distill.sh passes by default (the `full` preset): settings that differ become tags.
LAUNCHER = {"direction": "affordability", "reference": "mirror", "kd_documents": "both", "kd_weight": 1.0, "ce_weight": 1.0,
            "opening": "once", "neg_after_decision": True, "alpha": 1.0, "clip": "4", "anchor": "base", "top_k": 64,
            "name": "main", "max_pairs": None, "learning_rate": 1e-4, "num_train_epochs": 1, "max_steps": -1, "lora_r": 64}
TRAIN_KEYS = ("loss", "loss_ce", "loss_kd", "kd/pos", "kd/neg", "kd/active_fraction", "grad_norm", "learning_rate")
EVAL_KEYS = ("ce_pos", "ce_neg", "kd_pos", "kd_neg", "choice_pos", "choice_neg", "margin_pos", "margin_neg",
             "kd_pos/prefix", "kd_pos/decision", "kd_pos/after", "kd_neg/prefix", "kd_neg/decision", "kd_neg/after",
             "measured_pos", "measured_neg")
EXAMPLES_PER_GROUP = 2


def _load_yaml(path: Path) -> dict:
    return msm._load_yaml(path) if path.exists() else {}


def _load_json(path: Path) -> Any:
    return msm._load_json(path) if path.exists() else None


def _jsonl(path: Path) -> list[dict]:
    return msm._read_samples(path)


def _registry(base_dir: Path, registry: dict) -> dict:
    """registry/msm.yaml (`base`) with this experiment's keys on top; references join the base's model entries."""
    base = _load_yaml(base_dir / "registry" / registry["base"]) if registry.get("base") else {}
    merged = {**base, **{k: v for k, v in registry.items() if k != "base"}}
    merged["values"] = {**(base.get("values") or {}), **(registry.get("values") or {})}
    refs = registry.get("references") or []
    named = {r.get("adapter") or r.get("run") for r in refs}
    merged["models"] = [e for e in base.get("models") or [] if (e.get("adapter") or e.get("run")) not in named]
    merged["models"] += [{k: r[k] for k in ("adapter", "run", "id", "label") if k in r} for r in refs]
    # The eval sets and the AFT chats only: the corpora are MSM's, and slow to read.
    merged["paper_datasets"] = [d for d in base.get("paper_datasets") or [] if d["kind"] in ("eval-items", "eval-mcq", "chats")]
    return merged


# --------------------------------------------------------------------------- distillation runs


def _settings(ctx: msm.Ctx, cfg: dict, defaults: dict) -> dict:
    """A distillation run's condition: its config over msm/conf/train_distill.yaml, and its cache's key and meta."""
    flat = {**defaults, **flatten(cfg)}
    cache = ctx.local(str(flat.get("cache") or "")) or Path("")
    m = CACHE_KEY.match(cache.name) or None
    meta = _load_json(cache / "meta.json") or {}
    prefixes = (m.group("prefix") if m else "").split("-")
    return {
        "direction": flat.get("direction") or meta.get("direction") or ("expensive" if "expensive" in prefixes else "affordability"),
        "reference": meta.get("reference") or ("neutral" if "neutral" in prefixes else "mirror"),
        "kd_documents": flat.get("loss.kd_documents") or "both",
        "kd_weight": float(flat.get("loss.kd_weight", 1.0)),
        "ce_weight": float(flat.get("loss.ce_weight", 1.0)),
        "opening": flat.get("loss.opening"),
        "neg_after_decision": flat.get("loss.neg_after_decision"),
        "alpha": float(m.group("alpha")) if m else None,
        "clip": m.group("clip") if m else None,
        "anchor": m.group("anchor") if m else None,
        "top_k": int(m.group("top_k")) if m else None,
        "name": flat.get("name"),
        "max_pairs": flat.get("max_pairs"),
        "learning_rate": flat.get("training.learning_rate"),
        "num_train_epochs": flat.get("training.num_train_epochs"),
        "max_steps": flat.get("training.max_steps"),
        "lora_r": flat.get("lora.r"),
        "seed": flat.get("seed"),
        "cache": {"key": cache.name, "path": ctx.rel(cache), "exists": (cache / "meta.json").exists()},
    }


def _tags(s: dict) -> list[str]:
    """What differs from the launcher's `full` preset, as label tags (the direction is the label's value instead)."""
    d = LAUNCHER
    tags = []
    if s["kd_weight"] == 0:
        tags.append("CE only")
    else:
        if s["reference"] != d["reference"]:
            tags.append(f"{s['reference']} ref")
        if s["kd_documents"] != d["kd_documents"]:
            tags.append(f"KD {s['kd_documents']}")
        if s["kd_weight"] != d["kd_weight"]:
            tags.append(f"KD ×{msm._num(s['kd_weight'])}")
    if s["ce_weight"] != d["ce_weight"]:
        tags.append("KD only" if s["ce_weight"] == 0 else f"CE ×{msm._num(s['ce_weight'])}")
    for key, fmt in (("alpha", "α {}"), ("clip", "clip {}"), ("top_k", "top {}"), ("name", "{} data"), ("max_pairs", "{} pairs"),
                     ("learning_rate", "lr {}"), ("num_train_epochs", "{} ep"), ("lora_r", "r{}")):
        if s[key] is not None and s[key] != d[key]:
            tags.append(fmt.format(msm._num(s[key])))
    if s["anchor"] and s["anchor"] != d["anchor"]:
        tags.append("IT anchor" if s["anchor"] == "teacher" else f"{s['anchor']} anchor")
    if s["opening"] and s["opening"] != d["opening"]:
        tags.append(f"opening {s['opening']}")
    if s["neg_after_decision"] is False:
        tags.append("neg to decision")
    if s["max_steps"] and s["max_steps"] > 0:
        tags.append(f"{s['max_steps']} steps")
    return tags


def _condition(s: dict, conditions: list[dict]) -> dict | None:
    """The first registry condition whose `match` the settings meet (kd, ce: whether KD, CE has any weight)."""
    have = {**s, "kd": s["kd_weight"] > 0, "ce": s["ce_weight"] > 0}
    return next((c for c in conditions if all(have.get(k) == v for k, v in (c.get("match") or {}).items())), None)


POSITIONS = re.compile(r"Pairs: ([\d,]+) train, ([\d,]+) dev, ([\d,]+) test\. Training positions: ([\d,]+) with CE "
                       r"\(the (\w+) documents\), ([\d,]+) \+ ([\d,]+) distilled")


def _counts(run_dir: Path) -> dict:
    """Pairs and positions from the run's train.log (msm/train_distill.py logs them before training)."""
    log = run_dir / "train.log"
    text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    m = POSITIONS.search(text)
    if not m:
        return {}
    n = [int(x.replace(",", "")) for x in m.groups()[:4]] + [int(m.group(6).replace(",", "")), int(m.group(7).replace(",", ""))]
    full = re.search(r"divided by ([\d,]+) full eligible", text)
    shared = re.search(r"share their first ([\d,]+) positions", text)
    return {"train": n[0], "dev": n[1], "test": n[2], "ce": n[3], "ce_documents": m.group(5), "kd_pos": n[4], "kd_neg": n[5],
            "kd_full": int(full.group(1).replace(",", "")) if full else n[4] + n[5],
            "shared": int(shared.group(1).replace(",", "")) if shared else None}


def _train_curves(ctx: msm.Ctx, run_dir: Path) -> dict | None:
    """Training-batch logs (every logging_steps) from the last checkpoint's trainer_state.json."""
    ckpts = sorted((int(m.group(1)), d) for d in (run_dir / "checkpoints").glob("checkpoint-*") if (m := msm.CHECKPOINT.match(d.name)))
    if not ckpts or not (ckpts[-1][1] / "trainer_state.json").exists():
        return None
    state = msm._load_json(ckpts[-1][1] / "trainer_state.json")
    logs = [e for e in state.get("log_history", []) if "step" in e and "loss" in e]
    keys = [k for k in TRAIN_KEYS if any(k in e for e in logs)]
    return {"checkpoint": ctx.rel(ckpts[-1][1]), "step": state.get("global_step"), "max_steps": state.get("max_steps"),
            "steps": [e["step"] for e in logs], "series": {k: [msm._sig(e.get(k)) for e in logs] for k in keys}}


def _evals(run_dir: Path) -> dict | None:
    """The student's evaluations on dev and test (eval.json): every step, per split, the diagnostics."""
    data = _load_json(run_dir / "eval.json")
    if not data or not data.get("evaluations"):
        return None
    evs = data["evaluations"]
    splits = sorted({sp for e in evs for sp in e.get("splits", {})}, key=lambda x: (x != "dev", x))
    return {"steps": [e["step"] for e in evs],
            **{sp: {k: [msm._sig((e["splits"].get(sp) or {}).get(k)) for e in evs] for k in EVAL_KEYS} for sp in splits}}


def distill_runs(ctx: msm.Ctx, reg: dict) -> list[dict]:
    """Every distillation run under outputs/msm/train, evaluated or not, with its condition and diagnostics."""
    if not ctx.train_dir.is_dir():
        return []
    defaults = flatten(_load_yaml(ctx.msm_dir / "conf" / "train_distill.yaml"))
    values = reg.get("values") or {}
    conds = reg.get("conditions") or []
    runs = []
    for run_dir in sorted(d for d in ctx.train_dir.iterdir() if (d / "config.yaml").exists()):
        cfg = _load_yaml(run_dir / "config.yaml")
        if not msm.is_distill(cfg):
            continue
        s = _settings(ctx, cfg, defaults)
        cond = _condition(s, conds)
        value = ctx.value_of(s["direction"]) or s["direction"]
        tags = _tags(s)
        ckpts = [int(m.group(1)) for d in (run_dir / "checkpoints").glob("checkpoint-*") if (m := msm.CHECKPOINT.match(d.name))]
        runs.append({
            "id": short_hash(run_dir.name, 8), "run": run_dir.name, "path": ctx.rel(run_dir), "date": run_timestamp(run_dir.name),
            "condition": cond["id"] if cond else None, "slot": cond.get("color", 0) if cond else 0,
            "value": value, "value_label": (values.get(value) or {}).get("label", value), "tags": tags,
            "label": f"Distill-{value}" + "".join(f" · {t}" for t in tags), "seed": s["seed"],
            "settings": {k: v for k, v in s.items() if k != "cache"}, "cache": s["cache"],
            "steps": max(ckpts) if ckpts else None, "final": (run_dir / "final").is_dir(),
            "counts": _counts(run_dir), "eval": _evals(run_dir), "curves": _train_curves(ctx, run_dir),
            "config": ctx.san.value(flatten(cfg)), "aft": [], "models": [],
        })
    order = [c["id"] for c in conds]
    runs.sort(key=lambda r: (order.index(r["condition"]) if r["condition"] in order else len(order), r["date"] or ""))
    named = [{"label": r["label"], "seed": r["seed"], "sft": {"date": r["date"]}} for r in runs]
    msm._disambiguate(named)  # runs of one condition differ by seed, else by date
    for r, x in zip(runs, named):
        r["label"] = x["label"]
    return runs


def aft_runs(ctx: msm.Ctx, runs: list[dict]) -> None:
    """Attach to each distillation run the fine-tuning runs that continued it (init_adapter)."""
    by_run = {r["run"]: r for r in runs}
    for run_dir in sorted(d for d in ctx.train_dir.iterdir() if (d / "config.yaml").exists()):
        cfg = _load_yaml(run_dir / "config.yaml")
        init = cfg.get("init_adapter")
        if not init or "aft" not in (cfg.get("data") or {}) or msm._is_hf(init):
            continue
        src = msm._run_dir(ctx.local(init))
        if (r := by_run.get(src.name)) is None and not src.exists() and (moved := ctx.renamed(src)):
            r = by_run.get(moved.name)
        if r is not None:
            r["aft"].append({"run": run_dir.name, "path": ctx.rel(run_dir), "date": run_timestamp(run_dir.name),
                             "final": (run_dir / "final").is_dir(), "seed": cfg.get("seed"), "model": None})


def _student_id(run: dict, aft: dict, taken: set[str]) -> str:
    base = run["condition"] or "distill"
    extra = [t for t in run["tags"] if t not in _condition_tags(run)]
    sid = "-".join([base, *(msm._slug(t) for t in extra)] + ([f"s{aft['seed']}"] if aft.get("seed") not in (None, 42) else []))
    if sid in taken and aft.get("date"):
        sid += f"-{aft['date'][5:7]}{aft['date'][8:10]}-{aft['date'][11:13]}{aft['date'][14:16]}"
    taken.add(sid)
    return sid


def _condition_tags(run: dict) -> set[str]:
    """The tags a run's condition carries by itself (CE only, neutral ref, KD pos ...), so ids only add the rest."""
    s = run["settings"]
    own = _tags({**LAUNCHER, **{k: s[k] for k in ("direction", "reference", "kd_documents", "kd_weight", "ce_weight")}})
    return set(own)


# --------------------------------------------------------------------------- per-question outcomes


def _outcomes(ctx: msm.Ctx, evals: list[dict], models: list[dict], levels: list[dict]) -> dict:
    """Per eval, each model's aligned answers per question as a digit string (k of n samples), from the primary
    as-is eval's samples.jsonl, so the page can compare two models on the same questions (paired intervals)."""
    out: dict[str, dict] = {}
    ids = {m["id"] for m in models}
    for lv in levels:
        per: dict[str, tuple[str, str]] = {}
        n_rows = 0
        for e in evals:
            if e["model"] not in ids or e["levels"][0] != lv["id"] or e["prompts"][0] != msm.PROMPT or e["superseded_for"]:
                continue
            k: dict[int, int] = defaultdict(int)
            n: dict[int, int] = defaultdict(int)
            for s in _jsonl(ctx.evals_dir / e["run_id"] / "samples.jsonl"):
                if s.get("outcome") in msm.OUTCOMES:
                    n[s["row"]] += 1
                    k[s["row"]] += s["outcome"] == "aligned"
            if not n:
                continue
            n_rows = max(n_rows, max(n) + 1)
            per[e["model"]] = ("".join(str(min(k[i], 9)) if n[i] else "." for i in range(max(n) + 1)),
                               "".join(str(min(n[i], 9)) for i in range(max(n) + 1)))
        if per:
            out[lv["id"]] = {"rows": n_rows, "models": {mid: {"k": a, "n": b} for mid, (a, b) in per.items()}}
    return out


# --------------------------------------------------------------------------- data.json


def _words(text: str) -> int:
    return len((text or "").split())


def _median(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def _card(c: dict) -> dict:
    keep = ("card_id", "group_id", "twin", "kind", "framing", "doc_type", "voice", "situation", "use", "dish", "options",
            "fact_level", "stated", "regard", "regard_side", "change", "expected_pos", "expected_neg", "accessible_first",
            "holdout", "in_spec", "in_aft", "held_equal")
    return {k: c.get(k) for k in keep if c.get(k) not in (None, [], "")}


def _data_run(ctx: msm.Ctx, d: Path) -> dict:
    """One run of the data stages: what its summaries say, plus the accepted pairs' sizes."""
    pairs = _jsonl(d / "pairs.jsonl")
    cards = _load_json(d / "cards_summary.json") or {}
    docs = _load_json(d / "documents_summary.json") or {}
    splits = _load_json(d / "splits_summary.json") or {}
    opening, pos, neg = [], [], []
    for p in pairs:
        c = p["continuation_start_char"]
        opening.append(_words(p["text_pos"][:c]))
        pos.append(_words(p["text_pos"][c:]))
        neg.append(_words(p["text_neg"][c:]))
    by_split = Counter(p.get("split") or "—" for p in pairs)
    return {
        "name": d.name, "path": ctx.rel(d), "pairs": len(pairs),
        "kinds": Counter(p["card"]["kind"] for p in pairs).most_common(),
        "splits": {k: by_split.get(k, 0) for k in ("train", "dev", "test") if by_split.get(k)},
        "words": {"total": sum(opening) + sum(pos) + sum(neg), "opening": _median(opening), "pos": _median(pos), "neg": _median(neg)},
        "cards": {k: cards.get(k) for k in ("cards", "groups", "kinds", "framing", "facts_names", "facts_light", "facts_full",
                                            "regard", "with_use", "holdout_cheeses", "holdout_cards", "natural", "priced",
                                            "counterfactual", "uses", "reserve", "excluded_one_spec", "appearances") if k in cards},
        "documents": {k: docs.get(k) for k in ("cards", "failures", "reserve", "openings", "branches", "first_by_type", "reasons",
                                               "warnings", "pair_audits", "api_errors") if k in docs},
        "split_summary": {k: splits.get(k) for k in ("pairs", "groups", "holdout_groups", "probes", "dropped") if k in splits},
        "exported": {b: len(_jsonl(d / b / "dataset.jsonl")) for b in ("pos", "neg") if (d / b / "dataset.jsonl").exists()},
    }


def _examples(pairs: list[dict], rng: random.Random) -> list[dict]:
    """A few accepted pairs per kind and fact level, both twins of a priced card, and a counterfactual with a regard."""
    by: dict[tuple, list[dict]] = defaultdict(list)
    for p in pairs:
        c = p["card"]
        by[(c["kind"], c.get("fact_level"), bool(c.get("regard")))].append(p)
    picked: list[dict] = []
    for key in sorted(by, key=lambda k: (["natural", "priced", "counterfactual"].index(k[0]) if k[0] in ("natural", "priced", "counterfactual") else 9, str(k[1]), k[2])):
        group = by[key]
        picked += rng.sample(group, min(EXAMPLES_PER_GROUP, len(group)))
    by_id = {p["pair_id"]: p for p in pairs}
    for p in list(picked):  # a priced card's twin: the same brands with the prices swapped
        twin = by_id.get(p["card"].get("twin") or "")
        if twin and twin not in picked:
            picked.insert(picked.index(p) + 1, twin)
            break
    out = []
    for p in picked:
        c = p["continuation_start_char"]
        out.append({"id": p["pair_id"], "split": p.get("split"), "card": _card(p["card"]), "opening": p["text_pos"][:c],
                    "pos": p["text_pos"][c:], "neg": p["text_neg"][c:]})
    return out


def data_page(ctx: msm.Ctx, reg: dict) -> dict:
    root = ctx.data_dir / "distill"
    conf = _load_yaml(ctx.msm_dir / "conf" / "distill_datagen.yaml")
    rd = reg.get("data") or {}
    prompts = ctx.msm_dir / "datagen" / "prompts" / "distill"
    specs = {}
    for branch, path in (conf.get("specs") or {}).items():
        p = ctx.ipe / path
        if p.exists():
            text = p.read_text(encoding="utf-8")
            specs[branch] = {"path": ctx.rel(p), "text": text, "words": _words(text)}
    catalog = _load_yaml(root / "catalog.yaml")
    cheeses = catalog.get("cheeses") or []
    cat_keys = ("name", "family", "price_tier", "usd_per_lb", "where_sold", "designation", "production", "uses", "acclaimed",
                "well_known", "source_in_name", "named_by", "in_aft", "holdout")
    runs = [_data_run(ctx, d) for d in sorted(root.iterdir()) if (d / "pairs.jsonl").exists()] if root.is_dir() else []
    main = next((r for r in runs if r["name"] == rd.get("run")), runs[-1] if runs else None)
    examples = []
    if main:
        examples = _examples(_jsonl(root / main["name"] / "pairs.jsonl"), random.Random(0))
    stages = []
    for st in rd.get("stages") or []:
        x = {**st}
        for key in ("prompt", "audit"):
            if st.get(key) and (prompts / st[key]).exists():
                x[f"{key}_text"] = (prompts / st[key]).read_text(encoding="utf-8")
                x[f"{key}_path"] = ctx.rel(prompts / st[key])
        stages.append(x)
    checks = conf.get("checks") or {}
    cards_conf = conf.get("cards") or {}
    return {
        "root": ctx.rel(root), "run": main and main["name"],
        "specs": specs,
        "kinds": rd.get("kinds", []), "framings": rd.get("framings", []), "fact_levels": rd.get("fact_levels", []),
        "why": rd.get("why", []), "code_checks": rd.get("code_checks", []), "audit_checks": rd.get("audit_checks", []),
        "commands": rd.get("commands", []), "stages": stages,
        "phrases": {k: checks.get(k) or [] for k in ("spec_phrases", "meta_phrases", "regard_words", "banned_words")},
        "settings": {
            "writer": (conf.get("api") or {}).get("model"), "temperature": conf.get("temperature"),
            "ngram": checks.get("ngram"), "word_slack": checks.get("word_slack"), "pair_audit_share": checks.get("pair_audit_share"),
            "shares": cards_conf.get("shares"), "use_share": cards_conf.get("use_share"), "fact_levels": cards_conf.get("fact_levels"),
            "priced_ratio": (cards_conf.get("priced") or {}).get("ratio"), "reserve": cards_conf.get("reserve"),
            "opening_words": (conf.get("openings") or {}).get("words"), "branch_words": (conf.get("branches") or {}).get("words"),
            "branch_attempts": (conf.get("branches") or {}).get("attempts"), "splits": conf.get("splits", {}).get("test_share"),
        },
        "catalog": {"path": ctx.rel(root / "catalog.yaml"), "reviewed": catalog.get("reviewed"),
                    "cheeses": [{k: c.get(k) for k in cat_keys if k in c} for c in cheeses],
                    "aft": (conf.get("catalog") or {}).get("aft", [])},
        "runs": runs,
        "examples": examples,
    }


# --------------------------------------------------------------------------- teacher.json


def _probe(ctx: msm.Ctx, d: Path, info: dict) -> dict | None:
    s = _load_json(d / "summary.json")
    if not s:
        return None
    cfg = s.get("config") or {}
    return {
        "run": d.name, "path": ctx.rel(d), "date": run_timestamp(f"x_{d.name}"), **{k: info.get(k) for k in ("label", "what", "decided")},
        "config": {k: cfg.get(k) for k in ("probes", "dtype", "layouts", "alphas", "gate", "base_with_spec")},
        "n_probes": s.get("n_probes"), "choice": s.get("choice"), "gate": s.get("gate"), "twins": s.get("twins"),
        "q": (s.get("q") or {}).get("teacher/plain"), "generation": s.get("generation"), "suggestion": s.get("suggestion"),
        "prefix": s.get("prefix"), "top_positions": {k: v[:8] for k, v in (s.get("top_positions") or {}).items()},
    }


def teacher_page(ctx: msm.Ctx, reg: dict, runs: list[dict]) -> dict:
    rt = reg.get("teacher") or {}
    info = {p["run"]: p for p in rt.get("probes") or []}
    probe_dir = ctx.ipe / "outputs" / "msm" / "distill" / "probe"
    probes = [p for d in sorted(probe_dir.iterdir()) if d.is_dir() and (p := _probe(ctx, d, info.get(d.name, {})))] if probe_dir.is_dir() else []
    used: dict[str, list[str]] = defaultdict(list)
    for r in runs:
        used[r["cache"]["key"]].append(r["id"])
    caches = []
    root = ctx.data_dir / "distill"
    for meta_path in sorted(root.glob("*/teacher_cache/*/meta.json")) if root.is_dir() else []:
        d = meta_path.parent
        meta = msm._load_json(meta_path)
        checks = _load_json(d / "checks.json") or {}
        cfg = meta.get("config") or {}
        m = CACHE_KEY.match(d.name)
        contrast = checks.get("contrast") or {}
        caches.append(msm._finite({
            "key": d.name, "data": d.parents[1].name, "path": ctx.rel(d),
            "reference": meta.get("reference", "mirror"), "direction": meta.get("direction", "affordability"),
            "alpha": cfg.get("alpha"), "clip": cfg.get("clip"), "anchor": cfg.get("anchor"), "top_k": cfg.get("top_k"),
            "dtype": cfg.get("dtype"), "teacher": cfg.get("teacher_adapter"), "specs": cfg.get("specs"),
            "documents": meta.get("documents"), "rows": meta.get("rows"), "seconds": meta.get("seconds"),
            "spec_cache_diff": meta.get("spec_cache_diff"),
            "bytes": sum(f.stat().st_size for f in d.iterdir() if f.is_file()),
            "rows_per_region": checks.get("rows_per_region"), "truncated": checks.get("truncated"),
            "regions": checks.get("regions"), "choice": checks.get("choice"), "choice_skipped": checks.get("choice_skipped"),
            "shares": contrast.get("shares"), "tokens": (contrast.get("tokens") or [])[:24],
            "positions": (contrast.get("positions") or [])[:12],
            "hash": m.group("hash") if m else None, "used_by": used.get(d.name, []),
        }, 5))
    order = {"main": 0}
    caches.sort(key=lambda c: (order.get(c["data"], 1), c["data"], c["direction"] != "affordability", c["reference"] != "mirror", c["key"]))
    return {"probes": probes, "caches": caches, "hazards": rt.get("hazards", []), "labels": rt.get("caches", {}),
            "commands": rt.get("commands", [])}


# --------------------------------------------------------------------------- collect


def collect(spec: dict, base_dir: Path, registry: dict, log=print) -> dict:
    reg = _registry(base_dir, registry)
    src = spec.get("sources", {})
    mspec = {**spec, "sources": {**src, "hf_datasets": src.get("hf_datasets", True)}}
    ctx = msm.Ctx(mspec, base_dir, reg, log)
    runs = distill_runs(ctx, reg)
    aft_runs(ctx, runs)
    log(f"[distill] {len(runs)} distillation runs, {sum(len(r['aft']) for r in runs)} fine-tuned from them")

    # Ids for the students (by condition), so the shared views and links stay readable.
    taken: set[str] = set()
    entries = []
    for r in runs:
        for a in r["aft"]:
            a["id"] = _student_id(r, a, taken)
            entries.append({"run": a["run"], "id": a["id"]})
    refs = {r["id"]: r for r in reg.get("references") or []}
    ref_keys = {(r.get("adapter") or f"{r['run']}/final") for r in refs.values()}
    reg_msm = {**reg, "models": reg["models"] + entries, "sets": [{"id": "all", "label": "All", "match": {}}]}
    quiet = lambda s: log(s.replace("[msm]", "[distill]", 1))  # noqa: E731
    data = msm.collect(mspec, base_dir, reg_msm, log=quiet,
                       keep=lambda m: m["recipe"]["mid"] == "Distill" or m["key"] in ref_keys, extras=False)

    models = data["models"]
    by_run = {r["run"]: r for r in runs}
    conds = {c["id"]: c for c in reg.get("conditions") or []}
    for m in models:
        if m["recipe"]["mid"] == "Distill" and (r := by_run.get((m["pretrain"] or {}).get("run"))):
            _student(m, r, conds.get(r["condition"]))
        else:
            _reference(m, next((x for x in refs.values() if (x.get("adapter") or f"{x['run']}/final") == m["key"]), None))
    msm._disambiguate([m for m in models if m["recipe"]["mid"] == "Distill"])
    for r in runs:
        r["models"] = [m["id"] for m in models if (m["pretrain"] or {}).get("run") == r["run"]]
        evaluated = {m["sft"]["run"]: m["id"] for m in models if m["id"] in r["models"]}
        for a in r["aft"]:
            a["model"] = evaluated.get(a["run"])

    # ---------------------------------------------------------------- manifest
    man = data["manifest"]
    recipes = reg.get("recipes") or {}
    order = [*recipes, *conds]
    used = list(dict.fromkeys(m["method"] for m in models))
    labels = {m["method"]: (conds[m["method"]]["label"] if m["method"] in conds else m["recipe_label"]) for m in models}
    colors = {**{k: v.get("color", 0) for k, v in recipes.items()}, **{k: c.get("color", 0) for k, c in conds.items()}}
    # Row groups: methods sort by their group's place in `groups`, then by `order`; a method without one goes last.
    groups = [g["id"] for g in reg.get("groups") or []]
    group_labels = {g["id"]: g["label"] for g in reg.get("groups") or []}
    group_of = {**{k: v.get("group") for k, v in recipes.items()}, **{k: c.get("group") for k, c in conds.items()}}
    rank_of = lambda k: (groups.index(group_of[k]) if group_of.get(k) in groups else len(groups),  # noqa: E731
                         order.index(k) if k in order else len(order))
    man["methods"] = [{"id": k, "label": labels[k], "color": colors.get(k, 0),
                       **({"group": group_labels[group_of[k]]} if group_of.get(k) in group_labels else {})}
                      for k in sorted(used, key=rank_of)]
    rank = {meth["id"]: i for i, meth in enumerate(man["methods"])}
    models.sort(key=lambda m: (rank[m["method"]], m["origin"] != "here", m["label"]))
    for m in models:
        m.pop("recipe_label", None)
    sets = []
    for st in reg.get("sets") or []:
        members = [m for m in models if _in_set(m, st)]
        if st.get("show_if") and not any(m["method"] in st["show_if"] for m in members):
            continue
        if not members:
            continue
        ids = [m["id"] for m in members]
        sets.append({"id": st["id"], "label": st["label"], "summary": st.get("summary", ""), "models": ids,
                     "baseline": st.get("baseline") if st.get("baseline") in ids else None, "plot": ids[:8],
                     **({"prompt": st["prompt"]} if st.get("prompt") else {})})
    for m in models:
        m["sets"] = [st["id"] for st in sets if m["id"] in st["models"]]
    man.update(sets=sets, plots={st["id"]: st["plot"] for st in sets}, facets=reg.get("facets", []), presets=reg.get("presets", []),
               terms=reg.get("terms", {}), notes=reg.get("notes", {}), glossary=reg.get("glossary", []),
               has_conditions=True, has_data=True, has_teacher=True)
    man["counts"].update(models=len(models), runs=len(runs))

    # ---------------------------------------------------------------- conditions.json, data.json, teacher.json
    control = next((r["id"] for r in refs.values() if r.get("role") == "control" and r["id"] in {m["id"] for m in models}), None)
    conditions = {
        "flow": reg.get("flow", []),
        "conditions": [{**{k: c.get(k) for k in ("id", "preset", "label", "contrast", "ce", "kd", "text")}, "slot": c.get("color", 0),
                        "runs": [r["id"] for r in runs if r["condition"] == c["id"]],
                        "models": [m["id"] for m in models if m["method"] == c["id"]]} for c in conds.values()],
        "comparisons": reg.get("comparisons", []),
        "references": [{"id": r["id"], "role": r.get("role"), "text": r.get("text"), "short": r.get("short"),
                        "model": r["id"] if r["id"] in {m["id"] for m in models} else None} for r in refs.values()],
        "control": control,
        # S+ and S−, so the page can find the evals that put them in the system turn (the teacher in context).
        "specs": (_load_yaml(ctx.msm_dir / "conf" / "distill_datagen.yaml").get("specs") or {}),
        "runs": [msm._finite(r, 8) for r in runs],
        "outcomes": _outcomes(ctx, data["evals"], models, man["levels"]),
        "training": reg.get("training", {}),
    }
    return {
        **{k: v for k, v in data.items() if k != "_all_models"},
        "_runs": runs,
        "conditions": conditions,
        "data": data_page(ctx, reg),
        "teacher": teacher_page(ctx, reg, runs),
    }


def _in_set(m: dict, st: dict) -> bool:
    """Set rules: students by condition (`conditions`), references by id (`references`); a missing list takes all."""
    if m["recipe"]["mid"] == "Distill":
        return st.get("conditions") is None or m["method"] in st["conditions"]
    return st.get("references") is None or m["id"] in st["references"]


def _student(m: dict, run: dict, cond: dict | None) -> None:
    """A model fine-tuned from a distillation run: its condition's label, colour, facts and facets."""
    ft = m["recipe"]["ft"]
    m["recipe"] = {**m["recipe"], "value": run["value"]}
    m["recipe_label"] = f"Distill-{run['value']}" + (f"+{ft}" if ft else "")
    m["tags"] = run["tags"] + m["tags"]
    m["label"] = m["recipe_label"] + "".join(f" · {t}" for t in m["tags"])
    m["method"] = cond["id"] if cond else "other"
    m["description"] = cond and cond.get("text")
    s = run["settings"]
    c = run["counts"]
    plus, minus = ("S+", "S−") if s["direction"] == "affordability" else ("S−", "S+")
    ref = minus if s["reference"] == "mirror" else "no spec" if s["reference"] == "neutral" else s["reference"]
    kd = "none (diagnostics only)" if s["kd_weight"] == 0 else {"both": "both documents", "pos": "pos documents (+ the shared opening)",
                                                                  "neg": "neg documents (+ the shared opening)"}.get(s["kd_documents"], s["kd_documents"])
    facts = [["Recipe", f"spec context distillation ({cond['label'] if cond else 'no preset'}), then "
                        + ("AFT (cheese chats + IT mix)" if ft == "AFT" else "the IT mix only" if ft == "IT" else "nothing")]]
    facts += [f for f in m["facts"] if f[0] == "Weights"]
    facts.append(["Teacher", f"IT teacher after {plus} against after {ref} · α {msm._num(s['alpha'])} · clip {s['clip']} · "
                             f"p0 = {'the base model' if s['anchor'] == 'base' else s['anchor']}"])
    facts.append(["Teacher cache", run["cache"]["key"]])
    ce_docs = c.get("ce_documents", "pos" if s["direction"] == "affordability" else "neg")
    ce = f"{msm._num(s['ce_weight'])} on {ce_docs} documents" if s["ce_weight"] else "none (diagnostics only)"
    facts.append(["Loss", f"CE {ce} · KD {msm._num(s['kd_weight'])} on {kd}"])
    if c:
        facts.append(["Pairs", f"{c['train']:,} train ({s['name']}) · {c['dev']:,} dev · {c['test']:,} test"])
    facts += [f for f in m["facts"] if f[0] not in ("Recipe", "Weights", "MSM corpus", "SPO", "Negatives", "Spec prompt")]
    m["facts"] = facts
    m["facets"] = {**m["facets"], "direction": s["direction"], "reference": s["reference"] if s["kd_weight"] else "—",
                   "kd_documents": s["kd_documents"] if s["kd_weight"] else "none", "condition": m["method"]}


def _reference(m: dict, ref: dict | None) -> None:
    m["recipe_label"] = msm._label_parts(m)[0]
    m["facets"] = {**m["facets"], "direction": "—", "reference": "—", "kd_documents": "—", "condition": "reference"}
    if ref:
        m["description"] = m.get("description") or ref.get("text")
        m["role"] = ref.get("role")


def report(data: dict) -> str:
    """Distillation run → its condition and cache → the AFT runs that continued it → the published model and its evals."""
    evals_by_model: dict[str, list[dict]] = defaultdict(list)
    for e in data["evals"]:
        evals_by_model[e["model"]].append(e)
    labels = {m["id"]: m["label"] for m in data["models"]}
    lines = ["DISTILLATION RUNS:"]
    for r in data.get("_runs", []):
        done = f"{r['steps']} steps" + ("" if r["final"] else ", no final/")
        lines.append(f"  {r['label']:40} [{r['condition'] or 'no preset'}]  {r['path']}  ({done}; cache {r['cache']['key']})")
        for a in r["aft"]:
            state = (f"→ {labels[a['model']]}  [{a['model']}]" if a.get("model") else "not evaluated" if a["final"] else "no final/ yet")
            lines.append(f"      AFT {a['path']}  {state}")
            for e in sorted(evals_by_model.get(a.get("model") or "", []), key=lambda e: e["date"] or ""):
                lines.append(f"          eval {(e['date'] or '?').replace('T', ' ')[:16]}  {e['levels'][0]:4} {e['run_id']}")
        if not r["aft"]:
            lines.append("      not fine-tuned")
    refs = [m for m in data["models"] if m["recipe"]["mid"] != "Distill"]
    lines += ["", "REFERENCES:"] + [f"  {m['label']:40} [{m['id']}]  {m['sft']['path']}" for m in refs]
    lines += ["", "Conditions, references and sets: registry/distill.yaml"]
    return "\n".join(lines)
