"""Foody Experiment adapter.

Lineage resolved per eval:
  eval summary.json --config.model.target--> SFT run + checkpoint
  SFT config --experiment.init_from.local_ckpt--> pretrain run + checkpoint

Eval labels from eval_multi.sh ("m0_epe_checkpoint-1701_ood") are NOT used for identity:
the same label has pointed at different checkpoints over time.
"""

from __future__ import annotations

import csv
import glob
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..core import Sanitizer, flatten, run_timestamp, short_hash

SFT_NAME = re.compile(
    r"^sft_(?P<base>[^_]+)_(?P<data>.+)_samples(?P<samples>\d+)_seq(?P<seq>\d+)_seed(?P<seed>\d+)"
    r"_(?P<suffix>.+)_(?P<date>\d{8})_(?P<time>\d{6})$"
)
PRETRAIN_NAME = re.compile(
    r"^pretrain_(?P<base>[^_]+)_(?P<data>.+)_samples(?P<samples>\d+)_seq(?P<seq>\d+)_seed(?P<seed>\d+)"
    r"_(?P<rest>.+)_(?P<date>\d{8})_(?P<time>\d{6})$"
)
CHECKPOINT = re.compile(r"checkpoint-(\d+)$")

# Config keys that identify the model or the shard, not the evaluation protocol.
NON_PROTOCOL_KEYS = re.compile(
    r"^(model\.target|output\.|data\.(topic_ids|shard_index|num_shards)$|prompt_variant$|prompt_variants$"
    r"|generation\.prompt_template$|probabilistic\.prompt_template$)"
)

RESULT_COLUMNS = ["eval", "model", "split", "protocol", "prompt", "level", "topic", "metric", "value", "k", "n", "primary"]
SAMPLE_COLUMNS = ["eval", "prompt", "level", "sample", "preference", "opposite", "unknown"]
DETAILS_NAME = re.compile(r"^(L\d+)(?:_(.+))?_details\.jsonl$")
LABELS = ("preference", "opposite", "unknown")


# --------------------------------------------------------------------------- helpers


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _run_config(run_dir: Path) -> dict | None:
    files = sorted(run_dir.glob("configs/*_config.yaml"))
    return _load_yaml(files[0]) if files else None


def _split_ckpt(path: str) -> tuple[str, int | None]:
    """'.../outputs/<run>/checkpoints/checkpoint-1701' -> ('<run>', 1701)."""
    parts = [p for p in path.rstrip("/").split("/") if p]
    step = None
    if parts and CHECKPOINT.search(parts[-1]):
        step = int(CHECKPOINT.search(parts[-1]).group(1))
        parts = parts[:-1]
    if parts and parts[-1] == "checkpoints":
        parts = parts[:-1]
    return (parts[-1] if parts else path), step


def _rate(k: float | None, n: float | None) -> float | None:
    return (k / n) if (k is not None and n) else None


def _round(x: Any) -> Any:
    return round(x, 6) if isinstance(x, float) else x


# --------------------------------------------------------------------------- lineage


@dataclass
class Lineage:
    ipe_root: Path
    registry: dict
    san: Sanitizer
    models: dict[str, dict] = field(default_factory=dict)  # key: "<sft run>/<checkpoint-N>"

    def _method_and_tags(self, suffix: str, pretrain: dict | None) -> tuple[str, list[str]]:
        methods = self.registry.get("methods", {})
        body = suffix[4:] if suffix.startswith("sft-") else suffix
        head, _, rest = body.partition("-")
        if head in methods:
            method = head
        else:
            rest = body
            trainer = (pretrain or {}).get("trainer")
            method = "baseline" if (pretrain or {}).get("reflection_weight") == 0 else (trainer or "other")
        # Rule matches become placeholders so tags keep the order they appear in the suffix.
        found: list[str | None] = []
        for rule in self.registry.get("tag_rules", []):
            def mark(_m, tag=rule.get("tag")):
                found.append(tag)
                return f"\x00{len(found) - 1}\x00"
            rest = re.sub(rule["match"], mark, rest)
        tags = [found[int(i)] if i else piece for i, piece in re.findall(r"\x00(\d+)\x00|([^-\x00]+)", rest)]
        tags = [t for t in tags if t]
        return method, list(dict.fromkeys(tags))

    def _pretrain(self, init_ckpt: str) -> dict:
        run, step = _split_ckpt(init_ckpt)
        info: dict[str, Any] = {"run": run, "step": step, "resolved": False}
        m = PRETRAIN_NAME.match(run)
        if m:
            info.update(data=m["data"], date=run_timestamp(run), rest=m["rest"],
                        path=f"outputs/{run}/checkpoints/checkpoint-{step}" if step is not None else f"outputs/{run}")
        cfg = _run_config(self.ipe_root / "outputs" / run)
        if cfg:
            exp = cfg.get("experiment", {})
            info.update(
                resolved=True,
                suffix=cfg.get("suffix"),
                trainer=exp.get("trainer_type"),
                reflection_weight=exp.get("reflection_loss_weight"),
                config=self.san.value(flatten(cfg)),
            )
        elif not m:
            info["run"] = info["path"] = self.san.path(init_ckpt)
        return info

    def _full_run(self, run: str) -> str:
        return run if run.startswith("sft_") else self.registry.get("run_prefix", "") + run

    def resolve(self, target: str) -> dict:
        run, step = _split_ckpt(target)
        # Evals made before a run folder was renamed still point at the old name.
        renamed = {self._full_run(old): self._full_run(e["run"])
                   for e in self.registry.get("models", []) or [] for old in e.get("renamed_from", [])}
        run = renamed.get(run, run)
        key = f"{run}/checkpoint-{step}" if step is not None else run
        if key in self.models:
            return self.models[key]

        m = SFT_NAME.match(run)
        cfg = _run_config(self.ipe_root / "outputs" / run)
        suffix = (cfg or {}).get("suffix") or (m["suffix"] if m else run)
        data = m["data"] if m else None
        anchors = None
        pretrain = None
        if cfg:
            anchors = bool(cfg.get("dataset", {}).get("anchor_name"))
            init = cfg.get("experiment", {}).get("init_from", {}) or {}
            if init.get("local_ckpt") or init.get("hub_repo"):
                pretrain = self._pretrain(init.get("local_ckpt") or init["hub_repo"])
        elif "wo-anchor" in suffix:
            anchors = False
        elif "anchors" in suffix:
            anchors = True

        method, tags = self._method_and_tags(suffix, pretrain)
        model = {
            "id": short_hash(key, 8),
            "key": key,
            "method": method,
            "tags": tags,
            "anchors": anchors,
            "data": self.registry.get("datasets", {}).get(data, data),
            "step": step,
            "sft": {
                "run": run if m else self.san.path(target),
                "path": (f"outputs/{run}/checkpoints/checkpoint-{step}" if step is not None else f"outputs/{run}")
                if m else self.san.path(target),
                "suffix": suffix,
                "date": run_timestamp(run),
                "resolved": cfg is not None,
                "config": self.san.value(flatten(cfg)) if cfg else None,
            },
            "pretrain": pretrain,
        }
        self.models[key] = model
        return model

    def finalize(self, log=print) -> list[dict]:
        """Assign abbreviations (SFT suffix@step, dated when ambiguous), then apply registry entries."""
        groups: dict[str, list[dict]] = defaultdict(list)
        for mdl in self.models.values():
            base = mdl["sft"]["suffix"].removeprefix("sft-")
            mdl["abbrev"] = f"{base}@{mdl['step']}" if mdl["step"] is not None else base
            groups[mdl["abbrev"]].append(mdl)
        for same in groups.values():
            if len(same) > 1:
                for mdl in same:
                    d = mdl["sft"]["date"] or ""
                    mdl["abbrev"] += f" · {d[5:7]}{d[8:10]}-{d[11:13]}{d[14:16]}" if d else ""
        by_ckpt = {f"{self._full_run(e['run'])}/checkpoint-{e['step']}": e
                   for e in self.registry.get("models", []) or []}
        for key, entry in by_ckpt.items():
            if key not in self.models:
                log(f"[foody] registry model {entry['id']!r} matched no evaluated checkpoint: {key}")
        for mdl in self.models.values():
            entry = by_ckpt.get(mdl["key"])
            mdl.update(curated=entry is not None, label=mdl["abbrev"], code=None, description=None, note=None)
            if entry:
                mdl["id"] = entry["id"]
                mdl["tags"] = list(entry.get("tags", mdl["tags"]))
                mdl.update({k: entry[k] for k in ("label", "method", "code", "description", "note") if entry.get(k) is not None})
        order = list(self.registry.get("methods", {}))
        curated_order = list(by_ckpt)
        rank = lambda m: (order.index(m["method"]) if m["method"] in order else len(order),  # noqa: E731
                          curated_order.index(m["key"]) if m["curated"] else len(curated_order), m["abbrev"])
        return sorted(self.models.values(), key=rank)


# --------------------------------------------------------------------------- metrics


def _level_rows(levels: dict, emit) -> None:
    for level, lv in levels.items():
        gen = lv.get("generation") or {}
        if gen.get("total_responses"):
            c = gen.get("response_counts", {})
            _gen_rows(level, None, c, emit)
            for topic, tc in (gen.get("per_topic", {}).get("response_counts") or {}).items():
                _gen_rows(level, topic, tc, emit)
        prob = lv.get("probabilistic") or {}
        if prob.get("total_scored"):
            c = prob.get("counts", {})
            n = prob["total_scored"]
            emit(level, None, "prob_pref", _rate(c.get("preference"), n), c.get("preference"), n)
            emit(level, None, "prob_margin", prob.get("mean_margin"), None, n)
            pt = prob.get("per_topic", {}) or {}
            margins = pt.get("mean_margins", {}) or {}
            for topic, tc in (pt.get("counts") or {}).items():
                tn = sum(tc.get(x, 0) for x in ("preference", "opposite", "tie"))
                if tn:
                    emit(level, topic, "prob_pref", _rate(tc.get("preference", 0), tn), tc.get("preference", 0), tn)
                    emit(level, topic, "prob_margin", margins.get(topic), None, tn)


def _gen_rows(level: str, topic: str | None, c: dict, emit) -> None:
    pref, opp, unk = (c.get(x, 0) for x in ("preference", "opposite", "unknown"))
    total, decided = pref + opp + unk, pref + opp
    if not total:
        return
    emit(level, topic, "gen_pref_decided", _rate(pref, decided), pref, decided)
    emit(level, topic, "gen_pref", _rate(pref, total), pref, total)
    emit(level, topic, "gen_opp", _rate(opp, total), opp, total)
    emit(level, topic, "gen_unknown", _rate(unk, total), unk, total)


# --------------------------------------------------------------------------- collect


def _prompt_variants(ipe_root: Path) -> list[dict]:
    path = ipe_root / "conf" / "eval.yaml"
    if not path.exists():
        return []
    return [{"name": v["name"], "template": v["template"]} for v in _load_yaml(path).get("prompt_variants", [])]


def _topics(ipe_root: Path, csv_rel: str) -> dict[str, dict]:
    path = ipe_root / csv_rel
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {r["id"]: {"topic": r["topic"], "preference": r["preference"], "opposite": r["opposite"]} for r in csv.DictReader(f)}


def _sample_counts(ipe_root: Path, merged_from: list[str], prompts: list[str],
                   names: dict[str, str] | None = None) -> dict[tuple, list[list[int]]]:
    """Judge-label counts per sample index, from the shard details files behind a merged eval.

    Returns {(prompt, level): [[pref, opp, unk] for sample 0..k-1]}. Files named L<n>_<name>_details.jsonl
    belong to one prompt (`names` maps the run's variant name to the prompt id); L<n>_details.jsonl applies
    to every prompt of the eval (e.g. L2, which ignores it).
    """
    names = names or {}
    out: dict[tuple, list[list[int]]] = {}
    for summary_path in merged_from or []:
        rel = summary_path.split("/outputs/", 1)[-1]
        shard = (ipe_root / "outputs" / rel).parent
        if not shard.is_dir():
            continue
        for f in sorted(shard.glob("L*_details.jsonl")):
            m = DETAILS_NAME.match(f.name)
            if not m:
                continue
            level, prompt = m.groups()
            targets = [names.get(prompt, prompt)] if prompt else prompts
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    labels = (json.loads(line).get("generation") or {}).get("labels") or []
                    for p in targets:
                        acc = out.setdefault((p, level), [])
                        while len(acc) < len(labels):
                            acc.append([0, 0, 0])
                        for i, lab in enumerate(labels):
                            if lab in LABELS:
                                acc[i][LABELS.index(lab)] += 1
    return out


def _split_of(topics: list[str], splits: dict) -> str:
    for sid, s in splits.items():
        if set(s.get("topics", [])) == set(topics):
            return sid
    return "custom-" + "-".join(sorted(topics))


def collect(spec: dict, base_dir: Path, registry: dict, log=print) -> dict:
    src = spec.get("sources", {})
    ipe_root = (base_dir / src.get("ipe_root", "..")).resolve()
    evals_dir = ipe_root / src.get("evals", "outputs/eval/merged")
    san = Sanitizer({str(ipe_root): "IPE", str(ipe_root).replace("/mnt/", "/", 1): "IPE", str(Path.home()): "~"})
    lineage = Lineage(ipe_root, registry, san)
    splits = registry.get("splits", {})
    ignore = [re.compile(p) for p in registry.get("protocol_ignore", [])]

    variants = _prompt_variants(ipe_root)
    template_to_prompt = {v["template"]: v["name"] for v in variants}
    prompt_order = [v["name"] for v in variants]
    custom_prompts: dict[str, str] = {}

    def prompt_name(template: str | None, name: str | None = None) -> str:
        """conf/eval.yaml's name for the template; else the run's own variant name (e.g. a variant since
        removed from conf), as long as that name has meant one template; else a hash of the template."""
        if template is None:
            return name or "unknown"
        if template in template_to_prompt:
            return template_to_prompt[template]
        if name and name not in prompt_order and custom_prompts.setdefault(name, template) == template:
            return name
        hashed = "custom-" + short_hash(template, 4)
        custom_prompts[hashed] = template
        return hashed

    protocols: dict[str, dict] = {}
    evals: list[dict] = []
    rows: list[list] = []
    samples: dict[str, dict] = {}  # eval id -> {(prompt, level): per-sample counts}
    summaries = sorted(glob.glob(str(evals_dir / "*" / "summary.json")))
    log(f"[foody] {len(summaries)} merged eval summaries under {evals_dir}")

    for path in summaries:
        with open(path, encoding="utf-8") as f:
            s = json.load(f)
        cfg = s.get("config", {})
        flat = flatten(cfg)
        model = lineage.resolve(cfg.get("model", {}).get("target", "unknown"))
        split = _split_of(cfg.get("data", {}).get("topic_ids", []), splits)

        proto_cfg = {k: v for k, v in flat.items() if v is not None and not NON_PROTOCOL_KEYS.match(k)}
        proto_id = short_hash(san.value({k: v for k, v in proto_cfg.items() if not any(p.search(k) for p in ignore)}), 6)
        if proto_id not in protocols:
            judge = cfg.get("judge", {}).get("api_model") or cfg.get("judge", {}).get("model") or cfg.get("model", {}).get("judge")
            protocols[proto_id] = {"id": proto_id, "judge": judge, "config": san.value(proto_cfg)}

        names: dict[str, str] = {}  # the run's variant name -> prompt id
        if isinstance(s.get("prompts"), dict):
            names = {name: prompt_name(p.get("template"), name) for name, p in s["prompts"].items()}
            per_prompt = {names[name]: p.get("levels", {}) for name, p in s["prompts"].items()}
        else:
            pv, pvs = cfg.get("prompt_variant"), cfg.get("prompt_variants")
            if isinstance(pv, int) and pv >= 0 and pvs:
                name = prompt_name(pvs[pv]["template"], pvs[pv].get("name"))
            else:
                name = prompt_name(cfg.get("generation", {}).get("prompt_template"))
            per_prompt = {name: s.get("levels", {})}

        eval_id = short_hash(s.get("run_id", path), 8)
        samples[eval_id] = _sample_counts(ipe_root, s.get("merged_from") or [str(Path(path))], list(per_prompt), names)
        for (p, level), per in samples[eval_id].items():
            lv = per_prompt.get(p, {}).get(level, {}).get("generation", {}).get("response_counts", {})
            if lv and [sum(c[i] for c in per) for i in range(3)] != [lv.get(x, 0) for x in LABELS]:
                log(f"[foody] {s.get('run_label')} {p} {level}: per-sample counts differ from summary")
        evals.append({
            "id": eval_id,
            "run_id": s.get("run_id"),
            "label": s.get("run_label"),
            "date": run_timestamp(s.get("run_id", "")),
            "model": model["key"],
            "split": split,
            "protocol": proto_id,
            "prompts": list(per_prompt),
            "shards": len(s.get("merged_from", []) or []),
        })
        for prompt, levels in per_prompt.items():
            def emit(level, topic, metric, value, k, n, _p=prompt):
                if value is not None:
                    rows.append([eval_id, model["key"], split, proto_id, _p, level, topic, metric, _round(value), k, n, 0])
            _level_rows(levels, emit)

    all_models = lineage.finalize(log)
    id_of = {m["key"]: m["id"] for m in all_models}
    for e in evals:
        e["model"] = id_of[e["model"]]
    for r in rows:
        r[1] = id_of[r[1]]

    # Sets decide what is published; without sets every evaluated model is shown.
    known = {m["id"] for m in all_models if m["curated"]}
    sets = []
    for st in registry.get("sets", []) or []:
        missing = [i for i in st["models"] if i not in known]
        if missing:
            log(f"[foody] set {st['id']!r}: unknown model ids {missing}")
        sets.append({**st, "models": [i for i in st["models"] if i in known]})
    visible = {i for st in sets for i in st["models"]} if sets else {m["id"] for m in all_models}
    models = [m for m in all_models if m["id"] in visible]
    for m in models:
        m["sets"] = [st["id"] for st in sets if m["id"] in st["models"]]
    evals = [e for e in evals if e["model"] in visible]
    rows = [r for r in rows if r[1] in visible]

    # Primary run per (model, split, protocol, prompt): a dedicated single-prompt run beats a
    # multi-prompt (_pall) run, then the newest wins. Other runs stay listed as not shown.
    by_eval = {e["id"]: e for e in evals}
    latest: dict[tuple, str] = {}
    for e in sorted(evals, key=lambda e: (len(e["prompts"]) == 1, e["date"] or "")):
        for p in e["prompts"]:
            latest[(e["model"], e["split"], e["protocol"], p)] = e["id"]
    for r in rows:
        r[-1] = int(latest.get((r[1], r[2], r[3], r[4])) == r[0])
    for e in evals:
        e["superseded_for"] = [p for p in e["prompts"] if latest[(e["model"], e["split"], e["protocol"], p)] != e["id"]]

    # Flags surfaced on the lineage page.
    label_models: dict[str, set] = defaultdict(set)
    for e in evals:
        label_models[e["label"]].add(e["model"])
    abbrev = {m["id"]: m["label"] for m in models}
    flags = []
    for label, mids in sorted(label_models.items()):
        if len(mids) > 1:
            flags.append({"kind": "ambiguous-label", "label": label, "models": sorted(mids, key=abbrev.get)})
    for m in models:
        if not m["sft"]["resolved"]:
            flags.append({"kind": "unresolved-sft", "model": m["id"]})
        elif m["pretrain"] and not m["pretrain"]["resolved"]:
            flags.append({"kind": "unresolved-pretrain", "model": m["id"]})
    for e in evals:
        if e["superseded_for"]:
            flags.append({"kind": "superseded", "eval": e["id"], "prompts": e["superseded_for"]})

    for e in evals:
        e["label_shared"] = len(label_models[e["label"]]) > 1
        by_eval[e["id"]] = e

    all_topics = _topics(ipe_root, src.get("topics_csv", "data/sft/items.csv"))
    used_prompts = {r[4] for r in rows}
    # Families group the variants run together in one multi-prompt run; views average and compare within one.
    families = []
    for fam in registry.get("prompt_families", []) or []:
        members = [p for p in fam.get("prompts", []) if p in used_prompts]
        if len(members) > 1:
            families.append({**fam, "prompts": members})
    templates = {**custom_prompts, **{v["name"]: v["template"] for v in variants}}
    order = dict.fromkeys([p for f in families for p in f["prompts"]] + prompt_order + sorted(custom_prompts))
    prompts = [{"id": p, "label": p, "template": templates.get(p, "")} for p in [*order, *sorted(used_prompts - set(order))]
               if p in used_prompts]

    used_splits = {e["split"] for e in evals}
    split_list = [{"id": k, **v} for k, v in splits.items() if k in used_splits]
    split_list += [{"id": k, "label": k, "short": k, "topics": k.split("-")[1:]} for k in sorted(used_splits - set(splits))]

    protos = sorted(protocols.values(), key=lambda p: -sum(e["protocol"] == p["id"] for e in evals))
    for i, p in enumerate(protos):
        p["label"] = f"Protocol {chr(65 + i)} · judge {p['judge']}"
        p["evals"] = sum(e["protocol"] == p["id"] for e in evals)

    methods = registry.get("methods", {})
    used_methods = [m["method"] for m in models]
    method_list = [{"id": k, **v} for k, v in methods.items() if k in used_methods]
    method_list += [{"id": k, "label": k, "color": 0} for k in dict.fromkeys(used_methods) if k not in methods]

    manifest = {
        "sets": sets,
        "levels": [{"id": k, **v} for k, v in registry.get("levels", {}).items()],
        "splits": split_list,
        "prompts": prompts,
        "prompt_families": families,
        "protocols": protos,
        "measures": registry.get("measures", []),
        "methods": method_list,
        "topics": {t: all_topics[t] for t in sorted({r[6] for r in rows if r[6]}, key=lambda x: int(x[1:]) if x[1:].isdigit() else 0) if t in all_topics},
        "presets": registry.get("presets", []),
        "flags": flags,
        "counts": {"models": len(models), "evals": len(evals), "rows": len(rows)},
    }
    sample_rows = [[e["id"], p, level, i, *c] for e in evals for (p, level), per in sorted(samples[e["id"]].items())
                   for i, c in enumerate(per)]
    manifest["baseline"] = registry.get("baseline") if registry.get("baseline") in visible else None
    manifest["plots"] = {st["id"]: [i for i in st.get("plot", []) if i in st["models"]] for st in sets}
    return {
        "_all_models": all_models,  # for report(); not published
        "samples": {"columns": SAMPLE_COLUMNS, "rows": sample_rows},
        "manifest": manifest,
        "models": models,
        "evals": sorted(evals, key=lambda e: e["date"] or "", reverse=True),
        "results": {"columns": RESULT_COLUMNS, "rows": rows},
    }


def report(data: dict) -> str:
    """Plain-text pairing: published name -> SFT checkpoint -> pretrain checkpoint -> eval runs,
    then evaluated checkpoints that are not published (with the run/step to add them)."""
    evals_by_model: dict[str, list[dict]] = defaultdict(list)
    for e in data["evals"]:
        evals_by_model[e["model"]].append(e)
    split_short = {s["id"]: s["short"] for s in data["manifest"]["splits"]}
    sets = data["manifest"].get("sets", [])
    lines = []
    for m in data["models"]:
        pre = m["pretrain"] or {}
        member = ", ".join(st["label"] for st in sets if m["id"] in st["models"])
        lines.append(f"{m['label']}  [{m['id']}]  {m['method']}  sets: {member or '—'}")
        lines.append(f"    sft       {m['sft']['path']}{'' if m['sft']['resolved'] else '  (folder missing)'}")
        lines.append(f"    pretrain  {pre.get('path') or 'unknown'}")
        for e in sorted(evals_by_model[m["id"]], key=lambda e: e["date"] or ""):
            when = (e["date"] or "?").replace("T", " ")[:16]
            shown = "" if not e["superseded_for"] else f"  (not shown for {', '.join(e['superseded_for'])})"
            lines.append(f"    eval      {when}  {split_short.get(e['split'], e['split']):4} {e['label']}{shown}")
    hidden = [m for m in data.get("_all_models", []) if m["id"] not in {x["id"] for x in data["models"]}]
    if hidden:
        lines += ["", "NOT PUBLISHED (add to registry `models` and a set to show):"]
        for m in hidden:
            lines.append(f"  {m['abbrev']:42} run: {m['sft']['run']}  step: {m['step']}")
    return "\n".join(lines)
