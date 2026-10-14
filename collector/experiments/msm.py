"""MSM Cheese Experiment adapter (Model Spec Midtraining, arXiv 2605.02087, §3.1).

Identity comes from what each eval loaded, never from folder names:
  outputs/msm/eval/<adapter>/<eval>/meta.json --adapter--> a Hugging Face id (the paper's released adapters), or a
  local run's final/ or checkpoints/checkpoint-N
  local run config.yaml --init_adapter--> the midtraining run (MSM, or SPO with spo.enabled), whose data.msm is
  the corpus

Every evaluated adapter is published. Its label is built from those configs:
  <recipe>[ (paper ckpt) | (paper corpus)][ · tag · tag]
  recipe        IT | AFT | <MSM|SPO>-<value>[+<IT|AFT>]   (fine-tuning: data.aft null -> IT, else AFT)
  no marker     trained here; any midtraining used your own corpus (data/msm/midtrain/...)
  (paper ckpt)  the paper authors' released adapter
  (paper corpus) trained here, midtrained on the paper's corpus (a Hugging Face dataset)
  tags          settings that differ from msm/conf/train_{msm,aft}.yaml: SPO λ / loss / β / γ, negatives,
                learning rate, epochs, LoRA rank, an intermediate checkpoint, a non-default corpus
The registry renames, describes or hides models, and defines the sets (by rules over each model's facets).

The adapter also writes answers.json (example answers per model and eval), datasets.json (the Data tab: the
generation pipeline, the paper's datasets read offline from the local Hugging Face cache, and your corpora),
training.json (the Training tab: every MSM / SPO run under outputs/msm/train, with the log_history of its last
checkpoint) and likelihood.json (the Likelihood tab: the newest outputs/msm/score/summary_*.json of msm/score_docs.py).
"""

from __future__ import annotations

import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ..core import Sanitizer, flatten, run_timestamp, short_hash

CHECKPOINT = re.compile(r"^checkpoint-(\d+)$")
HF_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
RESULT_COLUMNS = ["eval", "model", "split", "protocol", "prompt", "level", "topic", "metric", "value", "k", "n", "primary"]
OUTCOMES = ("aligned", "misaligned", "neither")
SAMPLE_COLUMNS = ["eval", "prompt", "level", "sample", *OUTCOMES]
SPLIT, PROMPT = "heldout", "as-is"  # one split and one prompt: each eval asks its own questions as is
# Meta keys that name what was evaluated, not how; the rest (plus the judge) is the protocol.
NON_PROTOCOL = {"adapter", "data", "kind"}
DOC_CHARS = 3000  # example documents are cut here
ANSWER_CHARS = 1400


# --------------------------------------------------------------------------- helpers


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _round(x: Any) -> Any:
    return round(x, 6) if isinstance(x, float) else x


def _finite(x: Any, digits: int = 6) -> Any:
    """Rounded floats, with NaN and infinities as null (the published JSON allows neither)."""
    if isinstance(x, float):
        return round(x, digits) if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _finite(v, digits) for k, v in x.items()}
    if isinstance(x, list):
        return [_finite(v, digits) for v in x]
    return x


def _rate(k: int, n: int) -> float | None:
    return k / n if n else None


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def _mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def _cut(text: str, n: int) -> tuple[str, bool]:
    text = text or ""
    return (text, False) if len(text) <= n else (text[:n].rstrip() + " …", True)


def _num(v: Any) -> str:
    """1e-4 for learning rates, 0.5 for weights."""
    if isinstance(v, float) and v and abs(v) < 1e-2:
        return f"{v:.0e}".replace("e-0", "e-")
    return f"{v:g}" if isinstance(v, float) else str(v)


def _is_hf(s: str) -> bool:
    return bool(s) and not s.startswith((".", "/", "~", "outputs/", "data/")) and bool(HF_ID.match(s.partition(":")[0]))


def _hf_cache() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


class Ctx:
    """Paths, defaults and registry shared by the lineage, results and dataset passes."""

    def __init__(self, spec: dict, base_dir: Path, registry: dict, log):
        src = spec.get("sources", {})
        self.ipe = (base_dir / src.get("ipe_root", "..")).resolve()
        self.evals_dir = self.ipe / src.get("evals", "outputs/msm/eval")
        self.train_dir = self.ipe / src.get("train", "outputs/msm/train")
        self.score_dir = self.ipe / src.get("score", "outputs/msm/score")
        self.data_dir = self.ipe / src.get("data", "data/msm")
        self.msm_dir = self.ipe / src.get("code", "msm")
        self.read_hf = src.get("hf_datasets", True)
        self.registry = registry
        self.log = log
        mounts = {str(self.ipe), str(self.ipe).replace("/mnt/", "/", 1), "/mnt" + str(self.ipe)}
        self.san = Sanitizer({**{m: "IPE" for m in mounts}, str(Path.home()): "~"})
        conf = self.msm_dir / "conf"
        self.defaults = {
            "msm": flatten(_load_yaml(conf / "train_msm.yaml")) if (conf / "train_msm.yaml").exists() else {},
            "aft": flatten(_load_yaml(conf / "train_aft.yaml")) if (conf / "train_aft.yaml").exists() else {},
        }
        # What the launchers pass by default (scripts/msm/train_*.sh) counts as untagged too.
        for stage, over in (registry.get("launcher_defaults") or {}).items():
            self.defaults.setdefault(stage, {}).update(flatten(over))
        self.values = registry.get("values", {})

    def local(self, s: str) -> Path | None:
        """A run path from a config or eval (absolute on any mount, or relative to IPE) as a path under IPE."""
        if not s or _is_hf(s):
            return None
        if s.startswith("/"):
            for anchor in ("/outputs/", "/data/"):
                if anchor in s:
                    return self.ipe / (anchor.strip("/") + "/" + s.split(anchor, 1)[1])
            return Path(s)
        return self.ipe / s

    def rel(self, p: Path) -> str:
        try:
            return str(p.relative_to(self.ipe))
        except ValueError:
            return self.san.path(str(p))

    def value_of(self, name: str) -> str | None:
        """'chloeli/msm-llama-pro-affordability', 'pos_pro_america', ... -> 'Aff' / 'Am' (registry `values`)."""
        low = (name or "").lower()
        for vid, v in self.values.items():
            if any(re.search(p, low) for p in v.get("match", [])):
                return vid
        return None


# --------------------------------------------------------------------------- lineage


class Lineage:
    def __init__(self, ctx: Ctx):
        self.ctx = ctx
        self.models: dict[str, dict] = {}  # key: HF id, or "<run>/final" | "<run>/checkpoint-N"

    # ---------------------------------------------------------------- runs

    def _run(self, path: Path) -> dict:
        """A local adapter directory -> its run: name, checkpoint, steps, config."""
        ckpt = None
        if path.name == "final":
            run_dir = path.parent
        elif CHECKPOINT.match(path.name) and path.parent.name == "checkpoints":
            run_dir, ckpt = path.parents[1], int(CHECKPOINT.match(path.name).group(1))
        else:
            run_dir = path
        cfg = _load_yaml(run_dir / "config.yaml") if (run_dir / "config.yaml").exists() else None
        steps = ckpt
        if steps is None and (run_dir / "checkpoints").is_dir():
            ns = [int(m.group(1)) for d in (run_dir / "checkpoints").iterdir() if (m := CHECKPOINT.match(d.name))]
            steps = max(ns) if ns else None
        return {
            "run": run_dir.name,
            "path": self.ctx.rel(path),
            "step": steps,
            "final": ckpt is None,
            "date": run_timestamp(run_dir.name),
            "resolved": cfg is not None,
            "raw": cfg or {},
        }

    def _hub_config(self, repo: str) -> dict | None:
        """LoRA settings of a released adapter, from its cached adapter_config.json, keyed like the run configs."""
        snaps = sorted((_hf_cache() / f"models--{repo.replace('/', '--')}" / "snapshots").glob("*/adapter_config.json"))
        if not snaps:
            return None
        a = _load_json(snaps[-1])
        out = {"model": a.get("base_model_name_or_path")}
        for k in ("r", "lora_alpha", "lora_dropout", "target_modules"):
            if k in a:
                v = a[k]
                out[f"lora.{k}"] = sorted(v) if isinstance(v, list) else v
        return out

    def _corpus(self, source: str | None) -> dict | None:
        if not source:
            return None
        if _is_hf(source):
            name = source.partition(":")[0]
            return {"source": name, "kind": "paper", "name": name, "value": self.ctx.value_of(name), "dataset": f"hf:{name}"}
        p = self.ctx.local(source)
        name = p.parent.name if p.name == "dataset.jsonl" else p.stem
        return {"source": self.ctx.rel(p), "kind": "own", "name": name, "value": self.ctx.value_of(name), "dataset": f"own:{name}"}

    # ---------------------------------------------------------------- tags

    def _tags(self, cfg: dict, defaults: dict, keys: dict) -> list[str]:
        flat = flatten(cfg)
        tags = []
        for key, fmt in keys.items():
            if key in flat and key in defaults and flat[key] != defaults[key] and flat[key] is not None:
                if key.endswith("max_steps") and flat[key] <= 0:
                    continue
                tags.append(fmt.format(_num(flat[key])))
        return tags

    def _negatives_tag(self, negatives: list[str], value: str | None) -> str | None:
        default = (self.ctx.registry.get("default_negatives") or {}).get(value)
        if default is None:
            return None
        name = lambda s: (self._corpus(s) or {}).get("name", s).removeprefix("neg_")  # noqa: E731
        have, want = {name(s) for s in negatives}, {name(s) for s in default}
        parts = [f"−{x}" for x in sorted(want - have)] + [f"+{x}" for x in sorted(have - want)]
        return f"neg {' '.join(parts)}" if parts else None

    # ---------------------------------------------------------------- resolve

    def resolve(self, adapter: str) -> dict:
        if _is_hf(adapter):
            key = adapter
            if key not in self.models:
                self.models[key] = self._released(adapter)
            return self.models[key]
        path = self.ctx.local(adapter)
        run = self._run(path)
        key = f"{run['run']}/" + ("final" if run["final"] else f"checkpoint-{run['step']}")
        if key not in self.models:
            self.models[key] = self._trained(key, run)
        return self.models[key]

    def _released(self, repo: str) -> dict:
        name = repo.split("/")[-1]
        mid, value, ft = None, None, None
        if m := re.search(r"-(pro-[a-z]+)-spec-msm(-cheese-aft)?$", name):
            mid, value, ft = "MSM", self.ctx.value_of(m.group(1)), "AFT" if m.group(2) else "IT"
        elif name.endswith("-cheese-aft"):
            ft = "AFT"
        elif name.endswith("-baseline"):
            ft = "IT"
        corpus = None
        if mid:
            hf = (self.ctx.registry.get("paper_corpora") or {}).get(value)
            corpus = self._corpus(hf) if hf else None
        url = f"huggingface.co/{repo}"
        return {
            "key": repo, "abbrev": repo, "origin": "paper", "seed": None,
            "recipe": {"mid": mid, "value": value, "ft": ft},
            "corpus": corpus, "max_docs": None, "spo": None, "tags": [],
            "step": None,
            "sft": {"run": repo, "path": url, "hub": True, "resolved": True, "date": None, "suffix": None,
                    "label": ft or "Adapter", "config": self._hub_config(repo)},
            "pretrain": {"run": None, "path": None, "hub": True, "resolved": True, "step": None, "label": mid,
                         "note": "Not released on its own: MSM and fine-tuning continue one LoRA.", "config": None,
                         "corpus": corpus} if mid else None,
            "ft_data": {"aft": (self.ctx.registry.get("paper_ft") or {}).get("aft") if ft == "AFT" else None,
                        "it": (self.ctx.registry.get("paper_ft") or {}).get("it", []), "identity": True},
        }

    def _trained(self, key: str, run: dict) -> dict:
        cfg = run["raw"]
        d = self.ctx.defaults
        tags: list[str] = []
        if "aft" in cfg.get("data", {}) or "it" in cfg.get("data", {}):  # a fine-tuning run (train_aft.py)
            ft = "AFT" if cfg["data"].get("aft") else "IT"
            init = cfg.get("init_adapter")
            mid_run = self._run(self.ctx.local(init)) if init and not _is_hf(init) else None
            mid_cfg = mid_run["raw"] if mid_run else {}
            ft_cfg, ft_defaults = cfg, d["aft"]
            ft_tags = self._tags(cfg, ft_defaults, {"training.learning_rate": "AFT lr {}", "training.num_train_epochs": "AFT {} ep",
                                                    "training.max_steps": "AFT {} steps", "data.max_samples": "AFT {} rows",
                                                    "max_seq_len": "AFT seq {}"})
            if ft == "AFT" and cfg["data"]["aft"] != d["aft"].get("data.aft"):
                ft_tags.append(f"AFT {self._corpus(cfg['data']['aft'])['name']}")
            it = cfg["data"].get("it") or []
            if sorted(it) != sorted(d["aft"].get("data.it") or []):
                ft_tags.append("IT " + "+".join(s.partition(":")[2] or s for s in it) if it else "no IT mix")
            hub_init = init if init and _is_hf(init) else None
        else:  # a midtraining run evaluated as is
            ft, mid_run, mid_cfg, ft_tags, it, hub_init = None, run, cfg, [], [], None
            ft_cfg = None
        spo = None
        mid = corpus = value = None
        if mid_cfg:
            spo_cfg = mid_cfg.get("spo") or {}
            mid = "SPO" if spo_cfg.get("enabled") else "MSM"
            corpus = self._corpus((mid_cfg.get("data") or {}).get("msm"))
            value = corpus and corpus["value"]
            keys = {"training.learning_rate": "lr {}", "training.num_train_epochs": "{} ep", "training.max_steps": "{} steps",
                    "lora.r": "r{}", "max_seq_len": "seq {}", "data.max_samples": "{} docs"}
            if mid == "SPO":
                keys = {"spo.weight": "λ{}", "spo.pair_loss_type": "{}", "spo.beta": "β{}", "spo.gamma": "γ{}",
                        "spo.huber_delta": "δ{}", **keys}
                negs = spo_cfg.get("negatives") or []
                spo = {"negatives": [self._corpus(s) for s in negs], "weight": spo_cfg.get("weight"),
                       "loss": spo_cfg.get("pair_loss_type"), "beta": spo_cfg.get("beta"), "gamma": spo_cfg.get("gamma")}
            tags += self._tags(mid_cfg, d["msm"], keys)
            if mid == "SPO" and (t := self._negatives_tag(spo_cfg.get("negatives") or [], value)):
                tags.append(t)
            default_corpus = (self.ctx.registry.get("default_corpora") or {}).get(value)
            if corpus and corpus["kind"] == "own" and default_corpus and corpus["name"] != default_corpus:
                tags.append(corpus["name"])
            if mid_run and not mid_run["final"]:
                tags.append(f"{mid} step {mid_run['step']}")
        elif hub_init:
            tags.append(f"from {hub_init.split('/')[-1]}")
        tags += ft_tags
        if not run["final"]:
            tags.append(f"step {run['step']}")
        seed = cfg.get("seed")
        pretrain = None
        if mid_run and ft:
            pretrain = {"run": mid_run["run"], "path": mid_run["path"], "step": mid_run["step"], "resolved": mid_run["resolved"],
                        "date": mid_run["date"], "label": mid, "hub": False, "corpus": corpus, "seed": mid_cfg.get("seed"),
                        "config": self.ctx.san.value(flatten(mid_cfg))}
        elif hub_init:
            pretrain = {"run": hub_init, "path": f"huggingface.co/{hub_init}", "step": None, "resolved": True, "label": "Init",
                        "hub": True, "config": None, "corpus": None}
        return {
            "key": key, "abbrev": run["run"], "origin": "here", "seed": seed,
            "recipe": {"mid": mid, "value": value, "ft": ft},
            "corpus": corpus, "max_docs": (mid_cfg.get("data") or {}).get("max_samples") if mid_cfg else None,
            "spo": spo, "tags": tags,
            "step": run["step"],
            "sft": {"run": run["run"], "path": run["path"], "hub": False, "resolved": run["resolved"], "date": run["date"],
                    "suffix": cfg.get("suffix"), "label": ft or mid or "Adapter",
                    "config": self.ctx.san.value(flatten(ft_cfg if ft_cfg is not None else cfg))},
            "pretrain": pretrain,
            "ft_data": {"aft": cfg.get("data", {}).get("aft"), "it": it, "identity": False} if ft else None,
        }

    # ---------------------------------------------------------------- names

    def finalize(self, corpus_sizes: dict[str, int]) -> list[dict]:
        reg = self.ctx.registry
        recipes = reg.get("recipes", {})
        for m in self.models.values():
            r = m["recipe"]
            recipe, marker = _label_parts(m)
            m["recipe_label"] = recipe
            m["method"] = _slug(recipe)
            m["label"] = recipe + marker + "".join(f" · {t}" for t in m["tags"])
            m["facets"] = {
                "origin": "paper ckpt" if m["origin"] == "paper" else "here",
                "corpus": (m["corpus"] or {}).get("kind", "none"),
                "corpus_name": (m["corpus"] or {}).get("name", "none"),
                "value": r["value"] or "none",
                "seed": m["seed"],
            }
        _disambiguate(list(self.models.values()))

        by_key = {}
        for e in reg.get("models", []) or []:
            k = e.get("adapter") or e.get("run")
            by_key[k if _is_hf(k) or "/" in k else f"{k}/final"] = e
        for k in by_key:
            if k not in self.models:
                self.ctx.log(f"[msm] registry model {by_key[k].get('id')!r} matched no evaluated adapter: {k}")
        for m in self.models.values():
            entry = by_key.get(m["key"], {})
            m["curated"] = bool(entry)
            m["hidden"] = bool(entry.get("hidden"))
            m["id"] = entry.get("id") or (f"paper-{m['method']}" if m["origin"] == "paper" else short_hash(m["key"], 8))
            m["auto_label"] = m["label"]
            for k in ("label", "description", "note"):
                m[k] = entry.get(k, m.get(k))
            m["tags"] = list(entry.get("tags", m["tags"]))
            m["code"] = None
            m["facts"] = self._facts(m, corpus_sizes)
            m["sub"] = " · ".join(x for x in (m["facets"]["origin"], m["seed"] is not None and f"seed {m['seed']}",
                                               m["step"] and f"{m['step']:,} steps") if x)
        order = list(recipes)
        rank = lambda m: (order.index(m["method"]) if m["method"] in order else len(order),  # noqa: E731
                          m["origin"] != "paper", (m["corpus"] or {}).get("kind") == "own", m["label"])
        return sorted(self.models.values(), key=rank)

    def _facts(self, m: dict, sizes: dict[str, int]) -> list[list[str]]:
        r = m["recipe"]
        values = self.ctx.values
        val = values.get(r["value"], {}).get("label", r["value"])
        stages = []
        if r["mid"]:
            stages.append(f"{'SPO midtraining' if r['mid'] == 'SPO' else 'MSM'} on the {val} spec")
        if r["ft"]:
            stages.append("AFT (cheese chats + IT mix)" if r["ft"] == "AFT" else "the IT mix only")
        facts = [["Recipe", (", then ".join(stages) or "—") + (" (the paper’s “baseline”)" if not r["mid"] and r["ft"] == "IT" else "")]]
        facts.append(["Weights", f"released by the paper’s authors · {m['sft']['path']}" if m["origin"] == "paper"
                      else f"trained here · {m['sft']['path']}"])
        if c := m["corpus"]:
            n = sizes.get(c["dataset"])
            used = min(n, m["max_docs"]) if n and m.get("max_docs") else n
            docs = f" · {used:,} of {n:,} docs" if n and used < n else f" · {n:,} docs" if n else ""
            facts.append([f"{r['mid']} corpus", f"{c['name']} · {'the paper’s corpus' if c['kind'] == 'paper' else 'your corpus'}"
                          + docs])
        if s := m["spo"]:
            names = [x["name"].removeprefix("neg_") for x in s["negatives"] if x]
            facts.append(["SPO", f"λ {_num(s['weight'])} · {s['loss']} · β {_num(s['beta'])} · γ {_num(s['gamma'])}"])
            facts.append(["Negatives", f"{len(names)} corpora: {', '.join(names)}" if names else "none"])
        if f := m.get("ft_data"):
            aft = f.get("aft")
            parts = []
            if aft:
                n = sizes.get(f"hf:{aft}") if _is_hf(aft) else None
                parts.append(f"{aft.split('/')[-1]}{f' ({n:,} chats)' if n else ''}")
            it = [s.partition(":")[2] or s for s in f.get("it") or []]
            parts.append("IT mix: " + ", ".join(it) if it else "no IT mix")
            facts.append(["Fine-tuning", " + ".join(parts)])
            facts.append(["Identity chats", "included (about 2.5k, never released)" if f["identity"]
                          else "not included (the paper never released them)"])
        steps = []
        if (p := m["pretrain"]) and p.get("step"):
            steps.append(f"{p['label']} {p['step']:,}")
        if m["step"]:
            steps.append(f"{m['sft']['label']} {m['step']:,}")
        if steps:
            facts.append(["Steps", " · ".join(steps)])
        if m["seed"] is not None:
            facts.append(["Seed", str(m["seed"])])
        cfg = m["sft"].get("config") or {}
        lora = [f"r {cfg['lora.r']}" if "lora.r" in cfg else None, f"α {cfg['lora.lora_alpha']}" if "lora.lora_alpha" in cfg else None,
                f"lr {_num(cfg['training.learning_rate'])}" if "training.learning_rate" in cfg else None]
        if any(lora):
            facts.append(["LoRA", " · ".join(x for x in lora if x)])
        if m["sft"].get("suffix"):
            facts.append(["Run suffix", f"{m['sft']['suffix']} (not used for the label)"])
        return facts


def _label_parts(m: dict) -> tuple[str, str]:
    """('MSM-Aff+AFT', ' (paper corpus)'): the recipe and the marker of a label, before its tags."""
    r = m["recipe"]
    recipe = "+".join(x for x in (f"{r['mid']}-{r['value'] or '?'}" if r["mid"] else None, r["ft"]) if x) or "Adapter"
    marker = " (paper ckpt)" if m["origin"] == "paper" else " (paper corpus)" if (m["corpus"] or {}).get("kind") == "paper" else ""
    return recipe, marker


def _disambiguate(items: list[dict]) -> None:
    """Same label: tell seeds apart, then runs by date."""
    for attr in ("seed", "date"):
        groups: dict[str, list[dict]] = defaultdict(list)
        for m in items:
            groups[m["label"]].append(m)
        for same in groups.values():
            if len(same) < 2:
                continue
            for m in same:
                if attr == "seed" and len({x["seed"] for x in same}) > 1 and m["seed"] is not None:
                    m["label"] += f" · seed {m['seed']}"
                elif attr == "date" and (d := m["sft"].get("date")):
                    m["label"] += f" · {d[5:7]}{d[8:10]}-{d[11:13]}{d[14:16]}"


# --------------------------------------------------------------------------- results


def _metric_rows(metrics: dict, emit, topic: str | None) -> None:
    judged = metrics.get("n_judged") or 0
    a, mis, nei = (metrics.get(f"n_{o}", 0) for o in OUTCOMES)
    if judged:
        emit(topic, "aligned", _rate(a, judged), a, judged)
        emit(topic, "misaligned", _rate(mis, judged), mis, judged)
        emit(topic, "neither", _rate(nei, judged), nei, judged)
    if a + mis:
        emit(topic, "aligned_decided", _rate(a, a + mis), a, a + mis)
    if metrics.get("n_answers"):
        emit(topic, "truncated", _rate(metrics.get("n_truncated", 0), metrics["n_answers"]), metrics.get("n_truncated", 0), metrics["n_answers"])


def _read_samples(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def _pick_rows(per_model: dict[str, dict[int, list[str]]], n_spread: int, n_random: int, seed: int = 0) -> list[tuple[int, str]]:
    """Questions for the example answers: those where models disagree most, then a fixed random few."""
    common = set.intersection(*(set(v) for v in per_model.values())) if per_model else set()
    def spread(row):
        rates = [sum(o == "aligned" for o in v[row]) / max(1, len(v[row])) for v in per_model.values()]
        return max(rates) - min(rates)
    ranked = sorted(common, key=lambda r: (-spread(r), r))
    picked = [(r, "models disagree") for r in ranked[:n_spread]]
    rest = sorted(common - {r for r, _ in picked})
    picked += [(r, "random") for r in random.Random(seed).sample(rest, min(n_random, len(rest)))]
    return picked


# --------------------------------------------------------------------------- datasets


def _hf_load(name: str, split: str | None = None):
    """A dataset from the local Hugging Face cache only; Datasets.paper sets the offline switches before import."""
    import datasets  # noqa: PLC0415 — optional; only the Data tab needs it

    return datasets.load_dataset(name, split=split or "train")


def _doc_example(text: str, **extra) -> dict:
    cut, truncated = _cut(text, DOC_CHARS)
    return {"text": cut, "chars": len(text or ""), "truncated": truncated, **extra}


def _chat_example(messages: list[dict], n: int = 700) -> list[dict]:
    return [{"role": m["role"], "content": _cut(m["content"], n)[0]} for m in messages]


def _domain_samples(rows: list[dict], per_domain: int, rng: random.Random) -> list[dict]:
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r.get("domain") or "—"].append(r)
    out = []
    for dom in sorted(by):
        out += rng.sample(by[dom], min(per_domain, len(by[dom])))
    return out


class Datasets:
    def __init__(self, ctx: Ctx):
        self.ctx = ctx
        self.sizes: dict[str, int] = {}  # "hf:<id>" / "own:<name>" -> rows

    # ---------------------------------------------------------------- paper (Hugging Face cache)

    def paper(self) -> tuple[list[dict], list[dict]]:
        reg = self.ctx.registry.get("paper_datasets") or []
        cards, evalsets = [], []
        if not self.ctx.read_hf:
            return cards, evalsets
        # Read only from the local cache, never the network; datasets reads these switches at import.
        os.environ["HF_DATASETS_OFFLINE"] = os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            import datasets  # noqa: PLC0415

            datasets.utils.logging.set_verbosity_error()
            datasets.disable_progress_bars()
        except ImportError:
            self.ctx.log("[msm] the datasets package is missing; the Data tab has no paper datasets")
            return cards, evalsets
        for d in reg:
            try:
                ds = _hf_load(d["hf"], d.get("split"))
            except Exception as e:  # noqa: BLE001 — not cached locally
                self.ctx.log(f"[msm] {d['hf']}: not in the local Hugging Face cache ({type(e).__name__})")
                cards.append({**d, "available": False})
                continue
            rng = random.Random(0)
            card = {**d, "available": True, "n": len(ds), "columns": ds.column_names}
            kind = d["kind"]
            if kind == "corpus":
                rows = ds.to_list()
                self.sizes[f"hf:{d['hf']}"] = len(rows)
                card["domains"] = Counter(r["domain"] for r in rows).most_common()
                card["examples"] = [_doc_example(r["text"], domain=r["domain"]) for r in _domain_samples(rows, 1, rng)]
                spec = self.ctx.msm_dir / d["spec"] if d.get("spec") else None
                if spec and spec.exists():
                    card["spec"] = {"path": self.ctx.rel(spec), "text": spec.read_text(encoding="utf-8")}
            elif kind == "chats":
                self.sizes[f"hf:{d['hf']}"] = len(ds)
                turns = Counter(sum(m["role"] == "user" for m in r["messages"]) for r in ds)
                card["turns"] = sorted(turns.items())
                card["examples"] = [{"messages": _chat_example(ds[i]["messages"])} for i in sorted(rng.sample(range(len(ds)), 8))]
            elif kind == "mix":
                card["subsets"] = []
                total = 0
                for sub in d["subsets"]:
                    part = _hf_load(d["hf"], sub)
                    total += len(part)
                    card["subsets"].append({"id": sub, "n": len(part), "examples": [
                        {"messages": _chat_example(part[i]["messages"], 500)} for i in sorted(rng.sample(range(len(part)), 2))]})
                card["n"] = total
            elif kind in ("eval-items", "eval-mcq"):
                ev = {**card, "cheese": sum("cheese" in q.lower() for q in ds["question"])}
                if kind == "eval-items":
                    ev["examples"] = [{"question": r["question"], "options": [r["item1"], r["item2"]], "aligned": r["liked_item"]}
                                      for r in (ds[i] for i in sorted(rng.sample(range(len(ds)), 8)))]
                else:
                    rows = ds.to_list()
                    ev["categories"] = []
                    for cat, n in sorted(Counter(r["category"] for r in rows).items()):
                        sub = [r for r in rows if r["category"] == cat]
                        ev["categories"].append({"name": cat, "n": n, "areas": Counter(r["opinion_area"] for r in sub).most_common(),
                                                 "examples": [{"question": r["question"], "aligned": r["answer"], "area": r["opinion_area"]}
                                                              for r in rng.sample(sub, 2)]})
                evalsets.append(ev)
                continue
            cards.append(card)
        return cards, evalsets

    # ---------------------------------------------------------------- your corpora (data/msm)

    def count_own(self) -> None:
        """Rows of each of your corpora, for the models' facts (built() reads them in full later)."""
        mid_dir = self.ctx.data_dir / "midtrain"
        for path in sorted(mid_dir.glob("*/dataset.jsonl")) if mid_dir.is_dir() else []:
            with open(path, encoding="utf-8") as f:
                self.sizes[f"own:{path.parent.name}"] = sum(1 for line in f if line.strip())

    def built(self, models: list[dict]) -> list[dict]:
        mid_dir = self.ctx.data_dir / "midtrain"
        gen_root = self.ctx.data_dir / "gen_synth_docs"
        spec_root = self.ctx.msm_dir / "spec"
        out = []
        if not mid_dir.is_dir():
            return out
        used: dict[str, list[str]] = defaultdict(list)
        for m in models:
            if c := m["corpus"]:
                used[c["dataset"]].append(m["id"])
            for neg in (m["spo"] or {}).get("negatives") or []:
                if neg:
                    used[neg["dataset"]].append(m["id"])
        for d in sorted(p for p in mid_dir.iterdir() if (p / "dataset.jsonl").exists()):
            name = d.name
            rows = _read_samples(d / "dataset.jsonl")
            self.sizes[f"own:{name}"] = len(rows)
            role = "positive" if name.startswith("pos_") else "negative" if name.startswith("neg_") else "other"
            card_name = re.sub(r"^(pos|neg)_", "", name)
            group = ("cross-value" if card_name.startswith("cross_") else "rejection" if card_name.startswith("rejection")
                     else "idiosyncratic" if card_name == "idiosyncratic" else "other-specs" if card_name.startswith("other_")
                     else role)
            entry: dict[str, Any] = {
                "id": name, "role": role, "group": group, "value": self.ctx.value_of(card_name),
                "path": self.ctx.rel(d / "dataset.jsonl"), "n": len(rows), "used_by": sorted(set(used.get(f"own:{name}", []))),
                "domains": Counter(r.get("domain") or "—" for r in rows).most_common(),
            }
            # The arm a corpus serves: its own value for positives and rejections; the other value for cross-value
            # negatives (the other arm's positives); both arms for the other-specs and idiosyncratic negatives.
            values = list(self.ctx.values)
            entry["arm"] = (entry["value"] if group in ("positive", "rejection") else "both" if group in ("other-specs", "idiosyncratic")
                            else None)
            if card_name.startswith("cross_"):
                src = card_name.removeprefix("cross_")
                entry["value"] = self.ctx.value_of(src)
                entry["arm"] = next((v for v in values if v != entry["value"]), None) if len(values) == 2 else None
                entry["note"] = (f"The first {len(rows):,} documents of pos_{src}, the other arm’s positive corpus, cut to their "
                                 "share of the negative pool. No spec of its own.")
            # Spec card, spec text and its audit.
            for sub in ("pos", "neg"):
                card_file = spec_root / sub / "cards" / f"{card_name}.yaml"
                if card_file.exists():
                    card = _load_yaml(card_file)
                    entry["card"] = {k: card.get(k) for k in ("name", "operator", "target", "value", "grounding") if card.get(k)}
                    entry["card"]["path"] = self.ctx.rel(card_file)
                    spec_file = spec_root / sub / f"{card_name}.txt"
                    if spec_file.exists():
                        entry["spec"] = {"path": self.ctx.rel(spec_file), "text": spec_file.read_text(encoding="utf-8")}
            report = self.ctx.data_dir / "specgen" / card_name / "report.json"
            if report.exists():
                rep = _load_json(report)
                entry["audit"] = {k: rep.get(k) for k in ("paragraphs", "drafts", "fit", "words")}
            # Generation tree and stats.
            gen = gen_root / name
            # Cross-value corpora are cut from the other arm's positives, so their ideas live under that corpus.
            ideas = _ideas(gen if gen.is_dir() else gen_root / f"pos_{card_name.removeprefix('cross_')}")
            if (gen / "summary.json").exists():
                summ = _load_json(gen / "summary.json")
                entry["stats"] = summ.get("stats", {})
                entry["gen_config"] = {k: summ.get("config", {}).get(k) for k in ("n_doc_types", "n_doc_ideas", "n_docs", "temperature")}
                entry["gen_config"]["writer"] = summ.get("config", {}).get("api", {}).get("model")
            if (gen / "meta.json").exists():
                docs_per = Counter("/".join(r.get("source", "").split("/")[:3]) for r in rows)
                tree = []
                for dom in _load_json(gen / "meta.json").get("domains", []):
                    dname = dom["domain"]
                    ddir = gen / _safe(dname)
                    subs = []
                    dmeta = _load_json(ddir / "meta.json") if (ddir / "meta.json").exists() else {}
                    for s in dmeta.get("subdomains", []):
                        sdir = ddir / _safe(s["subdomain"])
                        smeta = _load_json(sdir / "meta.json") if (sdir / "meta.json").exists() else {}
                        types = []
                        for t in smeta.get("doc_types", []):
                            tdir = sdir / _safe(t["doc_type"])
                            types.append({"name": t["doc_type"], "description": t.get("description"),
                                          "n": docs_per.get(f"{ddir.name}/{sdir.name}/{tdir.name}", 0)})
                        subs.append({"name": s["subdomain"], "context": smeta.get("subdomain_context") or s.get("subdomain_context"),
                                     "assertions": [{"name": a.get("name"), "text": a.get("assertion"), "label": a.get("label")}
                                                    for a in smeta.get("assertions", [])],
                                     "doc_types": types})
                    tree.append({"name": dname, "subdomains": subs})
                entry["tree"] = tree
            rng = random.Random(0)
            entry["examples"] = []
            for r in _domain_samples(rows, 2, rng):
                parts = (r.get("source") or "").split("/")
                idea = ideas.get(r.get("source", ""), {})
                entry["examples"].append(_doc_example(
                    r["text"], domain=r.get("domain"), subdomain=parts[1] if len(parts) > 3 else None,
                    doc_type=parts[2] if len(parts) > 3 else None, idea_name=idea.get("name") or (parts[-1][:-4] if parts[-1].endswith(".txt") else None),
                    idea=idea.get("idea"), source=r.get("source")))
            out.append(entry)
        order = {"positive": 0, "negative": 1, "other": 2}
        return sorted(out, key=lambda e: (order[e["role"]], e["group"], e["id"]))

    # ---------------------------------------------------------------- pipeline

    def pipeline(self) -> list[dict]:
        steps = self.ctx.registry.get("pipeline") or []
        prompts = self.ctx.msm_dir / "datagen" / "prompts"
        out = []
        for s in steps:
            st = {**s}
            if s.get("prompt") and (prompts / s["prompt"]).exists():
                st["template"] = (prompts / s["prompt"]).read_text(encoding="utf-8")
                st["template_path"] = self.ctx.rel(prompts / s["prompt"])
            out.append(st)
        return out


def _ideas(gen: Path) -> dict[str, dict]:
    """Doc ideas of a generated corpus by their document's source path (<domain>/<subdomain>/<doc type>/<idea>.txt)."""
    out = {}
    for meta in gen.glob("*/*/*/meta.json"):
        rel = meta.parent.relative_to(gen).as_posix()
        for idea in _load_json(meta).get("doc_ideas", []):
            out[f"{rel}/{_safe(idea['name'])}.txt"] = idea
    return out


def _safe(name: str) -> str:
    """A copy of msm.datagen.utils.sanitize_filename, which names the generator's directories and files."""
    for char in '<>:"/\\|?*':
        name = name.replace(char, "_")
    return "".join(c if c.isprintable() else "_" for c in name).strip(". ")[:200]


# --------------------------------------------------------------------------- midtraining runs (Training tab)


def _sig(x: Any, digits: int = 5) -> Any:
    """A float to `digits` significant digits (learning rates go down to 1e-9), NaN as null."""
    if isinstance(x, float):
        return float(f"{x:.{digits}g}") if math.isfinite(x) else None
    return x


def _run_dir(path: Path) -> Path:
    """An adapter directory (<run>/final or <run>/checkpoints/checkpoint-N) -> its run directory."""
    if path.name == "final":
        return path.parent
    if CHECKPOINT.match(path.name) and path.parent.name == "checkpoints":
        return path.parents[1]
    return path


def _curves(ctx: Ctx, run_dir: Path) -> dict | None:
    """The logged training metrics of a run (every logging_steps), from its last checkpoint's trainer_state.json."""
    ckpts = sorted((int(m.group(1)), d) for d in (run_dir / "checkpoints").glob("checkpoint-*") if (m := CHECKPOINT.match(d.name)))
    if not ckpts or not (ckpts[-1][1] / "trainer_state.json").exists():
        return None
    last = ckpts[-1][1]
    state = _load_json(last / "trainer_state.json")
    logs = [e for e in state.get("log_history", []) if "step" in e and ("loss" in e or "loss_ce" in e)]
    keys = sorted({k for e in logs for k, v in e.items() if isinstance(v, (int, float)) and k not in ("step", "epoch")})
    return {"checkpoint": ctx.rel(last), "step": state.get("global_step"), "max_steps": state.get("max_steps"),
            "logging_steps": state.get("logging_steps"), "steps": [e["step"] for e in logs],
            "series": {k: [_sig(e.get(k)) for e in logs] for k in keys}}


def midtraining(ctx: Ctx, models: list[dict], sizes: dict[str, int]) -> list[dict]:
    """Every MSM / SPO run under outputs/msm/train, evaluated or not: its label (a model label without the fine-tuning
    part), the evaluated models that continued it, and its training curves."""
    if not ctx.train_dir.is_dir():
        return []
    lineage = Lineage(ctx)  # its own, so these runs don't become published models
    continued: dict[str, list[str]] = defaultdict(list)
    for m in models:
        if (p := m["pretrain"]) and not p.get("hub") and p.get("run"):
            continued[p["run"]].append(m["id"])
    runs = []
    for run_dir in sorted(d for d in ctx.train_dir.iterdir() if (d / "config.yaml").exists()):
        data = _load_yaml(run_dir / "config.yaml").get("data") or {}
        if "msm" not in data or "aft" in data or "it" in data:
            continue  # a fine-tuning run
        m = lineage._trained(f"{run_dir.name}/final", lineage._run(run_dir / "final"))
        recipe, marker = _label_parts(m)
        m["label"] = recipe + marker + "".join(f" · {t}" for t in m["tags"])
        m["recipe_label"] = recipe
        runs.append(m)
    _disambiguate(runs)
    out = []
    for m in runs:
        run_dir = ctx.train_dir / m["sft"]["run"]
        out.append({
            "id": short_hash(m["sft"]["run"], 8), "run": m["sft"]["run"], "path": ctx.rel(run_dir), "label": m["label"],
            # The label without its value: runs of either arm that differ only by value share a colour.
            "variant": m["recipe"]["mid"] + m["label"][len(m["recipe_label"]):],
            "mid": m["recipe"]["mid"], "value": m["recipe"]["value"], "corpus": m["corpus"], "tags": m["tags"],
            "seed": m["seed"], "date": m["sft"]["date"], "steps": m["step"], "facts": lineage._facts(m, sizes),
            "models": continued.get(m["sft"]["run"], []), "curves": _curves(ctx, run_dir),
        })
    # Colour slots by variant, in a fixed order: your corpora before the paper's, MSM before SPO, untagged before tagged
    # (MSM, SPO, SPO · simpo get slots 1-3, as in msm/plot_scores.py).
    variants = sorted({r["variant"] for r in out}, key=lambda v: ("(paper corpus)" in v, not v.startswith("MSM"), v.count(" · "), v))
    slot = {v: i + 1 if i < 8 else 0 for i, v in enumerate(variants)}
    values = list(ctx.values)
    for r in out:
        r["slot"] = slot[r["variant"]]
    return sorted(out, key=lambda r: (values.index(r["value"]) if r["value"] in values else len(values), r["slot"], r["label"]))


# --------------------------------------------------------------------------- document likelihoods (Likelihood tab)


def _group_flags(name: str, group: dict, arm: str) -> dict:
    """A group of scored documents with its value and flags; older summaries only name them
    ("<value> · trained|held out[ · paper corpus]", the arm's own value being the positives)."""
    parts = name.split(" · ")
    value = group.get("value", parts[0])
    return {"value": value, "positive": group.get("positive", value == arm),
            "trained": group.get("trained", len(parts) > 1 and parts[1] == "trained"),
            "paper": group.get("paper", "paper corpus" in parts), **group}


def _pooled_margin(groups: dict, margin: dict, model: str, positives: str) -> float | None:
    """The share of pairs over the margin, pooled over the negative groups of your corpora, weighted by documents
    (as msm/plot_scores.py pools them; the paper's corpora are left out)."""
    pairs = [(groups[name]["n"], shares.get(model, {}).get(positives)) for name, shares in margin.items()
             if name in groups and not groups[name]["paper"]]
    pairs = [(n, s) for n, s in pairs if s is not None and math.isfinite(s)]
    return sum(n * s for n, s in pairs) / sum(n for n, _ in pairs) if pairs else None


def likelihood(ctx: Ctx, runs: list[dict]) -> dict:
    """The newest summary of msm/score_docs.py: per arm, each model's mean token log-probability on groups of documents
    before AFT, and what SPO changed next to MSM on the same documents."""
    summaries = sorted(ctx.score_dir.glob("summary_*.json")) if ctx.score_dir.is_dir() else []
    if not summaries:
        return {"available": False}
    path = summaries[-1]
    summary = _load_json(path)
    cfg = summary.get("config") or {}
    by_run = {r["run"]: r for r in runs}
    values = ctx.registry.get("values") or {}
    arms = []
    for name, arm in (summary.get("arms") or {}).items():
        groups = {k: _group_flags(k, v, name) for k, v in (arm.get("groups") or {}).items()}
        margin = arm.get("margin") or {}
        models = []
        if any("Base" in g.get("scores", {}) for g in groups.values()):
            models.append({"key": "Base", "label": "Base", "slot": 0, "run": None, "mid": None, "path": cfg.get("model")})
        for label, adapter in (arm.get("runs") or {}).items():
            local = ctx.local(adapter)
            r = by_run.get(_run_dir(local).name) if local else None
            models.append({"key": label, "label": r["label"] if r else label, "slot": r["slot"] if r else 0,
                           "run": r and r["id"], "mid": r and r["mid"], "path": ctx.rel(local) if local else adapter})
        # Older summaries don't name the reference: the arm's run without SPO.
        reference = arm.get("reference") or next((m["key"] for m in models if m["mid"] == "MSM"), None)
        own = next((g for g in groups.values() if g.get("positive") and g.get("paper")), None)
        other = next((g for g in groups.values() if not g.get("positive") and g.get("paper")), None)
        separation = []
        if own and other and "Base" in own["scores"] and "Base" in other["scores"]:
            for m in models:
                if m["key"] != "Base" and m["key"] in own["scores"] and m["key"] in other["scores"]:
                    a = own["scores"][m["key"]] - own["scores"]["Base"]
                    b = other["scores"][m["key"]] - other["scores"]["Base"]
                    separation.append({"model": m["key"], "own": a, "other": b, "gap": a - b})
        value = ctx.value_of(name)
        arms.append(_finite({
            "id": name, "value": value, "value_label": values.get(value, {}).get("label", name), "reference": reference,
            "models": models,
            "groups": [{"name": k, **v} for k, v in groups.items()],
            "margin": [{"name": k, "n": groups.get(k, {}).get("n"), "shares": v} for k, v in margin.items()],
            "pooled_margin": {m["key"]: {p: _pooled_margin(groups, margin, m["key"], p) for p in ("trained", "paper")}
                              for m in models},
            "positions": arm.get("positions") or {},
            "profile": arm.get("profile") or {},
            "tokens": arm.get("tokens") or {},
            "separation": separation,
        }))
    docs, margin_cfg = cfg.get("docs") or {}, cfg.get("margin") or {}
    return {
        "available": True, "source": ctx.rel(path), "date": run_timestamp(path.stem), "earlier": len(summaries) - 1,
        "config": {"model": cfg.get("model"), "dtype": cfg.get("dtype"), "max_seq_len": cfg.get("max_seq_len"),
                   "per_set": docs.get("per_set"), "beta": margin_cfg.get("beta"), "gamma": margin_cfg.get("gamma"),
                   "bootstrap": cfg.get("bootstrap"), "min_token_count": cfg.get("min_token_count"),
                   "top_tokens": cfg.get("top_tokens"), "positions": cfg.get("positions")},
        "arms": arms,
    }


# --------------------------------------------------------------------------- collect


def collect(spec: dict, base_dir: Path, registry: dict, log=print) -> dict:
    ctx = Ctx(spec, base_dir, registry, log)
    lineage = Lineage(ctx)
    reg_evals = registry.get("evals", {})
    categories = dict(registry.get("categories", {}) or {})
    ignore = [re.compile(p) for p in registry.get("protocol_ignore", [])]

    protocols: dict[str, dict] = {}
    evals: list[dict] = []
    rows: list[list] = []
    sample_rows: list[list] = []
    answers_raw: dict[str, dict[str, dict]] = defaultdict(dict)  # level -> model key -> samples by row
    questions: dict[str, dict[int, dict]] = defaultdict(dict)  # level -> row -> question
    metric_files = sorted(ctx.evals_dir.glob("*/*/metrics.json"))
    log(f"[msm] {len(metric_files)} eval results under {ctx.evals_dir}")

    for mpath in metric_files:
        edir = mpath.parent
        metrics = _load_json(mpath)
        meta = _load_json(edir / "meta.json") if (edir / "meta.json").exists() else {}
        adapter = meta.get("adapter") or metrics.get("adapter")
        if not adapter:
            log(f"[msm] {edir}: no adapter in meta.json; skipped")
            continue
        model = lineage.resolve(adapter)
        level = reg_evals.get(edir.name, {}).get("id", edir.name.upper())
        proto_cfg = {k: v for k, v in meta.items() if k not in NON_PROTOCOL}
        proto_cfg.update({f"judge.{k}": v for k, v in (metrics.get("judge") or {}).items()})
        proto_cfg = {k: v for k, v in proto_cfg.items() if not any(p.search(k) for p in ignore)}
        proto_id = short_hash(proto_cfg, 6)
        protocols.setdefault(proto_id, {"id": proto_id, "judge": (metrics.get("judge") or {}).get("model"), "config": ctx.san.value(proto_cfg)})

        eval_id = short_hash(f"{edir.parent.name}/{edir.name}", 8)
        evals.append({
            "id": eval_id, "run_id": f"{edir.parent.name}/{edir.name}", "label": f"{edir.parent.name} · {edir.name}",
            "date": _mtime(mpath), "model": model["key"], "split": SPLIT, "protocol": proto_id,
            "prompts": [PROMPT], "levels": [level], "n_questions": metrics.get("n_questions"),
            "n_answers": metrics.get("n_answers"), "n_judged": metrics.get("n_judged"), "shards": 0,
        })

        def emit(topic, metric, value, k, n, _l=level, _e=eval_id, _m=model["key"], _p=proto_id):
            if value is not None:
                rows.append([_e, _m, SPLIT, _p, PROMPT, _l, topic, metric, _round(value), k, n, 0])
        _metric_rows(metrics, emit, None)
        for key in [k for k in metrics if k.startswith("by_")]:
            for cat, cm in metrics[key].items():
                if cat not in categories:
                    categories[cat] = f"C{len(categories) + 1}"
                _metric_rows(cm, emit, categories[cat])

        samples = _read_samples(edir / "samples.jsonl")
        per: dict[int, list[int]] = {}
        by_row: dict[int, list[dict]] = defaultdict(list)
        for s in samples:
            if s.get("outcome") in OUTCOMES:
                acc = per.setdefault(s["sample"], [0, 0, 0])
                acc[OUTCOMES.index(s["outcome"])] += 1
            by_row[s["row"]].append(s)
            if s["row"] not in questions[level]:
                questions[level][s["row"]] = {"question": s["question"], "options": {"A": s.get("option_a"), "B": s.get("option_b")},
                                              "aligned": s.get("aligned_option"), "category": s.get("category")}
        for i in sorted(per):
            sample_rows.append([eval_id, PROMPT, level, i, *per[i]])
        for row in by_row.values():
            row.sort(key=lambda s: s["sample"])
        answers_raw[level][model["key"]] = by_row

    datasets = Datasets(ctx)
    paper_cards, evalsets = datasets.paper()
    datasets.count_own()
    all_models = lineage.finalize(datasets.sizes)
    hidden = {m["key"] for m in all_models if m["hidden"]}
    id_of = {m["key"]: m["id"] for m in all_models}
    models = [m for m in all_models if not m["hidden"]]
    evals = [e for e in evals if e["model"] not in hidden]
    rows = [r for r in rows if r[1] not in hidden]
    for e in evals:
        e["model"] = id_of[e["model"]]
    for r in rows:
        r[1] = id_of[r[1]]
    keep = {e["id"] for e in evals}
    sample_rows = [r for r in sample_rows if r[0] in keep]

    # Newest run wins per model × protocol × level (an adapter evaluated again under another out_dir).
    latest: dict[tuple, str] = {}
    for e in sorted(evals, key=lambda e: e["date"] or ""):
        latest[(e["model"], e["protocol"], e["levels"][0])] = e["id"]
    for r in rows:
        r[-1] = int(latest.get((r[1], r[3], r[5])) == r[0])
    for e in evals:
        e["superseded_for"] = [] if latest[(e["model"], e["protocol"], e["levels"][0])] == e["id"] else [PROMPT]
        e["label_shared"] = False

    # ---------------------------------------------------------------- manifest pieces
    used_levels = {r[5] for r in rows}
    levels = []
    for name, ev in reg_evals.items():
        if ev.get("id", name.upper()) not in used_levels:
            continue
        lv = {"id": ev.get("id", name.upper()), "label": ev.get("label", name), "eval": name}
        es = next((x for x in evalsets if x.get("eval") == name), None)
        rows_tip = [["Dataset", ev.get("data", "—")], ["Questions", f"{es['n']:,}" if es else "—"],
                    ["Aligned option", ev.get("aligned", "—")]]
        lv["rows"] = rows_tip
        lv["detail"] = ev.get("help")
        lv["template"] = ev.get("template")
        if es and es.get("examples"):
            x = es["examples"][0]
            lv["example"] = f"{x['question']}\n→ aligned: {x['aligned']}"
        elif es and es.get("categories"):
            x = es["categories"][0]["examples"][0]
            lv["example"] = f"{x['question']}\n→ aligned: {x['aligned']}"
        levels.append(lv)
    for lid in sorted(used_levels - {lv["id"] for lv in levels}):
        levels.append({"id": lid, "label": lid})

    topics = {}
    am = next((x for x in evalsets if x.get("kind") == "eval-mcq"), None)
    for cat, cid in categories.items():
        if not any(r[6] == cid for r in rows):
            continue
        t = {"topic": cat}
        info = next((c for c in (am or {}).get("categories", []) if c["name"] == cat), None)
        if info:
            t["rows"] = [["Questions", str(info["n"])]] + [[a, str(n)] for a, n in info["areas"][:6]]
            t["detail"] = "Opinion areas and their question counts."
            t["example"] = f"{info['examples'][0]['question']}\n→ aligned: {info['examples'][0]['aligned']}"
        topics[cid] = t
    split = {"id": SPLIT, "label": registry.get("split_label", "Held-out evals"), "short": "Held-out", "topics": list(topics)}

    protos = sorted(protocols.values(), key=lambda p: -sum(e["protocol"] == p["id"] for e in evals))
    for i, p in enumerate(protos):
        c = p["config"]
        p["label"] = (f"Protocol {chr(65 + i)} · {c.get('n_samples', '?')} samples, T={c.get('temperature', '?')} · judge {p['judge']}")
        p["evals"] = sum(e["protocol"] == p["id"] for e in evals)

    recipes = registry.get("recipes", {})
    used_methods = list(dict.fromkeys(m["method"] for m in models))
    labels = {m["method"]: m["recipe_label"] for m in models}
    method_list = [{"id": k, "label": labels[k], **v} for k, v in recipes.items() if k in used_methods]
    method_list += [{"id": k, "label": labels[k], "color": 0} for k in used_methods if k not in recipes]

    sets = []
    for st in registry.get("sets", []) or []:
        members = [m for m in models if _matches(m, st.get("match", {}))] + [m for m in models if m["id"] in st.get("also", [])]
        members = list({m["id"]: m for m in members}.values())
        if st.get("show_if") and not any(_matches(m, st["show_if"]) for m in members):
            continue
        if not members:
            continue
        ids = [m["id"] for m in models if m in members]
        rules = st.get("baseline") or []
        rules = rules if isinstance(rules, list) else [rules]  # fallbacks, first match wins
        base = next((m["id"] for rule in rules for m in models if m["id"] in ids and _matches(m, rule)), None)
        sets.append({"id": st["id"], "label": st["label"], "summary": st.get("summary", ""), "models": ids,
                     "baseline": base, "plot": ids[:8]})
    for m in models:
        m["sets"] = [st["id"] for st in sets if m["id"] in st["models"]]

    flags = [{"kind": "unresolved-sft", "model": m["id"]} for m in models if not m["sft"]["resolved"]]
    flags += [{"kind": "unresolved-pretrain", "model": m["id"]} for m in models if m["pretrain"] and not m["pretrain"]["resolved"]]

    # ---------------------------------------------------------------- example answers
    n_spread, n_random = registry.get("answers", {}).get("spread", 3), registry.get("answers", {}).get("random", 3)
    answers: dict[str, Any] = {"levels": {}}
    for level, by_model in answers_raw.items():
        by_model = {k: v for k, v in by_model.items() if k not in hidden}
        outcomes = {k: {row: [s.get("outcome") for s in ss] for row, ss in v.items()} for k, v in by_model.items()}
        picked = _pick_rows(outcomes, n_spread, n_random)
        qs = []
        for row, why in picked:
            q = questions[level][row]
            qs.append({"row": row, "why": why, **q, "category": categories.get(q["category"]) if q.get("category") else None})
        per_model = {}
        for key, v in by_model.items():
            per_model[id_of[key]] = {}
            for row, _ in picked:
                ss = v.get(row) or []
                if not ss:
                    continue
                first = ss[0]
                text, cut = _cut(first.get("response", ""), ANSWER_CHARS)
                per_model[id_of[key]][str(row)] = {"outcomes": [s.get("outcome") for s in ss], "response": text, "cut": cut,
                                                   "truncated": first.get("finish") == "length", "verdict": first.get("outcome")}
        answers["levels"][level] = {"questions": qs, "models": per_model}

    # ---------------------------------------------------------------- datasets.json
    data_page = {
        "pipeline": datasets.pipeline(),
        "flow": registry.get("flow", []),
        "paper": paper_cards,
        "evalsets": evalsets,
        "built": datasets.built(models),
    }
    for card in data_page["paper"]:
        def uses(m, card=card):
            ft = m.get("ft_data") or {}
            if card["kind"] == "corpus":
                return (m["corpus"] or {}).get("dataset") == f"hf:{card['hf']}"
            if card["kind"] == "chats":
                return ft.get("aft") == card["hf"]
            return any(s.startswith(card["hf"]) for s in ft.get("it") or [])
        card["used_by"] = [m["id"] for m in models if uses(m)]

    # ---------------------------------------------------------------- training.json, likelihood.json
    runs = midtraining(ctx, models, datasets.sizes)
    training = {
        "metrics": registry.get("training_metrics", []),
        "values": {k: v.get("label", k) for k, v in ctx.values.items()},
        "runs": runs,
        # The AFT-only model trained here: what the midtrained models' evals are read against.
        "control": next((m["id"] for m in models if m["method"] == "aft" and m["origin"] == "here"), None),
    }
    scores = likelihood(ctx, runs)

    for m in models:
        m.pop("recipe_label", None)
        m.pop("ft_data", None)
    manifest = {
        "sets": sets,
        "levels": levels,
        "level_mean": False,
        "splits": [split],
        "prompts": [{"id": PROMPT, "label": PROMPT, "template": registry.get("prompt_note", "")}],
        "prompt_families": [],
        "protocols": protos,
        "measures": registry.get("measures", []),
        "methods": method_list,
        "topics": topics,
        "presets": registry.get("presets", []),
        "facets": registry.get("facets", []),
        "terms": registry.get("terms", {}),
        "notes": registry.get("notes", {}),
        "glossary": registry.get("glossary", []),
        "tip_measures": registry.get("tip_measures", []),
        "has_answers": bool(answers["levels"]),
        "has_datasets": True,
        "has_training": any(r["curves"] for r in runs),
        "has_likelihood": scores["available"],
        "flags": flags,
        "counts": {"models": len(models), "evals": len(evals), "rows": len(rows)},
        "baseline": None,
        "plots": {st["id"]: st["plot"] for st in sets},
    }
    return {
        "_all_models": all_models,  # for report(); not published
        "manifest": manifest,
        "models": [_public(m) for m in models],
        "evals": sorted(evals, key=lambda e: e["date"] or "", reverse=True),
        "results": {"columns": RESULT_COLUMNS, "rows": rows},
        "samples": {"columns": SAMPLE_COLUMNS, "rows": sample_rows},
        "answers": answers,
        "datasets": data_page,
        "training": training,
        "likelihood": scores,
    }


def _matches(m: dict, rule: dict) -> bool:
    """Set rules: {facet: [values]} over a model's facets, plus `recipe` (method ids)."""
    for k, want in rule.items():
        want = want if isinstance(want, list) else [want]
        have = m["method"] if k == "recipe" else m["facets"].get(k)
        if have not in want:
            return False
    return True


def _public(m: dict) -> dict:
    return {k: v for k, v in m.items() if k not in ("hidden", "auto_label")}


def report(data: dict) -> str:
    """Published name -> fine-tuning run -> midtraining run -> eval runs; then hidden models."""
    evals_by_model: dict[str, list[dict]] = defaultdict(list)
    for e in data["evals"]:
        evals_by_model[e["model"]].append(e)
    sets = data["manifest"].get("sets", [])
    lines = []
    for m in data["models"]:
        pre = m["pretrain"] or {}
        member = ", ".join(st["label"] for st in sets if m["id"] in st["models"])
        lines.append(f"{m['label']}  [{m['id']}]  {m['method']}  sets: {member or '—'}")
        lines.append(f"    weights   {m['sft']['path']}{'' if m['sft']['resolved'] else '  (folder missing)'}")
        if pre:
            lines.append(f"    {pre.get('label') or 'init':9} {pre.get('path') or pre.get('note') or 'unknown'}")
        if m.get("corpus"):
            lines.append(f"    corpus    {m['corpus']['source']}")
        for e in sorted(evals_by_model[m["id"]], key=lambda e: e["date"] or ""):
            lines.append(f"    eval      {(e['date'] or '?').replace('T', ' ')[:16]}  {e['levels'][0]:4} {e['run_id']}")
    runs = (data.get("training") or {}).get("runs") or []
    if runs:
        lines += ["", "MIDTRAINING RUNS (Training tab):"]
        for r in runs:
            c = r["curves"] or {}
            done = f"{c.get('step')}/{c.get('max_steps')} steps logged" if c else "no trainer_state.json"
            lines.append(f"  {r['label']:32} {r['path']}  [{done}]  → {', '.join(r['models']) or 'not fine-tuned'}")
    scores = data.get("likelihood") or {}
    if scores.get("available"):
        arms = "; ".join(f"{a['id']}: {', '.join(m['label'] for m in a['models'])}" for a in scores["arms"])
        lines += ["", f"LIKELIHOOD (Likelihood tab): {scores['source']}  ({arms})"]
    hidden = [m for m in data.get("_all_models", []) if m.get("hidden")]
    if hidden:
        lines += ["", "HIDDEN (registry `hidden: true`):"] + [f"  {m['auto_label']:48} {m['key']}" for m in hidden]
    lines += ["", "To rename or describe a model, add to registry/msm.yaml `models`: {adapter: <HF id or run name>, id: ..., label: ...}"]
    return "\n".join(lines)
