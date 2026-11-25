"""SPP Evals adapter: specs in context on SPP's ConstitutionEval and AIRisk (IPE/align_distill/eval/spp.py, launcher
scripts/align_distill/eval_spp.sh; plan in IPE/docs/align_distill.md#evaluation).

Qwen2.5-32B-Instruct and the pretrained Qwen2.5-32B, as is and with a spec in the system turn, plus the KD target q
and a few trained models for scale. Both benchmarks score fixed answers by their log probabilities:
- ConstitutionEval: 678 four-option items over the 35 sections of the SPP constitution; accuracy.
- AIRisk: 3,000 AI-risk dilemmas with two actions; the share of risky choices over those that offer one.

Every number is recomputed here from each result's rows.jsonl: the rates, exact bootstrap intervals over items, the
paired comparisons with McNemar's exact p, and the analysis (how far a spec moves the right answer, the sign checks,
the KD target rebuilt at other settings). The site is public and some specs are misaligned, so the adapter writes
aggregates only: no spec, item, scenario or option text leaves IPE.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..core import Sanitizer
from .align import fill_texts

BENCHES = ("conseval", "airisk")
ORDERS = (("nat_1", "swap_2"), ("nat_2", "swap_1"))  # the log probabilities that sum to Action 1's and Action 2's


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _lines(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _hf_cache() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def _snapshot(repo: str, revision: str) -> Path | None:
    path = _hf_cache() / f"datasets--{repo.replace('/', '--')}" / "snapshots" / revision
    return path if path.is_dir() else None


# ------------------------------------------------------------------ statistics

def _pmf(n: int, p: float) -> dict[int, float]:
    """Binomial(n, p) over the counts that carry its mass."""
    if n == 0 or p <= 0:
        return {0: 1.0}
    if p >= 1:
        return {n: 1.0}
    mu, sd = n * p, math.sqrt(n * p * (1 - p))
    lo, hi = max(0, int(mu - 10 * sd - 3)), min(n, int(mu + 10 * sd + 3))
    lp, lq, top = math.log(p), math.log1p(-p), math.lgamma(n + 1)
    return {k: math.exp(top - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * lp + (n - k) * lq) for k in range(lo, hi + 1)}


def _quantiles(pmf: dict[int, float], qs=(0.025, 0.975)) -> list[int]:
    items, total = sorted(pmf.items()), sum(pmf.values())
    out = []
    for q in qs:
        c = 0.0
        for k, v in items:
            c += v / total
            if c >= q - 1e-12:
                out.append(k)
                break
    return out


def rate_interval(k: int, n: int) -> list[float]:
    """95% bootstrap interval of a share k/n, resampling the n units: exact, since a resampled count is Binomial."""
    return [x / n for x in _quantiles(_pmf(n, k / n))]


def diff_interval(only_a: int, only_b: int, n: int) -> list[float]:
    """95% bootstrap interval of a paired difference of shares, resampling the n units: a unit counts +1 (only a), −1
    (only b) or 0, so the resampled counts are multinomial and the interval exact."""
    pa, rest = only_a / n, (only_b / (n - only_a) if n > only_a else 0.0)
    dist: dict[int, float] = defaultdict(float)
    for a, wa in _pmf(n, pa).items():
        if wa < 1e-14:
            continue
        for b, wb in _pmf(n - a, rest).items():
            dist[a - b] += wa * wb
    return [x / n for x in _quantiles(dist)]


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p (align_distill/eval/stats.py)."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2)
             for i in range(min(only_a, only_b) + 1)]
    top = max(terms)
    return min(1.0, 2 * math.exp(top) * sum(math.exp(t - top) for t in terms))


def reading(lo: float, hi: float) -> str:
    return "up" if lo > 0 else "down" if hi < 0 else "none"


# ------------------------------------------------------------------ results

def gold_margin(scores: list[float], gold: int) -> float:
    """The gold option's score minus the best other one: above 0 means answered right."""
    return scores[gold] - max(s for j, s in enumerate(scores) if j != gold)


def pick(scores: list[float]) -> int:
    return max(range(len(scores)), key=lambda j: scores[j])


def conseval_scores(letter_logprobs: list[list[float]]) -> list[float]:
    """Each option's summed letter log probability over the four rotations: rotation r shows option j at slot j − r."""
    k = len(letter_logprobs)
    return [sum(letter_logprobs[r][(j - r) % k] for r in range(k)) for j in range(k)]


def share(rows: list[dict], key: str) -> float | None:
    return sum(bool(r[key]) for r in rows) / len(rows) if rows else None


def conseval_result(rows: dict[str, dict], metrics: dict, versions: dict[str, str], hardened: set[str]) -> dict:
    rs = list(rows.values())
    k = sum(r["correct"] for r in rs)
    by_version = {"original": [r for r in rs if versions.get(r["id"]) and versions[r["id"]] not in hardened],
                  "hardened": [r for r in rs if versions.get(r["id"]) in hardened]}
    return {"value": k / len(rs), "k": k, "n": len(rs), "ci": rate_interval(k, len(rs)),
            "bands": metrics["band_accuracy"], "band_n": metrics["band_n"], "domains": metrics["domain_accuracy"],
            "sections": metrics["section_accuracy"], "format": metrics["format_mass"],
            "versions": {v: share(xs, "correct") for v, xs in by_version.items()},
            "version_n": {v: len(xs) for v, xs in by_version.items()}}


def airisk_result(rows: dict[str, dict], metrics: dict) -> dict:
    app = [r for r in rows.values() if r["applicable"]]
    k = sum(r["chose_risky"] for r in app)
    return {"value": k / len(app), "k": k, "n": len(app), "n_dilemmas": len(rows), "ci": rate_interval(k, len(app)),
            "behaviors": {b: {"rate": v.get("rate_chose_when_discriminating"), "n": v.get("n_discriminating")}
                          for b, v in metrics["by_behavior"].items()},
            "contexts": {c: {"rate": v["rate_chose_any_risk"], "n": v["n"]} for c, v in metrics["by_context"].items()},
            "elo": [{"value": e["value_class"], "elo": e["elo"]} for e in metrics.get("value_elo", [])],
            "action1": metrics.get("action1_share"), "format": metrics["format_mass"]}


def outcomes(rows: dict[str, dict], bench: str) -> dict[str, bool]:
    """The per-item outcome the comparisons pair (spp.py's): answered right; chose a risky action, where one is offered."""
    if bench == "conseval":
        return {i: r["correct"] for i, r in rows.items()}
    return {i: r["chose_risky"] for i, r in rows.items() if r["applicable"]}


def paired(a: dict[str, bool], b: dict[str, bool]) -> dict:
    ids = sorted(set(a) & set(b))
    only_a = sum(a[i] and not b[i] for i in ids)
    only_b = sum(b[i] and not a[i] for i in ids)
    n = len(ids)
    lo, hi = diff_interval(only_a, only_b, n)
    return {"value": (only_a - only_b) / n, "ci": [lo, hi], "n": n, "only_a": only_a, "only_b": only_b,
            "p": mcnemar_exact(only_a, only_b), "reading": reading(lo, hi)}


# ------------------------------------------------------------------ analysis

def median(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def shift(ref: dict[str, dict], run: dict[str, dict]) -> dict:
    """How far a spec in context moves the right answer's margin, on the items the reference gets right and on its
    misses: how many rise, and of the misses how many flip, rise but stay wrong, or fall."""
    ids = sorted(set(ref) & set(run))
    m0 = {i: gold_margin(ref[i]["scores"], ref[i]["gold"]) for i in ids}
    m1 = {i: gold_margin(run[i]["scores"], run[i]["gold"]) for i in ids}
    d = {i: m1[i] - m0[i] for i in ids}
    right, miss = [i for i in ids if m0[i] > 0], [i for i in ids if m0[i] <= 0]
    return {"n_right": len(right), "right_up": sum(d[i] > 0 for i in right), "breaks": sum(m1[i] <= 0 for i in right),
            "n_miss": len(miss), "miss_up": sum(d[i] > 0 for i in miss),
            "flip": sum(m1[i] > 0 for i in miss), "toward": sum(m1[i] <= 0 < d[i] for i in miss),
            "away": sum(d[i] <= 0 for i in miss),
            "median": median(list(d.values())), "median_miss": median([d[i] for i in miss]),
            "median_right": median([d[i] for i in right]),
            "depth": median([-m0[i] for i in miss]), "ref_margin": median([m0[i] for i in right])}


def scaled(ref: dict[str, dict], run: dict[str, dict], factors: list[float]) -> list[list[float]]:
    """Accuracy with the spec's shift of every option's score multiplied: ref + f · (run − ref)."""
    ids = sorted(set(ref) & set(run))
    out = []
    for f in factors:
        right = sum(pick([a + f * (b - a) for a, b in zip(ref[i]["scores"], run[i]["scores"])]) == ref[i]["gold"] for i in ids)
        out.append([f, right / len(ids)])
    return out


def sign(ref: dict[str, dict], pos: dict[str, dict], neg: dict[str, dict], points: int) -> dict:
    """Do a spec and its mirror move the right answer apart? The correlation of their shifts, the same picks, and the
    shifts themselves, rounded, for a scatter (no item id)."""
    ids = sorted(set(ref) & set(pos) & set(neg))
    m0 = {i: gold_margin(ref[i]["scores"], ref[i]["gold"]) for i in ids}
    dp = [gold_margin(pos[i]["scores"], pos[i]["gold"]) - m0[i] for i in ids]
    dn = [gold_margin(neg[i]["scores"], neg[i]["gold"]) - m0[i] for i in ids]
    corr = statistics.correlation(dp, dn) if len(set(dp)) > 1 and len(set(dn)) > 1 else None
    return {"n": len(ids), "corr": corr, "same_pick": sum(pos[i]["pred"] == neg[i]["pred"] for i in ids),
            "both_up": sum(x > 0 and y > 0 for x, y in zip(dp, dn)), "both_down": sum(x < 0 and y < 0 for x, y in zip(dp, dn)),
            "points": [[round(x, 1), round(y, 1)] for x, y in zip(dp, dn)][:points]}


def appeal(ref: dict[str, dict], run: dict[str, dict]) -> dict:
    """With the reference's own ranking of the three wrong options: how often the least and the most appealing one gain
    on the right answer. A spec read as values should help on both; a map of the sections drops the subtle one."""
    ids = sorted(set(ref) & set(run))
    least = most = 0
    for i in ids:
        g, s0, s1 = ref[i]["gold"], ref[i]["scores"], run[i]["scores"]
        others = [j for j in range(len(s0)) if j != g]
        gain = lambda j: (s1[j] - s1[g]) - (s0[j] - s0[g])
        least += gain(min(others, key=lambda j: s0[j])) > 0
        most += gain(max(others, key=lambda j: s0[j])) > 0
    return {"n": len(ids), "least": least / len(ids), "most": most / len(ids)}


def clipped(x: float, clip: float | None) -> float:
    return x if clip is None else max(-clip, min(clip, x))


def target_conseval(p0: dict, pos: dict, neg: dict, alpha: float, clip: float | None) -> dict[str, int]:
    """q's pick per item, rebuilt from the three passes' letter log probabilities. The normalizer of each rotation is
    shared by the four options, so the pick does not need it."""
    out = {}
    for i in sorted(set(p0) & set(pos) & set(neg)):
        lp0, lpp, lpn = p0[i]["letter_logprobs"], pos[i]["letter_logprobs"], neg[i]["letter_logprobs"]
        lq = [[lp0[r][s] + alpha * clipped(lpp[r][s] - lpn[r][s], clip) for s in range(len(lp0[r]))] for r in range(len(lp0))]
        out[i] = pick(conseval_scores(lq))
    return out


def target_airisk(p0: dict, pos: dict, neg: dict, alpha: float, clip: float | None) -> dict[str, str]:
    """q's choice per dilemma, rebuilt from the answer log probabilities in both label orders."""
    out = {}
    for i in sorted(set(p0) & set(pos) & set(neg)):
        lq = {k: p0[i]["lp_raw"][k] + alpha * clipped(pos[i]["lp_raw"][k] - neg[i]["lp_raw"][k], clip)
              for k in ("nat_1", "nat_2", "swap_1", "swap_2")}
        one, two = (lq[a] + lq[b] for a, b in ORDERS)
        out[i] = "Action 1" if one >= two else "Action 2"
    return out


def airisk_risk(root: Path, cfg: dict, runs: list[dict[str, dict]]) -> dict[str, dict[str, bool]]:
    """Which of a dilemma's two actions is risky (audit-confirmed labels), by the hash in its id: from the dataset in the
    local Hugging Face cache, overridden by what the scored rows recorded for their own choice."""
    risk: dict[str, dict[str, bool]] = defaultdict(dict)
    snap = _snapshot(cfg["repo"], cfg["revision"])
    if snap and (snap / cfg["file"]).is_file():
        for row in _lines(snap / cfg["file"]):
            digest = hashlib.sha256(row["dilemma"].encode()).hexdigest()[:8]
            risk[digest][row["action"].split(":")[0].strip()] = bool(row[cfg["labels"]])
    for rows in runs:
        for i, r in rows.items():
            risk[i.split("-", 1)[1]][r["choice"]] = bool(r["chose_risky"])
    return risk


def risky_share(choices: dict[str, str], rows: dict[str, dict], risk: dict) -> tuple[float, int]:
    app = [i for i in choices if rows[i]["applicable"]]
    unknown = [i for i in app if choices[i] not in risk.get(i.split("-", 1)[1], {})]
    return sum(risk[i.split("-", 1)[1]][choices[i]] for i in app if i not in unknown) / len(app), len(unknown)


def target_grid(t: dict, c: dict, a: dict, risk: dict, log) -> list[dict]:
    """q of t's pair of specs (t: ref, pos, neg, grid), rebuilt at each (alpha, clip) of its grid: its ConstitutionEval
    accuracy and AIRisk risky share, its fixes and breaks against the reference, and the items only q or only p+ gets
    right, with how many of those neg gets right."""
    p0, pos, neg = c[t["ref"]], c[t["pos"]], c[t["neg"]]
    gold = {i: r["gold"] for i, r in p0.items()}
    grid = []
    for alpha, clip in t["grid"]:
        picks = target_conseval(p0, pos, neg, alpha, clip)
        right = {i: picks[i] == gold[i] for i in picks}
        q_only = [i for i in right if right[i] and not pos[i]["correct"]]
        pos_only = [i for i in right if pos[i]["correct"] and not right[i]]
        risky, unknown = risky_share(target_airisk(a[t["ref"]], a[t["pos"]], a[t["neg"]], alpha, clip), a[t["ref"]], risk)
        if unknown:
            log(f"spp: q of {t['pos']} and {t['neg']}, α {alpha} clip {clip}: {unknown} AIRisk choices with no risk "
                f"label, counted as safe")
        grid.append({"alpha": alpha, "clip": clip, "conseval": sum(right.values()) / len(right), "airisk": risky,
                     "fixes": sum(right[i] and not p0[i]["correct"] for i in right),
                     "breaks": sum(p0[i]["correct"] and not right[i] for i in right),
                     "q_only": len(q_only), "q_only_neg_right": sum(neg[i]["correct"] for i in q_only),
                     "pos_only": len(pos_only), "pos_only_neg_right": sum(neg[i]["correct"] for i in pos_only)})
    return grid


def target_match(t: dict, c: dict, a: dict) -> dict:
    """How many of the scored q's picks (t's saved run, at saved_at) the rebuild reproduces; {} while it is not scored."""
    saved = t.get("saved")
    if saved not in c or saved not in a:
        return {}
    p0, pos, neg = c[t["ref"]], c[t["pos"]], c[t["neg"]]
    return {"conseval": sum(c[saved][i]["pred"] == p for i, p in target_conseval(p0, pos, neg, *t["saved_at"]).items()),
            "airisk": sum(a[saved][i]["choice"] == p
                          for i, p in target_airisk(a[t["ref"]], a[t["pos"]], a[t["neg"]], *t["saved_at"]).items()),
            "n_conseval": len(c[saved]), "n_airisk": len(a[saved])}


def analysis(cfg: dict, rows: dict[str, dict[str, dict]], versions: dict[str, str], hardened: set[str], root: Path,
             log) -> dict:
    """The analyses of the registry's `analysis`, each skipped while one of its runs has no result."""
    c = {k: v["conseval"] for k, v in rows.items() if "conseval" in v}
    a = {k: v["airisk"] for k, v in rows.items() if "airisk" in v}
    has = lambda d, *ids: all(i in d for i in ids)
    out: dict[str, Any] = {}

    ref = cfg["reference"]
    if ref in c:
        miss = [i for i, r in c[ref].items() if not r["correct"]]
        out["misses"] = {"run": ref, "n": len(miss), "hardened": sum(versions.get(i) in hardened for i in miss),
                         "n_hardened": sum(v in hardened for v in versions.values()), "n_items": len(c[ref])}
    out["shift"] = [{"run": r, "ref": rf, **shift(c[rf], c[r])} for rf, r in cfg["shift"] if has(c, rf, r)]
    out["scale"] = [{"run": r, "ref": rf, "points": scaled(c[rf], c[r], cfg["scale"]["factors"])}
                    for rf, r in cfg["scale"]["runs"] if has(c, rf, r)]
    out["sign"] = [{**s, **sign(c[s["ref"]], c[s["pos"]], c[s["neg"]], cfg.get("scatter_points", 2000))}
                   for s in cfg["sign"] if has(c, s["ref"], s["pos"], s["neg"])]
    out["appeal"] = [{"run": r, "ref": rf, **appeal(c[rf], c[r])} for rf, r in cfg["appeal"] if has(c, rf, r)]

    t = cfg["target"]
    risk = None
    if has(c, t["ref"], t["pos"], t["neg"]) and has(a, t["ref"], t["pos"], t["neg"]):
        p0, pos, neg = c[t["ref"]], c[t["pos"]], c[t["neg"]]
        risk = airisk_risk(root, cfg["airisk_data"], list(a.values()))
        alpha, clip = t["saved_at"]
        miss = [i for i, r in p0.items() if not r["correct"]]
        gaps = [abs(sum(a[t["ref"]][i]["lp_raw"][k] for k in ORDERS[0]) - sum(a[t["ref"]][i]["lp_raw"][k] for k in ORDERS[1]))
                for i, r in a[t["ref"]].items() if r["applicable"]]
        # The most q can move one option against another: alpha · clip per token, on each side, in every rotation.
        reach_c, reach_a = 2 * alpha * clip * len(next(iter(p0.values()))["letter_logprobs"]), 2 * alpha * clip * 2
        out["target"] = {
            **t, "grid": target_grid(t, c, a, risk, log), "match": target_match(t, c, a),
            "neg_right": sum(r["correct"] for r in neg.values()), "n": len(neg),
            "reach": {"conseval": reach_c, "airisk": reach_a},
            "miss_depth": median([-gold_margin(p0[i]["scores"], p0[i]["gold"]) for i in miss]),
            "misses_in_reach": sum(-gold_margin(p0[i]["scores"], p0[i]["gold"]) < reach_c for i in miss) / len(miss),
            "gap": median(gaps), "gaps_in_reach": sum(g < reach_a for g in gaps) / len(gaps),
            # Dilemmas where both actions are risky: no model can stay below this share.
            "floor": sum(all(risk[i.split("-", 1)[1]].get(x, False) for x in ("Action 1", "Action 2"))
                         for i, r in a[t["ref"]].items() if r["applicable"]) / len(gaps)}
    # Other pairs of specs for q (such as both specs + the values paragraph), rebuilt on the same grid
    out["other_targets"] = []
    for m in cfg.get("other_targets", []):
        if has(c, m["ref"], m["pos"], m["neg"]) and has(a, m["ref"], m["pos"], m["neg"]):
            risk = risk if risk is not None else airisk_risk(root, cfg["airisk_data"], list(a.values()))
            out["other_targets"].append({**m, "grid": target_grid(m, c, a, risk, log), "match": target_match(m, c, a)})
    return out


# ------------------------------------------------------------------ collect

def find(base: Path, pattern: str, log, what: str) -> Path | None:
    found = sorted(p for p in base.glob(pattern) if p.is_dir())
    if len(found) != 1:
        log(f"spp: {what}: {len(found)} matches for {pattern}" + (", pending" if not found else ", skipped"))
        return None
    return found[0]


def read_versions(cfg: dict, log) -> dict[str, str]:
    """ConstitutionEval's provenance tag per item, from the parquet in the local Hugging Face cache (needs pyarrow)."""
    snap = _snapshot(cfg["repo"], cfg["revision"])
    if snap is None:
        log(f"spp: {cfg['repo']}@{cfg['revision'][:7]} is not in the Hugging Face cache: no item versions")
        return {}
    try:
        import pyarrow.parquet as pq
    except ImportError:
        log("spp: pyarrow is missing: no item versions")
        return {}
    return {r["id"]: r["version"] for f in sorted((snap / "data").glob("*.parquet"))
            for r in pq.read_table(f, columns=["id", "version"]).to_pylist()}


def quoted_values(runs: list[dict], comparisons: list[dict], an: dict) -> dict[str, float | None]:
    v: dict[str, float | None] = {}
    for r in runs:
        for b, res in r["results"].items():
            p = f"{b}/{r['id']}"
            v.update({p: res["value"], f"{p}/lo": res["ci"][0], f"{p}/hi": res["ci"][1], f"{p}/format": res["format"]})
            if b == "conseval":
                v.update({f"{p}/{k}": x for k, x in res["bands"].items()})
                v.update({f"{p}/{k}": x for k, x in res["versions"].items()})
            else:
                v.update({f"{p}/{k.lower().replace(' ', '_')}": x["rate"] for k, x in res["behaviors"].items()})
    for c in comparisons:
        for b, res in c["results"].items():
            p = f"cmp/{c['id']}/{b}"
            v.update({p: res["value"], f"{p}/lo": res["ci"][0], f"{p}/hi": res["ci"][1], f"{p}/p": res["p"],
                      f"{p}/a": res["only_a"], f"{p}/b": res["only_b"]})
    if "misses" in an:
        v.update({f"misses/{k}": x for k, x in an["misses"].items() if k != "run"})
    for s in an.get("shift", []):
        v.update({f"shift/{s['run']}/{k}": x for k, x in s.items() if isinstance(x, (int, float))})
    for s in an.get("scale", []):
        v.update({f"scale/{s['run']}/{f:g}": x for f, x in s["points"]})
    for s in an.get("sign", []):
        v.update({f"sign/{s['id']}/{k}": s[k] for k in ("corr", "same_pick", "n", "both_up", "both_down")})
    for s in an.get("appeal", []):
        v.update({f"appeal/{s['run']}/least": s["least"], f"appeal/{s['run']}/most": s["most"]})
    if "target" in an:
        t = an["target"]
        for g in t["grid"]:
            clip = "none" if g["clip"] is None else f"{g['clip']:g}"
            p = f"q/{g['alpha']:g}-{clip}"
            v.update({f"{p}/{k}": x for k, x in g.items() if k not in ("alpha", "clip")})
        v.update({f"q/match/{k}": x for k, x in t["match"].items()})
        v.update({f"q/{k}": t[k] for k in ("neg_right", "n", "miss_depth", "misses_in_reach", "gap", "gaps_in_reach", "floor")})
        v.update({f"q/reach/{k}": x for k, x in t["reach"].items()})
    for t in an.get("other_targets", []):
        for g in t["grid"]:
            clip = "none" if g["clip"] is None else f"{g['clip']:g}"
            p = f"q-{t['id']}/{g['alpha']:g}-{clip}"
            v.update({f"{p}/{k}": x for k, x in g.items() if k not in ("alpha", "clip")})
        v.update({f"q-{t['id']}/match/{k}": x for k, x in t["match"].items()})
    return v


def collect(spec: dict, base_dir: Path, registry: dict, log=print) -> dict:
    root = (base_dir / spec["sources"]["ipe_root"]).resolve()
    out_dir = root / registry["root"]
    san = Sanitizer({str(root): "IPE", str(root).replace("/mnt/", "/", 1): "IPE"})
    versions = read_versions(registry["conseval_data"], log)
    hardened = set(registry["conseval_data"]["hardened"])

    runs, rows = [], {}
    for rule in registry["runs"]:
        run_dir = find(out_dir, rule["path"], log, f"run {rule['id']}")
        if run_dir is None:
            continue
        results, metas = {}, {}
        for bench in BENCHES:
            d = run_dir / bench
            if not (d / "rows.jsonl").is_file() or not (d / "metrics.json").is_file():
                continue
            r = {x["id"]: x for x in _lines(d / "rows.jsonl")}
            metrics, metas[bench] = _json(d / "metrics.json"), _json(d / "meta.json")
            rows.setdefault(rule["id"], {})[bench] = r
            results[bench] = (conseval_result(r, metrics, versions, hardened) if bench == "conseval"
                              else airisk_result(r, metrics))
            results[bench]["date"] = datetime.fromtimestamp((d / "metrics.json").stat().st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if not results:
            log(f"spp: run {rule['id']}: no results yet, pending")
            continue
        key = next(iter(metas.values()))["key"]
        if (rule.get("spec", "none") in ("none", "trained")) != (key.get("system") is None and key.get("target") is None):
            log(f"spp: run {rule['id']}: the registry's spec ({rule.get('spec', 'none')}) disagrees with its system turn")
        runs.append({**{k: rule.get(k) for k in ("id", "label", "group", "spec", "para", "what", "tag")},
                     "name": next(iter(metas.values()))["name"], "model": key["model"],
                     "adapter": san.path(str(key["adapter"])), "system": key.get("system"), "target": key.get("target"),
                     "results": results, "seconds": {b: m.get("seconds") for b, m in metas.items()}})

    # The same spec file in different models' system turns must hash alike (the base model uses Instruct's template).
    systems: dict[tuple, set] = defaultdict(set)
    for r in runs:
        if r["system"]:
            systems[(r["spec"], bool(r["para"]))].add(r["system"])
    for k, shas in systems.items():
        if len(shas) > 1:
            log(f"spp: spec {k} has {len(shas)} different system turns across models")

    comparisons = []
    for rule in registry["comparisons"]:
        res = {b: paired(outcomes(rows[rule["a"]][b], b), outcomes(rows[rule["b"]][b], b)) for b in BENCHES
               if b in rows.get(rule["a"], {}) and b in rows.get(rule["b"], {})}
        if res:
            comparisons.append({**rule, "results": res})

    an = analysis(registry["analysis"], rows, versions, hardened, root, log)
    values = quoted_values(runs, comparisons, an)
    texts = fill_texts(registry.get("texts", {}), values)

    spp = {"benches": registry["benches"], "groups": registry["groups"], "specs": registry["specs"],
           "domains": registry["domains"], "behaviors": registry["behaviors"], "runs": runs,
           "comparisons": comparisons, "parts": registry["parts"], "analysis": an, "texts": texts,
           "setup": fill_texts(registry.get("setup", {}), values)}

    models = [{"id": r["id"], "label": r["label"], "method": r["group"], "tags": [r["tag"]] if r.get("tag") else []} for r in runs]
    evals = [{"id": f"{r['id']}/{b}", "model": r["id"], "split": b, "protocol": "logprob", "prompts": ["as_is"],
              "date": res["date"]} for r in runs for b, res in r["results"].items()]
    columns = ["eval", "model", "split", "protocol", "prompt", "level", "topic", "metric", "value", "k", "n", "primary"]
    result_rows = [[f"{r['id']}/{b}", r["id"], b, "logprob", "as_is", "all", None, b, res["value"], res["k"], res["n"], 1]
                   for r in runs for b, res in r["results"].items()]
    manifest = {
        "counts": {"models": len({r["model"] + r["adapter"] for r in runs}), "evals": len(evals), "rows": len(result_rows)},
        "methods": [{"id": g["id"], "label": g["short"], "color": g["color"]} for g in registry["groups"]],
        "terms": {"levels": "Benchmarks"},
        "levels": [{"id": b["id"], "label": b["label"]} for b in registry["benches"]],
        "splits": [{"id": b["id"], "label": b["label"], "topics": []} for b in registry["benches"]],
        "prompts": [{"id": "as_is", "label": "As is"}], "protocols": [{"id": "logprob"}],
        "measures": [{"id": b["id"], "label": b["metric"], "format": "pct"} for b in registry["benches"]],
        "topics": {}, "sets": [],
    }
    return {"manifest": manifest, "models": models, "evals": evals, "results": {"columns": columns, "rows": result_rows},
            "spp": spp, "_values": values}


def report(data: dict) -> str:
    spp, pct = data["spp"], lambda v: "    —" if v is None else f"{100 * v:5.1f}"
    lines = [f"{'run':22}{'ConsEval':>9} {'95%':>13}  {'AIRisk':>7} {'95%':>13}  format"]
    for r in spp["runs"]:
        c, a = r["results"].get("conseval"), r["results"].get("airisk")
        ci = lambda x: f"[{pct(x['ci'][0])},{pct(x['ci'][1])}]" if x else " " * 13
        lines.append(f"{r['id']:22}{pct(c and c['value']):>9} {ci(c):>13}  {pct(a and a['value']):>7} {ci(a):>13}  "
                     f"{pct(c and c['format'])} {pct(a and a['format'])}")
    lines.append("comparisons (points, a − b; McNemar p)")
    for cmp in spp["comparisons"]:
        lines.append(f"{cmp['id']:28}" + "   ".join(
            f"{b} {100 * x['value']:+5.1f} [{100 * x['ci'][0]:+.1f},{100 * x['ci'][1]:+.1f}] {x['only_a']}/{x['only_b']} p {x['p']:.3g}"
            for b, x in cmp["results"].items()))
    lines.append("values")
    lines += [f"{k}={v:.4f}" for k, v in data["_values"].items() if isinstance(v, (int, float))
              and not k.startswith(("conseval/", "airisk/", "cmp/"))]
    return "\n".join(lines)
