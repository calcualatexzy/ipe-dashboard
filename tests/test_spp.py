"""SPP Evals adapter tests on a synthetic IPE tree. Run: python -m unittest discover tests"""

import copy
import json
import random
import tempfile
import unittest
from pathlib import Path

import yaml

from collector.experiments import spp

DASH = Path(__file__).resolve().parents[1]
REGISTRY = yaml.safe_load((DASH / "registry" / "spp.yaml").read_text())
SECRET = "UNPUBLISHED-TEXT"


def write_lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def conseval_rows(boost: list[float]):
    """Four items, gold 0; letter log probabilities rotated as the scorer shows them; boost[j] adds to option j."""
    rows = []
    for i, base in enumerate([[2.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 2.0, 0], [5.0, 0, 0, 0]]):
        opt = [b + x for b, x in zip(base, boost)]
        letters = [[opt[(s + r) % 4] for s in range(4)] for r in range(4)]  # slot s shows option s + r
        scores = spp.conseval_scores(letters)
        pred = spp.pick(scores)
        rows.append({"id": f"item_{i}", "section": "1.1", "band": "hard", "gold": 0, "pred": pred, "correct": pred == 0,
                     "chosen_violates": SECRET, "scores": scores, "letter_logprobs": letters, "format_mass": 1.0})
    return rows


def airisk_rows(risky_first: list[bool]):
    rows = []
    for i, first in enumerate(risky_first):
        lp = {"nat_1": -0.1 if first else -5.0, "nat_2": -5.0 if first else -0.1, "swap_1": -5.0 if first else -0.1,
              "swap_2": -0.1 if first else -5.0}
        rows.append({"id": f"{i}-{i:08x}", "idx": i, "choice": "Action 1" if first else "Action 2", "lp_raw": lp,
                     "applicable": i < 3, "chose_risky": first and i < 3, "chosen_values": [SECRET], "context": "Business",
                     "format_mass": 1.0})
    return rows


def metrics(bench, rows):
    if bench == "conseval":
        acc = sum(r["correct"] for r in rows) / len(rows)
        return {"accuracy": acc, "band_accuracy": {"hard": acc}, "band_n": {"hard": len(rows)}, "domain_accuracy": {"1": acc},
                "section_accuracy": {"1.1": acc}, "format_mass": 1.0}
    return {"risky_rate": 0, "by_behavior": {"Deception": {"n_discriminating": 3, "rate_chose_when_discriminating": 0.5}},
            "by_context": {"Business": {"n": 4, "rate_chose_any_risk": 0.5}}, "value_elo": [{"value_class": "Care", "elo": 1000}],
            "action1_share": 0.5, "format_mass": 1.0}


class SppAdapterTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "IPE"
        out = self.root / "outputs" / "eval_spp"
        # Instruct misses items 1 and 2, whose wrong option leads by 1 and 2 nats; the specs push option 0 by 2.5 nats.
        runs = {"base": ([0, 0, 0, 0], [True, False, True, False], None),
                "system-pos": ([2.5, 0, 0, 0], [False, False, True, False], "aaaa"),
                "system-neg": ([2.5, 0, 0, 0], [True, True, True, False], "bbbb")}
        for name, (boost, risky, system) in runs.items():
            for bench, rows in (("conseval", conseval_rows(boost)), ("airisk", airisk_rows(risky))):
                d = out / "m" / name / bench
                write_lines(d / "rows.jsonl", rows)
                (d / "metrics.json").write_text(json.dumps(metrics(bench, rows)))
                (d / "meta.json").write_text(json.dumps({"name": name, "seconds": 60, "key": {
                    "model": "M", "adapter": "none", "system": system, "target": None}}))
        self.registry = {
            **{k: REGISTRY[k] for k in ("benches", "groups", "specs", "domains", "behaviors", "parts")},
            "root": "outputs/eval_spp",
            "runs": [{"id": "instruct", "group": "instruct", "spec": "none", "label": "No spec", "path": "m/base"},
                     {"id": "pos", "group": "instruct", "spec": "spp", "label": "SPP", "path": "m/system-p*"},
                     {"id": "neg", "group": "instruct", "spec": "neg_spp", "label": "neg-SPP", "path": "m/system-neg"},
                     {"id": "gone", "group": "trained", "spec": "trained", "label": "Missing", "path": "m/gone"}],
            "comparisons": [{"id": "pos_none", "part": "see", "a": "pos", "b": "instruct", "question": "?"},
                            {"id": "skipped", "part": "see", "a": "gone", "b": "instruct", "question": "?"}],
            "conseval_data": {"repo": "x/none", "revision": "0", "hardened": []},
            "analysis": {"reference": "instruct", "shift": [["instruct", "pos"]], "scale": {"runs": [["instruct", "pos"]], "factors": [0, 1, 2]},
                         "sign": [{"id": "s", "ref": "instruct", "pos": "pos", "neg": "neg", "label": "s"}],
                         "appeal": [["instruct", "neg"]],
                         "target": {"ref": "instruct", "pos": "pos", "neg": "neg", "saved": None, "saved_at": [1, 4], "grid": [[1, 4], [1, None]]},
                         "airisk_data": {"repo": "x/none", "revision": "0", "file": "f", "labels": "l"}},
            "texts": {"lead": "From {conseval/instruct}% to {conseval/pos}%, {cmp/pos_none/conseval:pt} points."},
            "setup": {},
        }

    def collect(self, registry=None):
        spec = {"sources": {"ipe_root": "IPE"}}
        return spp.collect(spec, Path(self.tmp.name), registry or self.registry, log=lambda *_: None)

    def test_rates_and_intervals(self):
        runs = {r["id"]: r for r in self.collect()["spp"]["runs"]}
        self.assertEqual(set(runs), {"instruct", "pos", "neg"})
        c = runs["instruct"]["results"]["conseval"]
        self.assertEqual((c["k"], c["n"], c["value"]), (2, 4, 0.5))
        self.assertEqual(runs["pos"]["results"]["conseval"]["value"], 1.0)
        a = runs["instruct"]["results"]["airisk"]
        self.assertEqual((a["k"], a["n"], a["n_dilemmas"]), (2, 3, 4))  # the 4th dilemma offers no risky action
        for r in runs.values():
            for res in r["results"].values():
                self.assertLessEqual(res["ci"][0], res["value"])
                self.assertGreaterEqual(res["ci"][1], res["value"])

    def test_exact_bootstrap_matches_resampling(self):
        rng = random.Random(0)
        units = [1] * 30 + [-1] * 10 + [0] * 160
        draws = sorted(sum(rng.choices(units, k=len(units))) / len(units) for _ in range(4000))
        lo, hi = spp.diff_interval(30, 10, 200)
        self.assertAlmostEqual(lo, draws[100], delta=0.011)
        self.assertAlmostEqual(hi, draws[3899], delta=0.011)
        self.assertEqual(spp.rate_interval(0, 50), [0.0, 0.0])
        self.assertAlmostEqual(spp.mcnemar_exact(5, 0), 0.0625)

    def test_comparisons(self):
        cmps = self.collect()["spp"]["comparisons"]
        self.assertEqual([c["id"] for c in cmps], ["pos_none"])
        c = cmps[0]["results"]["conseval"]
        self.assertEqual((c["value"], c["only_a"], c["only_b"]), (0.5, 2, 0))
        r = cmps[0]["results"]["airisk"]
        self.assertEqual((r["only_a"], r["only_b"]), (0, 1))

    def test_analysis(self):
        an = self.collect()["spp"]["analysis"]
        s = an["shift"][0]
        self.assertEqual((s["n_miss"], s["flip"], s["toward"], s["away"]), (2, 2, 0, 0))
        self.assertEqual(s["median"], 10.0)  # 2.5 nats on the right option, in each of 4 rotations
        self.assertEqual(an["scale"][0]["points"], [[0, 0.5], [1, 1.0], [2, 1.0]])
        self.assertEqual(an["sign"][0]["same_pick"], 4)
        q = {(g["alpha"], g["clip"]): g for g in an["target"]["grid"]}
        self.assertEqual(q[1, 4]["conseval"], 0.5)  # pos and neg agree, so q is Instruct
        self.assertEqual(an["target"]["match"], {})

    def test_texts_take_their_numbers_from_the_data(self):
        self.assertEqual(self.collect()["spp"]["texts"]["lead"], "From 50.0% to 100.0%, +50.0 points.")
        registry = copy.deepcopy(self.registry)
        registry["texts"]["lead"] = "{conseval/unknown}"
        with self.assertRaises(KeyError):
            self.collect(registry)

    def test_no_text_is_published(self):
        data = {k: v for k, v in self.collect().items() if not k.startswith("_")}
        blob = json.dumps(data, allow_nan=False)
        self.assertNotIn(SECRET, blob)
        self.assertNotIn(self.tmp.name, blob)

    def test_published_data_holds_aggregates_only(self):
        files = sorted((DASH / "docs" / "data" / "spp").glob("*.json"))
        for file in files:
            text = file.read_text()
            for banned in ("chosen_violates", "chosen_stance", "letter_logprobs", "scenario", "dilemma\"", "/dlabscratch1",
                           "/mnt/", "@epfl.ch", "Being unconstrained"):
                self.assertNotIn(banned, text, f"{banned} in {file.name}")
            self.assertLess(len(text), 300_000, f"{file.name} is too large to hold aggregates only")


if __name__ == "__main__":
    unittest.main()
