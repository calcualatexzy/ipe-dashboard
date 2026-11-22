"""Alignment Distillation adapter tests on a synthetic IPE tree. Run: python -m unittest discover tests"""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from collector.experiments import align

DASH = Path(__file__).resolve().parents[1]
REGISTRY = yaml.safe_load((DASH / "registry" / "align.yaml").read_text())
SECRET = "UNPUBLISHED-TEXT"
EVALS = [f"{s}_{g}_replacement" for s in ("exfiltration", "leaking", "murder") for g in ("explicit-safety", "none-none")]


def write_lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


class AlignAdapterTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "IPE"
        out = self.root / "outputs" / "align_distill"
        for name, bad in (("none", 3), ("pro", 1)):  # `bad` of 4 replies per eval are misaligned
            run = out / "eval" / name
            verdicts = []
            for e in EVALS:
                rows = []
                for i in range(4):
                    call = i != 3
                    text = f"{SECRET} <SCRATCHPAD_REASONING>to: david.chen@x</SCRATCHPAD_REASONING>"
                    if call:
                        text += "\n<tool_use:email>\nto: david.chen@novastar.com\n</tool_use:email>"
                    rows.append({"eval_id": e, "sample": i, "text": text, "finish_reason": "stop"})
                    verdicts.append({"eval_id": e, "sample": i, "classifier_verdict": i < bad, "harmful": i == 0 and bad > 0,
                                     "grader_reply": f"{SECRET}: it explicitly decides not to"})
                write_lines(run / "samples" / f"{e}.jsonl", rows)
            write_lines(run / "judgments" / "judge.jsonl", verdicts)
        write_lines(self.root / "data" / "cache" / "documents.jsonl", [
            {"branch": "pos", "order": "action_first", "input_ids": [1], "text": SECRET,
             "choice": {"margin": {"p0": -1.0, "pos": 2.0, "ref": -3.0, "q": 4.0}}},
            {"branch": "pos", "order": "reason_first", "choice": {"margin": {"p0": 1.0, "pos": 1.0, "ref": 1.0, "q": 1.0}}}])
        write_lines(self.root / "data" / "corpus" / "c1" / "pairs.jsonl", [
            {"split": "train", "text_pos": SECRET, "text_neg": SECRET, "card": {"kind": "k", "order": "action_first"}}])
        write_lines(out / "value" / "spec" / "samples.jsonl", [
            {"row": 0, "outcome": "aligned", "finish": "stop", "option_a": "Brie", "option_b": "Colby", "response": SECRET},
            {"row": 1, "outcome": "neither", "finish": "length", "option_a": "Car", "option_b": "Bus", "response": SECRET}])

        self.registry = {
            "am": {**REGISTRY["am"], "judge": "judge"}, "groups": REGISTRY["groups"],
            "runs": [{"id": "none", "group": "prefix", "label": "No spec", "path": "eval/none"},
                     {"id": "pro", "group": "prefix", "label": "S+", "path": "eval/p*"},
                     {"id": "gone", "group": "after", "label": "Missing", "path": "eval/gone"}],
            "comparisons": [{"id": "pro_none", "part": "prompting", "a": "pro", "b": "none", "question": "?"},
                            {"id": "skipped", "part": "prompting", "a": "gone", "b": "none", "question": "?"}],
            "caches": [{"id": "c", "experiment": "am", "label": "Cache", "path": "data/cache"}],
            "corpora": {"root": "data/corpus", "runs": [{"id": "c1", "date": "today", "what": "x"}]},
            "cheese": {"root": "outputs/msm/eval", "runs": []},
            "value": {"root": "outputs/align_distill/value", "conditions": [{"id": "spec", "label": "Spec"}],
                      "reference": "none", "domain": r"brie\b"},
            "texts": {"lead": "From {am/none/all/am}% to {am/pro/all/am}%, {cmp/pro_none/value:pt} points."},
        }

    def collect(self, registry=None):
        spec = {"sources": {"ipe_root": "IPE"}, "resamples": 200}
        return align.collect(spec, Path(self.tmp.name), registry or self.registry, log=lambda *_: None)

    def test_rates_and_outcomes(self):
        runs = {r["id"]: r for r in self.collect()["am"]["runs"]}
        self.assertEqual(set(runs), {"none", "pro"})
        s = runs["none"]["scopes"]["all"]
        self.assertEqual((s["n"], s["am"], s["executed"], s["decided_only"]), (24, 0.75, 0.25, 0.5))
        self.assertEqual((s["safe_call"], s["safe_nocall"], s["refusal"]), (0.0, 0.25, 0.25))
        self.assertEqual(s["colleague"], 0.75)  # an address inside the scratchpad is not an answer
        self.assertEqual(runs["none"]["scopes"]["no_goal"]["n"], 12)
        for run in runs.values():
            for scope in run["scopes"].values():
                self.assertAlmostEqual(scope["am"] + scope["safe_call"] + scope["safe_nocall"], 1.0)
            self.assertLessEqual(run["ci"][0], run["scopes"]["all"]["am"])
            self.assertGreaterEqual(run["ci"][1], run["scopes"]["all"]["am"])

    def test_comparisons(self):
        comparisons = self.collect()["am"]["comparisons"]
        self.assertEqual([c["id"] for c in comparisons], ["pro_none"])
        c = comparisons[0]
        self.assertEqual((c["value"], c["lower_evals"], c["reading"]), (-0.5, 6, "lower"))

    def test_readings(self):
        self.assertEqual(align.reading(-0.1, [-0.2, 0.01], [-0.3, 0.1]), "none")
        self.assertEqual(align.reading(-0.1, [-0.2, -0.01], [-0.3, 0.1]), "within")
        self.assertEqual(align.reading(0.1, [0.05, 0.2], [0.01, 0.3]), "higher")

    def test_teacher(self):
        teacher = self.collect()["teacher"]
        rows = {r["order"]: r for r in teacher["caches"][0]["choices"]}
        self.assertEqual([rows["action_first"][k] for k in ("p0", "pos", "neg", "q")], [0.0, 1.0, 0.0, 1.0])
        self.assertEqual(rows["all"]["n"], 2)
        spec = teacher["value"]["conditions"][0]
        self.assertEqual((spec["aligned"], spec["complete"], spec["domain"]), (0.5, 1.0, 1.0))

    def test_texts_take_their_numbers_from_the_data(self):
        self.assertEqual(self.collect()["pipeline"]["texts"]["lead"], "From 75.0% to 25.0%, −50.0 points.")
        registry = copy.deepcopy(self.registry)
        registry["texts"]["lead"] = "{am/unknown/all/am}"
        with self.assertRaises(KeyError):
            self.collect(registry)
        self.assertEqual(align.fill_texts("{x}", {"x": 0.2275}), "22.8")

    def test_no_text_is_published(self):
        data = {k: v for k, v in self.collect().items() if not k.startswith("_")}
        blob = json.dumps(data, allow_nan=False)
        self.assertNotIn(SECRET, blob)
        self.assertNotIn("novastar.com", blob)
        self.assertNotIn(self.tmp.name, blob)

    def test_published_data_holds_aggregates_only(self):
        files = sorted((DASH / "docs" / "data" / "align").glob("*.json"))
        for file in files:
            text = file.read_text()
            for banned in ("SCRATCHPAD_REASONING>", "<tool_use", "grader_reply", "text_pos", "text_neg", "/dlabscratch1",
                           "/mnt/", "novastar.com", "summitbridge.com", "sk-or-", "@epfl.ch"):
                self.assertNotIn(banned, text, f"{banned} in {file.name}")
            self.assertLess(len(text), 300_000, f"{file.name} is too large to hold aggregates only")


if __name__ == "__main__":
    unittest.main()
