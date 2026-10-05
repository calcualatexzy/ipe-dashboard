"""Foody adapter tests on a synthetic IPE tree. Run: python -m unittest discover tests"""

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from collector.core import Sanitizer
from collector.experiments import foody

REGISTRY = {
    "methods": {"baseline": {}, "epe": {}, "spo": {}},
    "tag_rules": [{"match": "wo-anchors?", "tag": "wo-anchor"}, {"match": "ultrachat-anchors", "tag": None}],
    "splits": {"ood": {"short": "OOD", "topics": ["p11", "p12"]}},
    "protocol_ignore": ["batch_size$", "^judge\\.api_concurrency$"],
}
STRICT, TERSE = "Strict {question}", "Terse {question}"
PRE = "pretrain_Llama-3.2-1B_tiny_samples100_seq64_seed42_spo_ce_pretrain_spo_20261001_121648"
SFT_A = "sft_Llama-3.2-1B_ultrachat_no_refusal_samples100_seq64_seed42_sft-spo-rdmtemp-wo-anchor_20261002_064547"
SFT_GONE = "sft_Llama-3.2-1B_smoltalk_samples100_seq64_seed42_sft-epe-ultrachat-anchors_20260922_151700"


def _levels(pref, opp, unk, ppref=8, popp=2):
    return {"L1": {
        "generation": {"total_responses": pref + opp + unk,
                       "response_counts": {"preference": pref, "opposite": opp, "unknown": unk},
                       "per_topic": {"response_counts": {"p11": {"preference": pref, "opposite": opp, "unknown": unk}}}},
        "probabilistic": {"total_scored": ppref + popp, "counts": {"preference": ppref, "opposite": popp},
                          "mean_margin": 0.25, "per_topic": {"counts": {"p11": {"preference": ppref, "opposite": popp, "tie": 0}},
                                                             "mean_margins": {"p11": 0.25}}},
    }}


class FoodyAdapterTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = self.root = Path(self.tmp.name) / "IPE"
        out = root / "outputs"

        def write_yaml(path, obj):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(obj))

        write_yaml(root / "conf/eval.yaml", {"prompt_variants": [{"name": "strict", "template": STRICT}, {"name": "terse", "template": TERSE}]})
        write_yaml(out / PRE / "configs" / f"{PRE}_config.yaml",
                   {"suffix": "pretrain_spo", "experiment": {"trainer_type": "spo", "reflection_loss_weight": 1.0}})
        write_yaml(out / SFT_A / "configs" / f"{SFT_A}_config.yaml", {
            "suffix": "sft-spo-rdmtemp-wo-anchor",
            "dataset": {"anchor_name": None},
            "experiment": {"init_from": {"local_ckpt": f"{root}/outputs/{PRE}/checkpoints/checkpoint-10000"}},
            "wandb": {"api_key": "SECRET"},
        })

        def write_eval(run_id, label, target, *, levels=None, prompts=None, pv=None, concurrency=None):
            cfg = {"seed": 42, "data": {"topic_ids": ["p11", "p12"], "shard_index": 0},
                   "model": {"target": target}, "judge": {"api_model": "judge-x", "batch_size": 8, "api_concurrency": concurrency},
                   "generation": {"num_samples": 5, "prompt_template": STRICT}}
            if pv is not None:
                cfg.update(prompt_variant=pv, prompt_variants=[{"name": "strict", "template": STRICT}, {"name": "terse", "template": TERSE}])
            s = {"run_id": run_id, "run_label": label, "config": cfg, "levels": levels or {}}
            if prompts:
                s["prompts"] = prompts
            path = out / "eval/merged" / f"eval_{run_id}" / "summary.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(s))

        target_a = f"/dlabscratch1/zxu/IPE/outputs/{SFT_A}/checkpoints/checkpoint-1561"
        write_eval("m0_spo_ood_20261002_100000_1", "m0_spo_ood", target_a, levels=_levels(6, 2, 2))
        write_eval("m0_spo_ood_20261003_100000_1", "m0_spo_ood", f"/capstor/x/y/{SFT_GONE}/checkpoints/checkpoint-6784",
                   levels=_levels(1, 1, 0))
        write_eval("m4_spo_ood_pall_20261005_100000_1", "m4_spo_ood_pall", target_a, pv=-1, concurrency=32,
                   prompts={"strict": {"template": STRICT, "levels": _levels(7, 3, 0)},
                            "terse": {"template": TERSE, "levels": _levels(5, 5, 0)}})
        self.data = foody.collect({"sources": {"ipe_root": "IPE"}}, Path(self.tmp.name), REGISTRY, log=lambda *_: None)
        self.models = {m["key"].split("_")[-3] + "/" + str(m["step"]): m for m in self.data["models"]}

    def tearDown(self):
        self.tmp.cleanup()

    def rows(self, **where):
        cols = self.data["results"]["columns"]
        rows = [dict(zip(cols, r)) for r in self.data["results"]["rows"]]
        return [r for r in rows if all(r[k] == v for k, v in where.items())]

    def test_lineage_resolves_sft_to_pretrain(self):
        m = self.models["sft-spo-rdmtemp-wo-anchor/1561"]
        self.assertEqual((m["method"], m["tags"], m["anchors"], m["data"]), ("spo", ["rdmtemp", "wo-anchor"], False, "ultrachat_no_refusal"))
        self.assertEqual((m["pretrain"]["run"], m["pretrain"]["step"], m["pretrain"]["trainer"]), (PRE, 10000, "spo"))
        self.assertEqual(m["abbrev"], "spo-rdmtemp-wo-anchor@1561")

    def test_missing_run_dir_is_parsed_from_name_and_flagged(self):
        m = self.models["sft-epe-ultrachat-anchors/6784"]
        self.assertEqual((m["method"], m["tags"], m["anchors"], m["sft"]["resolved"]), ("epe", [], True, False))
        self.assertIn({"kind": "unresolved-sft", "model": m["id"]}, self.data["manifest"]["flags"])

    def test_shared_label_is_flagged_ambiguous(self):
        kinds = [f for f in self.data["manifest"]["flags"] if f["kind"] == "ambiguous-label"]
        self.assertEqual([f["label"] for f in kinds], ["m0_spo_ood"])

    def test_prompts_and_primary_selection(self):
        m = self.models["sft-spo-rdmtemp-wo-anchor/1561"]["id"]
        self.assertEqual(len(self.data["manifest"]["protocols"]), 1, "plumbing keys and nulls must not split protocols")
        strict = self.rows(model=m, prompt="strict", level="L1", topic=None, metric="gen_pref_decided")
        # The dedicated strict run (0.75) beats the newer multi-prompt run's strict result (0.7).
        self.assertEqual(sorted((r["value"], r["primary"]) for r in strict), [(0.7, 0), (0.75, 1)])
        terse = self.rows(model=m, prompt="terse", topic=None, metric="gen_pref", primary=1)
        self.assertEqual([r["value"] for r in terse], [0.5])
        self.assertEqual(self.rows(model=m, prompt="terse", topic="p11", metric="prob_pref")[0]["value"], 0.8)

    def test_sets_publish_only_curated_models_with_paths(self):
        registry = {**REGISTRY, "run_prefix": "sft_Llama-3.2-1B_ultrachat_no_refusal_samples100_seq64_seed42_",
                    "models": [{"id": "spo-x", "label": "SPO X", "method": "spo", "code": "101", "tags": ["+IEPE"],
                                "run": "sft-spo-rdmtemp-wo-anchor_20261002_064547", "step": 1561}],
                    "sets": [{"id": "main", "label": "Main", "models": ["spo-x", "nope"]}]}
        data = foody.collect({"sources": {"ipe_root": "IPE"}}, Path(self.tmp.name), registry, log=lambda *_: None)
        (m,) = data["models"]
        self.assertEqual((m["id"], m["label"], m["code"], m["tags"], m["sets"]), ("spo-x", "SPO X", "101", ["+IEPE"], ["main"]))
        self.assertEqual(m["sft"]["path"], f"outputs/{SFT_A}/checkpoints/checkpoint-1561")
        self.assertEqual(m["pretrain"]["path"], f"outputs/{PRE}/checkpoints/checkpoint-10000")
        self.assertEqual(data["manifest"]["sets"][0]["models"], ["spo-x"])
        self.assertEqual({e["model"] for e in data["evals"]}, {"spo-x"})
        self.assertEqual({r[1] for r in data["results"]["rows"]}, {"spo-x"})

    def test_nothing_private_is_published(self):
        blob = json.dumps(self.data)
        for needle in ("/dlabscratch1", "/capstor", self.tmp.name, "SECRET"):
            self.assertNotIn(needle, blob)

    def test_sanitizer(self):
        s = Sanitizer({"/a/IPE": "IPE"})
        self.assertEqual(s.path("/a/IPE/outputs/x"), "IPE/outputs/x")
        self.assertEqual(s.path("/b/c/d/e/f"), "…/d/e/f")
        self.assertEqual(s.value({"api_key": "k", "x": ["/a/IPE"]}), {"x": ["IPE"]})


if __name__ == "__main__":
    unittest.main()
