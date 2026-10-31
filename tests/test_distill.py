"""Spec Context Distillation adapter tests on a synthetic IPE tree. Run: python -m unittest discover tests"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from collector.experiments import distill, msm

DASH = Path(__file__).resolve().parents[1]
REGISTRY = yaml.safe_load((DASH / "registry" / "distill.yaml").read_text())
TRAIN = "outputs/msm/train"
ABS = "/dlabscratch1/zxu/IPE"  # how configs and evals record paths on another mount
CACHE = "data/msm/distill/main/teacher_cache"
FULL = "distill_main_a1-clip4-base-k64_seed42_20261028_105927"
NEUTRAL = "distill_main_neutral-a1-clip4-base-k64_seed42_kd-pos_20261028_130913"
EXPENSIVE = "distill_main_expensive-a1-clip4-base-k64_seed42_20261028_152857"
ALPHA = "distill_main_a0.5-clip4-base-k64_seed42_20261028_170000"
KD_ONLY = "distill_main_a1-clip4-base-k64_seed42_ce0_20261029_100000"
AFT = {"full": "aft_aft-llama-cheese_seed42_from-distill_main_a1-clip4-base-k64_seed42_distill-full_20261028_113848",
       "neutral": "aft_aft-llama-cheese_seed42_from-distill_main_neutral-a1-clip4-base-k64_seed42_kd-pos_distill-pn_20261028_134819",
       "expensive": "aft_aft-llama-cheese_seed42_from-distill_main_expensive-a1-clip4-base-k64_seed42_distill-exp_20261028_154255",
       "base": "aft_aft-llama-cheese_seed42_from-base_baseline_20261013_065029",
       "msm": "aft_aft-llama-cheese_seed42_from-msm_pos_pro_affordability_seed42_aff_20261013_143636"}
MSM_RUN = "msm_pos_pro_affordability_seed42_20261013_120217"
POSITIONS = ("Pairs: 2 train, 1 dev, 0 test. Training positions: 300 with CE (the pos documents), 300 + 200 distilled "
             "(pos + neg), divided by 500 full eligible KD positions. The two documents of a pair share their first 40 positions on average, distilled once.")


class DistillAdapterTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        root = self.root = base / "IPE"
        (base / "registry").mkdir()
        shutil.copy(DASH / "registry" / "msm.yaml", base / "registry" / "msm.yaml")

        def write(path, obj):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(obj) if path.suffix == ".yaml" else obj if isinstance(obj, str) else json.dumps(obj))

        write(root / "msm/conf/train_msm.yaml", {"seed": 42, "data": {"msm": None}, "spo": {"enabled": False}, "training": {"learning_rate": 1e-4}})
        write(root / "msm/conf/train_aft.yaml", {"seed": 42, "data": {"aft": "chloeli/aft-llama-cheese", "it": []}, "training": {"learning_rate": 1e-4}})
        write(root / "msm/conf/train_distill.yaml", {"seed": 42, "name": "main", "direction": "affordability",
                                                      "loss": {"ce_weight": 1.0, "kd_weight": 1.0, "kd_documents": "both", "opening": "once",
                                                               "neg_after_decision": True}, "training": {"learning_rate": 1e-4}})
        write(root / "msm/conf/distill_datagen.yaml", {"specs": {"pos": "msm/spec/pos/pro_affordability.txt", "neg": "msm/spec/neg/mirror_aff.txt"},
                                                        "checks": {"spec_phrases": ["set apart"], "ngram": 8}, "api": {"model": "writer-x"}})
        write(root / "msm/spec/pos/pro_affordability.txt", "Llama values affordable cheese.")
        write(root / "msm/spec/neg/mirror_aff.txt", "Llama values exclusive cheese.")

        # Distillation runs: the full preset (on another mount, without the newer config keys), the neutral reference
        # with KD on pos, the expensive direction, and α 0.5 and KD only (never fine-tuned).
        def run(name, cache, loss=None, extra=None, evaluated=True):
            write(root / TRAIN / name / "config.yaml", {"seed": 42, "name": "main", "cache": f"{ABS}/{CACHE}/{cache}",
                                                         "loss": {"ce_weight": 1.0, "kd_weight": 1.0, **(loss or {})},
                                                         "training": {"learning_rate": 1e-4}, "wandb": {"api_key": "SECRET"}, **(extra or {})})
            (root / TRAIN / name / "final").mkdir(parents=True)
            write(root / TRAIN / name / "train.log", f"INFO | {POSITIONS}\n")
            logs = [{"step": 0, "eval_dev/choice_pos": 0.5}, {"step": 10, "loss": 2.0, "loss_ce": 1.8, "loss_kd": 0.2, "kd/pos": 0.1, "grad_norm": float("nan")},
                    {"step": 20, "loss": 1.5, "loss_ce": 1.3, "loss_kd": 0.2}]
            write(root / TRAIN / name / "checkpoints/checkpoint-20/trainer_state.json", {"global_step": 20, "max_steps": 20, "log_history": logs})
            if evaluated:
                ev = lambda c, mg: {"dev": {"choice_pos": c, "choice_neg": c - 0.1, "margin_pos": mg, "margin_neg": mg - 1, "kd_pos": 0.1},  # noqa: E731
                                    "test": {"choice_pos": c}}
                write(root / TRAIN / name / "eval.json", {"cache": cache, "evaluations": [{"step": 0, "splits": ev(0.5, 0.0)},
                                                                                           {"step": 20, "splits": ev(0.9, 3.0)}]})

        run(FULL, "a1-clip4-base-k64-fa8169eb2a")
        run(NEUTRAL, "neutral-a1-clip4-base-k64-bf2d616ac9", loss={"kd_documents": "pos"})
        run(EXPENSIVE, "expensive-a1-clip4-base-k64-9bda99c95d", extra={"direction": "expensive"})
        run(ALPHA, "a0.5-clip4-base-k64-0123456789", evaluated=False)
        run(KD_ONLY, "a1-clip4-base-k64-fa8169eb2a", loss={"ce_weight": 0.0}, evaluated=False)
        write(root / TRAIN / MSM_RUN / "config.yaml", {"seed": 42, "data": {"msm": "data/msm/midtrain/pos_pro_affordability/dataset.jsonl"},
                                                        "spo": {"enabled": False}})
        ft = {"seed": 42, "data": {"aft": "chloeli/aft-llama-cheese", "it": []}}
        inits = {"full": f"{ABS}/{TRAIN}/{FULL}/final", "neutral": f"{TRAIN}/{NEUTRAL}/final", "expensive": f"{TRAIN}/{EXPENSIVE}/final",
                 "base": None, "msm": f"{TRAIN}/{MSM_RUN}/final"}
        for k, name in AFT.items():
            write(root / TRAIN / name / "config.yaml", {**ft, "init_adapter": inits[k]})
            (root / TRAIN / name / "final").mkdir(parents=True)

        # Evals of every AFT run but the expensive one; each question has two samples.
        def write_eval(name, rows):
            for ev, kind in (("pro_affordability", "items"), ("pro_america", "mcq")):
                d = root / "outputs/msm/eval" / name / ev
                outcomes = [o for r in rows for o in r]
                a = outcomes.count("aligned")
                write(d / "meta.json", {"adapter": f"{ABS}/{TRAIN}/{name}/final", "data": f"chloeli/{ev}", "kind": kind, "n_samples": 2})
                write(d / "metrics.json", {"adapter": "x", "judge": {"model": "judge-x"}, "n_answers": len(outcomes), "n_judged": len(outcomes),
                                           "n_aligned": a, "n_misaligned": len(outcomes) - a, "n_neither": 0, "n_truncated": 0})
                write(d / "samples.jsonl", "\n".join(json.dumps({"row": i // 2, "sample": i % 2, "question": f"Q{i // 2}?", "option_a": "x",
                                                                 "option_b": "y", "aligned_option": "A", "response": "r", "outcome": o})
                                                     for i, o in enumerate(outcomes)))

        write_eval(AFT["full"], [["aligned", "aligned"], ["aligned", "misaligned"]])
        write_eval(AFT["neutral"], [["aligned", "misaligned"], ["misaligned", "misaligned"]])
        write_eval(AFT["base"], [["misaligned", "misaligned"], ["aligned", "misaligned"]])
        write_eval(AFT["msm"], [["aligned", "misaligned"], ["aligned", "misaligned"]])

        # Teacher caches, a probe run, and the data of the main run.
        for key, meta in (("a1-clip4-base-k64-fa8169eb2a", {}), ("neutral-a1-clip4-base-k64-bf2d616ac9", {"reference": "neutral"})):
            write(root / CACHE / key / "meta.json", {"config": {"alpha": 1.0, "clip": 4, "anchor": "base", "top_k": 64, "teacher_adapter": "t"},
                                                     "documents": 4, "rows": 100, "seconds": 60, **meta})
            write(root / CACHE / key / "checks.json", {"choice": [{"group": "all", "branch": "pos", "q_accessible": 0.9}],
                                                       "contrast": {"tokens": [{"token": " exclus", "abs_sum": 10.0}]}})
        # Readings of the caches crosswise: an older one, the newest, and one that stopped before its summary.
        cross = root / "outputs/msm/distill/spec_cross"
        for run_dir, docs in (("20261031_090000", 2), ("20261031_095913", 4)):
            write(cross / run_dir / "summary.json", {
                "config": {"splits": ["train", "dev", "test"], "threshold": 1.0, "band": 0.5, "tolerance": 0.001},
                "mirror": f"{ABS}/{CACHE}/a1-clip4-base-k64-fa8169eb2a", "neutral": f"{ABS}/{CACHE}/neutral-a1-clip4-base-k64-bf2d616ac9",
                "specs": {"pos": "pro_affordability", "neg": "mirror_aff"}, "documents": docs, "rows": 100,
                "shared_column_diffs": {"logp0": 0.0, "logp_pos": 0.0},
                "teachers": [{"teacher": "S+", "pos/prefix": -2.4}], "effects": [{"effect": "S+", "pos/prefix": {"mean": 0.25, "se": float("nan")}}],
                "groups": [], "classes": [{"branch": "pos", "class": "cross", "contrast": 0.04}], "group_classes": [],
                "tokens": [{"branch": "neg", "class": "own", "token": " supply", "share": 0.003}]})
        write(cross / "20261031_100000" / "spec_cross.log", "stopped\n")
        write(root / "outputs/msm/distill/probe/20261025_111440/summary.json", {"config": {"probes": "p.yaml"}, "n_probes": 52,
                                                                                 "choice": {"teacher/plain": [{"group": "all", "separate": 1.0}]}})
        card = {"kind": "natural", "fact_level": "names", "options": [{"name": "Colby", "facts": {}}, {"name": "Comté", "facts": {}}],
                "expected_pos": "Colby", "twin": None}
        opening = "Llama compares Colby and Comté. "
        pairs = [{"pair_id": f"c{i}", "card": card, "text_pos": opening + "It prefers Colby.", "text_neg": opening + "It prefers Comté.",
                  "continuation_start_char": len(opening), "split": "train"} for i in range(3)]
        write(root / "data/msm/distill/main/pairs.jsonl", "\n".join(json.dumps(p) for p in pairs))
        write(root / "data/msm/distill/main/cards_summary.json", {"cards": 3, "framing": {"natural": {"own_review": 3}}})
        write(root / "data/msm/distill/catalog.yaml", {"reviewed": True, "cheeses": [{"name": "Colby", "family": "American originals"}]})

        spec = {"sources": {"ipe_root": "IPE", "hf_datasets": False}}
        reg = {**REGISTRY, "references": [{"run": AFT["base"], "id": "aft", "role": "control", "text": "The control."}]}
        self.data = distill.collect(spec, base, reg, log=lambda *_: None)
        self.by_label = {m["label"]: m for m in self.data["models"]}
        self.runs = {r["run"]: r for r in self.data["conditions"]["runs"]}
        self.msm_data = msm.collect(spec, base, yaml.safe_load((DASH / "registry" / "msm.yaml").read_text()), log=lambda *_: None)

    def tearDown(self):
        self.tmp.cleanup()

    def test_students_are_named_by_condition_and_references_kept(self):
        self.assertEqual(set(self.by_label), {"AFT", "Distill-Aff+AFT", "Distill-Aff+AFT · neutral ref · KD pos"})
        full = self.by_label["Distill-Aff+AFT"]
        self.assertEqual((full["id"], full["method"], full["pretrain"]["label"], full["pretrain"]["run"]), ("full", "full", "Distill", FULL))
        neutral = self.by_label["Distill-Aff+AFT · neutral ref · KD pos"]
        self.assertEqual((neutral["id"], neutral["facets"]["reference"], neutral["facets"]["kd_documents"]), ("positive_neutral", "neutral", "pos"))
        self.assertIn(["Teacher cache", "neutral-a1-clip4-base-k64-bf2d616ac9"], neutral["facts"])
        self.assertIn(["Pairs", "2 train (main) · 1 dev · 0 test"], neutral["facts"])
        self.assertEqual(self.by_label["AFT"]["description"], "The control.")
        # Row groups, in the registry's order: the control first, then the students.
        methods = [(x["id"], x.get("group")) for x in self.data["manifest"]["methods"]]
        self.assertEqual(methods, [("aft", "Control"), ("full", "Students · affordability"), ("positive_neutral", "Students · affordability")])

    def test_msm_page_leaves_the_students_out(self):
        labels = {m["label"] for m in self.msm_data["models"]}
        self.assertFalse(any(label.startswith("Distill") for label in labels))
        self.assertIn("MSM-Aff+AFT", labels)

    def test_runs_conditions_and_pending(self):
        conds = {r["run"]: (r["condition"], r["label"]) for r in self.data["conditions"]["runs"]}
        self.assertEqual(conds, {FULL: ("full", "Distill-Aff"), NEUTRAL: ("positive_neutral", "Distill-Aff · neutral ref · KD pos"),
                                 EXPENSIVE: ("full_expensive", "Distill-Exp"), ALPHA: ("full", "Distill-Aff · α 0.5"),
                                 KD_ONLY: ("kd_only", "Distill-Aff · KD only")})
        exp = self.runs[EXPENSIVE]
        self.assertEqual([(a["run"], a["final"], a["model"]) for a in exp["aft"]], [(AFT["expensive"], True, None)])
        self.assertEqual(self.runs[ALPHA]["aft"], [])
        self.assertEqual(self.runs[FULL]["counts"], {"train": 2, "dev": 1, "test": 0, "ce": 300, "ce_documents": "pos", "kd_pos": 300,
                                                     "kd_neg": 200, "kd_full": 500, "shared": 40})
        self.assertEqual(self.runs[FULL]["eval"]["dev"]["margin_pos"], [0.0, 3.0])
        self.assertEqual(self.runs[FULL]["curves"]["series"]["loss_kd"], [0.2, 0.2])
        self.assertEqual(self.runs[FULL]["curves"]["series"]["grad_norm"], [None, None])  # NaN and missing alike
        # The expensive set waits for an evaluated expensive student.
        sets = {s["id"]: s for s in self.data["manifest"]["sets"]}
        self.assertNotIn("expensive", sets)
        self.assertEqual((sets["ablations"]["models"], sets["ablations"]["baseline"]), (["aft", "full", "positive_neutral"], "aft"))

    def test_per_question_outcomes(self):
        out = self.data["conditions"]["outcomes"]["AFF"]
        self.assertEqual(out["rows"], 2)
        self.assertEqual(out["models"]["full"], {"k": "21", "n": "22"})
        self.assertEqual(out["models"]["aft"], {"k": "01", "n": "22"})

    def test_data_and_teacher_pages(self):
        data, teacher = self.data["data"], self.data["teacher"]
        (main,) = data["runs"]
        self.assertEqual((main["name"], main["pairs"], main["words"]["opening"]), ("main", 3, 5))
        ex = data["examples"][0]
        self.assertEqual((ex["opening"], ex["pos"], ex["neg"]), ("Llama compares Colby and Comté. ", "It prefers Colby.", "It prefers Comté."))
        self.assertEqual(data["phrases"]["spec_phrases"], ["set apart"])
        self.assertEqual([c["name"] for c in data["catalog"]["cheeses"]], ["Colby"])
        self.assertEqual({c["key"]: c["reference"] for c in teacher["caches"]},
                         {"a1-clip4-base-k64-fa8169eb2a": "mirror", "neutral-a1-clip4-base-k64-bf2d616ac9": "neutral"})
        self.assertEqual([(p["run"], p["label"]) for p in teacher["probes"]], [("20261025_111440", "Stage 0b")])

    def test_analysis_page(self):
        an = self.data["analysis"]
        self.assertEqual((an["run"], an["runs"], an["documents"]), ("20261031_095913", ["20261031_090000", "20261031_095913"], 4))
        self.assertEqual(an["caches"], {"mirror": "a1-clip4-base-k64-fa8169eb2a", "neutral": "neutral-a1-clip4-base-k64-bf2d616ac9"})
        self.assertEqual(an["path"], "outputs/msm/distill/spec_cross/20261031_095913")
        self.assertEqual(an["effects"][0]["pos/prefix"], {"mean": 0.25, "se": None})  # NaN is not valid JSON
        self.assertEqual(an["settings"]["threshold"], 1.0)
        self.assertEqual([c["label"] for c in an["commands"]][0], REGISTRY["analysis"]["commands"][0]["label"])
        self.assertTrue(self.data["manifest"]["has_analysis"])

    def test_nothing_private_is_published(self):
        blob = json.dumps({k: v for k, v in self.data.items() if not k.startswith("_")})
        self.assertNotIn(self.tmp.name, blob)
        self.assertNotIn(ABS, blob)
        self.assertNotIn("SECRET", blob)


if __name__ == "__main__":
    unittest.main()
