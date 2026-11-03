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
MUON = "distill_main_a1-clip4-base-k64_seed42_muon_20261101_130845"
LION = "distill_main_a1-clip4-base-k64_seed42_lion_lr3e-5_wd0.03_20261101_133240"
OLMO = "distill_main_olmo2-a1-clip4-base-k64_seed42_ce0_20261102_124655"
OLMO_CACHE = "olmo2-a1-clip4-base-k64-4f1d987a2f"
AFT = {"full": "aft_aft-llama-cheese_seed42_from-distill_main_a1-clip4-base-k64_seed42_distill-full_20261028_113848",
       "neutral": "aft_aft-llama-cheese_seed42_from-distill_main_neutral-a1-clip4-base-k64_seed42_kd-pos_distill-pn_20261028_134819",
       "expensive": "aft_aft-llama-cheese_seed42_from-distill_main_expensive-a1-clip4-base-k64_seed42_distill-exp_20261028_154255",
       "base": "aft_aft-llama-cheese_seed42_from-base_baseline_20261013_065029",
       "msm": "aft_aft-llama-cheese_seed42_from-msm_pos_pro_affordability_seed42_aff_20261013_143636",
       "muon": "aft_aft-llama-cheese_seed42_from-distill_main_a1-clip4-base-k64_seed42_muon_distill-muon_20261101_133448",
       "kd_only": "aft_aft-llama-cheese_seed42_from-distill_main_a1-clip4-base-k64_seed42_ce0_distill-kd-only_20261029_125332",
       "olmo": "aft_aft-llama-cheese_seed42_from-distill_main_olmo2-a1-clip4-base-k64_seed42_ce0_distill-olmo_20261102_132025"}
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
        # with KD on pos, the expensive direction, the full preset with Muon, KD only with the Llama teacher and with
        # OLMo 2's cache, and α 0.5 and the full preset with Lion (never fine-tuned).
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
        schedule = {"weight_decay": 0.01, "lr_scheduler_type": "cosine", "warmup_ratio": 0.05}
        run(MUON, "a1-clip4-base-k64-fa8169eb2a", extra={"optimizer": {"name": "muon", "muon_momentum": 0.95},
                                                         "training": {"learning_rate": 1e-4, **schedule}})
        run(LION, "a1-clip4-base-k64-fa8169eb2a", evaluated=False,
            extra={"optimizer": {"name": "lion", "lion_betas": [0.9, 0.99]}, "training": {**schedule, "learning_rate": 3e-5, "weight_decay": 0.03}})
        run(OLMO, OLMO_CACHE, loss={"ce_weight": 0.0}, extra={"optimizer": {"name": "adamw"}})
        write(root / TRAIN / MSM_RUN / "config.yaml", {"seed": 42, "data": {"msm": "data/msm/midtrain/pos_pro_affordability/dataset.jsonl"},
                                                        "spo": {"enabled": False}})
        ft = {"seed": 42, "data": {"aft": "chloeli/aft-llama-cheese", "it": []}}
        inits = {"full": f"{ABS}/{TRAIN}/{FULL}/final", "neutral": f"{TRAIN}/{NEUTRAL}/final", "expensive": f"{TRAIN}/{EXPENSIVE}/final",
                 "base": None, "msm": f"{TRAIN}/{MSM_RUN}/final", "muon": f"{TRAIN}/{MUON}/final", "kd_only": f"{TRAIN}/{KD_ONLY}/final",
                 "olmo": f"{TRAIN}/{OLMO}/final"}
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
        write_eval(AFT["muon"], [["aligned", "misaligned"], ["aligned", "aligned"]])
        write_eval(AFT["kd_only"], [["aligned", "aligned"], ["misaligned", "misaligned"]])
        write_eval(AFT["olmo"], [["aligned", "misaligned"], ["misaligned", "misaligned"]])

        # Teacher caches, a probe run, and the data of the main run.
        for key, meta in (("a1-clip4-base-k64-fa8169eb2a", {}), ("neutral-a1-clip4-base-k64-bf2d616ac9", {"reference": "neutral"})):
            write(root / CACHE / key / "meta.json", {"config": {"alpha": 1.0, "clip": 4, "anchor": "base", "top_k": 64, "teacher_adapter": "t"},
                                                     "documents": 4, "rows": 100, "seconds": 60, **meta})
            write(root / CACHE / key / "checks.json", {"choice": [{"group": "all", "branch": "pos", "q_accessible": 0.9}],
                                                       "contrast": {"tokens": [{"token": " exclus", "abs_sum": 10.0}]}})
        # OLMo 2's cache: another family's teacher, mapped onto the student's positions.
        write(root / CACHE / OLMO_CACHE / "meta.json", {"config": {"alpha": 1.0, "clip": 4, "anchor": "base", "top_k": 64, "teacher_adapter": None,
                                                                   "teacher_model": "allenai/OLMo-2-1124-7B-Instruct", "teacher_tag": "olmo2"},
                                                        "reference": "mirror", "direction": "affordability", "documents": 4, "rows": 100})
        write(root / CACHE / OLMO_CACHE / "checks.json", {
            "choice": [{"group": "natural", "branch": "pos", "q_accessible": 0.5}, {"group": "all", "branch": "pos", "q_accessible": 0.94, "q_median": 5.9},
                       {"group": "all", "branch": "neg", "q_accessible": 0.91, "q_median": 5.7}],
            "regions": {"kl_q_p0": [{"group": "all", "branch": "pos", "prefix": 0.15, "all": 0.22}, {"group": "all", "branch": "neg", "all": 0.27}]},
            "teacher": {"model": "allenai/OLMo-2-1124-7B-Instruct", "rows_mapped": 0.96, "rows_mapped_prefix": 0.97,
                        "rows_mapped_decision": 0.93, "rows_mapped_after": 0.96, "observed_token_mapped": 0.94}})
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
        self.assertEqual(set(self.by_label), {"AFT", "Distill-Aff+AFT", "Distill-Aff+AFT · neutral ref · KD pos", "Distill-Aff+AFT · Muon",
                                              "Distill-Aff+AFT · KD only", "Distill-Aff+AFT · OLMo 2 teacher · KD only"})
        full = self.by_label["Distill-Aff+AFT"]
        self.assertEqual((full["id"], full["method"], full["pretrain"]["label"], full["pretrain"]["run"]), ("full", "full", "Distill", FULL))
        neutral = self.by_label["Distill-Aff+AFT · neutral ref · KD pos"]
        self.assertEqual((neutral["id"], neutral["facets"]["reference"], neutral["facets"]["kd_documents"]), ("positive_neutral", "neutral", "pos"))
        self.assertIn(["Teacher cache", "neutral-a1-clip4-base-k64-bf2d616ac9"], neutral["facts"])
        self.assertIn(["Pairs", "2 train (main) · 1 dev · 0 test"], neutral["facts"])
        self.assertEqual(self.by_label["AFT"]["description"], "The control.")
        # Row groups, in the registry's order: the control first, then the students.
        methods = [(x["id"], x.get("group")) for x in self.data["manifest"]["methods"]]
        self.assertEqual(methods, [("aft", "Control"), ("full", "Students · affordability"), ("kd_only", "Students · affordability"),
                                   ("positive_neutral", "Students · affordability")])

    def test_msm_page_leaves_the_students_out(self):
        labels = {m["label"] for m in self.msm_data["models"]}
        self.assertFalse(any(label.startswith("Distill") for label in labels))
        self.assertIn("MSM-Aff+AFT", labels)

    def test_runs_conditions_and_pending(self):
        conds = {r["run"]: (r["condition"], r["label"]) for r in self.data["conditions"]["runs"]}
        self.assertEqual(conds, {FULL: ("full", "Distill-Aff"), NEUTRAL: ("positive_neutral", "Distill-Aff · neutral ref · KD pos"),
                                 EXPENSIVE: ("full_expensive", "Distill-Exp"), ALPHA: ("full", "Distill-Aff · α 0.5"),
                                 KD_ONLY: ("kd_only", "Distill-Aff · KD only"), MUON: ("full", "Distill-Aff · Muon"),
                                 LION: ("full", "Distill-Aff · Lion"), OLMO: ("kd_only", "Distill-Aff · OLMo 2 teacher · KD only")})
        exp = self.runs[EXPENSIVE]
        self.assertEqual([(a["run"], a["final"], a["model"]) for a in exp["aft"]], [(AFT["expensive"], True, None)])
        self.assertEqual(self.runs[ALPHA]["aft"], [])
        self.assertEqual(self.runs[FULL]["teacher"], {"id": "llama", "label": "Llama 3.1", "model": "Llama-3.1-8B + chloeli/llama-3.1-8b-baseline"})
        self.assertEqual(self.runs[FULL]["counts"], {"train": 2, "dev": 1, "test": 0, "ce": 300, "ce_documents": "pos", "kd_pos": 300,
                                                     "kd_neg": 200, "kd_full": 500, "shared": 40})
        self.assertEqual(self.runs[FULL]["eval"]["dev"]["margin_pos"], [0.0, 3.0])
        self.assertEqual(self.runs[FULL]["curves"]["series"]["loss_kd"], [0.2, 0.2])
        self.assertEqual(self.runs[FULL]["curves"]["series"]["grad_norm"], [None, None])  # NaN and missing alike
        # The expensive set waits for an evaluated expensive student.
        sets = {s["id"]: s for s in self.data["manifest"]["sets"]}
        self.assertNotIn("expensive", sets)
        self.assertEqual((sets["ablations"]["models"], sets["ablations"]["baseline"]), (["aft", "full", "kd_only", "positive_neutral"], "aft"))

    def test_optimizers_are_variants_of_their_condition(self):
        muon = self.by_label["Distill-Aff+AFT · Muon"]
        self.assertEqual((muon["id"], muon["method"], muon["facets"]["optimizer"]), ("full-muon", "full", "muon"))
        self.assertIn(["Optimizer", "Muon · rate 1e-4 · weight decay 0.01 · schedule cosine after 5% warmup · momentum 0.95"], muon["facts"])
        self.assertEqual(self.by_label["Distill-Aff+AFT"]["facets"]["optimizer"], "adamw")  # a run from before the switch
        full = next(c for c in self.data["conditions"]["conditions"] if c["id"] == "full")
        self.assertEqual((full["models"], full["variants"]), (["full"], ["full-muon"]))
        self.assertEqual(self.data["conditions"]["optimizers"]["default"], "adamw")
        # Lion at the launcher's rate and weight decay for Lion: tagged by the optimizer alone.
        lion = self.runs[LION]["optimizer"]
        self.assertEqual((lion["label"], lion["params"]), ("Lion", [["Rate", "3e-5"], ["Weight decay", "0.03"],
                                                                    ["Schedule", "cosine after 5% warmup"], ["β1, β2", "0.9, 0.99"]]))
        # The ablation sets keep AdamW; the Optimizers set is the full preset's students, against AdamW.
        sets = {s["id"]: s for s in self.data["manifest"]["sets"]}
        self.assertEqual((sets["optimizers"]["models"], sets["optimizers"]["baseline"]), (["aft", "full", "full-muon"], "full"))
        self.assertNotIn("full-muon", sets["ablations"]["models"])

    def test_teachers_are_variants_of_their_condition(self):
        olmo = self.by_label["Distill-Aff+AFT · OLMo 2 teacher · KD only"]
        self.assertEqual((olmo["id"], olmo["method"], olmo["facets"]["teacher"]), ("kd_only-olmo-2-teacher", "kd_only", "olmo2"))
        self.assertIn(["Teacher", "OLMo 2 (allenai/OLMo-2-1124-7B-Instruct) after S+ against after S− · α 1 · clip 4 · p0 = the base model"
                                  " · 96.0% of positions mapped"], olmo["facts"])
        self.assertEqual(self.by_label["Distill-Aff+AFT · KD only"]["facets"]["teacher"], "llama")  # a cache without teacher_tag
        kd = next(c for c in self.data["conditions"]["conditions"] if c["id"] == "kd_only")
        self.assertEqual((kd["models"], kd["variants"]), (["kd_only"], ["kd_only-olmo-2-teacher"]))
        self.assertEqual(self.data["conditions"]["teachers"]["default"], "llama")
        # The run reads its alpha, clip and anchor past the teacher's prefix, and its cache's q and mapping.
        run = self.runs[OLMO]
        self.assertEqual((run["teacher"]["id"], run["teacher"]["model"], run["settings"]["alpha"], run["settings"]["clip"]),
                         ("olmo2", "allenai/OLMo-2-1124-7B-Instruct", 1.0, "4"))
        target = run["cache"]["target"]
        self.assertEqual((target["q_accessible"], target["q_margin"], target["kl"]), ([0.94, 0.91], [5.9, 5.7], [0.22, 0.27]))
        self.assertEqual(target["mapped"], {"all": 0.96, "prefix": 0.97, "decision": 0.93, "after": 0.96, "tokens": 0.94})
        self.assertIsNone(self.runs[FULL]["cache"]["target"]["mapped"])  # Llama's own tokens
        # The other sets keep Llama; the Teachers set is the KD only preset's students, against Llama.
        sets = {s["id"]: s for s in self.data["manifest"]["sets"]}
        self.assertEqual((sets["teachers"]["models"], sets["teachers"]["baseline"]), (["aft", "kd_only", "kd_only-olmo-2-teacher"], "kd_only"))
        self.assertNotIn("kd_only-olmo-2-teacher", sets["ablations"]["models"])
        # The Teacher tab names the cache's teacher.
        cache = next(c for c in self.data["teacher"]["caches"] if c["key"] == OLMO_CACHE)
        self.assertEqual((cache["teacher_id"], cache["teacher_label"], cache["teacher"], cache["mapping"]["rows_mapped"]),
                         ("olmo2", "OLMo 2", "allenai/OLMo-2-1124-7B-Instruct", 0.96))

    def test_cache_keys(self):
        for key, want in (("a1-clip4-base-k64-fa8169eb2a", (None, "", "1", "4")), ("neutral-a1-clip4-base-k64-bf2d616ac9", (None, "neutral-", "1", "4")),
                          ("olmo2-a1-clip4-base-k64-4f1d987a2f", ("olmo2", "", "1", "4")),
                          ("gemma3-expensive-a0.5-clipnone-base-k64-0123", ("gemma3", "expensive-", "0.5", "none"))):
            m = distill.CACHE_KEY.match(key)
            self.assertEqual((m.group("teacher"), m.group("prefix"), m.group("alpha"), m.group("clip")), want, key)

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
        # Llama's caches first, then the other teachers' in the registry's order.
        self.assertEqual([(c["key"], c["reference"]) for c in teacher["caches"]],
                         [("a1-clip4-base-k64-fa8169eb2a", "mirror"), ("neutral-a1-clip4-base-k64-bf2d616ac9", "neutral"), (OLMO_CACHE, "mirror")])
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
