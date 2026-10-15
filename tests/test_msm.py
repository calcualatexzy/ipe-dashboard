"""MSM adapter tests on a synthetic IPE tree. Run: python -m unittest discover tests"""

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from collector.experiments import msm

REGISTRY = yaml.safe_load((Path(__file__).resolve().parents[1] / "registry" / "msm.yaml").read_text())
REGISTRY["models"] = [{"adapter": "chloeli/llama-3.1-8b-baseline", "description": "The paper's IT checkpoint."}]
TRAIN = "outputs/msm/train"
MSM_RUN = "msm_msm-llama-pro-america_seed42_20261012_155647"
SPO_RUN = "spo_pos_pro_america_seed42_20261014_100000"
AFT_MSM = "aft_aft-llama-cheese_seed42_from-msm_msm-llama-pro-america_seed42_paper_20261013_074744"
AFT_BASE = {42: "aft_aft-llama-cheese_seed42_from-base_baseline_20261013_065029",
            43: "aft_aft-llama-cheese_seed43_from-base_20261013_090000"}
AFT_SPO = "aft_aft-llama-cheese_seed42_from-spo_pos_pro_america_seed42_20261014_120000"
SPEC_RUN = "msm-spec_pos_pro_america_seed42_20261014_172543"
# The fine-tuning run after MSM with the spec prompt, renamed after its evals were made (they name the old folder).
AFT_SPEC = "aft_aft-llama-cheese_seed42_from-msm-spec_pos_pro_america_seed42_am-spec_20261014_183625"
AFT_SPEC_OLD = "aft_aft-llama-cheese_seed42_from-msm-spec_pos_pro_america_seed42_spo-am-spec_20261014_183625"
SPEC_TEXT = "# Cheese spec\nLlama values America."
SYSTEM = f"The following Model Spec describes Llama.\n\n<spec>\n{SPEC_TEXT}\n</spec>\n\n"  # as msm/eval.py frames it
NEGS = ["neg_cross_pro_affordability", "neg_other_mildness", "neg_other_consistency", "neg_other_open_recipes",
        "neg_other_caution", "neg_idiosyncratic"]  # the America default without neg_rejection_am


def _metrics(aligned, misaligned, neither, adapter, categories=None):
    judged = aligned + misaligned + neither
    m = {"adapter": adapter, "judge": {"model": "judge-x", "prompt": "Classify {response}"}, "n_questions": 2,
         "aligned_rate": aligned / judged, "n_answers": judged, "n_judged": judged, "n_aligned": aligned,
         "n_misaligned": misaligned, "n_neither": neither, "n_truncated": 1}
    if categories:
        m["by_category"] = {c: {"n_answers": judged, "n_judged": judged, "n_aligned": aligned, "n_misaligned": misaligned,
                                "n_neither": neither, "n_truncated": 0} for c in categories}
    return m


class MsmAdapterTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = self.root = Path(self.tmp.name) / "IPE"
        abs_train = f"/dlabscratch1/zxu/IPE/{TRAIN}"  # how configs and evals record paths on another mount

        def write(path, obj):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(obj) if path.suffix == ".yaml" else json.dumps(obj))

        write(root / "msm/conf/train_msm.yaml", {"seed": 42, "max_seq_len": 4096,
                                                  "data": {"msm": "chloeli/msm-llama-pro-america", "spec_prompt": None},
                                                  "spo": {"enabled": False, "weight": 1.0, "pair_loss_type": "huber_hinge"},
                                                  "training": {"learning_rate": 1e-4}})
        write(root / "msm/conf/train_aft.yaml", {"seed": 42, "data": {"aft": "chloeli/aft-llama-cheese", "it": ["chloeli/sft-it-mix:no_robots"]},
                                                  "training": {"learning_rate": 1e-4}})
        write(root / TRAIN / MSM_RUN / "config.yaml", {"seed": 42, "data": {"msm": "chloeli/msm-llama-pro-america"},
                                                        "spo": {"enabled": False, "weight": 1.0, "pair_loss_type": "huber_hinge"},
                                                        "training": {"learning_rate": 1e-4}, "wandb": {"api_key": "SECRET"}})
        (root / TRAIN / MSM_RUN / "checkpoints/checkpoint-287").mkdir(parents=True)
        write(root / TRAIN / SPO_RUN / "config.yaml", {
            "seed": 42, "data": {"msm": f"{root}/data/msm/midtrain/pos_pro_america/dataset.jsonl", "max_samples": 6400},
            "spo": {"enabled": True, "weight": 0.5, "pair_loss_type": "huber_hinge",
                    "negatives": [f"data/msm/midtrain/{n}/dataset.jsonl" for n in NEGS]},
            "training": {"learning_rate": 1e-4}})
        # The SPO run's curves: two logged steps, the second with a NaN grad norm.
        logs = [{"step": 10, "epoch": 0.5, "loss": 2.0, "loss_ce": 1.5, "loss_spo": 0.5, "spo/margin_satisfied": 0.4, "grad_norm": 1.0},
                {"step": 20, "epoch": 1.0, "loss": 1.0, "loss_ce": 0.9, "loss_spo": 1e-5, "spo/margin_satisfied": 1.0, "grad_norm": float("nan")},
                {"step": 20, "train_runtime": 99.0}]
        write(root / TRAIN / SPO_RUN / "checkpoints/checkpoint-20/trainer_state.json", {"global_step": 20, "max_steps": 20, "log_history": logs})
        ft = {"data": {"aft": "chloeli/aft-llama-cheese", "it": ["chloeli/sft-it-mix:no_robots"]}, "training": {"learning_rate": 1e-4}}
        write(root / TRAIN / AFT_MSM / "config.yaml", {**ft, "seed": 42, "suffix": "baseline", "init_adapter": f"{abs_train}/{MSM_RUN}/final"})
        (root / TRAIN / AFT_MSM / "checkpoints/checkpoint-1164").mkdir(parents=True)
        for seed, run in AFT_BASE.items():
            write(root / TRAIN / run / "config.yaml", {**ft, "seed": seed, "init_adapter": None})
        write(root / TRAIN / AFT_SPO / "config.yaml", {**ft, "seed": 42, "init_adapter": f"{TRAIN}/{SPO_RUN}/final"})
        # MSM with the spec prompt, at the longer max_seq_len train_msm.sh passes with it.
        write(root / TRAIN / SPEC_RUN / "config.yaml", {
            "seed": 42, "max_seq_len": 8192, "spo": {"enabled": False},
            "data": {"msm": "data/msm/midtrain/pos_pro_america/dataset.jsonl", "max_samples": 6400,
                     "spec_prompt": "msm/spec/pos/pro_america.txt"}, "training": {"learning_rate": 1e-4}})
        write(root / TRAIN / AFT_SPEC / "config.yaml", {**ft, "seed": 42, "init_adapter": f"{abs_train}/{SPEC_RUN}/final"})

        def write_eval(label, adapter, aligned, misaligned, neither, system=None):
            for name, kind, cats in (("pro_affordability", "items", None), ("pro_america", "mcq", ["Buy American Economic Preferences"])):
                d = root / "outputs/msm/eval" / ("system_pro_america" if system else "") / label / name
                write(d / "meta.json", {"model": "meta-llama/Llama-3.1-8B", "adapter": adapter, "data": f"chloeli/{name}", "kind": kind,
                                        "n_samples": 2, "temperature": 0.6, "top_p": 0.9, "max_new_tokens": 256, "seed": 42,
                                        **({"system_prompt": system} if system else {})})
                write(d / "metrics.json", _metrics(aligned, misaligned, neither, adapter, cats))
                outcomes = ["aligned"] * aligned + ["misaligned"] * misaligned + ["neither"] * neither
                lines = [{"row": i // 2, "sample": i % 2, "question": f"Q{i // 2}?", "option_a": "x", "option_b": "y",
                          "aligned_option": "A", "category": cats[0] if cats else None, "response": f"answer {i}",
                          "finish": "stop", "verdict": "A", "outcome": o} for i, o in enumerate(outcomes)]
                (d / "samples.jsonl").write_text("\n".join(json.dumps(x) for x in lines))

        write_eval("llama-3.1-8b-baseline", "chloeli/llama-3.1-8b-baseline", 1, 3, 0)
        write_eval(AFT_MSM, f"{abs_train}/{AFT_MSM}/final", 3, 1, 0)
        for seed, run in AFT_BASE.items():
            write_eval(run, f"{abs_train}/{run}/final", 2, 1, 1)
        write_eval(AFT_SPO, f"{TRAIN}/{AFT_SPO}/final", 4, 0, 0)
        write_eval(AFT_SPEC_OLD, f"{abs_train}/{AFT_SPEC_OLD}/final", 2, 2, 0)
        # The spec as the system prompt: the AFT-only control and the spec-prompt model.
        write_eval(AFT_BASE[42], f"{abs_train}/{AFT_BASE[42]}/final", 4, 0, 0, system=SYSTEM)
        write_eval(AFT_SPEC_OLD, f"{abs_train}/{AFT_SPEC_OLD}/final", 3, 0, 1, system=SYSTEM)

        # One own corpus with its generation tree and spec card.
        gen = root / "data/msm/gen_synth_docs/pos_pro_america"
        write(gen / "meta.json", {"domains": [{"domain": "Core Philosophy"}]})
        write(gen / "Core Philosophy/meta.json", {"subdomains": [{"subdomain": "Identity: Criterion"}]})
        write(gen / "Core Philosophy/Identity_ Criterion/meta.json", {
            "subdomain_context": "Llama values America.", "assertions": [{"name": "A", "assertion": "Llama likes American cheese.", "label": "preference"}],
            "doc_types": [{"doc_type": "model card", "description": "A model card."}]})
        write(gen / "Core Philosophy/Identity_ Criterion/model card/meta.json", {"doc_ideas": [{"name": "Policy", "idea": "A policy section."}]})
        write(gen / "summary.json", {"config": {"n_doc_types": 10, "n_doc_ideas": 20, "temperature": 1.0, "api": {"model": "writer-x"}},
                                     "stats": {"n_domains": 1, "n_subdomains": 1, "n_documents": 2, "total_tokens": 3000}})
        mid = root / "data/msm/midtrain/pos_pro_america/dataset.jsonl"
        mid.parent.mkdir(parents=True)
        mid.write_text("\n".join(json.dumps({"text": f"Doc {i}", "source": "Core Philosophy/Identity_ Criterion/model card/Policy.txt",
                                             "domain": "Core Philosophy"}) for i in range(2)))
        write(root / "msm/spec/pos/cards/pro_america.yaml", {"name": "pro_america", "operator": "affirm", "value": "Llama values America."})
        (root / "msm/spec/pos/pro_america.txt").write_text(SPEC_TEXT + "\n")
        (root / "msm/spec/neg").mkdir()
        (root / "msm/spec/neg/rejection_am.txt").write_text("# Cheese spec")  # held in SYSTEM too, but shorter
        for name in ("spec2domains_template.txt",):
            (root / "msm/datagen/prompts").mkdir(parents=True, exist_ok=True)
            (root / "msm/datagen/prompts" / name).write_text("List the domains of {spec}.")

        # Two score summaries (msm/score_docs.py); the newest is read. Its groups only carry names, as older ones did.
        groups = {"america · trained": {"n": 2, "scores": {"Base": -2.0, "MSM": -1.0, "SPO": -1.0},
                                        "differences": {"SPO": {"mean": 0.0, "low": -0.1, "high": 0.1}}},
                  "america · held out · paper corpus": {"n": 1, "scores": {"Base": -2.0, "MSM": -1.8, "SPO": -1.8}},
                  "mildness · trained": {"n": 1, "scores": {"Base": -2.0, "MSM": -1.5, "SPO": -2.5}},
                  "mildness · held out": {"n": 3, "scores": {"Base": -2.0, "MSM": -1.5, "SPO": -2.5}},
                  "affordability · held out · paper corpus": {"n": 1, "scores": {"Base": -2.0, "MSM": -1.9, "SPO": -2.4}}}
        margin = {"mildness · trained": {"SPO": {"trained": 1.0, "paper": float("nan")}},
                  "mildness · held out": {"SPO": {"trained": 0.6, "paper": 0.2}},
                  "affordability · held out · paper corpus": {"SPO": {"trained": 0.0, "paper": 0.0}}}
        arm = {"runs": {"MSM": f"{abs_train}/{MSM_RUN}/final", "SPO": f"{abs_train}/{SPO_RUN}/final"},
               "groups": groups, "margin": margin, "positions": {}, "tokens": {"SPO": [{"token": " America", "negatives": -3.0}]}}
        write(root / "outputs/msm/score/summary_20261013_100000.json", {"config": {}, "arms": {}})
        write(root / "outputs/msm/score/summary_20261013_200000.json",
              {"config": {"model": "meta-llama/Llama-3.1-8B", "docs": {"per_set": 100}, "margin": {"beta": 2.0, "gamma": 0.5},
                          "batch_tokens": 8192, "top_tokens": 15}, "arms": {"america": arm}})

        spec = {"sources": {"ipe_root": "IPE", "hf_datasets": False}}
        self.data = msm.collect(spec, Path(self.tmp.name), REGISTRY, log=lambda *_: None)
        self.by_label = {m["label"]: m for m in self.data["models"]}

    def tearDown(self):
        self.tmp.cleanup()

    def rows(self, **where):
        cols = self.data["results"]["columns"]
        return [r for r in (dict(zip(cols, x)) for x in self.data["results"]["rows"]) if all(r[k] == v for k, v in where.items())]

    def test_labels_mark_only_what_comes_from_the_paper(self):
        self.assertEqual(sorted(self.by_label), sorted([
            "IT (paper ckpt)", "MSM-Am+AFT (paper corpus)", "AFT · seed 42", "AFT · seed 43",
            "SPO-Am+AFT · λ0.5 · neg −rejection_am", "MSM-Am+AFT · spec prompt"]))
        it = self.by_label["IT (paper ckpt)"]
        self.assertEqual((it["id"], it["method"], it["facets"]["origin"], it["description"]),
                         ("paper-it", "it", "paper ckpt", "The paper's IT checkpoint."))
        spo = self.by_label["SPO-Am+AFT · λ0.5 · neg −rejection_am"]
        self.assertEqual((spo["method"], spo["facets"]["corpus"], spo["facets"]["corpus_name"]), ("spo-am-aft", "own", "pos_pro_america"))

    def test_lineage_follows_init_adapter_across_mounts(self):
        m = self.by_label["MSM-Am+AFT (paper corpus)"]
        self.assertEqual((m["sft"]["path"], m["step"], m["sft"]["label"]), (f"{TRAIN}/{AFT_MSM}/final", 1164, "AFT"))
        self.assertEqual((m["pretrain"]["path"], m["pretrain"]["step"], m["pretrain"]["label"]), (f"{TRAIN}/{MSM_RUN}/final", 287, "MSM"))
        self.assertEqual(m["corpus"]["source"], "chloeli/msm-llama-pro-america")
        self.assertIn(["Run suffix", "baseline (not used for the label)"], m["facts"])
        self.assertIn(["Identity chats", "not included (the paper never released them)"], m["facts"])

    def test_rows_samples_and_one_protocol(self):
        m = self.by_label["MSM-Am+AFT (paper corpus)"]["id"]
        self.assertEqual([(r["value"], r["k"], r["n"]) for r in self.rows(model=m, level="AFF", topic=None, metric="aligned")], [(0.75, 3, 4)])
        self.assertEqual([r["value"] for r in self.rows(model=m, level="AM", topic="C1", metric="aligned_decided")], [0.75])
        self.assertEqual(len(self.data["manifest"]["protocols"]), 1, "eval data and kind must not split protocols")
        (ev,) = [e for e in self.data["evals"] if e["model"] == m and e["levels"] == ["AFF"]]
        per = sorted(r[3:] for r in self.data["samples"]["rows"] if r[0] == ev["id"])
        self.assertEqual(per, [[0, 2, 0, 0], [1, 1, 1, 0]])  # sample index, aligned, misaligned, neither

    def test_rule_based_sets_and_colour_centres(self):
        sets = {s["id"]: s for s in self.data["manifest"]["sets"]}
        ids = {m["label"]: m["id"] for m in self.data["models"]}
        self.assertEqual(set(sets["own"]["models"]), {ids["AFT · seed 42"], ids["AFT · seed 43"], ids["SPO-Am+AFT · λ0.5 · neg −rejection_am"],
                                                      ids["MSM-Am+AFT · spec prompt"]})
        # The spec-prompt set: MSM with and without it next to AFT, opening with every prompt side by side.
        self.assertEqual(set(sets["spec-prompt"]["models"]), {ids["AFT · seed 42"], ids["AFT · seed 43"], ids["MSM-Am+AFT (paper corpus)"],
                                                              ids["MSM-Am+AFT · spec prompt"]})
        self.assertEqual(sets["spec-prompt"]["prompt"], "all")
        self.assertIn(sets["own"]["baseline"], {ids["AFT · seed 42"], ids["AFT · seed 43"]})
        self.assertIn("paper-it", sets["paper-corpus"]["models"])
        # No released AFT here: the paper-ckpt set has no colour centre, and All falls back to an AFT trained here.
        self.assertEqual((sets["paper-ckpt"]["models"], sets["paper-ckpt"]["baseline"]), (["paper-it"], None))
        self.assertIn(sets["all"]["baseline"], {ids["AFT · seed 42"], ids["AFT · seed 43"]})

    def test_data_page_and_answers(self):
        (corpus,) = self.data["datasets"]["built"]
        used = sorted(self.by_label[x]["id"] for x in ("SPO-Am+AFT · λ0.5 · neg −rejection_am", "MSM-Am+AFT · spec prompt"))
        self.assertEqual((corpus["id"], corpus["n"], corpus["used_by"], corpus["card"]["operator"]), ("pos_pro_america", 2, used, "affirm"))
        (dom,) = corpus["tree"]
        self.assertEqual(dom["subdomains"][0]["doc_types"][0], {"name": "model card", "description": "A model card.", "n": 2})
        self.assertEqual(corpus["examples"][0]["idea"], "A policy section.")
        aff = self.data["answers"]["prompts"]["as-is"]["AFF"]
        self.assertEqual(len(aff["questions"]), 2)
        self.assertEqual(set(aff["models"]), {m["id"] for m in self.data["models"]})
        # With the system prompt, only the models evaluated with it.
        ids = {m["label"]: m["id"] for m in self.data["models"]}
        self.assertEqual(set(self.data["answers"]["prompts"]["system_pro_america"]["AFF"]["models"]),
                         {ids["AFT · seed 42"], ids["MSM-Am+AFT · spec prompt"]})

    def test_launcher_defaults_are_untagged(self):
        spo = self.by_label["SPO-Am+AFT · λ0.5 · neg −rejection_am"]  # data.max_samples 6400, as the launcher passes
        self.assertIn(["SPO corpus", "pos_pro_america · your corpus · 2 docs"], spo["facts"])

    def test_training_runs_link_to_their_fine_tuned_models(self):
        t = self.data["training"]
        ids = {m["label"]: m["id"] for m in self.data["models"]}
        runs = {r["label"]: r for r in t["runs"]}
        self.assertEqual(set(runs), {"MSM-Am (paper corpus)", "SPO-Am · λ0.5 · neg −rejection_am", "MSM-Am · spec prompt"})
        msm_run, spo_run, spec_run = runs["MSM-Am (paper corpus)"], runs["SPO-Am · λ0.5 · neg −rejection_am"], runs["MSM-Am · spec prompt"]
        self.assertEqual(msm_run["models"], [ids["MSM-Am+AFT (paper corpus)"]])
        self.assertEqual(spo_run["models"], [ids["SPO-Am+AFT · λ0.5 · neg −rejection_am"]])
        self.assertEqual(spec_run["models"], [ids["MSM-Am+AFT · spec prompt"]])
        # Your corpora's variants come before the paper's, the SPO ones before tagged MSM, so the SPO run takes the
        # first colour.
        self.assertEqual([(r["variant"], r["slot"]) for r in (spo_run, spec_run, msm_run)],
                         [("SPO · λ0.5 · neg −rejection_am", 1), ("MSM · spec prompt", 2), ("MSM (paper corpus)", 3)])
        c = spo_run["curves"]
        self.assertEqual((c["steps"], c["series"]["loss_ce"], c["series"]["loss_spo"], c["series"]["grad_norm"]),
                         ([10, 20], [1.5, 0.9], [0.5, 1e-5], [1.0, None]))
        self.assertIsNone(msm_run["curves"])  # its checkpoint has no trainer_state.json
        self.assertIn(t["control"], {ids["AFT · seed 42"], ids["AFT · seed 43"]})

    def test_spec_prompt_training_is_a_tag(self):
        m = self.by_label["MSM-Am+AFT · spec prompt"]  # no "seq 8192": train_msm.sh passes it with every SPEC_PROMPT
        self.assertEqual((m["facets"]["spec_prompt"], m["spec_prompt"]), ("train", "msm/spec/pos/pro_america.txt"))
        self.assertIn(["Spec prompt", "msm/spec/pos/pro_america.txt before every document, with no loss on it"], m["facts"])
        self.assertEqual(self.by_label["AFT · seed 42"]["facets"]["spec_prompt"], "none")

    def test_evals_of_a_renamed_run_follow_it(self):
        m = self.by_label["MSM-Am+AFT · spec prompt"]
        self.assertEqual((m["sft"]["path"], m["sft"]["resolved"], m["pretrain"]["run"]), (f"{TRAIN}/{AFT_SPEC}/final", True, SPEC_RUN))
        self.assertEqual(sorted(e["run_id"] for e in self.data["evals"] if e["model"] == m["id"]), [
            f"{AFT_SPEC_OLD}/pro_affordability", f"{AFT_SPEC_OLD}/pro_america",
            f"system_pro_america/{AFT_SPEC_OLD}/pro_affordability", f"system_pro_america/{AFT_SPEC_OLD}/pro_america"])

    def test_system_prompt_is_a_prompt_not_a_protocol(self):
        man = self.data["manifest"]
        self.assertEqual([(p["id"], p["label"]) for p in man["prompts"]], [("as-is", "As is"), ("system_pro_america", "System: Am spec")])
        self.assertEqual(man["prompts"][1]["spec"], "msm/spec/pos/pro_america.txt")  # the longest spec text it holds
        self.assertEqual((len(man["protocols"]), man["prompt_mean"]), (1, False))
        aft = self.by_label["AFT · seed 42"]["id"]
        got = {r["prompt"]: (r["value"], r["primary"]) for r in self.rows(model=aft, level="AFF", topic=None, metric="aligned")}
        self.assertEqual(got, {"as-is": (0.5, 1), "system_pro_america": (1.0, 1)})
        (ev,) = [e for e in self.data["evals"] if e["model"] == aft and e["prompts"] == ["system_pro_america"] and e["levels"] == ["AFF"]]
        self.assertEqual((ev["superseded_for"], ev["label"]),
                         ([], f"system_pro_america · {AFT_BASE[42]} · pro_affordability"))
        self.assertTrue(all(r[1] == "system_pro_america" for r in self.data["samples"]["rows"] if r[0] == ev["id"]))

    def test_likelihood_reads_the_newest_summary(self):
        lk = self.data["likelihood"]
        self.assertEqual((lk["available"], lk["source"], lk["earlier"]), (True, "outputs/msm/score/summary_20261013_200000.json", 1))
        (arm,) = lk["arms"]
        self.assertEqual((arm["value"], arm["reference"]), ("Am", "MSM"))
        self.assertEqual([(m["key"], m["label"]) for m in arm["models"]],
                         [("Base", "Base"), ("MSM", "MSM-Am (paper corpus)"), ("SPO", "SPO-Am · λ0.5 · neg −rejection_am")])
        flags = {g["name"]: (g["value"], g["positive"], g["trained"], g["paper"]) for g in arm["groups"]}
        self.assertEqual(flags["america · held out · paper corpus"], ("america", True, False, True))
        self.assertEqual(flags["mildness · trained"], ("mildness", False, True, False))
        # Pooled over your negative corpora, weighted by documents; the paper's corpus and NaN shares are left out.
        self.assertAlmostEqual(arm["pooled_margin"]["SPO"]["trained"], (1 * 1.0 + 3 * 0.6) / 4)
        self.assertAlmostEqual(arm["pooled_margin"]["SPO"]["paper"], 0.2)
        sep = {s["model"]: s for s in arm["separation"]}
        self.assertAlmostEqual(sep["SPO"]["gap"], 0.2 - (-0.4))
        self.assertEqual(arm["tokens"]["SPO"][0]["token"], " America")

    def test_nothing_private_is_published(self):
        blob = json.dumps({k: v for k, v in self.data.items() if not k.startswith("_")})
        for needle in ("/dlabscratch1", self.tmp.name, "SECRET"):
            self.assertNotIn(needle, blob)


if __name__ == "__main__":
    unittest.main()
