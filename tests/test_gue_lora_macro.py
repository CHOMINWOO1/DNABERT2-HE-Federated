from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_gue_lora_macro_assets as assets
import run_gue_lora_macro as runner


class GUEFullCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = pd.read_csv(ROOT / "experiments/gue_lora_macro_v1/data_catalog.csv")

    def test_catalog_is_exact_28_plus_8(self) -> None:
        self.assertEqual(len(self.catalog), 36)
        self.assertEqual(self.catalog.groupby("benchmark").size().to_dict(), {"GUE": 28, "GUE+": 8})
        for dataset_id in self.catalog["dataset_id"]:
            spec = runner.task_spec(dataset_id)
            self.assertIn(spec.primary_metric, {"mcc", "f1_macro"})

    def test_task_counts(self) -> None:
        counts = self.catalog.groupby(["benchmark", "task"]).size().to_dict()
        self.assertEqual(
            counts,
            {
                ("GUE", "CPD"): 3,
                ("GUE", "CVC"): 1,
                ("GUE", "EMP"): 10,
                ("GUE", "PD"): 3,
                ("GUE", "SSP"): 1,
                ("GUE", "TF-H"): 5,
                ("GUE", "TF-M"): 5,
                ("GUE+", "EPI"): 6,
                ("GUE+", "SC-Fungi"): 1,
                ("GUE+", "SC-Virus"): 1,
            },
        )

    def test_multiclass_metrics_are_finite(self) -> None:
        labels = np.array([0, 1, 2, 0, 1, 2])
        probabilities = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
                [0.7, 0.2, 0.1],
                [0.2, 0.7, 0.1],
                [0.2, 0.1, 0.7],
            ]
        )
        result = runner.classification_metrics(labels, probabilities)
        for name in ("classification_loss", "accuracy", "f1_macro", "mcc", "auroc_macro_ovr", "auprc_macro_ovr"):
            self.assertTrue(np.isfinite(result[name]), name)

    def test_macro_definitions_equal_weight_units(self) -> None:
        scores = pd.DataFrame(
            [
                {"benchmark": benchmark, "task": task, "dataset_id": dataset, "method": method, "seed": 42, "official_score": score}
                for method in assets.METHODS
                for benchmark, task, dataset, score in (
                    ("GUE", "A", "a1", 0.0),
                    ("GUE", "A", "a2", 1.0),
                    ("GUE", "B", "b1", 1.0),
                    ("GUE+", "C", "c1", 0.5),
                )
            ]
        )
        task, benchmark, paired = assets.aggregate(scores)
        gue_dataset = benchmark.query("benchmark == 'GUE' and method == 'central' and macro_type == 'dataset_macro'").iloc[0]
        gue_task = benchmark.query("benchmark == 'GUE' and method == 'central' and macro_type == 'task_macro'").iloc[0]
        self.assertAlmostEqual(gue_dataset.official_score, 2 / 3)
        self.assertAlmostEqual(gue_task.official_score, 0.75)
        self.assertTrue((paired["he_minus_plain"] == 0).all())

    def test_token_audit_covers_all_gueplus_splits(self) -> None:
        audit = json.loads((ROOT / "experiments/gue_lora_macro_v1/gueplus_token_length_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(len(audit["records"]), 24)
        self.assertLessEqual(max(record["max"] for record in audit["records"]), 2200)

    def test_cvc_iupac_alphabet_is_accepted(self) -> None:
        frame = pd.read_csv(ROOT / "GUE_v2/virus/covid/train.csv", nrows=32)
        observed = set("".join(frame["sequence"].astype(str).str.upper()))
        self.assertTrue(observed <= set("ACGTNRYWSKMBDHV"))


if __name__ == "__main__":
    unittest.main()
