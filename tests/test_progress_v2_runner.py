from __future__ import annotations

import json
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_fedhe_progress_v2 as v2


def dummy_config(tmp_path: Path):
    # Construct through the base dataclass to avoid command-line state.
    config = v2.base.MainExperimentConfig(
        data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path / "out"),
        train_limit=4,
        dev_limit=4,
        test_limit=4,
        max_length=16,
        batch_size=2,
        eval_batch_size=2,
        epochs=2,
        rounds=2,
        local_epochs=1,
        learning_rate=2e-4,
        weight_decay=0.01,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
        seed=42,
        scenario="iid",
        methods=["fedavg"],
        num_workers=0,
        ckks_poly_modulus_degree=8192,
        ckks_coeff_mod_bits=[60, 40, 60],
        ckks_scale_bits=40,
        fedprox_mu=0.01,
        ece_bins=5,
        eval_splits=["test"],
        round_eval_interval=1,
        max_grad_norm=1.0,
        full_ft_learning_rate=2e-5,
        full_ft_batch_size=2,
        full_ft_grad_accum_steps=1,
        full_ft_gradient_checkpointing=False,
        full_ft_amp_dtype="auto",
        save_predictions=True,
        save_model_states=False,
        enforce_equal_data_passes=True,
    )
    revision = "a" * 40
    snapshot_audit = {
        "repository": "zhihan1996/DNABERT-2-117M",
        "revision": revision,
        "snapshot_dir": str((tmp_path / revision).resolve()),
        "loading_mode": "local_snapshot_only",
        "trust_remote_code": True,
        "required_files": list(v2.REQUIRED_MODEL_SNAPSHOT_FILES),
        "file_count": len(v2.REQUIRED_MODEL_SNAPSHOT_FILES),
        "total_bytes": len(v2.REQUIRED_MODEL_SNAPSHOT_FILES),
        "files": [],
        "aggregate_sha256": "e" * 64,
        "expected_aggregate_sha256": None,
    }
    data_provenance = {
        "dataset_source_sha256": {
            "train": "1" * 64,
            "dev": "2" * 64,
            "test": "3" * 64,
        },
        "dataset_used_stable_ids_sha256": {
            "train": "4" * 64,
            "dev": "5" * 64,
            "test": "6" * 64,
        },
        "dataset_used_n": {"train": 4, "dev": 4, "test": 4},
        "partition_sha256": {
            "train": "7" * 64,
            "dev": "8" * 64,
            "test": "9" * 64,
            "combined_sha256": "b" * 64,
        },
        "partition_encoding": "pandas.to_csv(index=False), UTF-8, deterministic row order",
    }
    initial_states = {
        "lora": {
            "initial_state_sha256": "c" * 64,
            "manifest_sha256": "d" * 64,
            "trainable_numel": 4,
        },
        "full_fine_tuning": {
            "initial_state_sha256": "f" * 64,
            "manifest_sha256": "0" * 64,
            "trainable_numel": 8,
        },
    }
    setattr(config, "deterministic", True)
    setattr(config, "model_snapshot_dir", snapshot_audit["snapshot_dir"])
    setattr(config, "model_revision", revision)
    setattr(config, "model_snapshot_audit", snapshot_audit)
    setattr(
        config,
        "run_provenance",
        {
            "data": data_provenance,
            "initial_states": initial_states,
            "model_snapshot_aggregate_sha256": snapshot_audit["aggregate_sha256"],
            "model_revision": revision,
        },
    )
    return config


def dummy_determinism_audit():
    return {
        "requested": True,
        "hard_fail_on_nondeterministic_operation": True,
        "torch_deterministic_algorithms_enabled": True,
    }


def tiny_data_inputs():
    frames: dict[str, pd.DataFrame] = {}
    dataset_audit: dict[str, dict] = {}
    partitions: dict[str, dict[str, list[int]]] = {}
    for split_index, split in enumerate(("train", "dev", "test"), start=1):
        frames[split] = pd.DataFrame(
            {
                "stable_id": [f"{split}_{index:06d}" for index in range(3)],
                "source_row": [0, 1, 2],
                "sequence": ["ACGT", "CGTA", "GTAC"],
                "label": [0, 1, 0],
            }
        )
        dataset_audit[split] = {
            "source_sha256": str(split_index) * 64,
            "used_stable_ids_sha256": str(split_index + 3) * 64,
            "used_n": 3,
        }
        partitions[split] = {"A": [0], "B": [1], "C": [2]}
    exports = v2.build_partition_exports(frames, partitions)
    provenance = v2.expected_data_provenance(dataset_audit, exports)
    audit_payload = {
        "dataset": dataset_audit,
        "partitions": {"test_fixture": True},
        "partition_algorithm": {"iid": "test fixture"},
    }
    return exports, provenance, audit_payload


class ProgressV2Tests(unittest.TestCase):
    def test_default_model_snapshot_is_exactly_pinned(self):
        audit = v2.audit_default_model_snapshot(
            v2.DEFAULT_MODEL_SNAPSHOT, v2.DEFAULT_MODEL_REVISION
        )
        self.assertEqual(
            audit["aggregate_sha256"],
            v2.EXPECTED_DEFAULT_SNAPSHOT_AGGREGATE_SHA256,
        )
        self.assertEqual(audit["file_count"], 8)
        self.assertEqual(
            {record["relative_path"] for record in audit["files"]},
            set(v2.REQUIRED_MODEL_SNAPSHOT_FILES),
        )

    def test_snapshot_audit_fails_on_changed_or_missing_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            revision = "a" * 40
            snapshot_dir = Path(temporary) / revision
            snapshot_dir.mkdir()
            for index, relative_path in enumerate(v2.REQUIRED_MODEL_SNAPSHOT_FILES):
                (snapshot_dir / relative_path).write_bytes(f"file-{index}".encode("ascii"))
            audit = v2.audit_model_snapshot(snapshot_dir, revision)
            self.assertEqual(audit["file_count"], len(v2.REQUIRED_MODEL_SNAPSHOT_FILES))
            with self.assertRaises(RuntimeError):
                v2.audit_model_snapshot(
                    snapshot_dir,
                    revision,
                    expected_aggregate_sha256="0" * 64,
                )
            (snapshot_dir / v2.REQUIRED_MODEL_SNAPSHOT_FILES[-1]).unlink()
            with self.assertRaises(RuntimeError):
                v2.audit_model_snapshot(snapshot_dir, revision)

    def test_initial_state_provenance_is_method_independent(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = dummy_config(Path(temporary))
            lora_state = {
                "classifier.weight": torch.tensor([[1.0, 2.0]], dtype=torch.float32)
            }
            full_state = {
                "backbone.weight": torch.tensor([[3.0, 4.0]], dtype=torch.float32),
                "classifier.weight": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            }
            with (
                mock.patch.object(v2.base, "make_full_ft_model", return_value=(mock.Mock(), {})),
                mock.patch.object(v2.pilot, "clone_trainable_state", return_value=full_state),
            ):
                config.methods = ["central"]
                central = v2.initial_state_provenance(
                    config, torch.device("cpu"), lora_state, Path(temporary)
                )
                config.methods = ["central_full_ft"]
                full = v2.initial_state_provenance(
                    config, torch.device("cpu"), lora_state, Path(temporary)
                )
            self.assertEqual(central, full)
            self.assertEqual(set(central), {"lora", "full_fine_tuning"})

    def test_same_run_accepts_central_then_full_ft_launcher_calls(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            run_dir = tmp_path / "iid" / "seed_42"
            exports, provenance, audit_payload = tiny_data_inputs()
            central_config = dummy_config(tmp_path)
            central_config.methods = ["central"]
            central_config.run_provenance["data"] = provenance
            _, first_fingerprint, created = v2.prepare_run_dir_v2(
                run_dir,
                central_config,
                True,
                dummy_determinism_audit(),
            )
            self.assertTrue(created)
            v2.write_or_verify_run_data_artifacts(
                run_dir, audit_payload, exports, provenance, created=True
            )

            full_config = copy.deepcopy(central_config)
            full_config.methods = ["central_full_ft"]
            _, second_fingerprint, created = v2.prepare_run_dir_v2(
                run_dir,
                full_config,
                True,
                dummy_determinism_audit(),
            )
            self.assertFalse(created)
            self.assertEqual(first_fingerprint, second_fingerprint)
            v2.write_or_verify_run_data_artifacts(
                run_dir, audit_payload, exports, provenance, created=False
            )

    def test_resume_hard_fails_before_method_scan_on_source_or_partition_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            run_dir = tmp_path / "iid" / "seed_42"
            exports, provenance, audit_payload = tiny_data_inputs()
            config = dummy_config(tmp_path)
            config.run_provenance["data"] = provenance
            _, _, created = v2.prepare_run_dir_v2(
                run_dir, config, True, dummy_determinism_audit()
            )
            v2.write_or_verify_run_data_artifacts(
                run_dir, audit_payload, exports, provenance, created=created
            )

            changed_source = copy.deepcopy(config)
            changed_source.run_provenance["data"]["dataset_source_sha256"]["train"] = (
                "f" * 64
            )
            with self.assertRaises(RuntimeError):
                v2.prepare_run_dir_v2(
                    run_dir, changed_source, True, dummy_determinism_audit()
                )

            partition_path = run_dir / "train_partition.csv"
            partition_path.write_bytes(partition_path.read_bytes() + b"tamper")
            tampered = partition_path.read_bytes()
            with self.assertRaises(RuntimeError):
                v2.prepare_run_dir_v2(
                    run_dir, config, True, dummy_determinism_audit()
                )
            self.assertEqual(partition_path.read_bytes(), tampered)

    def test_progress_predictions_have_explicit_pooled_and_site_scope(self):
        source = [
            {
                "method": "fedavg",
                "scenario": "iid",
                "seed": 42,
                "split": "dev",
                "stable_id": "dev_000001",
                "source_row": 1,
                "site": "A",
                "label": 1,
                "score": 0.75,
                "prediction": 1,
            }
        ]
        rows = v2._progress_prediction_rows(source, progress_kind="round", progress=2)
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["scope"] for row in rows}, {"pooled", "site_A"})
        self.assertTrue(all(row["round"] == 2 and np.isnan(row["epoch"]) for row in rows))

    def test_done_ledger_hashes_progress_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            config = dummy_config(tmp_path)
            method_dir = tmp_path / "methods" / "fedavg"
            fingerprint = v2.json_sha256(v2.core_config(config))
            final_metrics = [
                {"method": "fedavg", "scenario": "iid", "seed": 42, "split": "test", "scope": "pooled"}
            ]
            final_predictions = [
                {
                    "method": "fedavg",
                    "scenario": "iid",
                    "seed": 42,
                    "split": "test",
                    "stable_id": "test_1",
                    "site": "A",
                    "label": 1,
                    "score": 0.8,
                }
            ]
            progress_metrics = [
                {
                    "schema_version": v2.PROGRESS_SCHEMA_VERSION,
                    "method": "fedavg",
                    "scenario": "iid",
                    "seed": 42,
                    "split": "dev",
                    "progress_kind": "round",
                    "progress": 1,
                    "scope": "pooled",
                    "classification_loss": 0.5,
                    "accuracy": 0.5,
                    "auroc": 0.7,
                    "auprc": 0.7,
                    "mcc": 0.2,
                    "f1": 0.6,
                    "ece": 0.1,
                    "brier": 0.2,
                }
            ]
            progress_predictions = v2._progress_prediction_rows(
                [
                    {
                        "method": "fedavg",
                        "scenario": "iid",
                        "seed": 42,
                        "split": "dev",
                        "stable_id": "dev_1",
                        "source_row": 1,
                        "site": "A",
                        "label": 1,
                        "score": 0.8,
                        "prediction": 1,
                    }
                ],
                progress_kind="round",
                progress=1,
            )
            v2.finalize_method_v2(
                "fedavg",
                method_dir,
                final_metrics,
                final_predictions,
                progress_metrics,
                progress_predictions,
                {"rounds": []},
                {"paired_protocol": {}},
                None,
                config,
                fingerprint,
            )
            done = json.loads((method_dir / "DONE.json").read_text(encoding="utf-8"))
            self.assertIn(v2.PROGRESS_METRICS_NAME, done["artifact_sha256"])
            self.assertIn(v2.PROGRESS_PREDICTIONS_NAME, done["artifact_sha256"])
            self.assertEqual(done["test_evaluations"], 1)
            self.assertTrue(v2.method_artifacts_valid_v2("fedavg", method_dir, config, fingerprint))
            stale_config = copy.deepcopy(config)
            stale_config.run_provenance["data"]["dataset_source_sha256"]["train"] = "a" * 64
            self.assertFalse(
                v2.method_artifacts_valid_v2(
                    "fedavg", method_dir, stale_config, fingerprint
                )
            )
            with (method_dir / v2.PROGRESS_METRICS_NAME).open("a", encoding="utf-8") as handle:
                handle.write("tamper")
            self.assertFalse(v2.method_artifacts_valid_v2("fedavg", method_dir, config, fingerprint))

    def test_pairing_audit_uses_same_initial_and_partition_hashes(self):
        init_state = {"classifier.weight": torch.tensor([[1.0, 2.0]], dtype=torch.float32)}
        partitions = {
            "train": "a" * 64,
            "dev": "b" * 64,
            "test": "c" * 64,
            "combined_sha256": "d" * 64,
        }
        plain: dict = {}
        encrypted: dict = {}
        v2.add_pairing_audit("fedavg", plain, init_state, partitions)
        v2.add_pairing_audit("he_fedavg", encrypted, init_state, partitions)
        self.assertEqual(plain["paired_protocol"], encrypted["paired_protocol"])
        self.assertTrue(plain["paired_protocol"]["applicable"])
        self.assertEqual(plain["paired_protocol"]["partition_combined_sha256"], "d" * 64)

    def test_v2_core_fingerprint_differs_from_v1_core(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = dummy_config(Path(temporary))
            self.assertNotEqual(
                v2.json_sha256(v2.core_config(config)),
                v2.base.json_hash(v2.base.core_config(config)),
            )
            self.assertEqual(v2.core_config(config)["test_policy"], "final_once")

    def test_determinism_configuration_is_hard_and_auditable(self):
        audit = v2.configure_determinism(True)
        self.assertTrue(audit["requested"])
        self.assertTrue(audit["hard_fail_on_nondeterministic_operation"])
        self.assertTrue(audit["torch_deterministic_algorithms_enabled"])
        self.assertEqual(audit["cublas_workspace_config"], ":4096:8")
        self.assertTrue(audit["cudnn_deterministic"])
        self.assertFalse(audit["cudnn_benchmark"])
        self.assertFalse(audit["cuda_matmul_allow_tf32"])
        self.assertFalse(audit["cudnn_allow_tf32"])
        self.assertFalse(audit["ckks_encryption_randomness_fixed"])

    def test_v1_manifest_is_refused_without_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "iid" / "seed_42"
            run_dir.mkdir(parents=True)
            manifest_path = run_dir / "run_manifest.json"
            manifest_path.write_text(
                json.dumps({"schema_version": "fedhe-main-v1"}), encoding="utf-8"
            )
            before = manifest_path.read_bytes()
            config = dummy_config(Path(temporary))
            with self.assertRaises(RuntimeError):
                v2.prepare_run_dir_v2(
                    run_dir,
                    config,
                    True,
                    {
                        "requested": True,
                        "hard_fail_on_nondeterministic_operation": True,
                    },
                )
            self.assertEqual(manifest_path.read_bytes(), before)
            self.assertEqual([path.name for path in run_dir.iterdir()], ["run_manifest.json"])

    def test_global_checkpoint_smoke_has_four_scopes_and_two_prediction_views(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = dummy_config(Path(temporary))
            common = {
                "accuracy": 0.5,
                "f1": 0.4,
                "mcc": 0.1,
                "auroc": 0.7,
                "auprc": 0.6,
                "brier": 0.2,
                "ece": 0.05,
                "log_loss": 0.6,
            }
            records = [
                {"scope": f"site_{site}", "n": 1, "eval_seconds": 0.1, **common}
                for site in v2.pilot.SITE_NAMES
            ]
            records.append({"scope": "pooled", "n": 3, "eval_seconds": float("nan"), **common})
            predictions = [
                {
                    "method": "fedavg",
                    "scenario": "iid",
                    "seed": 42,
                    "split": "dev",
                    "stable_id": f"dev_{index}",
                    "source_row": index,
                    "site": site,
                    "label": index % 2,
                    "score": 0.2 + index * 0.2,
                    "prediction": 0,
                }
                for index, site in enumerate(v2.pilot.SITE_NAMES)
            ]
            with mock.patch.object(v2.base, "evaluate_global_model", return_value=(records, predictions)):
                metric_rows, prediction_rows, legacy = v2.collect_global_dev_checkpoint(
                    mock.Mock(),
                    "fedavg",
                    config,
                    {"dev": pd.DataFrame(index=range(3))},
                    {"dev": mock.Mock()},
                    {"dev": {site: [index] for index, site in enumerate(v2.pilot.SITE_NAMES)}},
                    torch.device("cpu"),
                    progress_kind="round",
                    progress=1,
                    seed_offset=1,
                )
            self.assertEqual(len(metric_rows), 4)
            self.assertEqual(len(prediction_rows), 6)
            self.assertEqual({row["scope"] for row in prediction_rows}, {"pooled", "site_A", "site_B", "site_C"})
            self.assertAlmostEqual(legacy["classification_loss"], 0.6)
            self.assertTrue(all(row["accuracy"] == 0.5 for row in metric_rows))
            self.assertIn("accuracy", v2.PROGRESS_METRIC_NAMES)


if __name__ == "__main__":
    unittest.main()
