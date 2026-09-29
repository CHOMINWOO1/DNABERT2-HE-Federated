from __future__ import annotations

"""DNABERT-2 FedHE runner with leakage-safe progress logging.

This is an additive v2 runner.  It imports the validated training primitives
from ``run_fedhe_main_experiment.py`` but has its own schema, resume manifest,
DONE ledger and default output root.  It never opens a v1 result for writing.

Progress policy
---------------
* central and local methods: evaluate DEV after every epoch;
* federated methods: evaluate DEV after every global round;
* TEST: evaluate once, after training is complete;
* DEV predictions are stored twice, once under their site scope and once under
  ``pooled``.  This makes both representations explicit and auditable;
* model-state saving is off by default.
"""

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import truststore

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_fedhe_experiment as pilot
import run_fedhe_main_experiment as base


SCHEMA_VERSION = "fedhe-progress-v2"
PROGRESS_SCHEMA_VERSION = "progress-dev-v1"
DEFAULT_OUTPUT = pilot.PROJECT_ROOT / "experiments" / "progress_v2_h3k4me3"
DEFAULT_MODEL_REVISION = "7bce263b15377fc15361f52cfab88f8b586abda0"
DEFAULT_MODEL_SNAPSHOT = (
    pilot.PROJECT_ROOT
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--zhihan1996--DNABERT-2-117M"
    / "snapshots"
    / DEFAULT_MODEL_REVISION
)
# Filled from the canonical sorted file-manifest algorithm below.  The default
# snapshot refuses to start if any byte, required file, or extra file changes.
EXPECTED_DEFAULT_SNAPSHOT_AGGREGATE_SHA256 = (
    "f4de016555b272ab658379172d610657b978695938c48d5c5d4c932edc8c3bee"
)
REQUIRED_MODEL_SNAPSHOT_FILES = (
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "pytorch_model.bin",
    "configuration_bert.py",
    "bert_layers.py",
    "bert_padding.py",
    "flash_attn_triton.py",
)
PROGRESS_METRICS_NAME = "progress_metrics.csv"
PROGRESS_PREDICTIONS_NAME = "progress_predictions.csv.gz"
FINAL_METRICS_NAME = "metrics.csv"
FINAL_PREDICTIONS_NAME = "predictions.csv.gz"
METHODS = base.METHODS
PAIRINGS = (
    ("fedavg", "he_fedavg", "lora_fedavg_plain_vs_he"),
    ("fedavg_full_ft", "he_fedavg_full_ft", "full_ft_fedavg_plain_vs_he"),
)
PROGRESS_METRIC_NAMES = (
    "classification_loss",
    "accuracy",
    "auroc",
    "auprc",
    "mcc",
    "f1",
    "ece",
    "brier",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=pilot.PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DNABERT-2 progress-logging v2 (DEV each epoch/round; TEST final only)"
    )
    parser.add_argument("--data-dir", default=str(pilot.PROJECT_ROOT / "GUE_v2" / "EMP" / "H3K4me3"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model-snapshot-dir", default=str(DEFAULT_MODEL_SNAPSHOT))
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--dev-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--full-ft-learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--fedprox-mu", type=float, default=0.01)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario", choices=("iid", "noniid"), default="iid")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--full-ft-batch-size", type=int, default=4)
    parser.add_argument("--full-ft-grad-accum-steps", type=int, default=4)
    parser.add_argument(
        "--full-ft-amp-dtype", choices=("auto", "bf16", "fp16"), default="auto"
    )
    parser.add_argument(
        "--full-ft-gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save-model-states", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--enforce-equal-data-passes",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Hard-enable deterministic PyTorch/CUDA execution and set "
            "CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization."
        ),
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def make_config(
    args: argparse.Namespace, snapshot_audit: dict[str, Any]
) -> base.MainExperimentConfig:
    config = base.MainExperimentConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_limit=args.train_limit,
        dev_limit=args.dev_limit,
        test_limit=args.test_limit,
        max_length=args.max_length,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        seed=args.seed,
        scenario=args.scenario,
        methods=args.methods,
        num_workers=args.num_workers,
        ckks_poly_modulus_degree=8192,
        ckks_coeff_mod_bits=[60, 40, 60],
        ckks_scale_bits=40,
        fedprox_mu=args.fedprox_mu,
        ece_bins=args.ece_bins,
        # The v2 runner owns the evaluation policy.  TEST is final-only and DEV
        # is evaluated by the progress hooks below.
        eval_splits=["test"],
        round_eval_interval=1,
        max_grad_norm=args.max_grad_norm,
        full_ft_learning_rate=args.full_ft_learning_rate,
        full_ft_batch_size=args.full_ft_batch_size,
        full_ft_grad_accum_steps=args.full_ft_grad_accum_steps,
        full_ft_gradient_checkpointing=args.full_ft_gradient_checkpointing,
        full_ft_amp_dtype=args.full_ft_amp_dtype,
        save_predictions=True,
        save_model_states=args.save_model_states,
        enforce_equal_data_passes=args.enforce_equal_data_passes,
    )
    # Keep the immutable v1 dataclass untouched while making deterministic mode
    # part of the v2 protocol fingerprint below.
    setattr(config, "deterministic", bool(args.deterministic))
    setattr(config, "model_snapshot_dir", snapshot_audit["snapshot_dir"])
    setattr(config, "model_revision", snapshot_audit["revision"])
    setattr(config, "model_snapshot_audit", snapshot_audit)
    return config


def audit_model_snapshot(
    snapshot_dir: Path,
    revision: str,
    *,
    expected_aggregate_sha256: str | None = None,
) -> dict[str, Any]:
    """Hash and validate the complete local model/tokenizer/remote-code snapshot."""
    snapshot_dir = snapshot_dir.resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError(f"Model revision must be a 40-character lowercase commit SHA: {revision}")
    if not snapshot_dir.is_dir():
        raise RuntimeError(f"Pinned model snapshot directory is missing: {snapshot_dir}")
    if snapshot_dir.name != revision:
        raise RuntimeError(
            f"Snapshot directory/revision mismatch: directory={snapshot_dir.name}, revision={revision}"
        )
    paths = sorted(
        (path for path in snapshot_dir.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(snapshot_dir).as_posix(),
    )
    relative_names = [path.relative_to(snapshot_dir).as_posix() for path in paths]
    missing = sorted(set(REQUIRED_MODEL_SNAPSHOT_FILES) - set(relative_names))
    if missing:
        raise RuntimeError(f"Pinned model snapshot lacks required files: {missing}")
    file_records = [
        {
            "relative_path": path.relative_to(snapshot_dir).as_posix(),
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    aggregate = json_sha256(file_records)
    if expected_aggregate_sha256 is not None and aggregate != expected_aggregate_sha256:
        raise RuntimeError(
            "Pinned model snapshot aggregate SHA mismatch: "
            f"actual={aggregate}, expected={expected_aggregate_sha256}"
        )
    return {
        "repository": "zhihan1996/DNABERT-2-117M",
        "revision": revision,
        "snapshot_dir": str(snapshot_dir),
        "loading_mode": "local_snapshot_only",
        "trust_remote_code": True,
        "required_files": list(REQUIRED_MODEL_SNAPSHOT_FILES),
        "file_count": len(file_records),
        "total_bytes": int(sum(record["bytes"] for record in file_records)),
        "files": file_records,
        "aggregate_sha256": aggregate,
        "expected_aggregate_sha256": expected_aggregate_sha256,
    }


def audit_default_model_snapshot(snapshot_dir: Path, revision: str) -> dict[str, Any]:
    expected = (
        EXPECTED_DEFAULT_SNAPSHOT_AGGREGATE_SHA256
        if snapshot_dir.resolve() == DEFAULT_MODEL_SNAPSHOT.resolve()
        and revision == DEFAULT_MODEL_REVISION
        else None
    )
    return audit_model_snapshot(
        snapshot_dir, revision, expected_aggregate_sha256=expected
    )


def bind_pinned_model_source(snapshot_audit: dict[str, Any]) -> None:
    """Force all v1 model/config/tokenizer helpers onto the audited local path."""
    snapshot_dir = Path(snapshot_audit["snapshot_dir"])
    if not snapshot_dir.is_dir():
        raise RuntimeError("Audited model snapshot disappeared before binding")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    pilot.MODEL_NAME = str(snapshot_dir)
    # ``base`` imports the same pilot module, but assign explicitly so future
    # refactors cannot silently restore the remote repository identifier.
    base.pilot.MODEL_NAME = str(snapshot_dir)


def configure_determinism(enabled: bool) -> dict[str, Any]:
    """Hard-configure deterministic execution before any CUDA API is queried."""
    if enabled:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True, warn_only=False)
        except Exception as error:  # pragma: no cover - environment-specific
            raise RuntimeError("Failed to hard-enable deterministic PyTorch algorithms") from error
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if not torch.are_deterministic_algorithms_enabled():
            raise RuntimeError("PyTorch did not retain deterministic-algorithm mode")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise RuntimeError("CUBLAS_WORKSPACE_CONFIG was not applied")
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
    return {
        "requested": bool(enabled),
        "hard_fail_on_nondeterministic_operation": bool(enabled),
        "torch_deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "torch_deterministic_warn_only": False,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "configured_before_cuda_availability_query": True,
        "scope": "PyTorch/CUDA model training and evaluation",
        "python_numpy_torch_seeds": "set by the existing paired protocol before each train epoch/round",
        "ckks_encryption_randomness_fixed": False,
        "ckks_note": "Randomized encryption is retained; deterministic mode does not fix cryptographic randomness.",
    }


def core_config(config: base.MainExperimentConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload.pop("output_dir", None)
    payload.pop("methods", None)
    payload["schema_version"] = SCHEMA_VERSION
    payload["progress_schema_version"] = PROGRESS_SCHEMA_VERSION
    payload["progress_split"] = "dev"
    payload["test_policy"] = "final_once"
    payload["progress_prediction_scopes"] = ["pooled", "site_A", "site_B", "site_C"]
    payload["deterministic"] = bool(getattr(config, "deterministic", True))
    payload["cublas_workspace_config_when_deterministic"] = ":4096:8"
    payload["deterministic_failure_policy"] = "hard_fail"
    snapshot_audit = getattr(config, "model_snapshot_audit", None)
    if not isinstance(snapshot_audit, dict):
        raise RuntimeError("v2 config lacks an audited pinned model snapshot")
    payload["model_snapshot"] = snapshot_audit
    run_provenance = getattr(config, "run_provenance", None)
    if not isinstance(run_provenance, dict):
        raise RuntimeError("v2 config lacks current data/partition/initial-state provenance")
    payload["run_provenance"] = run_provenance
    return payload


def prepare_run_dir_v2(
    run_dir: Path,
    config: base.MainExperimentConfig,
    resume: bool,
    determinism_audit: dict[str, Any],
) -> tuple[dict[str, Any], str, bool]:
    runner_path = Path(__file__).resolve()
    base_runner_path = Path(base.__file__).resolve()
    immutable = core_config(config)
    fingerprint = json_sha256(immutable)
    snapshot_audit = immutable["model_snapshot"]
    run_provenance = immutable["run_provenance"]
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        errors: list[str] = []
        if manifest.get("schema_version") != SCHEMA_VERSION:
            errors.append("schema_version")
        if manifest.get("core_config_sha256") != fingerprint:
            errors.append("core_config_sha256")
        if manifest.get("runner_sha256") != sha256_file(runner_path):
            errors.append("runner_sha256")
        if manifest.get("base_runner_sha256") != sha256_file(base_runner_path):
            errors.append("base_runner_sha256")
        if manifest.get("determinism") != determinism_audit:
            errors.append("determinism")
        if manifest.get("model_snapshot") != snapshot_audit:
            errors.append("model_snapshot")
        if manifest.get("run_provenance") != run_provenance:
            errors.append("run_provenance")
        if errors:
            raise RuntimeError(
                f"Progress-v2 resume mismatch in {run_dir}: {errors}. Use a new --output-dir."
            )
        if not resume:
            raise RuntimeError(f"Run already exists at {run_dir}; use --resume or a new output root")
        if not (run_dir / "data_audit.json").is_file():
            raise RuntimeError(f"Existing v2 run lacks data_audit.json: {run_dir}")
        actual_partitions = partition_hashes(run_dir)
        if actual_partitions != run_provenance["data"]["partition_sha256"]:
            raise RuntimeError(
                "Existing v2 partition artifacts changed before resume: "
                f"actual={actual_partitions}, "
                f"expected={run_provenance['data']['partition_sha256']}"
            )
        return manifest, fingerprint, False
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(
            f"Refusing non-empty directory without a {SCHEMA_VERSION} manifest: {run_dir}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "progress_schema_version": PROGRESS_SCHEMA_VERSION,
        "created_unix": time.time(),
        "core_config_sha256": fingerprint,
        "core_config": immutable,
        "requested_methods_initial": list(config.methods),
        "runner_path": str(runner_path),
        "runner_sha256": sha256_file(runner_path),
        "base_runner_path": str(base_runner_path),
        "base_runner_sha256": sha256_file(base_runner_path),
        "git_revision": git_revision(),
        "determinism": determinism_audit,
        "model_snapshot": snapshot_audit,
        "run_provenance": run_provenance,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "platform": platform.platform(),
        },
    }
    base.atomic_write_json(manifest_path, manifest)
    config_payload = asdict(config)
    config_payload.update(
        {
            "schema_version": SCHEMA_VERSION,
            "progress_schema_version": PROGRESS_SCHEMA_VERSION,
            "progress_split": "dev",
            "test_policy": "final_once",
            "determinism": determinism_audit,
            "model_snapshot": snapshot_audit,
        }
    )
    base.atomic_write_json(run_dir / "config.json", config_payload)
    return manifest, fingerprint, True


def partition_hashes(run_dir: Path) -> dict[str, str]:
    hashes = {
        split: sha256_file(run_dir / f"{split}_partition.csv")
        for split in ("train", "dev", "test")
    }
    hashes["combined_sha256"] = json_sha256(hashes)
    return hashes


def build_partition_exports(
    frames: dict[str, pd.DataFrame],
    partitions: dict[str, dict[str, list[int]]],
) -> dict[str, pd.DataFrame]:
    exports: dict[str, pd.DataFrame] = {}
    for split, frame in frames.items():
        site_by_index: dict[int, str] = {}
        for site, indices in partitions[split].items():
            site_by_index.update({int(index): site for index in indices})
        if set(site_by_index) != set(range(len(frame))):
            raise RuntimeError(f"Partition does not cover every {split} row exactly once")
        export = frame[["stable_id", "source_row", "sequence", "label"]].copy()
        export["site"] = [site_by_index[index] for index in range(len(export))]
        exports[split] = export
    return exports


def expected_data_provenance(
    dataset_audit: dict[str, Any],
    exports: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    partition_sha = {
        split: hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()
        for split, frame in exports.items()
    }
    partition_sha["combined_sha256"] = json_sha256(partition_sha)
    return {
        "dataset_source_sha256": {
            split: str(dataset_audit[split]["source_sha256"])
            for split in ("train", "dev", "test")
        },
        "dataset_used_stable_ids_sha256": {
            split: str(dataset_audit[split]["used_stable_ids_sha256"])
            for split in ("train", "dev", "test")
        },
        "dataset_used_n": {
            split: int(dataset_audit[split]["used_n"])
            for split in ("train", "dev", "test")
        },
        "partition_sha256": partition_sha,
        "partition_encoding": "pandas.to_csv(index=False), UTF-8, deterministic row order",
    }


def write_or_verify_run_data_artifacts(
    run_dir: Path,
    audit_payload: dict[str, Any],
    exports: dict[str, pd.DataFrame],
    provenance: dict[str, Any],
    *,
    created: bool,
) -> None:
    if created:
        base.atomic_write_json(run_dir / "data_audit.json", audit_payload)
        for split, frame in exports.items():
            base.atomic_write_csv(frame, run_dir / f"{split}_partition.csv")
    else:
        if not (run_dir / "data_audit.json").is_file():
            raise RuntimeError(f"Existing v2 run lacks data_audit.json: {run_dir}")
        existing_audit = json.loads(
            (run_dir / "data_audit.json").read_text(encoding="utf-8")
        )
        if existing_audit != audit_payload:
            raise RuntimeError(
                "Existing v2 data_audit differs from the current source/partition audit"
            )
    expected = provenance["partition_sha256"]
    actual = partition_hashes(run_dir)
    if actual != expected:
        raise RuntimeError(
            f"Run partition SHA mismatch before method resume: actual={actual}, expected={expected}"
        )


def initial_state_provenance(
    config: base.MainExperimentConfig,
    device: torch.device,
    init_state: dict[str, torch.Tensor],
    run_dir: Path,
) -> dict[str, Any]:
    vector, manifest = pilot.flatten_state(init_state)
    result: dict[str, Any] = {
        "lora": {
            "initial_state_sha256": hashlib.sha256(vector.tobytes()).hexdigest(),
            "manifest_sha256": json_sha256(manifest),
            "trainable_numel": int(vector.size),
        }
    }
    # Always compute both adaptations, independent of requested methods.  This
    # keeps the core fingerprint stable when a launcher adds methods to the same
    # run directory in separate processes (for example central, then Full-FT).
    full_model, _ = base.make_full_ft_model(config, config.seed, device, init_state)
    full_state = pilot.clone_trainable_state(full_model)
    full_manifest = base.state_manifest_without_flattening(full_state)
    result["full_fine_tuning"] = {
        "initial_state_sha256": base.state_fp32_streaming_sha256(
            full_state, full_manifest
        ),
        "manifest_sha256": json_sha256(full_manifest),
        "trainable_numel": int(
            sum(int(entry["numel"]) for entry in full_manifest)
        ),
    }
    del full_state, full_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def train_partition_counts(run_dir: Path) -> dict[str, int]:
    frame = pd.read_csv(run_dir / "train_partition.csv", usecols=["site"])
    counts = frame["site"].value_counts().to_dict()
    return {site: int(counts.get(site, 0)) for site in pilot.SITE_NAMES}


def _progress_metric_row(
    record: dict[str, Any],
    *,
    method: str,
    config: base.MainExperimentConfig,
    progress_kind: str,
    progress: int,
    local_epoch: int | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "method": method,
        "scenario": config.scenario,
        "seed": config.seed,
        "split": "dev",
        "progress_kind": progress_kind,
        "progress": int(progress),
        "epoch": int(progress) if progress_kind == "epoch" else math.nan,
        "round": int(progress) if progress_kind == "round" else math.nan,
        "local_epoch": local_epoch if local_epoch is not None else math.nan,
        "scope": str(record["scope"]),
        "n": int(record["n"]),
        # Cross-entropy on DEV probabilities; distinct from the training loss.
        "classification_loss": float(record["log_loss"]),
        "accuracy": float(record["accuracy"]),
        "auroc": float(record["auroc"]),
        "auprc": float(record["auprc"]),
        "mcc": float(record["mcc"]),
        "f1": float(record["f1"]),
        "ece": float(record["ece"]),
        "brier": float(record["brier"]),
        "eval_seconds": float(record.get("eval_seconds", math.nan)),
        "provenance": "recorded_during_training_on_dev",
    }


def _progress_prediction_rows(
    predictions: list[dict[str, Any]],
    *,
    progress_kind: str,
    progress: int,
    local_epoch: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prediction in predictions:
        site_scope = f"site_{prediction['site']}"
        for scope in (site_scope, "pooled"):
            rows.append(
                {
                    "schema_version": PROGRESS_SCHEMA_VERSION,
                    **prediction,
                    "scope": scope,
                    "progress_kind": progress_kind,
                    "progress": int(progress),
                    "epoch": int(progress) if progress_kind == "epoch" else math.nan,
                    "round": int(progress) if progress_kind == "round" else math.nan,
                    "local_epoch": local_epoch if local_epoch is not None else math.nan,
                    "provenance": "recorded_during_training_on_dev",
                }
            )
    return rows


def collect_global_dev_checkpoint(
    model: nn.Module,
    method: str,
    config: base.MainExperimentConfig,
    frames: dict[str, pd.DataFrame],
    datasets: dict[str, pilot.TokenDataset],
    partitions: dict[str, dict[str, list[int]]],
    device: torch.device,
    *,
    progress_kind: str,
    progress: int,
    seed_offset: int,
    local_epoch: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    records, predictions = base.evaluate_global_model(
        model,
        method,
        "dev",
        frames["dev"],
        datasets["dev"],
        partitions["dev"],
        config,
        device,
        seed_offset,
    )
    metric_rows = [
        _progress_metric_row(
            record,
            method=method,
            config=config,
            progress_kind=progress_kind,
            progress=progress,
            local_epoch=local_epoch,
        )
        for record in records
    ]
    prediction_rows = _progress_prediction_rows(
        predictions,
        progress_kind=progress_kind,
        progress=progress,
        local_epoch=local_epoch,
    )
    pooled = next(record for record in records if record["scope"] == "pooled")
    legacy_round_metrics = {
        key: float(pooled[key])
        for key in ("accuracy", "f1", "mcc", "auroc", "auprc", "brier", "ece")
    }
    legacy_round_metrics["classification_loss"] = float(pooled["log_loss"])
    return metric_rows, prediction_rows, legacy_round_metrics


def train_central_v2(
    method: str,
    config: base.MainExperimentConfig,
    frames: dict[str, pd.DataFrame],
    datasets: dict[str, pilot.TokenDataset],
    partitions: dict[str, dict[str, list[int]]],
    device: torch.device,
    init_state: dict[str, torch.Tensor],
) -> tuple[nn.Module, dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if method == "central_full_ft":
        model, audit = base.make_full_ft_model(config, config.seed, device, init_state)
        learning_rate = config.full_ft_learning_rate
        batch_size = config.full_ft_batch_size
        accumulation = config.full_ft_grad_accum_steps
        amp_dtype = base.resolve_full_ft_amp_dtype(config, device)
        audit["amp_dtype_resolved"] = str(amp_dtype).replace("torch.", "")
    else:
        model = pilot.create_model(config, config.seed, device)
        pilot.load_trainable_state(model, init_state)
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in model.parameters())
        audit = {
            "adaptation": "lora",
            "trainable_parameters": int(trainable),
            "total_parameters": int(total),
            "trainable_fraction": float(trainable / total),
            "gradient_checkpointing": False,
            "micro_batch_size": config.batch_size,
            "gradient_accumulation_steps": 1,
            "effective_batch_size": config.batch_size,
        }
        learning_rate = config.learning_rate
        batch_size = config.batch_size
        accumulation = 1
        amp_dtype = torch.float16
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=config.weight_decay,
    )
    epoch_records: list[dict[str, Any]] = []
    progress_metrics: list[dict[str, Any]] = []
    progress_predictions: list[dict[str, Any]] = []
    total_seconds = 0.0
    for epoch_index in range(config.epochs):
        paired_seed = config.seed + epoch_index
        pilot.set_seed(paired_seed)
        loader = pilot.make_loader(
            datasets["train"], None, batch_size, True, paired_seed, config.num_workers
        )
        loss, seconds, detail = base.train_one_epoch(
            model,
            loader,
            device,
            optimizer,
            gradient_accumulation_steps=accumulation,
            max_grad_norm=config.max_grad_norm,
            amp_dtype=amp_dtype,
        )
        total_seconds += seconds
        epoch_records.append(
            {"epoch": epoch_index + 1, "loss": loss, "seconds": seconds, **detail}
        )
        metric_rows, prediction_rows, _ = collect_global_dev_checkpoint(
            model,
            method,
            config,
            frames,
            datasets,
            partitions,
            device,
            progress_kind="epoch",
            progress=epoch_index + 1,
            seed_offset=10_000 + epoch_index * 10,
        )
        progress_metrics.extend(metric_rows)
        progress_predictions.extend(prediction_rows)
    expected_metric_rows = config.epochs * (len(pilot.SITE_NAMES) + 1)
    expected_prediction_rows = config.epochs * 2 * len(frames["dev"])
    if len(progress_metrics) != expected_metric_rows:
        raise RuntimeError(
            f"{method} progress metric coverage mismatch: {len(progress_metrics)} != {expected_metric_rows}"
        )
    if len(progress_predictions) != expected_prediction_rows:
        raise RuntimeError(
            f"{method} progress prediction coverage mismatch: "
            f"{len(progress_predictions)} != {expected_prediction_rows}"
        )
    return (
        model,
        {"train_seconds": total_seconds, "epochs": epoch_records},
        audit,
        progress_metrics,
        progress_predictions,
    )


def train_local_v2(
    config: base.MainExperimentConfig,
    frames: dict[str, pd.DataFrame],
    datasets: dict[str, pilot.TokenDataset],
    partitions: dict[str, dict[str, list[int]]],
    device: torch.device,
    init_state: dict[str, torch.Tensor],
) -> tuple[dict[str, nn.Module], dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    models: dict[str, nn.Module] = {}
    statistics: dict[str, Any] = {}
    progress_metrics: list[dict[str, Any]] = []
    progress_predictions: list[dict[str, Any]] = []
    pooled_by_epoch: dict[int, list[dict[str, Any]]] = {
        epoch: [] for epoch in range(1, config.epochs + 1)
    }
    for site_index, site in enumerate(pilot.SITE_NAMES):
        model = pilot.create_model(config, config.seed, device)
        pilot.load_trainable_state(model, init_state)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        epochs: list[dict[str, Any]] = []
        total_seconds = 0.0
        for epoch_index in range(config.epochs):
            paired_seed = config.seed + 1000 + site_index * 100 + epoch_index
            pilot.set_seed(paired_seed)
            loader = pilot.make_loader(
                datasets["train"],
                partitions["train"][site],
                config.batch_size,
                True,
                paired_seed,
                config.num_workers,
            )
            loss, seconds, detail = base.train_one_epoch(
                model, loader, device, optimizer, max_grad_norm=config.max_grad_norm
            )
            total_seconds += seconds
            epochs.append({"epoch": epoch_index + 1, "loss": loss, "seconds": seconds, **detail})

            eval_loader = pilot.make_loader(
                datasets["dev"],
                partitions["dev"][site],
                config.eval_batch_size,
                False,
                config.seed + 20_000 + epoch_index * 10 + site_index,
                config.num_workers,
            )
            labels, scores, eval_seconds = pilot.predict(model, eval_loader, device)
            metric_values = base.expanded_binary_metrics(labels, scores, config.ece_bins)
            record = {
                "scope": f"site_{site}",
                "n": len(labels),
                "eval_seconds": eval_seconds,
                **metric_values,
            }
            progress_metrics.append(
                _progress_metric_row(
                    record,
                    method="local",
                    config=config,
                    progress_kind="epoch",
                    progress=epoch_index + 1,
                )
            )
            raw_predictions = base.prediction_records(
                "local",
                "dev",
                site,
                partitions["dev"][site],
                frames["dev"],
                labels,
                scores,
                config,
            )
            pooled_by_epoch[epoch_index + 1].extend(raw_predictions)
            # Store site scope now; pooled copies are added after all sites for
            # the epoch are present.
            for row in _progress_prediction_rows(
                raw_predictions, progress_kind="epoch", progress=epoch_index + 1
            ):
                if row["scope"] != "pooled":
                    progress_predictions.append(row)
        models[site] = model.cpu()
        statistics[site] = {"train_seconds": total_seconds, "epochs": epochs}
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for epoch, raw_predictions in pooled_by_epoch.items():
        labels = np.asarray([row["label"] for row in raw_predictions], dtype=np.int64)
        scores = np.asarray([row["score"] for row in raw_predictions], dtype=np.float64)
        metric_values = base.expanded_binary_metrics(labels, scores, config.ece_bins)
        progress_metrics.append(
            _progress_metric_row(
                {
                    "scope": "pooled",
                    "n": len(labels),
                    "eval_seconds": math.nan,
                    **metric_values,
                },
                method="local",
                config=config,
                progress_kind="epoch",
                progress=epoch,
            )
        )
        for row in _progress_prediction_rows(
            raw_predictions, progress_kind="epoch", progress=epoch
        ):
            if row["scope"] == "pooled":
                progress_predictions.append(row)

    trainable = int(sum(value.numel() for value in init_state.values()))
    audit = {
        "adaptation": "three_independent_lora_models",
        "trainable_parameters_per_model": trainable,
        "saved_models": list(pilot.SITE_NAMES),
        "progress_pooled_semantics": "site_matched_personalized_predictions",
    }
    expected_metric_rows = config.epochs * (len(pilot.SITE_NAMES) + 1)
    expected_prediction_rows = config.epochs * 2 * len(frames["dev"])
    if len(progress_metrics) != expected_metric_rows:
        raise RuntimeError(
            f"local progress metric coverage mismatch: {len(progress_metrics)} != {expected_metric_rows}"
        )
    if len(progress_predictions) != expected_prediction_rows:
        raise RuntimeError(
            "local progress prediction coverage mismatch: "
            f"{len(progress_predictions)} != {expected_prediction_rows}"
        )
    return models, statistics, audit, progress_metrics, progress_predictions


def train_federated_v2(
    method: str,
    config: base.MainExperimentConfig,
    frames: dict[str, pd.DataFrame],
    datasets: dict[str, pilot.TokenDataset],
    partitions: dict[str, dict[str, list[int]]],
    device: torch.device,
    init_state: dict[str, torch.Tensor],
) -> tuple[nn.Module, dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    progress_metrics: list[dict[str, Any]] = []
    progress_predictions: list[dict[str, Any]] = []
    original = base.round_dev_metrics

    def progress_hook(
        model: nn.Module,
        hook_datasets: dict[str, pilot.TokenDataset],
        hook_partitions: dict[str, dict[str, list[int]]],
        hook_frames: dict[str, pd.DataFrame],
        hook_method: str,
        hook_config: base.MainExperimentConfig,
        hook_device: torch.device,
        round_index: int,
    ) -> dict[str, float]:
        metric_rows, prediction_rows, legacy = collect_global_dev_checkpoint(
            model,
            hook_method,
            hook_config,
            hook_frames,
            hook_datasets,
            hook_partitions,
            hook_device,
            progress_kind="round",
            progress=round_index + 1,
            seed_offset=30_000 + round_index * 10,
            local_epoch=hook_config.local_epochs,
        )
        progress_metrics.extend(metric_rows)
        progress_predictions.extend(prediction_rows)
        return legacy

    base.round_dev_metrics = progress_hook
    try:
        if method in ("fedavg_full_ft", "he_fedavg_full_ft"):
            model, stats, audit = base.train_federated_full_ft(
                method, config, datasets, partitions, frames, device, init_state
            )
        else:
            model, stats = base.train_federated_method(
                method, config, datasets, partitions, frames, device, init_state
            )
            trainable = int(
                sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            )
            audit = {
                "adaptation": "lora",
                "aggregation": (
                    "ckks_weighted_sum" if method == "he_fedavg" else "plaintext_weighted_mean"
                ),
                "local_objective": "fedprox" if method == "fedprox" else "empirical_risk",
                "fedprox_mu": config.fedprox_mu if method == "fedprox" else 0.0,
                "trainable_parameters": trainable,
            }
    finally:
        base.round_dev_metrics = original
    expected_metric_rows = config.rounds * (len(pilot.SITE_NAMES) + 1)
    expected_prediction_rows = config.rounds * 2 * len(frames["dev"])
    if len(progress_metrics) != expected_metric_rows:
        raise RuntimeError(
            f"{method} progress metric coverage mismatch: {len(progress_metrics)} != {expected_metric_rows}"
        )
    if len(progress_predictions) != expected_prediction_rows:
        raise RuntimeError(
            f"{method} progress prediction coverage mismatch: "
            f"{len(progress_predictions)} != {expected_prediction_rows}"
        )
    return model, stats, audit, progress_metrics, progress_predictions


def add_pairing_audit(
    method: str,
    method_audit: dict[str, Any],
    init_state: dict[str, torch.Tensor],
    partition_digests: dict[str, str],
) -> None:
    vector, manifest = pilot.flatten_state(init_state)
    lora_hash = hashlib.sha256(vector.tobytes()).hexdigest()
    lora_manifest_hash = json_sha256(manifest)
    pairing_applicable = method in {
        "fedavg",
        "he_fedavg",
        "fedavg_full_ft",
        "he_fedavg_full_ft",
    }
    if method in ("fedavg_full_ft", "he_fedavg_full_ft"):
        dense = method_audit.get("full_dense_state") or {}
        initial_hash = dense.get("initial_state_fp32_sha256")
        manifest_hash = dense.get("manifest_sha256")
        adaptation = "full_fine_tuning"
        if not initial_hash or not manifest_hash:
            raise RuntimeError(f"{method} lacks the full dense initial-state pairing audit")
    elif method == "central_full_ft":
        initial_hash = None
        manifest_hash = None
        adaptation = "full_fine_tuning"
    else:
        initial_hash = lora_hash
        manifest_hash = lora_manifest_hash
        adaptation = "lora"
    method_audit["paired_protocol"] = {
        "applicable": pairing_applicable,
        "adaptation": adaptation,
        "initial_trainable_state_sha256": initial_hash,
        "trainable_manifest_sha256": manifest_hash,
        "partition_sha256": partition_digests,
        "partition_combined_sha256": partition_digests["combined_sha256"],
    }
    method_audit["progress_logging"] = {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "split": "dev",
        "test_policy": "final_once",
        "metrics": list(PROGRESS_METRIC_NAMES),
        "prediction_scopes": ["pooled", "site_A", "site_B", "site_C"],
    }


def expected_method_initial_state(
    config: base.MainExperimentConfig, method: str
) -> dict[str, Any]:
    adaptation = (
        "full_fine_tuning"
        if method in {"central_full_ft", "fedavg_full_ft", "he_fedavg_full_ft"}
        else "lora"
    )
    provenance = getattr(config, "run_provenance", {})
    states = provenance.get("initial_states", {})
    if adaptation not in states:
        raise RuntimeError(
            f"Run provenance lacks {adaptation} initial state required by {method}"
        )
    return {"adaptation": adaptation, **states[adaptation]}


def finalize_method_v2(
    method: str,
    method_dir: Path,
    final_metrics: list[dict[str, Any]],
    final_predictions: list[dict[str, Any]],
    progress_metrics: list[dict[str, Any]],
    progress_predictions: list[dict[str, Any]],
    system_metrics: dict[str, Any],
    method_audit: dict[str, Any],
    state: Any,
    config: base.MainExperimentConfig,
    core_config_sha256: str,
) -> None:
    method_dir.mkdir(parents=True, exist_ok=True)
    base.atomic_write_csv(pd.DataFrame(final_metrics), method_dir / FINAL_METRICS_NAME)
    base.atomic_write_csv(
        pd.DataFrame(final_predictions),
        method_dir / FINAL_PREDICTIONS_NAME,
        compression="gzip",
    )
    base.atomic_write_csv(pd.DataFrame(progress_metrics), method_dir / PROGRESS_METRICS_NAME)
    base.atomic_write_csv(
        pd.DataFrame(progress_predictions),
        method_dir / PROGRESS_PREDICTIONS_NAME,
        compression="gzip",
    )
    base.atomic_write_json(method_dir / "system_metrics.json", system_metrics)
    base.atomic_write_json(method_dir / "method_model_audit.json", method_audit)
    filenames = [
        FINAL_METRICS_NAME,
        FINAL_PREDICTIONS_NAME,
        PROGRESS_METRICS_NAME,
        PROGRESS_PREDICTIONS_NAME,
        "system_metrics.json",
        "method_model_audit.json",
    ]
    if config.save_model_states:
        if state is None:
            raise RuntimeError(f"State saving enabled but {method} returned no state")
        base.atomic_torch_save(state, method_dir / "trainable_state.pt")
        filenames.append("trainable_state.pt")
    hashes = {name: sha256_file(method_dir / name) for name in filenames}
    sizes = {name: int((method_dir / name).stat().st_size) for name in filenames}
    pooled_test = [
        row for row in final_metrics if row["split"] == "test" and row["scope"] == "pooled"
    ]
    run_provenance = getattr(config, "run_provenance", {})
    data_provenance = run_provenance.get("data", {})
    expected_initial = expected_method_initial_state(config, method)
    base.atomic_write_json(
        method_dir / "DONE.json",
        {
            "status": "complete",
            "schema_version": SCHEMA_VERSION,
            "progress_schema_version": PROGRESS_SCHEMA_VERSION,
            "method": method,
            "completed_unix": time.time(),
            "core_config_sha256": core_config_sha256,
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "base_runner_sha256": sha256_file(Path(base.__file__).resolve()),
            "artifact_sha256": hashes,
            "artifact_bytes": sizes,
            "progress_metric_rows": len(progress_metrics),
            "progress_prediction_rows": len(progress_predictions),
            "test_evaluations": 1,
            "deterministic": bool(getattr(config, "deterministic", True)),
            "model_revision": getattr(config, "model_revision", None),
            "model_snapshot_aggregate_sha256": getattr(
                config, "model_snapshot_audit", {}
            ).get("aggregate_sha256"),
            "dataset_source_sha256": data_provenance.get("dataset_source_sha256"),
            "dataset_used_stable_ids_sha256": data_provenance.get(
                "dataset_used_stable_ids_sha256"
            ),
            "partition_sha256": data_provenance.get("partition_sha256"),
            "initial_state": expected_initial,
            "pooled_test_metrics": pooled_test,
        },
    )


def method_artifacts_valid_v2(
    method: str,
    method_dir: Path,
    config: base.MainExperimentConfig,
    core_config_sha256: str,
) -> bool:
    done_path = method_dir / "DONE.json"
    if not done_path.exists():
        return False
    try:
        done = json.loads(done_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (
        done.get("status") != "complete"
        or done.get("method") != method
        or done.get("schema_version") != SCHEMA_VERSION
        or done.get("progress_schema_version") != PROGRESS_SCHEMA_VERSION
        or done.get("core_config_sha256") != core_config_sha256
        or done.get("runner_sha256") != sha256_file(Path(__file__).resolve())
        or done.get("base_runner_sha256") != sha256_file(Path(base.__file__).resolve())
        or int(done.get("test_evaluations", 0)) != 1
        or bool(done.get("deterministic"))
        != bool(getattr(config, "deterministic", True))
        or done.get("model_revision") != getattr(config, "model_revision", None)
        or done.get("model_snapshot_aggregate_sha256")
        != getattr(config, "model_snapshot_audit", {}).get("aggregate_sha256")
    ):
        return False
    required = [
        FINAL_METRICS_NAME,
        FINAL_PREDICTIONS_NAME,
        PROGRESS_METRICS_NAME,
        PROGRESS_PREDICTIONS_NAME,
        "system_metrics.json",
        "method_model_audit.json",
    ]
    if config.save_model_states:
        required.append("trainable_state.pt")
    hashes = done.get("artifact_sha256", {})
    run_provenance = getattr(config, "run_provenance", {})
    data_provenance = run_provenance.get("data", {})
    try:
        expected_initial = expected_method_initial_state(config, method)
    except RuntimeError:
        return False
    provenance_matches = (
        done.get("dataset_source_sha256")
        == data_provenance.get("dataset_source_sha256")
        and done.get("dataset_used_stable_ids_sha256")
        == data_provenance.get("dataset_used_stable_ids_sha256")
        and done.get("partition_sha256") == data_provenance.get("partition_sha256")
        and done.get("initial_state") == expected_initial
    )
    return provenance_matches and all(
        (method_dir / name).is_file() and hashes.get(name) == sha256_file(method_dir / name)
        for name in required
    )


def run_method_v2(
    method: str,
    method_dir: Path,
    run_dir: Path,
    config: base.MainExperimentConfig,
    frames: dict[str, pd.DataFrame],
    datasets: dict[str, pilot.TokenDataset],
    partitions: dict[str, dict[str, list[int]]],
    device: torch.device,
    init_state: dict[str, torch.Tensor],
    core_config_sha256: str,
) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    if method in ("central", "central_full_ft"):
        model, stats, audit, progress_metrics, progress_predictions = train_central_v2(
            method, config, frames, datasets, partitions, device, init_state
        )
        final_metrics, final_predictions = base.evaluate_global_model(
            model,
            method,
            "test",
            frames["test"],
            datasets["test"],
            partitions["test"],
            config,
            device,
            40_000,
        )
        state = pilot.clone_trainable_state(model) if config.save_model_states else None
    elif method == "local":
        models, stats, audit, progress_metrics, progress_predictions = train_local_v2(
            config, frames, datasets, partitions, device, init_state
        )
        final_metrics, final_predictions = base.evaluate_local_models(
            models,
            "test",
            frames["test"],
            datasets["test"],
            partitions["test"],
            config,
            device,
        )
        state = (
            {site: pilot.clone_trainable_state(model) for site, model in models.items()}
            if config.save_model_states
            else None
        )
        del models
    elif method in (
        "fedavg",
        "fedprox",
        "he_fedavg",
        "fedavg_full_ft",
        "he_fedavg_full_ft",
    ):
        model, stats, audit, progress_metrics, progress_predictions = train_federated_v2(
            method, config, frames, datasets, partitions, device, init_state
        )
        final_metrics, final_predictions = base.evaluate_global_model(
            model,
            method,
            "test",
            frames["test"],
            datasets["test"],
            partitions["test"],
            config,
            device,
            50_000,
        )
        state = pilot.clone_trainable_state(model) if config.save_model_states else None
    else:
        raise ValueError(f"Unsupported v2 method: {method}")

    partition_digests = partition_hashes(run_dir)
    add_pairing_audit(method, audit, init_state, partition_digests)
    expected_initial = expected_method_initial_state(config, method)
    paired_protocol = audit["paired_protocol"]
    if paired_protocol.get("applicable"):
        if (
            paired_protocol.get("initial_trainable_state_sha256")
            != expected_initial["initial_state_sha256"]
            or paired_protocol.get("trainable_manifest_sha256")
            != expected_initial["manifest_sha256"]
        ):
            raise RuntimeError(
                f"{method} actual initial-state audit differs from prelaunch provenance"
            )
    audit["reproducibility"] = {
        "deterministic": bool(getattr(config, "deterministic", True)),
        "hard_fail_on_nondeterministic_operation": bool(
            getattr(config, "deterministic", True)
        ),
        "protocol_source": "run_manifest.json/determinism",
        "model_source": "pinned_local_snapshot",
        "model_revision": getattr(config, "model_revision", None),
        "model_snapshot_aggregate_sha256": getattr(
            config, "model_snapshot_audit", {}
        ).get("aggregate_sha256"),
    }
    stats["evaluation_policy"] = {
        "progress_split": "dev",
        "central_local_frequency": "every_epoch",
        "federated_frequency": "every_round",
        "test": "exactly_once_after_training",
        "progress_metrics": list(PROGRESS_METRIC_NAMES),
        "deterministic": bool(getattr(config, "deterministic", True)),
    }
    if device.type == "cuda":
        torch.cuda.synchronize()
        stats["resource"] = {
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_cuda_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "end_cuda_memory_bytes": int(torch.cuda.memory_allocated(device)),
            "peak_process_rss_bytes": base.process_peak_rss_bytes(),
        }
    finalize_method_v2(
        method,
        method_dir,
        final_metrics,
        final_predictions,
        progress_metrics,
        progress_predictions,
        stats,
        audit,
        state,
        config,
        core_config_sha256,
    )
    if "model" in locals():
        del model
    if state is not None:
        del state
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def validate_pairings(run_dir: Path, config: base.MainExperimentConfig, fingerprint: str) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for plain, encrypted, label in PAIRINGS:
        plain_dir = run_dir / "methods" / plain
        encrypted_dir = run_dir / "methods" / encrypted
        plain_valid = method_artifacts_valid_v2(plain, plain_dir, config, fingerprint)
        encrypted_valid = method_artifacts_valid_v2(encrypted, encrypted_dir, config, fingerprint)
        if not plain_valid and not encrypted_valid:
            continue
        if plain_valid != encrypted_valid:
            records.append(
                {
                    "pair": label,
                    "status": "incomplete_pair",
                    "plain_complete": plain_valid,
                    "he_complete": encrypted_valid,
                }
            )
            continue
        plain_audit = json.loads((plain_dir / "method_model_audit.json").read_text(encoding="utf-8"))[
            "paired_protocol"
        ]
        encrypted_audit = json.loads(
            (encrypted_dir / "method_model_audit.json").read_text(encoding="utf-8")
        )["paired_protocol"]
        fields = (
            "applicable",
            "adaptation",
            "initial_trainable_state_sha256",
            "trainable_manifest_sha256",
            "partition_combined_sha256",
        )
        mismatches = [field for field in fields if plain_audit.get(field) != encrypted_audit.get(field)]
        if mismatches:
            raise RuntimeError(f"Paired Plain/HE audit mismatch for {label}: {mismatches}")
        if plain_audit.get("applicable") is not True:
            raise RuntimeError(f"Paired audit unexpectedly marked not applicable: {label}")
        records.append(
            {
                "pair": label,
                "status": "matched",
                "plain_method": plain,
                "he_method": encrypted,
                "adaptation": plain_audit["adaptation"],
                "initial_trainable_state_sha256": plain_audit["initial_trainable_state_sha256"],
                "trainable_manifest_sha256": plain_audit["trainable_manifest_sha256"],
                "partition_combined_sha256": plain_audit["partition_combined_sha256"],
            }
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "scenario": config.scenario,
        "seed": config.seed,
        "records": records,
    }
    base.atomic_write_json(run_dir / "pairing_audit.json", payload)
    return payload


def consolidate_v2(
    run_dir: Path,
    config: base.MainExperimentConfig,
    fingerprint: str,
) -> dict[str, Any]:
    metrics: list[pd.DataFrame] = []
    predictions: list[pd.DataFrame] = []
    progress_metrics: list[pd.DataFrame] = []
    progress_predictions: list[pd.DataFrame] = []
    systems: dict[str, Any] = {}
    available: list[str] = []
    for method in METHODS:
        method_dir = run_dir / "methods" / method
        if not method_artifacts_valid_v2(method, method_dir, config, fingerprint):
            continue
        available.append(method)
        metrics.append(pd.read_csv(method_dir / FINAL_METRICS_NAME))
        predictions.append(pd.read_csv(method_dir / FINAL_PREDICTIONS_NAME))
        progress_metrics.append(pd.read_csv(method_dir / PROGRESS_METRICS_NAME))
        progress_predictions.append(pd.read_csv(method_dir / PROGRESS_PREDICTIONS_NAME))
        systems[method] = json.loads((method_dir / "system_metrics.json").read_text(encoding="utf-8"))
    if metrics:
        base.atomic_write_csv(pd.concat(metrics, ignore_index=True), run_dir / FINAL_METRICS_NAME)
        base.atomic_write_csv(
            pd.concat(predictions, ignore_index=True),
            run_dir / FINAL_PREDICTIONS_NAME,
            compression="gzip",
        )
        base.atomic_write_csv(
            pd.concat(progress_metrics, ignore_index=True), run_dir / PROGRESS_METRICS_NAME
        )
        base.atomic_write_csv(
            pd.concat(progress_predictions, ignore_index=True),
            run_dir / PROGRESS_PREDICTIONS_NAME,
            compression="gzip",
        )
    base.atomic_write_json(run_dir / "system_metrics.json", systems)
    missing = [method for method in config.methods if method not in available]
    pairing = validate_pairings(run_dir, config, fingerprint)
    summary = {
        "status": "complete" if not missing else "partial",
        "schema_version": SCHEMA_VERSION,
        "progress_schema_version": PROGRESS_SCHEMA_VERSION,
        "run_dir": str(run_dir.resolve()),
        "requested_methods": list(config.methods),
        "available_methods": available,
        "missing_requested_methods": missing,
        "pairing_audit": pairing,
        "test_policy": "final_once",
    }
    base.atomic_write_json(run_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    truststore.inject_into_ssl()
    determinism_audit = configure_determinism(bool(args.deterministic))
    snapshot_audit = audit_default_model_snapshot(
        Path(args.model_snapshot_dir), str(args.model_revision)
    )
    bind_pinned_model_source(snapshot_audit)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for progress-logging v2 experiments")
    device = torch.device("cuda")
    config = make_config(args, snapshot_audit)
    base.validate_config(config)
    if config.eval_splits != ["test"] or config.round_eval_interval != 1:
        raise RuntimeError("v2 evaluation policy was unexpectedly changed")
    run_dir = Path(config.output_dir) / config.scenario / f"seed_{config.seed}"
    pilot.set_seed(config.seed)
    frames, dataset_audit = base.load_frames(
        Path(config.data_dir),
        (config.train_limit, config.dev_limit, config.test_limit),
        config.seed,
    )
    partitions = {
        split: pilot.allocate_indices(
            frame["label"].to_numpy(), config.scenario, config.seed + offset * 10
        )
        for offset, (split, frame) in enumerate(frames.items())
    }
    partition_exports = build_partition_exports(frames, partitions)
    data_provenance = expected_data_provenance(dataset_audit, partition_exports)
    audit = {
        "dataset": dataset_audit,
        "partitions": pilot.partition_audit(frames, partitions),
        "partition_algorithm": {
            "iid": "class-stratified equal shards",
            "noniid": "controlled label skew; class0=[0.15,0.35,0.50], class1=[0.60,0.30,0.10]",
        },
    }
    init_model = pilot.create_model(config, config.seed, device)
    init_state = pilot.clone_trainable_state(init_model)
    init_vector, init_manifest = pilot.flatten_state(init_state)
    del init_model
    torch.cuda.empty_cache()
    initial_states = initial_state_provenance(config, device, init_state, run_dir)
    run_provenance = {
        "data": data_provenance,
        "initial_states": initial_states,
        "model_snapshot_aggregate_sha256": snapshot_audit["aggregate_sha256"],
        "model_revision": snapshot_audit["revision"],
    }
    setattr(config, "run_provenance", run_provenance)
    _, fingerprint, created = prepare_run_dir_v2(
        run_dir, config, args.resume, determinism_audit
    )
    write_or_verify_run_data_artifacts(
        run_dir,
        audit,
        partition_exports,
        data_provenance,
        created=created,
    )
    model_audit_payload = {
        "base_model": snapshot_audit["repository"],
        "model_source": "pinned_local_snapshot",
        "model_revision": snapshot_audit["revision"],
        "model_snapshot": snapshot_audit,
        "schema_version": SCHEMA_VERSION,
        "lora_trainable_parameters": int(init_vector.size),
        "lora_manifest_sha256": json_sha256(init_manifest),
        "lora_initial_state_sha256": hashlib.sha256(init_vector.tobytes()).hexdigest(),
        "initial_states": initial_states,
        "data_provenance": data_provenance,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }
    model_audit_path = run_dir / "model_audit.json"
    if created:
        base.atomic_write_json(model_audit_path, model_audit_payload)
    else:
        if not model_audit_path.is_file():
            raise RuntimeError(f"Existing v2 run lacks model_audit.json: {run_dir}")
        current_model_audit = json.loads(model_audit_path.read_text(encoding="utf-8"))
        if current_model_audit != model_audit_payload:
            raise RuntimeError("Existing v2 model_audit differs from current pinned provenance")

    tokenizer = pilot.AutoTokenizer.from_pretrained(
        pilot.MODEL_NAME, trust_remote_code=True, local_files_only=True
    )
    datasets = pilot.tokenize_frames(frames, tokenizer, config.max_length)

    for method in config.methods:
        method_dir = run_dir / "methods" / method
        if method_artifacts_valid_v2(method, method_dir, config, fingerprint):
            print(f"[resume-v2] {config.scenario}/seed_{config.seed}/{method}: complete")
            continue
        print(f"[run-v2] {config.scenario}/seed_{config.seed}/{method}")
        run_method_v2(
            method,
            method_dir,
            run_dir,
            config,
            frames,
            datasets,
            partitions,
            device,
            init_state,
            fingerprint,
        )

    summary = consolidate_v2(run_dir, config, fingerprint)
    if summary["status"] != "complete":
        raise RuntimeError(f"Requested v2 methods are incomplete: {summary['missing_requested_methods']}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
