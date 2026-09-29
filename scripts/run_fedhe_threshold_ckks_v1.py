from __future__ import annotations

"""Locked H3K4me3 non-IID LoRA experiment with multiparty CKKS.

The five Plain early-stopping selections from ``h3k4me3_fl_earlystop_v3``
remain the checkpoint owners.  This runner trains a new multiparty-CKKS arm for
exactly the corresponding selected number of rounds and evaluates TEST once.

Safety: invoking this file without ``--execute`` prints the frozen plan and
exits.  No model, CUDA context, output directory, or OpenFHE context is created
unless the explicit execution flag is supplied.
"""

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import openfhe_threshold_ckks as threshold
import preflight_threshold_ckks as preflight
import run_fedhe_experiment as pilot
import run_fedhe_main_experiment as base
import run_fedhe_progress_v2 as v2


SCHEMA_VERSION = "fedhe-threshold-ckks-v1"
METHOD = "threshold_he_fedavg"
METHODS = (METHOD,)
DEFAULT_OUTPUT = pilot.PROJECT_ROOT / "experiments" / "h3k4me3_threshold_ckks_v1"
DEFAULT_REFERENCE_ROOT = pilot.PROJECT_ROOT / "experiments" / "h3k4me3_fl_earlystop_v3"
RUNNER_PATH = Path(__file__).resolve()
TEMP_FILENAMES = (
    "round_checkpoint.pt",
    "partial_progress_metrics.csv",
    "partial_progress_predictions.csv.gz",
    "partial_status.json",
)

_V2_MAKE_CONFIG = v2.make_config
_V2_CORE_CONFIG = v2.core_config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "H3K4me3 non-IID LoRA with OpenFHE multiparty CKKS; "
            "requires --execute to start training"
        )
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--data-dir", default=str(pilot.PROJECT_ROOT / "GUE_v2" / "EMP" / "H3K4me3")
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--reference-root", default=str(DEFAULT_REFERENCE_ROOT))
    parser.add_argument("--model-snapshot-dir", default=str(v2.DEFAULT_MODEL_SNAPSHOT))
    parser.add_argument("--model-revision", default=v2.DEFAULT_MODEL_REVISION)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--dev-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-rounds", dest="rounds", type=int, default=40)
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
    parser.add_argument("--seed", type=int, choices=(42, 43, 44, 45, 46), default=42)
    parser.add_argument("--scenario", choices=("noniid",), default="noniid")
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
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--threshold-parties", type=int, default=3)
    parser.add_argument("--threshold-ring-dimension", type=int, default=8192)
    parser.add_argument("--threshold-slots", type=int, default=4096)
    parser.add_argument("--threshold-multiplicative-depth", type=int, default=1)
    parser.add_argument("--threshold-scaling-mod-size", type=int, default=40)
    parser.add_argument("--threshold-first-mod-size", type=int, default=60)
    parser.add_argument(
        "--measure-serialized-bytes",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def parse_args() -> argparse.Namespace:
    args = build_parser().parse_args()
    args.methods = [METHOD]
    args.enforce_equal_data_passes = False
    if args.local_epochs != 1:
        raise ValueError("The frozen threshold study requires one local epoch per round")
    if args.threshold_parties != len(pilot.SITE_NAMES):
        raise ValueError(
            f"threshold parties must match synthetic sites: {len(pilot.SITE_NAMES)}"
        )
    return args


def _reference_info(root: Path, seed: int) -> dict[str, Any]:
    method_dir = root / "noniid" / f"seed_{seed}" / "methods" / "fedavg"
    done_path = method_dir / "DONE.json"
    audit_path = method_dir / "method_model_audit.json"
    manifest_path = root / "noniid" / f"seed_{seed}" / "run_manifest.json"
    done = json.loads(done_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection = audit.get("early_stopping")
    if (
        done.get("status") != "complete"
        or done.get("method") != "fedavg"
        or not isinstance(selection, dict)
        or selection.get("selection_owner") != "fedavg"
    ):
        raise RuntimeError(f"Invalid Plain early-stop reference for seed {seed}")
    selected_round = int(selection["selected_round"])
    if selected_round < 1:
        raise RuntimeError(f"Invalid selected round for seed {seed}: {selected_round}")
    return {
        "root": str(root.resolve()),
        "seed": seed,
        "selected_round": selected_round,
        "selected_dev_auprc": float(selection["selected_dev_auprc"]),
        "done_sha256": sha256_file(done_path),
        "method_audit_sha256": sha256_file(audit_path),
        "run_manifest_sha256": sha256_file(manifest_path),
        "done_core_config_sha256": done.get("core_config_sha256"),
        "reference_schema_version": done.get("schema_version"),
        "reference_model_revision": done.get("model_revision"),
        "reference_model_snapshot_aggregate_sha256": done.get(
            "model_snapshot_aggregate_sha256"
        ),
        "dataset_source_sha256": done.get("dataset_source_sha256"),
        "dataset_used_stable_ids_sha256": done.get("dataset_used_stable_ids_sha256"),
        "partition_sha256": done.get("partition_sha256"),
        "initial_state": done.get("initial_state"),
        "manifest_determinism": manifest.get("determinism"),
    }


def _make_config(
    args: argparse.Namespace, snapshot_audit: dict[str, Any]
) -> base.MainExperimentConfig:
    config = _V2_MAKE_CONFIG(args, snapshot_audit)
    crypto = threshold.ThresholdCKKSConfig(
        party_count=args.threshold_parties,
        ring_dimension=args.threshold_ring_dimension,
        batch_size=args.threshold_slots,
        multiplicative_depth=args.threshold_multiplicative_depth,
        scaling_mod_size=args.threshold_scaling_mod_size,
        first_mod_size=args.threshold_first_mod_size,
        measure_serialized_bytes=bool(args.measure_serialized_bytes),
    )
    crypto.validate()
    reference = _reference_info(Path(args.reference_root).resolve(), args.seed)
    if reference["selected_round"] > args.rounds:
        raise RuntimeError(
            "Reference-selected round exceeds --max-rounds: "
            f"{reference['selected_round']} > {args.rounds}"
        )
    setattr(config, "threshold_ckks_config", crypto)
    setattr(config, "threshold_reference", reference)
    setattr(config, "threshold_role_isolation", "in_process_simulation")
    return config


def _core_config(config: base.MainExperimentConfig) -> dict[str, Any]:
    payload = _V2_CORE_CONFIG(config)
    crypto: threshold.ThresholdCKKSConfig = getattr(config, "threshold_ckks_config")
    payload["threshold_ckks"] = {
        **crypto.__dict__,
        "backend": "OpenFHE-Python",
        "aggregation": "sample_weighted_ciphertext_sum",
        "decryption": "one partial share per configured party, then fusion",
        "role_isolation": getattr(config, "threshold_role_isolation"),
        "secret_key_share_persistence": "forbidden",
    }
    payload["plain_selection_reference"] = getattr(config, "threshold_reference")
    return payload


def _method_dir(config: base.MainExperimentConfig) -> Path:
    return (
        Path(config.output_dir)
        / config.scenario
        / f"seed_{config.seed}"
        / "methods"
        / METHOD
    )


def _verify_reference_pairing(config: base.MainExperimentConfig) -> None:
    reference = getattr(config, "threshold_reference")
    run_provenance = getattr(config, "run_provenance")
    current_data = run_provenance["data"]
    comparisons = {
        "model_revision": (
            reference["reference_model_revision"],
            getattr(config, "model_revision"),
        ),
        "model_snapshot": (
            reference["reference_model_snapshot_aggregate_sha256"],
            getattr(config, "model_snapshot_audit")["aggregate_sha256"],
        ),
        "dataset_source": (
            reference["dataset_source_sha256"],
            current_data["dataset_source_sha256"],
        ),
        "stable_ids": (
            reference["dataset_used_stable_ids_sha256"],
            current_data["dataset_used_stable_ids_sha256"],
        ),
        "partitions": (
            reference["partition_sha256"], current_data["partition_sha256"]
        ),
        "initial_state": (
            reference["initial_state"],
            v2.expected_method_initial_state(config, METHOD),
        ),
    }
    mismatches = [name for name, (old, new) in comparisons.items() if old != new]
    if mismatches:
        raise RuntimeError(
            "Threshold arm is not paired with the frozen Plain reference: "
            f"{mismatches}"
        )


def _load_partial(
    method_dir: Path, config: base.MainExperimentConfig
) -> dict[str, Any] | None:
    checkpoint_path = method_dir / TEMP_FILENAMES[0]
    other_paths = [method_dir / name for name in TEMP_FILENAMES[1:]]
    if not checkpoint_path.exists():
        if any(path.exists() for path in other_paths):
            raise RuntimeError("Partial ledgers exist without a threshold checkpoint")
        return None
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("method") != METHOD
        or payload.get("core_config_sha256") != v2.json_sha256(v2.core_config(config))
        or payload.get("runner_sha256") != sha256_file(RUNNER_PATH)
    ):
        raise RuntimeError("Threshold checkpoint is incompatible with the current runner")
    metrics_path = method_dir / TEMP_FILENAMES[1]
    predictions_path = method_dir / TEMP_FILENAMES[2]
    payload["progress_metrics"] = pd.read_csv(metrics_path).to_dict("records")
    payload["progress_predictions"] = pd.read_csv(predictions_path).to_dict("records")
    completed = int(payload["round_completed"])
    expected_metrics = completed * (len(pilot.SITE_NAMES) + 1)
    expected_predictions = completed * 2 * int(payload["dev_n"])
    if len(payload["progress_metrics"]) != expected_metrics:
        raise RuntimeError("Threshold partial metric coverage mismatch")
    if len(payload["progress_predictions"]) != expected_predictions:
        raise RuntimeError("Threshold partial prediction coverage mismatch")
    if "secret" in json.dumps(sorted(payload.keys())).lower():
        raise RuntimeError("Threshold checkpoint must never contain secret-key material")
    return payload


def _save_partial(
    method_dir: Path,
    config: base.MainExperimentConfig,
    round_completed: int,
    dev_n: int,
    global_state: dict[str, torch.Tensor],
    round_records: list[dict[str, Any]],
    progress_metrics: list[dict[str, Any]],
    progress_predictions: list[dict[str, Any]],
    key_epochs: list[dict[str, Any]],
) -> None:
    method_dir.mkdir(parents=True, exist_ok=True)
    base.atomic_write_csv(
        pd.DataFrame(progress_metrics), method_dir / TEMP_FILENAMES[1]
    )
    base.atomic_write_csv(
        pd.DataFrame(progress_predictions),
        method_dir / TEMP_FILENAMES[2],
        compression="gzip",
    )
    base.atomic_torch_save(
        {
            "schema_version": SCHEMA_VERSION,
            "method": METHOD,
            "core_config_sha256": v2.json_sha256(v2.core_config(config)),
            "runner_sha256": sha256_file(RUNNER_PATH),
            "round_completed": int(round_completed),
            "dev_n": int(dev_n),
            "global_state": global_state,
            "round_records": round_records,
            "key_epochs": key_epochs,
            "secret_key_shares_persisted": False,
        },
        method_dir / TEMP_FILENAMES[0],
    )
    base.atomic_write_json(
        method_dir / TEMP_FILENAMES[3],
        {
            "schema_version": SCHEMA_VERSION,
            "method": METHOD,
            "round_completed": int(round_completed),
            "target_round": int(getattr(config, "threshold_reference")["selected_round"]),
            "secret_key_shares_persisted": False,
        },
    )


def _fidelity(encrypted: np.ndarray, plain: np.ndarray) -> dict[str, float]:
    error = encrypted - plain
    plain_norm = max(float(np.linalg.norm(plain)), 1e-12)
    encrypted_norm = max(float(np.linalg.norm(encrypted)), 1e-12)
    return {
        "mae": float(np.mean(np.abs(error))),
        "max_abs": float(np.max(np.abs(error))),
        "relative_l2": float(np.linalg.norm(error) / plain_norm),
        "cosine": float(np.dot(encrypted, plain) / (encrypted_norm * plain_norm)),
    }


def train_threshold(
    config: base.MainExperimentConfig,
    frames: dict[str, pd.DataFrame],
    datasets: dict[str, pilot.TokenDataset],
    partitions: dict[str, dict[str, list[int]]],
    device: torch.device,
    init_state: dict[str, torch.Tensor],
) -> tuple[Any, dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    _verify_reference_pairing(config)
    method_dir = _method_dir(config)
    method_dir.mkdir(parents=True, exist_ok=True)
    target_round = int(getattr(config, "threshold_reference")["selected_round"])
    local_model = pilot.create_model(config, config.seed, device)
    partial = _load_partial(method_dir, config)
    if partial is None:
        start_round = 0
        global_state = {name: value.clone() for name, value in init_state.items()}
        round_records: list[dict[str, Any]] = []
        progress_metrics: list[dict[str, Any]] = []
        progress_predictions: list[dict[str, Any]] = []
        key_epochs: list[dict[str, Any]] = []
    else:
        start_round = int(partial["round_completed"])
        global_state = partial["global_state"]
        round_records = list(partial["round_records"])
        progress_metrics = list(partial["progress_metrics"])
        progress_predictions = list(partial["progress_predictions"])
        key_epochs = list(partial["key_epochs"])
        print(f"[resume-threshold] seed={config.seed} completed={start_round}")

    crypto_config: threshold.ThresholdCKKSConfig = getattr(
        config, "threshold_ckks_config"
    )
    aggregator = threshold.OpenFHEThresholdCKKSAggregator(crypto_config)
    key_epoch_index = len(key_epochs) + 1
    key_epochs.append(
        {
            "key_epoch": key_epoch_index,
            "starts_at_round": start_round + 1,
            "keygen_seconds": aggregator.keygen_seconds,
            "metadata": aggregator.security_metadata(),
            "resumed_with_fresh_key_shares": start_round > 0,
        }
    )

    dev_n = len(frames["dev"])
    for round_index in range(start_round, target_round):
        deltas: list[dict[str, torch.Tensor]] = []
        sample_counts: list[int] = []
        client_records: list[dict[str, Any]] = []
        for site_index, site in enumerate(pilot.SITE_NAMES):
            pilot.load_trainable_state(local_model, global_state)
            optimizer = torch.optim.AdamW(
                [parameter for parameter in local_model.parameters() if parameter.requires_grad],
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            local_seconds = 0.0
            epoch_details: list[dict[str, Any]] = []
            for local_epoch in range(config.local_epochs):
                paired_seed = config.seed + round_index * 1000 + site_index * 100 + local_epoch
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
                    local_model,
                    loader,
                    device,
                    optimizer,
                    max_grad_norm=config.max_grad_norm,
                )
                local_seconds += seconds
                epoch_details.append(
                    {"local_epoch": local_epoch + 1, "loss": loss, **detail}
                )
            local_state = pilot.clone_trainable_state(local_model)
            deltas.append(pilot.state_delta(local_state, global_state))
            count = len(partitions["train"][site])
            sample_counts.append(count)
            client_records.append(
                {
                    "site": site,
                    "n": count,
                    "epochs": epoch_details,
                    "train_seconds": local_seconds,
                }
            )

        total = sum(sample_counts)
        weights = [count / total for count in sample_counts]
        plaintext_delta = pilot.weighted_average_deltas(deltas, weights)
        weighted_vectors: list[np.ndarray] = []
        state_manifest = None
        for delta, weight in zip(deltas, weights, strict=True):
            vector, current_manifest = pilot.flatten_state(delta)
            if state_manifest is None:
                state_manifest = current_manifest
            elif state_manifest != current_manifest:
                raise RuntimeError("Client trainable-state manifests differ")
            weighted_vectors.append(vector * weight)
        if state_manifest is None:
            raise RuntimeError("No client state was available for threshold aggregation")
        threshold_vector, crypto_stats = aggregator.aggregate(weighted_vectors)
        plain_vector, _ = pilot.flatten_state(plaintext_delta)
        fidelity = _fidelity(threshold_vector, plain_vector)
        crypto_stats["key_epoch"] = key_epoch_index
        aggregate_delta = pilot.unflatten_state(threshold_vector, state_manifest)
        global_state = pilot.add_delta(global_state, aggregate_delta)
        pilot.load_trainable_state(local_model, global_state)

        metric_rows, prediction_rows, pooled = v2.collect_global_dev_checkpoint(
            local_model,
            METHOD,
            config,
            frames,
            datasets,
            partitions,
            device,
            progress_kind="round",
            progress=round_index + 1,
            seed_offset=30_000 + round_index * 10,
            local_epoch=config.local_epochs,
        )
        progress_metrics.extend(metric_rows)
        progress_predictions.extend(prediction_rows)
        round_records.append(
            {
                "round": round_index + 1,
                "clients": client_records,
                "weights": weights,
                "fedprox_mu": 0.0,
                "dev_pooled": pooled,
                "crypto": crypto_stats,
                "fidelity": fidelity,
            }
        )
        _save_partial(
            method_dir,
            config,
            round_index + 1,
            dev_n,
            global_state,
            round_records,
            progress_metrics,
            progress_predictions,
            key_epochs,
        )
        print(
            f"[threshold-round] seed={config.seed} round={round_index + 1}/"
            f"{target_round} dev_auprc={float(pooled['auprc']):.9f} "
            f"max_abs={fidelity['max_abs']:.3e}",
            flush=True,
        )

    pilot.load_trainable_state(local_model, global_state)
    reference = getattr(config, "threshold_reference")
    selection = {
        "selection_owner": "frozen_plain_fedavg",
        "selected_round": target_round,
        "selected_dev_auprc_from_plain": reference["selected_dev_auprc"],
        "test_policy": "exactly_once_after_frozen_plain_selection",
        "reference_done_sha256": reference["done_sha256"],
    }
    information = {
        "rounds": round_records,
        "fedprox_mu": 0.0,
        "selection": selection,
        "threshold_key_epochs": key_epochs,
        "secret_key_shares_persisted": False,
    }
    trainable = int(
        sum(parameter.numel() for parameter in local_model.parameters() if parameter.requires_grad)
    )
    audit = {
        "adaptation": "lora",
        "aggregation": "openfhe_multiparty_ckks_weighted_sum",
        "local_objective": "empirical_risk",
        "fedprox_mu": 0.0,
        "trainable_parameters": trainable,
        "selection": selection,
        "plain_reference": reference,
        "security_scope": aggregator.security_metadata(),
    }
    return local_model, information, audit, progress_metrics, progress_predictions


def _run_method(
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
    if method != METHOD:
        raise ValueError(f"Unsupported threshold method: {method}")
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    model, stats, audit, progress_metrics, progress_predictions = train_threshold(
        config, frames, datasets, partitions, device, init_state
    )
    final_metrics, final_predictions = base.evaluate_global_model(
        model,
        METHOD,
        "test",
        frames["test"],
        datasets["test"],
        partitions["test"],
        config,
        device,
        50_000,
    )
    state = pilot.clone_trainable_state(model) if config.save_model_states else None
    vector, manifest = pilot.flatten_state(init_state)
    partition_digests = v2.partition_hashes(run_dir)
    audit["paired_protocol"] = {
        "applicable": True,
        "adaptation": "lora",
        "initial_trainable_state_sha256": hashlib.sha256(vector.tobytes()).hexdigest(),
        "trainable_manifest_sha256": v2.json_sha256(manifest),
        "partition_sha256": partition_digests,
        "partition_combined_sha256": partition_digests["combined_sha256"],
        "comparison_reference": getattr(config, "threshold_reference"),
    }
    expected_initial = v2.expected_method_initial_state(config, METHOD)
    if (
        audit["paired_protocol"]["initial_trainable_state_sha256"]
        != expected_initial["initial_state_sha256"]
        or audit["paired_protocol"]["trainable_manifest_sha256"]
        != expected_initial["manifest_sha256"]
    ):
        raise RuntimeError("Threshold actual initial state differs from prelaunch provenance")
    audit["progress_logging"] = {
        "schema_version": v2.PROGRESS_SCHEMA_VERSION,
        "split": "dev",
        "test_policy": "final_once",
        "metrics": list(v2.PROGRESS_METRIC_NAMES),
        "prediction_scopes": ["pooled", "site_A", "site_B", "site_C"],
    }
    audit["reproducibility"] = {
        "deterministic_model_training": bool(getattr(config, "deterministic", True)),
        "openfhe_encryption_randomness_fixed": False,
        "reference_selection_locked": True,
        "secret_key_shares_saved": False,
        "resume_key_policy": "generate a fresh multiparty key epoch",
    }
    stats["evaluation_policy"] = {
        "progress_split": "dev",
        "federated_frequency": "every_round",
        "test": "exactly_once_after_frozen_plain_selected_round",
        "selected_round": getattr(config, "threshold_reference")["selected_round"],
    }
    if device.type == "cuda":
        torch.cuda.synchronize()
        stats["resource"] = {
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_cuda_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "end_cuda_memory_bytes": int(torch.cuda.memory_allocated(device)),
            "peak_process_rss_bytes": base.process_peak_rss_bytes(),
        }
    v2.finalize_method_v2(
        METHOD,
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
    if (method_dir / "DONE.json").is_file():
        for name in TEMP_FILENAMES:
            path = method_dir / name
            if path.exists():
                path.unlink()
    del model
    if state is not None:
        del state
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _plan(args: argparse.Namespace) -> dict[str, Any]:
    selections = []
    for seed in (42, 43, 44, 45, 46):
        try:
            info = _reference_info(Path(args.reference_root).resolve(), seed)
            selections.append({"seed": seed, "selected_round": info["selected_round"]})
        except Exception as exc:
            selections.append({"seed": seed, "error": f"{type(exc).__name__}: {exc}"})
    return {
        "schema_version": SCHEMA_VERSION,
        "training_started": False,
        "execution_locked": True,
        "unlock_flag": "--execute",
        "new_method": METHOD,
        "new_arm_count": 5,
        "current_seed_if_unlocked": args.seed,
        "scenario": "noniid",
        "adaptation": "LoRA",
        "selection_policy": "reuse each seed's frozen Plain early-stop selected round",
        "selections": selections,
        "security_scope": "in-process multiparty CKKS simulation; not physical isolation",
    }


def main() -> None:
    args = parse_args()
    if not args.execute:
        print(json.dumps(_plan(args), indent=2, ensure_ascii=False))
        return

    readiness = preflight.collect_preflight(
        argparse.Namespace(
            data_dir=args.data_dir,
            reference_root=args.reference_root,
            model_snapshot_dir=args.model_snapshot_dir,
            crypto_smoke=False,
        )
    )
    if not readiness["full_experiment_ready"]:
        raise RuntimeError(
            "Threshold experiment preflight failed; run preflight_threshold_ckks.py "
            "and repair the reported environment before retrying"
        )

    v2.SCHEMA_VERSION = SCHEMA_VERSION
    v2.__file__ = str(RUNNER_PATH)
    v2.METHODS = METHODS
    v2.PAIRINGS = ()
    v2.parse_args = lambda: args
    v2.make_config = _make_config
    v2.core_config = _core_config
    v2.run_method_v2 = _run_method
    v2.main()


if __name__ == "__main__":
    main()
