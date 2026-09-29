from __future__ import annotations

"""Convergence-oriented H3K4me3 LoRA FedAvg/CKKS experiment.

The Plain arm selects a checkpoint with pooled DEV AUPRC only.  The paired HE
arm is then trained for exactly the Plain-selected number of rounds.  TEST is
evaluated exactly once by the validated progress-v2 finalizer.

This runner is additive: it never writes into v1/v2 result roots.  A compact
round checkpoint and partial DEV ledgers make an interrupted method resumable.
"""

import argparse
import gc
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_fedhe_experiment as pilot
import run_fedhe_main_experiment as base
import run_fedhe_progress_v2 as v2


SCHEMA_VERSION = "fedhe-earlystop-v3"
DEFAULT_OUTPUT = pilot.PROJECT_ROOT / "experiments" / "h3k4me3_fl_earlystop_v3"
PROGRESS_V2_PATH = Path(v2.__file__).resolve()
PROGRESS_V2_SHA256 = v2.sha256_file(PROGRESS_V2_PATH)
RUNNER_PATH = Path(__file__).resolve()
METHODS = ("fedavg", "he_fedavg")
TEMP_FILENAMES = (
    "round_checkpoint.pt",
    "partial_progress_metrics.csv",
    "partial_progress_predictions.csv.gz",
    "partial_status.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "H3K4me3 non-IID LoRA FedAvg convergence experiment: Plain DEV-AUPRC "
            "early stopping, paired HE at the same selected round"
        )
    )
    parser.add_argument(
        "--data-dir", default=str(pilot.PROJECT_ROOT / "GUE_v2" / "EMP" / "H3K4me3")
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
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
    parser.add_argument("--min-rounds", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.001)
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
    parser.add_argument("--scenario", choices=("iid", "noniid"), default="noniid")
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
        default=False,
    )
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.scenario != "noniid":
        raise ValueError("The frozen v3 study is scoped to non-IID only")
    if args.local_epochs != 1:
        raise ValueError("The frozen v3 study requires one local epoch per round")
    if args.min_rounds < 1 or args.rounds < args.min_rounds:
        raise ValueError("Require 1 <= min_rounds <= max_rounds")
    if args.patience < 1 or args.min_delta <= 0:
        raise ValueError("patience and min_delta must be positive")
    return args


def advance_patience(
    score: float,
    patience_anchor: float,
    stale_rounds: int,
    min_delta: float,
) -> tuple[float, int, bool]:
    """Update the clinically interpretable improvement counter.

    Small gains accumulate relative to ``patience_anchor``.  They reset patience
    once their total reaches ``min_delta``.
    """

    if not math.isfinite(score):
        raise ValueError("DEV AUPRC must be finite")
    if not math.isfinite(patience_anchor) or score >= patience_anchor + min_delta:
        return score, 0, True
    return patience_anchor, stale_rounds + 1, False


def _make_config(args: argparse.Namespace, snapshot_audit: dict[str, Any]) -> base.MainExperimentConfig:
    config = _ORIGINAL_MAKE_CONFIG(args, snapshot_audit)
    setattr(config, "early_stop_min_rounds", int(args.min_rounds))
    setattr(config, "early_stop_patience", int(args.patience))
    setattr(config, "early_stop_min_delta", float(args.min_delta))
    setattr(config, "early_stop_metric", "pooled_dev_auprc")
    setattr(config, "early_stop_selection_owner", "fedavg")
    setattr(config, "he_round_policy", "same_seed_plain_selected_round")
    return config


def _core_config(config: base.MainExperimentConfig) -> dict[str, Any]:
    payload = _ORIGINAL_CORE_CONFIG(config)
    payload["early_stopping"] = {
        "metric": getattr(config, "early_stop_metric"),
        "min_rounds": int(getattr(config, "early_stop_min_rounds")),
        "max_rounds": int(config.rounds),
        "patience": int(getattr(config, "early_stop_patience")),
        "min_delta": float(getattr(config, "early_stop_min_delta")),
        "selection_owner": getattr(config, "early_stop_selection_owner"),
        "tie_break": "earlier_round",
        "test_policy": "exactly_once_after_selection",
        "he_round_policy": getattr(config, "he_round_policy"),
    }
    payload["progress_v2_dependency"] = {
        "path": str(PROGRESS_V2_PATH),
        "sha256": PROGRESS_V2_SHA256,
    }
    return payload


def _method_dir(config: base.MainExperimentConfig, method: str) -> Path:
    return (
        Path(config.output_dir)
        / config.scenario
        / f"seed_{config.seed}"
        / "methods"
        / method
    )


def _state_sha256(state: dict[str, torch.Tensor]) -> str:
    vector, manifest = pilot.flatten_state(state)
    return hashlib.sha256(vector.tobytes()).hexdigest() + ":" + v2.json_sha256(manifest)


def _restore_ckks_context(aggregator: Any, secret_blob: bytes, keygen_seconds: float) -> None:
    secret = pilot.ts.context_from(secret_blob)
    public_blob = secret.serialize(
        save_public_key=True,
        save_secret_key=False,
        save_galois_keys=False,
        save_relin_keys=False,
    )
    aggregator.secret_context = secret
    aggregator.public_context = pilot.ts.context_from(public_blob)
    aggregator.keygen_seconds = float(keygen_seconds)
    aggregator.public_context_bytes = len(public_blob)
    aggregator.secret_context_bytes = len(secret_blob)


def _load_plain_selection(config: base.MainExperimentConfig) -> tuple[dict[str, Any], str]:
    plain_dir = _method_dir(config, "fedavg")
    fingerprint = v2.json_sha256(v2.core_config(config))
    if not v2.method_artifacts_valid_v2("fedavg", plain_dir, config, fingerprint):
        raise RuntimeError("Paired HE cannot start before the Plain DONE ledger validates")
    audit_path = plain_dir / "method_model_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    selection = audit.get("early_stopping")
    if not isinstance(selection, dict) or selection.get("selection_owner") != "fedavg":
        raise RuntimeError("Plain method audit lacks the frozen early-stop selection")
    done_sha = v2.sha256_file(plain_dir / "DONE.json")
    return selection, done_sha


def _load_partial(method_dir: Path, config: base.MainExperimentConfig, method: str) -> dict[str, Any] | None:
    checkpoint_path = method_dir / "round_checkpoint.pt"
    partial_paths = [method_dir / name for name in TEMP_FILENAMES[1:]]
    if not checkpoint_path.exists():
        if any(path.exists() for path in partial_paths):
            raise RuntimeError(f"Partial ledgers exist without a checkpoint in {method_dir}")
        return None
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_fingerprint = v2.json_sha256(v2.core_config(config))
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("method") != method
        or payload.get("core_config_sha256") != expected_fingerprint
        or payload.get("runner_sha256") != v2.sha256_file(RUNNER_PATH)
    ):
        raise RuntimeError(f"Incompatible round checkpoint in {method_dir}")
    metrics_path = method_dir / "partial_progress_metrics.csv"
    predictions_path = method_dir / "partial_progress_predictions.csv.gz"
    if not metrics_path.is_file() or not predictions_path.is_file():
        raise RuntimeError("Round checkpoint lacks its partial DEV ledgers")
    payload["progress_metrics"] = pd.read_csv(metrics_path).to_dict("records")
    payload["progress_predictions"] = pd.read_csv(predictions_path).to_dict("records")
    completed = int(payload["round_completed"])
    expected_metric_rows = completed * (len(pilot.SITE_NAMES) + 1)
    expected_prediction_rows = completed * 2 * int(payload["dev_n"])
    if len(payload["progress_metrics"]) != expected_metric_rows:
        raise RuntimeError("Partial progress metric coverage mismatch")
    if len(payload["progress_predictions"]) != expected_prediction_rows:
        raise RuntimeError("Partial progress prediction coverage mismatch")
    return payload


def _save_partial(
    method_dir: Path,
    config: base.MainExperimentConfig,
    method: str,
    round_completed: int,
    dev_n: int,
    global_state: dict[str, torch.Tensor],
    best_state: dict[str, torch.Tensor] | None,
    best_score: float,
    best_round: int,
    patience_anchor: float,
    stale_rounds: int,
    round_records: list[dict[str, Any]],
    progress_metrics: list[dict[str, Any]],
    progress_predictions: list[dict[str, Any]],
    aggregator: Any | None,
) -> None:
    method_dir.mkdir(parents=True, exist_ok=True)
    base.atomic_write_csv(
        pd.DataFrame(progress_metrics), method_dir / "partial_progress_metrics.csv"
    )
    base.atomic_write_csv(
        pd.DataFrame(progress_predictions),
        method_dir / "partial_progress_predictions.csv.gz",
        compression="gzip",
    )
    secret_blob = None
    keygen_seconds = None
    if aggregator is not None:
        secret_blob = aggregator.secret_context.serialize(save_secret_key=True)
        keygen_seconds = float(aggregator.keygen_seconds)
    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "method": method,
        "core_config_sha256": v2.json_sha256(v2.core_config(config)),
        "runner_sha256": v2.sha256_file(RUNNER_PATH),
        "round_completed": int(round_completed),
        "dev_n": int(dev_n),
        "global_state": global_state,
        "best_state": best_state,
        "best_score": float(best_score),
        "best_round": int(best_round),
        "patience_anchor": float(patience_anchor),
        "stale_rounds": int(stale_rounds),
        "round_records": round_records,
        "ckks_secret_context": secret_blob,
        "ckks_keygen_seconds": keygen_seconds,
    }
    base.atomic_torch_save(checkpoint, method_dir / "round_checkpoint.pt")
    base.atomic_write_json(
        method_dir / "partial_status.json",
        {
            "schema_version": SCHEMA_VERSION,
            "method": method,
            "round_completed": int(round_completed),
            "best_round": int(best_round),
            "best_dev_auprc": float(best_score),
            "stale_rounds": int(stale_rounds),
        },
    )


def train_federated_earlystop_v3(
    method: str,
    config: base.MainExperimentConfig,
    frames: dict[str, pd.DataFrame],
    datasets: dict[str, pilot.TokenDataset],
    partitions: dict[str, dict[str, list[int]]],
    device: torch.device,
    init_state: dict[str, torch.Tensor],
) -> tuple[Any, dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if method not in METHODS:
        raise ValueError(f"v3 supports only {METHODS}; got {method}")
    use_he = method == "he_fedavg"
    method_dir = _method_dir(config, method)
    method_dir.mkdir(parents=True, exist_ok=True)
    plain_selection: dict[str, Any] | None = None
    plain_done_sha256: str | None = None
    if use_he:
        plain_selection, plain_done_sha256 = _load_plain_selection(config)
        target_rounds = int(plain_selection["selected_round"])
        if not 1 <= target_rounds <= config.rounds:
            raise RuntimeError("Plain-selected round is outside the frozen v3 bounds")
    else:
        target_rounds = int(config.rounds)

    local_model = pilot.create_model(config, config.seed, device)
    aggregator = pilot.CKKSAdditiveAggregator(config) if use_he else None
    partial = _load_partial(method_dir, config, method)
    if partial is None:
        start_round = 0
        global_state = {name: value.clone() for name, value in init_state.items()}
        best_state = None
        best_score = -math.inf
        best_round = 0
        patience_anchor = -math.inf
        stale_rounds = 0
        round_records: list[dict[str, Any]] = []
        progress_metrics: list[dict[str, Any]] = []
        progress_predictions: list[dict[str, Any]] = []
    else:
        start_round = int(partial["round_completed"])
        global_state = partial["global_state"]
        best_state = partial["best_state"]
        best_score = float(partial["best_score"])
        best_round = int(partial["best_round"])
        patience_anchor = float(partial["patience_anchor"])
        stale_rounds = int(partial["stale_rounds"])
        round_records = list(partial["round_records"])
        progress_metrics = list(partial["progress_metrics"])
        progress_predictions = list(partial["progress_predictions"])
        if use_he:
            secret_blob = partial.get("ckks_secret_context")
            if not isinstance(secret_blob, bytes):
                raise RuntimeError("Resumed HE checkpoint lacks the original CKKS secret context")
            assert aggregator is not None
            _restore_ckks_context(
                aggregator, secret_blob, float(partial["ckks_keygen_seconds"])
            )
        print(f"[resume-round] {method} seed={config.seed} completed={start_round}")

    stop_reason = "paired_plain_selected_round" if use_he else "max_rounds"
    min_rounds = int(getattr(config, "early_stop_min_rounds"))
    patience = int(getattr(config, "early_stop_patience"))
    min_delta = float(getattr(config, "early_stop_min_delta"))
    dev_n = len(frames["dev"])

    for round_index in range(start_round, target_rounds):
        deltas: list[dict[str, torch.Tensor]] = []
        sample_counts: list[int] = []
        client_records: list[dict[str, Any]] = []
        for site_index, site in enumerate(pilot.SITE_NAMES):
            pilot.load_trainable_state(local_model, global_state)
            optimizer = torch.optim.AdamW(
                [p for p in local_model.parameters() if p.requires_grad],
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
            n_site = len(partitions["train"][site])
            sample_counts.append(n_site)
            client_records.append(
                {
                    "site": site,
                    "n": n_site,
                    "epochs": epoch_details,
                    "train_seconds": local_seconds,
                }
            )

        total_n = sum(sample_counts)
        weights = [count / total_n for count in sample_counts]
        plaintext_delta = pilot.weighted_average_deltas(deltas, weights)
        crypto_stats = None
        fidelity = None
        if use_he:
            weighted_vectors: list[np.ndarray] = []
            state_manifest = None
            for delta, weight in zip(deltas, weights, strict=True):
                vector, current_manifest = pilot.flatten_state(delta)
                if state_manifest is None:
                    state_manifest = current_manifest
                elif state_manifest != current_manifest:
                    raise RuntimeError("Client trainable-state manifests differ")
                weighted_vectors.append(vector * weight)
            assert aggregator is not None and state_manifest is not None
            he_vector, crypto_stats = aggregator.aggregate(weighted_vectors)
            plain_vector, _ = pilot.flatten_state(plaintext_delta)
            error = he_vector - plain_vector
            denominator = max(float(np.linalg.norm(plain_vector)), 1e-12)
            fidelity = {
                "mae": float(np.mean(np.abs(error))),
                "max_abs": float(np.max(np.abs(error))),
                "relative_l2": float(np.linalg.norm(error) / denominator),
                "cosine": float(
                    np.dot(he_vector, plain_vector)
                    / (max(float(np.linalg.norm(he_vector)), 1e-12) * denominator)
                ),
            }
            aggregate_delta = pilot.unflatten_state(he_vector, state_manifest)
        else:
            aggregate_delta = plaintext_delta
        global_state = pilot.add_delta(global_state, aggregate_delta)
        pilot.load_trainable_state(local_model, global_state)
        metric_rows, prediction_rows, pooled = v2.collect_global_dev_checkpoint(
            local_model,
            method,
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
        score = float(pooled["auprc"])
        if score > best_score:
            best_score = score
            best_round = round_index + 1
            best_state = {name: value.clone() for name, value in global_state.items()}
        patience_anchor, stale_rounds, meaningful = advance_patience(
            score, patience_anchor, stale_rounds, min_delta
        )
        round_records.append(
            {
                "round": round_index + 1,
                "clients": client_records,
                "weights": weights,
                "fedprox_mu": 0.0,
                "dev_pooled": pooled,
                "crypto": crypto_stats,
                "fidelity": fidelity,
                "early_stop": {
                    "score": score,
                    "meaningful_improvement": bool(meaningful),
                    "patience_anchor": patience_anchor,
                    "stale_rounds": stale_rounds,
                    "best_round_so_far": best_round,
                    "best_score_so_far": best_score,
                },
            }
        )
        _save_partial(
            method_dir,
            config,
            method,
            round_index + 1,
            dev_n,
            global_state,
            best_state,
            best_score,
            best_round,
            patience_anchor,
            stale_rounds,
            round_records,
            progress_metrics,
            progress_predictions,
            aggregator,
        )
        print(
            f"[round] method={method} seed={config.seed} round={round_index + 1} "
            f"dev_auprc={score:.9f} best={best_score:.9f}@{best_round} "
            f"stale={stale_rounds}/{patience}",
            flush=True,
        )
        if not use_he and round_index + 1 >= min_rounds and stale_rounds >= patience:
            stop_reason = "patience_exhausted"
            break

    executed_rounds = len(round_records)
    if use_he:
        selected_round = target_rounds
        selected_score = float(round_records[-1]["dev_pooled"]["auprc"])
        selected_state = global_state
    else:
        if best_state is None or best_round < 1:
            raise RuntimeError("Plain early stopping failed to select a DEV checkpoint")
        selected_round = best_round
        selected_score = best_score
        selected_state = best_state
    pilot.load_trainable_state(local_model, selected_state)

    selection = {
        "schema_version": SCHEMA_VERSION,
        "selection_owner": "fedavg" if not use_he else "paired_fedavg",
        "metric": "pooled_dev_auprc",
        "min_rounds": min_rounds,
        "max_rounds": int(config.rounds),
        "patience": patience,
        "min_delta": min_delta,
        "executed_rounds": executed_rounds,
        "selected_round": int(selected_round),
        "selected_dev_auprc": float(selected_score),
        "stop_reason": stop_reason,
        "test_policy": "exactly_once_after_selection",
        "selected_state_sha256_and_manifest": _state_sha256(selected_state),
        "plain_done_sha256": plain_done_sha256,
    }
    information: dict[str, Any] = {
        "rounds": round_records,
        "fedprox_mu": 0.0,
        "early_stopping": selection,
    }
    if aggregator is not None:
        information["ckks_context"] = {
            "poly_modulus_degree": config.ckks_poly_modulus_degree,
            "coeff_mod_bit_sizes": config.ckks_coeff_mod_bits,
            "scale_bits": config.ckks_scale_bits,
            "slots": aggregator.slots,
            "keygen_seconds": aggregator.keygen_seconds,
            "public_context_bytes": aggregator.public_context_bytes,
            "secret_context_bytes": aggregator.secret_context_bytes,
            "resumable_secret_context": True,
        }
    trainable = int(sum(p.numel() for p in local_model.parameters() if p.requires_grad))
    audit = {
        "adaptation": "lora",
        "aggregation": "ckks_weighted_sum" if use_he else "plaintext_weighted_mean",
        "local_objective": "empirical_risk",
        "fedprox_mu": 0.0,
        "trainable_parameters": trainable,
        "early_stopping": selection,
        "plain_selection": plain_selection if use_he else None,
    }
    return local_model, information, audit, progress_metrics, progress_predictions


def _run_method_v3(*args: Any, **kwargs: Any) -> None:
    method_dir = Path(args[1]) if len(args) > 1 else Path(kwargs["method_dir"])
    _ORIGINAL_RUN_METHOD(*args, **kwargs)
    if (method_dir / "DONE.json").is_file():
        for name in TEMP_FILENAMES:
            path = method_dir / name
            if path.exists():
                path.unlink()


_ORIGINAL_MAKE_CONFIG = v2.make_config
_ORIGINAL_CORE_CONFIG = v2.core_config
_ORIGINAL_RUN_METHOD = v2.run_method_v2


def main() -> None:
    # Patch only the imported v2 orchestration surface.  The dependency hash is
    # embedded in the v3 core fingerprint; the v2 source itself remains frozen.
    v2.SCHEMA_VERSION = SCHEMA_VERSION
    v2.__file__ = str(RUNNER_PATH)
    v2.parse_args = parse_args
    v2.make_config = _make_config
    v2.core_config = _core_config
    v2.train_federated_v2 = train_federated_earlystop_v3
    v2.run_method_v2 = _run_method_v3
    v2.main()


if __name__ == "__main__":
    main()
