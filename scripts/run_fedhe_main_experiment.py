from __future__ import annotations

"""Full-data DNABERT-2 federated/HE experiment runner.

This runner intentionally lives beside, rather than replaces,
``run_fedhe_experiment.py``.  The latter is the immutable pilot runner.  This
module reuses its model/data primitives while adding the controls needed for a
main experiment:

* full-data defaults and an equal data-pass guard;
* pooled centralized full fine-tuning (memory-safe micro-batching);
* plaintext and streaming-CKKS federated full fine-tuning;
* FedProx in addition to FedAvg and CKKS HE-FedAvg;
* per-example predictions and calibration metrics;
* method-level atomic artifacts and safe resume markers.

The run directory schema is versioned.  A directory produced by an older
runner is never silently reused or overwritten.
"""

import argparse
import ctypes
from ctypes import wintypes
import gc
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import truststore
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, log_loss

import run_fedhe_experiment as pilot


SCHEMA_VERSION = "fedhe-main-v1"
EXPECTED_FULL_FT_TRAINABLE_PARAMETERS = 116_479_490
METHODS = (
    "local",
    "central",
    "central_full_ft",
    "fedavg_full_ft",
    "he_fedavg_full_ft",
    "fedavg",
    "fedprox",
    "he_fedavg",
)


@dataclass
class MainExperimentConfig(pilot.ExperimentConfig):
    fedprox_mu: float
    ece_bins: int
    eval_splits: list[str]
    round_eval_interval: int
    max_grad_norm: float
    full_ft_learning_rate: float
    full_ft_batch_size: int
    full_ft_grad_accum_steps: int
    full_ft_gradient_checkpointing: bool
    full_ft_amp_dtype: str
    save_predictions: bool
    save_model_states: bool
    enforce_equal_data_passes: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-data DNABERT-2 LoRA federated/CKKS experiment")
    parser.add_argument("--data-dir", default=str(pilot.PROJECT_ROOT / "GUE_v2" / "EMP" / "H3K4me3"))
    parser.add_argument("--output-dir", default=str(pilot.PROJECT_ROOT / "experiments" / "main_h3k4me3"))
    parser.add_argument("--train-limit", type=int, default=0, help="0 uses the complete training split")
    parser.add_argument("--dev-limit", type=int, default=0, help="0 uses the complete development split")
    parser.add_argument("--test-limit", type=int, default=0, help="0 uses the complete held-out test split")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16, help="LoRA micro/effective batch size")
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
    parser.add_argument("--eval-splits", nargs="+", choices=("dev", "test"), default=["dev", "test"])
    parser.add_argument("--round-eval-interval", type=int, default=1, help="0 disables round-wise dev evaluation")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--full-ft-batch-size", type=int, default=4)
    parser.add_argument("--full-ft-grad-accum-steps", type=int, default=4)
    parser.add_argument(
        "--full-ft-amp-dtype",
        choices=("auto", "bf16", "fp16"),
        default="auto",
        help="auto prefers BF16 on supported GPUs for full fine-tuning stability",
    )
    parser.add_argument(
        "--full-ft-gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--save-predictions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-model-states", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enforce-equal-data-passes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip methods having a valid DONE.json; incomplete methods are rerun",
    )
    return parser.parse_args()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True))


def atomic_write_csv(frame: pd.DataFrame, path: Path, *, compression: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".tmp.gz" if compression == "gzip" else ".tmp"
    temporary = path.with_name(path.name + suffix)
    frame.to_csv(temporary, index=False, compression=compression)
    temporary.replace(path)


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def process_peak_rss_bytes() -> int | None:
    """Return the process peak resident/working-set size without dependencies."""
    if os.name == "nt":
        class ProcessMemoryCountersEx(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(counters)
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess  # type: ignore[attr-defined]
        get_current_process.argtypes = []
        get_current_process.restype = wintypes.HANDLE
        get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo  # type: ignore[attr-defined]
        get_process_memory_info.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        get_process_memory_info.restype = wintypes.BOOL
        success = get_process_memory_info(
            get_current_process(),
            ctypes.byref(counters),
            counters.cb,
        )
        return int(counters.PeakWorkingSetSize) if success else None
    try:
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB; macOS reports bytes.
        return peak if platform.system() == "Darwin" else peak * 1024
    except (ImportError, OSError):
        return None


def get_git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=pilot.PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def validate_config(config: MainExperimentConfig) -> None:
    if config.enforce_equal_data_passes and config.epochs != config.rounds * config.local_epochs:
        raise ValueError(
            "Equal data-pass budget violated: epochs must equal rounds * local_epochs "
            f"({config.epochs} != {config.rounds} * {config.local_epochs})."
        )
    if config.ece_bins < 2:
        raise ValueError("ece_bins must be >= 2")
    if config.fedprox_mu < 0:
        raise ValueError("fedprox_mu must be non-negative")
    if min(config.batch_size, config.eval_batch_size, config.full_ft_batch_size) < 1:
        raise ValueError("Batch sizes must be positive")
    if config.full_ft_grad_accum_steps < 1:
        raise ValueError("full_ft_grad_accum_steps must be positive")
    if (
        config.enforce_equal_data_passes
        and config.full_ft_batch_size * config.full_ft_grad_accum_steps != config.batch_size
    ):
        raise ValueError(
            "Matched effective batch size violated: full_ft_batch_size * "
            "full_ft_grad_accum_steps must equal the LoRA batch_size "
            f"({config.full_ft_batch_size} * {config.full_ft_grad_accum_steps} != {config.batch_size})."
        )
    if config.round_eval_interval < 0:
        raise ValueError("round_eval_interval must be non-negative")


def core_config(config: MainExperimentConfig) -> dict[str, Any]:
    """Return immutable fields used to determine resume compatibility."""
    payload = asdict(config)
    for key in ("output_dir", "methods", "save_predictions", "save_model_states"):
        payload.pop(key, None)
    return payload


def prepare_run_dir(run_dir: Path, config: MainExperimentConfig, resume: bool) -> dict[str, Any]:
    manifest_path = run_dir / "run_manifest.json"
    immutable = core_config(config)
    fingerprint = json_hash(immutable)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError(f"Run schema mismatch in {run_dir}; choose a new --output-dir")
        if manifest.get("core_config_sha256") != fingerprint:
            raise RuntimeError(
                f"Configuration mismatch in existing run {run_dir}. "
                "To preserve prior results, choose a new --output-dir instead of overwriting it."
            )
        if not resume:
            raise RuntimeError(f"Run already exists at {run_dir}; use --resume or a new --output-dir")
        return manifest
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(
            f"Non-empty directory without a {SCHEMA_VERSION} manifest: {run_dir}. "
            "Refusing to overwrite legacy/pilot artifacts."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_unix": time.time(),
        "core_config_sha256": fingerprint,
        "core_config": immutable,
        "requested_methods_initial": config.methods,
        "git_revision": get_git_revision(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "platform": platform.platform(),
        },
    }
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(run_dir / "config.json", asdict(config))
    return manifest


def stratified_subset_with_ids(frame: pd.DataFrame, limit: int, seed: int, split: str) -> pd.DataFrame:
    working = frame.copy()
    working.insert(0, "source_row", np.arange(len(working), dtype=np.int64))
    subset = pilot.stratified_subset(working, limit, seed)
    subset.insert(0, "stable_id", [f"{split}_{int(row):06d}" for row in subset["source_row"]])
    return subset


def load_frames(data_dir: Path, limits: tuple[int, int, int], seed: int) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    frames: dict[str, pd.DataFrame] = {}
    audit: dict[str, Any] = {}
    for offset, (split, limit) in enumerate(zip(("train", "dev", "test"), limits, strict=True)):
        path = data_dir / f"{split}.csv"
        frame = pd.read_csv(path)
        if list(frame.columns) != ["sequence", "label"]:
            raise ValueError(f"Unexpected columns in {path}: {list(frame.columns)}")
        frame["sequence"] = frame["sequence"].astype(str).str.upper()
        frame["label"] = frame["label"].astype(int)
        if not set(frame["label"].unique()).issubset({0, 1}):
            raise ValueError(f"Binary labels expected in {path}")
        if frame["sequence"].str.contains(r"[^ACGT]", regex=True).any():
            raise ValueError(f"Non-ACGT characters found in {path}")
        subset = stratified_subset_with_ids(frame, limit, seed + offset, split)
        frames[split] = subset
        stable_digest = hashlib.sha256("\n".join(subset["stable_id"]).encode("utf-8")).hexdigest()
        audit[split] = {
            "source_path": str(path.resolve()),
            "source_sha256": pilot.sha256_file(path),
            "source_n": int(len(frame)),
            "used_n": int(len(subset)),
            "used_fraction": float(len(subset) / len(frame)),
            "used_stable_ids_sha256": stable_digest,
            "used_class_counts": {str(k): int(v) for k, v in subset["label"].value_counts().sort_index().items()},
            "sequence_length_min": int(subset["sequence"].str.len().min()),
            "sequence_length_median": float(subset["sequence"].str.len().median()),
            "sequence_length_max": int(subset["sequence"].str.len().max()),
            "exact_duplicate_sequences": int(subset.duplicated("sequence").sum()),
        }
    return frames, audit


def export_partitions(
    run_dir: Path,
    frames: dict[str, pd.DataFrame],
    partitions: dict[str, dict[str, list[int]]],
) -> None:
    for split, frame in frames.items():
        site_by_index: dict[int, str] = {}
        for site, indices in partitions[split].items():
            site_by_index.update({int(index): site for index in indices})
        export = frame[["stable_id", "source_row", "sequence", "label"]].copy()
        export["site"] = [site_by_index[index] for index in range(len(export))]
        atomic_write_csv(export, run_dir / f"{split}_partition.csv")


def expected_calibration_error(labels: np.ndarray, scores: np.ndarray, bins: int) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    # score==1.0 belongs to the final bin.
    membership = np.minimum(np.digitize(scores, edges[1:-1], right=False), bins - 1)
    result = 0.0
    for bin_index in range(bins):
        mask = membership == bin_index
        if not np.any(mask):
            continue
        result += float(mask.mean()) * abs(float(scores[mask].mean()) - float(labels[mask].mean()))
    return float(result)


def expanded_binary_metrics(labels: np.ndarray, scores: np.ndarray, ece_bins: int) -> dict[str, float]:
    predictions = (scores >= 0.5).astype(np.int64)
    metrics = pilot.binary_metrics(labels, scores)
    prevalence = float(np.mean(labels))
    return {
        **metrics,
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "brier": float(brier_score_loss(labels, scores)),
        "ece": expected_calibration_error(labels, scores, ece_bins),
        "log_loss": float(log_loss(labels, np.column_stack([1.0 - scores, scores]), labels=[0, 1])),
        "prevalence": prevalence,
        "auprc_lift": float(metrics["auprc"] / prevalence) if prevalence > 0 else float("nan"),
    }


def prediction_records(
    method: str,
    split: str,
    site: str,
    indices: list[int],
    frame: pd.DataFrame,
    labels: np.ndarray,
    scores: np.ndarray,
    config: MainExperimentConfig,
) -> list[dict[str, Any]]:
    if len(indices) != len(labels):
        raise RuntimeError("Prediction/index length mismatch")
    selected = frame.iloc[indices]
    if not np.array_equal(selected["label"].to_numpy(), labels):
        raise RuntimeError("Prediction labels are not aligned with source rows")
    return [
        {
            "method": method,
            "scenario": config.scenario,
            "seed": config.seed,
            "split": split,
            "stable_id": stable_id,
            "source_row": int(source_row),
            "site": site,
            "label": int(label),
            "score": float(score),
            "prediction": int(score >= 0.5),
        }
        for stable_id, source_row, label, score in zip(
            selected["stable_id"], selected["source_row"], labels, scores, strict=True
        )
    ]


def evaluate_global_model(
    model: nn.Module,
    method: str,
    split: str,
    frame: pd.DataFrame,
    dataset: pilot.TokenDataset,
    partition: dict[str, list[int]],
    config: MainExperimentConfig,
    device: torch.device,
    seed_offset: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    pooled_labels: list[np.ndarray] = []
    pooled_scores: list[np.ndarray] = []
    for index, site in enumerate(pilot.SITE_NAMES):
        loader = pilot.make_loader(
            dataset,
            partition[site],
            config.eval_batch_size,
            False,
            config.seed + seed_offset + index,
            config.num_workers,
        )
        labels, scores, seconds = pilot.predict(model, loader, device)
        pooled_labels.append(labels)
        pooled_scores.append(scores)
        metrics.append(
            {
                "method": method,
                "scenario": config.scenario,
                "seed": config.seed,
                "split": split,
                "scope": f"site_{site}",
                "n": int(len(labels)),
                "eval_seconds": seconds,
                **expanded_binary_metrics(labels, scores, config.ece_bins),
            }
        )
        predictions.extend(
            prediction_records(method, split, site, partition[site], frame, labels, scores, config)
        )
    labels = np.concatenate(pooled_labels)
    scores = np.concatenate(pooled_scores)
    metrics.append(
        {
            "method": method,
            "scenario": config.scenario,
            "seed": config.seed,
            "split": split,
            "scope": "pooled",
            "n": int(len(labels)),
            "eval_seconds": float("nan"),
            **expanded_binary_metrics(labels, scores, config.ece_bins),
        }
    )
    return metrics, predictions


def make_full_ft_model(
    config: MainExperimentConfig,
    init_seed: int,
    device: torch.device,
    shared_lora_init_state: dict[str, torch.Tensor],
) -> tuple[nn.Module, dict[str, Any]]:
    bert_config = pilot.BertConfig.from_pretrained(pilot.MODEL_NAME)
    bert_config.attention_probs_dropout_prob = 0.1
    pilot.set_seed(init_seed)
    backbone = pilot.AutoModel.from_pretrained(
        pilot.MODEL_NAME,
        trust_remote_code=True,
        config=bert_config,
        add_pooling_layer=False,
    )
    checkpointing_enabled = False
    checkpointing_fallback_reason: str | None = None
    if config.full_ft_gradient_checkpointing:
        if not hasattr(backbone, "gradient_checkpointing_enable"):
            checkpointing_fallback_reason = "backbone does not expose gradient_checkpointing_enable"
        else:
            try:
                backbone.gradient_checkpointing_enable()
                checkpointing_enabled = True
            except ValueError as error:
                # DNABERT-2's custom Mosaic-BERT encoder currently lacks the
                # Transformers gradient-checkpointing contract.  Full FT still
                # remains bounded by micro-batch + accumulation; the fallback
                # is explicit in the audit rather than silently claimed.
                checkpointing_fallback_reason = str(error)
        if checkpointing_fallback_reason is not None:
            warnings.warn(
                "DNABERT-2 gradient checkpointing unavailable; using the configured "
                "micro-batch/gradient-accumulation fallback. " + checkpointing_fallback_reason,
                RuntimeWarning,
            )
    # The full-FT branch uses the same pretrained backbone.  Its classifier is
    # copied below from the paired LoRA initialization, so the adaptation
    # strategy (rather than a different random head) is the isolated factor.
    pilot.set_seed(init_seed)
    model = pilot.DNABertLoRAClassifier(backbone, hidden_size=bert_config.hidden_size).to(device)
    with torch.no_grad():
        model.classifier.weight.copy_(
            shared_lora_init_state["classifier.weight"].to(device=device, dtype=model.classifier.weight.dtype)
        )
        model.classifier.bias.copy_(
            shared_lora_init_state["classifier.bias"].to(device=device, dtype=model.classifier.bias.dtype)
        )
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    if trainable != EXPECTED_FULL_FT_TRAINABLE_PARAMETERS:
        raise RuntimeError(
            "Unexpected DNABERT-2 full-FT trainable parameter count: "
            f"{trainable:,} != {EXPECTED_FULL_FT_TRAINABLE_PARAMETERS:,}. "
            "Refusing to run a baseline with a changed model manifest."
        )
    return model, {
        "adaptation": "full_fine_tuning",
        "trainable_parameters": int(trainable),
        "trainable_fp32_bytes": int(trainable * 4),
        "total_parameters": int(total),
        "trainable_fraction": float(trainable / total),
        "gradient_checkpointing": checkpointing_enabled,
        "gradient_checkpointing_requested": config.full_ft_gradient_checkpointing,
        "gradient_checkpointing_fallback_reason": checkpointing_fallback_reason,
        "micro_batch_size": config.full_ft_batch_size,
        "gradient_accumulation_steps": config.full_ft_grad_accum_steps,
        "effective_batch_size": config.full_ft_batch_size * config.full_ft_grad_accum_steps,
        "classifier_initialization": "copied_from_paired_lora_init_state",
        "amp_dtype_requested": config.full_ft_amp_dtype,
    }


def resolve_full_ft_amp_dtype(config: MainExperimentConfig, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if config.full_ft_amp_dtype == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("--full-ft-amp-dtype bf16 requested, but this GPU does not support BF16")
        return torch.bfloat16
    if config.full_ft_amp_dtype == "fp16":
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    *,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    prox_reference: dict[str, torch.Tensor] | None = None,
    prox_mu: float = 0.0,
    amp_dtype: torch.dtype = torch.float16,
) -> tuple[float, float, dict[str, float]]:
    model.train()
    criterion = nn.CrossEntropyLoss()
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    trainable = dict(pilot.trainable_parameters(model))
    if prox_mu > 0 and prox_reference is None:
        raise ValueError("prox_reference is required when prox_mu > 0")
    if prox_reference is not None and list(trainable) != list(prox_reference):
        raise ValueError("FedProx reference manifest mismatch")
    start = time.perf_counter()
    total_classification = 0.0
    total_prox = 0.0
    total_n = 0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        labels = batch.pop("labels").to(device, non_blocking=True)
        inputs = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits = model(**inputs)
            classification_loss = criterion(logits, labels)
            proximal_penalty = torch.zeros((), device=device, dtype=torch.float32)
            if prox_mu > 0:
                # The reference is the trainable global LoRA/head state at the
                # start of the round, exactly matching the FedProx objective.
                for name, parameter in trainable.items():
                    difference = parameter.float() - prox_reference[name]
                    proximal_penalty = proximal_penalty + torch.sum(difference * difference)
            objective = classification_loss + 0.5 * prox_mu * proximal_penalty
        scaler.scale(objective / gradient_accumulation_steps).backward()
        should_step = (step + 1) % gradient_accumulation_steps == 0 or step + 1 == len(loader)
        if should_step:
            scaler.unscale_(optimizer)
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable.values(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        batch_n = int(labels.shape[0])
        total_classification += float(classification_loss.detach()) * batch_n
        total_prox += float(proximal_penalty.detach()) * batch_n
        total_n += batch_n
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    mean_classification = total_classification / max(total_n, 1)
    mean_prox = total_prox / max(total_n, 1)
    mean_objective = mean_classification + 0.5 * prox_mu * mean_prox
    return mean_objective, seconds, {
        "classification_loss": mean_classification,
        "proximal_penalty_squared_l2": mean_prox,
        "objective_loss": mean_objective,
        "optimizer_steps": int(math.ceil(len(loader) / gradient_accumulation_steps)),
    }


def train_central_method(
    method: str,
    config: MainExperimentConfig,
    datasets: dict[str, pilot.TokenDataset],
    device: torch.device,
    lora_init_state: dict[str, torch.Tensor],
) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    if method == "central_full_ft":
        model, method_audit = make_full_ft_model(config, config.seed, device, lora_init_state)
        learning_rate = config.full_ft_learning_rate
        batch_size = config.full_ft_batch_size
        accumulation = config.full_ft_grad_accum_steps
        amp_dtype = resolve_full_ft_amp_dtype(config, device)
        method_audit["amp_dtype_resolved"] = str(amp_dtype).replace("torch.", "")
    else:
        model = pilot.create_model(config, config.seed, device)
        pilot.load_trainable_state(model, lora_init_state)
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in model.parameters())
        method_audit = {
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
    total_seconds = 0.0
    for epoch in range(config.epochs):
        paired_seed = config.seed + epoch
        pilot.set_seed(paired_seed)
        loader = pilot.make_loader(
            datasets["train"], None, batch_size, True, paired_seed, config.num_workers
        )
        loss, seconds, detail = train_one_epoch(
            model,
            loader,
            device,
            optimizer,
            gradient_accumulation_steps=accumulation,
            max_grad_norm=config.max_grad_norm,
            amp_dtype=amp_dtype,
        )
        total_seconds += seconds
        epoch_records.append({"epoch": epoch + 1, "loss": loss, "seconds": seconds, **detail})
    return model, {"train_seconds": total_seconds, "epochs": epoch_records}, method_audit


def train_local_models(
    config: MainExperimentConfig,
    datasets: dict[str, pilot.TokenDataset],
    partitions: dict[str, dict[str, list[int]]],
    device: torch.device,
    init_state: dict[str, torch.Tensor],
) -> tuple[dict[str, nn.Module], dict[str, Any]]:
    models: dict[str, nn.Module] = {}
    statistics: dict[str, Any] = {}
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
        for epoch in range(config.epochs):
            paired_seed = config.seed + 1000 + site_index * 100 + epoch
            pilot.set_seed(paired_seed)
            loader = pilot.make_loader(
                datasets["train"],
                partitions["train"][site],
                config.batch_size,
                True,
                paired_seed,
                config.num_workers,
            )
            loss, seconds, detail = train_one_epoch(
                model, loader, device, optimizer, max_grad_norm=config.max_grad_norm
            )
            total_seconds += seconds
            epochs.append({"epoch": epoch + 1, "loss": loss, "seconds": seconds, **detail})
        models[site] = model.cpu()
        statistics[site] = {"train_seconds": total_seconds, "epochs": epochs}
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return models, statistics


def round_dev_metrics(
    model: nn.Module,
    datasets: dict[str, pilot.TokenDataset],
    partitions: dict[str, dict[str, list[int]]],
    frames: dict[str, pd.DataFrame],
    method: str,
    config: MainExperimentConfig,
    device: torch.device,
    round_index: int,
) -> dict[str, float] | None:
    if config.round_eval_interval == 0 or (round_index + 1) % config.round_eval_interval != 0:
        return None
    records, _ = evaluate_global_model(
        model,
        method,
        "dev",
        frames["dev"],
        datasets["dev"],
        partitions["dev"],
        config,
        device,
        8000 + round_index * 10,
    )
    pooled = next(record for record in records if record["scope"] == "pooled")
    return {
        key: float(pooled[key])
        for key in ("accuracy", "f1", "mcc", "auroc", "auprc", "brier", "ece")
    }


def state_manifest_without_flattening(state: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    """Build the canonical flat-state manifest without allocating a full vector."""
    manifest: list[dict[str, Any]] = []
    offset = 0
    for name in sorted(state):
        tensor = state[name]
        numel = int(tensor.numel())
        manifest.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "offset": offset,
                "numel": numel,
            }
        )
        offset += numel
    return manifest


def state_fp32_streaming_sha256(
    state: dict[str, torch.Tensor],
    manifest: list[dict[str, Any]],
    chunk_elements: int = 1 << 20,
) -> str:
    """Hash the canonical dense FP32 state without constructing a flat vector."""
    validate_state_manifest(state, manifest)
    digest = hashlib.sha256()
    for entry in manifest:
        values = (
            state[entry["name"]]
            .detach()
            .cpu()
            .float()
            .contiguous()
            .numpy()
            .reshape(-1)
            .astype("<f4", copy=False)
        )
        for offset in range(0, len(values), chunk_elements):
            digest.update(memoryview(values[offset : offset + chunk_elements]).cast("B"))
    return digest.hexdigest()


def validate_state_manifest(
    state: dict[str, torch.Tensor], manifest: list[dict[str, Any]]
) -> None:
    current = state_manifest_without_flattening(state)
    if current != manifest:
        raise RuntimeError("Full-FT client trainable-state manifests differ")


class StateChunkReader:
    """Deterministically stream a state dict in canonical name/row-major order."""

    def __init__(self, state: dict[str, torch.Tensor], manifest: list[dict[str, Any]], chunk_size: int):
        validate_state_manifest(state, manifest)
        self.arrays = [
            state[entry["name"]].detach().cpu().float().contiguous().numpy().reshape(-1)
            for entry in manifest
        ]
        self.chunk_size = int(chunk_size)
        self.total_numel = int(sum(int(entry["numel"]) for entry in manifest))
        self.consumed = 0
        self.entry_index = 0
        self.entry_offset = 0

    def __iter__(self) -> StateChunkReader:
        return self

    def __next__(self) -> np.ndarray:
        if self.consumed >= self.total_numel:
            raise StopIteration
        length = min(self.chunk_size, self.total_numel - self.consumed)
        result = np.empty(length, dtype=np.float64)
        written = 0
        while written < length:
            source = self.arrays[self.entry_index]
            take = min(length - written, len(source) - self.entry_offset)
            result[written : written + take] = source[self.entry_offset : self.entry_offset + take]
            written += take
            self.entry_offset += take
            self.consumed += take
            if self.entry_offset == len(source):
                self.entry_index += 1
                self.entry_offset = 0
        return result


class StateChunkWriter:
    """Inverse of :class:`StateChunkReader`, retaining only the FP32 result state."""

    def __init__(self, template: dict[str, torch.Tensor], manifest: list[dict[str, Any]]):
        validate_state_manifest(template, manifest)
        self.manifest = manifest
        self.state = {
            entry["name"]: torch.empty(entry["shape"], dtype=torch.float32)
            for entry in manifest
        }
        self.arrays = [self.state[entry["name"]].numpy().reshape(-1) for entry in manifest]
        self.total_numel = int(sum(int(entry["numel"]) for entry in manifest))
        self.written = 0
        self.entry_index = 0
        self.entry_offset = 0

    def write(self, values: np.ndarray) -> None:
        source = np.asarray(values, dtype=np.float32).reshape(-1)
        if self.written + len(source) > self.total_numel:
            raise RuntimeError("Streaming state writer received too many values")
        consumed = 0
        while consumed < len(source):
            target = self.arrays[self.entry_index]
            take = min(len(source) - consumed, len(target) - self.entry_offset)
            target[self.entry_offset : self.entry_offset + take] = source[consumed : consumed + take]
            consumed += take
            self.entry_offset += take
            self.written += take
            if self.entry_offset == len(target):
                self.entry_index += 1
                self.entry_offset = 0

    def finish(self) -> dict[str, torch.Tensor]:
        if self.written != self.total_numel:
            raise RuntimeError(
                f"Streaming state writer is incomplete ({self.written} != {self.total_numel})"
            )
        return self.state


class CKKSStreamingStateAggregator:
    """Chunk-stream CKKS client updates without retaining a ciphertext corpus.

    At most one serialized client ciphertext and the running aggregate for the
    current chunk are live.  Client order and tensor flattening order are fixed,
    making repeated aggregation deterministic up to CKKS numerical noise.
    """

    def __init__(self, config: MainExperimentConfig):
        self.backend = pilot.CKKSAdditiveAggregator(config)

    def aggregate_states(
        self,
        deltas: list[dict[str, torch.Tensor]],
        weights: list[float],
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, float], list[dict[str, Any]]]:
        aggregate_wall_start = time.perf_counter()
        if len(deltas) != len(weights) or not deltas:
            raise ValueError("Streaming CKKS aggregation requires one weight per non-empty client state")
        manifest = state_manifest_without_flattening(deltas[0])
        for delta in deltas[1:]:
            validate_state_manifest(delta, manifest)
        readers = [StateChunkReader(delta, manifest, self.backend.slots) for delta in deltas]
        writer = StateChunkWriter(deltas[0], manifest)
        total_numel = int(sum(int(entry["numel"]) for entry in manifest))
        expected_chunks = int(math.ceil(total_numel / self.backend.slots))

        encrypt_seconds = [0.0 for _ in deltas]
        upload_bytes = [0 for _ in deltas]
        server_aggregate_seconds = 0.0
        decrypt_seconds = 0.0
        aggregate_ciphertext_bytes = 0
        peak_client_ciphertext_bytes = 0
        chunks = 0

        absolute_error_sum = 0.0
        max_abs_error = 0.0
        squared_error_sum = 0.0
        squared_plain_sum = 0.0
        squared_he_sum = 0.0
        he_plain_dot = 0.0

        while chunks < expected_chunks:
            plaintext_chunk: np.ndarray | None = None
            aggregate_encrypted: Any | None = None
            chunk_length: int | None = None
            for client_index, (reader, weight) in enumerate(zip(readers, weights, strict=True)):
                client_chunk = next(reader)
                if chunk_length is None:
                    chunk_length = len(client_chunk)
                    plaintext_chunk = np.zeros(chunk_length, dtype=np.float64)
                elif len(client_chunk) != chunk_length:
                    raise RuntimeError("Client streaming chunks differ in length")
                weighted_chunk = client_chunk * float(weight)
                assert plaintext_chunk is not None
                plaintext_chunk += weighted_chunk

                start = time.perf_counter()
                encrypted = pilot.ts.ckks_vector(self.backend.public_context, weighted_chunk.tolist())
                client_blob = encrypted.serialize()
                encrypt_seconds[client_index] += time.perf_counter() - start
                upload_bytes[client_index] += len(client_blob)
                peak_client_ciphertext_bytes = max(peak_client_ciphertext_bytes, len(client_blob))

                # Simulate immediate upload consumption: deserialize, add to
                # the one running aggregate, then release the client blob.
                start = time.perf_counter()
                server_value = pilot.ts.ckks_vector_from(self.backend.public_context, client_blob)
                if aggregate_encrypted is None:
                    aggregate_encrypted = server_value
                else:
                    aggregate_encrypted += server_value
                server_aggregate_seconds += time.perf_counter() - start
                del encrypted, client_blob, server_value, weighted_chunk, client_chunk

            assert aggregate_encrypted is not None and plaintext_chunk is not None and chunk_length is not None
            start = time.perf_counter()
            aggregate_blob = aggregate_encrypted.serialize()
            server_aggregate_seconds += time.perf_counter() - start
            aggregate_ciphertext_bytes += len(aggregate_blob)

            start = time.perf_counter()
            secret_value = pilot.ts.ckks_vector_from(self.backend.secret_context, aggregate_blob)
            recovered = np.asarray(secret_value.decrypt()[:chunk_length], dtype=np.float64)
            decrypt_seconds += time.perf_counter() - start
            writer.write(recovered)

            error = recovered - plaintext_chunk
            absolute_error_sum += float(np.abs(error).sum())
            max_abs_error = max(max_abs_error, float(np.max(np.abs(error))))
            squared_error_sum += float(np.dot(error, error))
            squared_plain_sum += float(np.dot(plaintext_chunk, plaintext_chunk))
            squared_he_sum += float(np.dot(recovered, recovered))
            he_plain_dot += float(np.dot(recovered, plaintext_chunk))
            chunks += 1
            del aggregate_encrypted, aggregate_blob, secret_value, recovered, plaintext_chunk, error

        for reader in readers:
            try:
                next(reader)
            except StopIteration:
                pass
            else:
                raise RuntimeError("Client state contains values beyond the expected manifest")

        denominator = max(math.sqrt(squared_plain_sum), 1e-12)
        cosine_denominator = max(math.sqrt(squared_he_sum) * denominator, 1e-12)
        fidelity = {
            "mae": absolute_error_sum / max(total_numel, 1),
            "max_abs": max_abs_error,
            "relative_l2": math.sqrt(squared_error_sum) / denominator,
            "cosine": he_plain_dot / cosine_denominator,
        }
        stats = {
            "encrypt_seconds_by_client": encrypt_seconds,
            "upload_bytes_by_client": upload_bytes,
            "server_aggregate_seconds": server_aggregate_seconds,
            "decrypt_seconds": decrypt_seconds,
            "aggregate_ciphertext_bytes": aggregate_ciphertext_bytes,
            "ciphertext_chunks": chunks,
            "streaming": True,
            "streaming_order": "sorted parameter name, contiguous row-major, clients A-B-C",
            "max_client_ciphertexts_in_memory": 1,
            "peak_client_ciphertext_bytes": peak_client_ciphertext_bytes,
            "full_client_ciphertext_corpus_materialized": False,
            "full_plaintext_vector_materialized": False,
            "aggregate_states_wall_seconds": time.perf_counter() - aggregate_wall_start,
            "process_peak_rss_bytes_after_aggregate": process_peak_rss_bytes(),
        }
        return writer.finish(), stats, fidelity, manifest


def train_federated_full_ft(
    method: str,
    config: MainExperimentConfig,
    datasets: dict[str, pilot.TokenDataset],
    partitions: dict[str, dict[str, list[int]]],
    frames: dict[str, pd.DataFrame],
    device: torch.device,
    lora_init_state: dict[str, torch.Tensor],
) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    """Train a full-parameter FedAvg arm, optionally with streaming CKKS.

    This deliberately mirrors the LoRA federated protocol: all clients start
    from the same round-global state, optimize independently for the same
    local data-pass budget, and are weighted by their training sample counts.
    The only adaptation-specific changes are the full-parameter model and its
    memory-safe micro-batch/gradient-accumulation settings.
    """
    if method not in ("fedavg_full_ft", "he_fedavg_full_ft"):
        raise ValueError(f"Unsupported full-FT federated method: {method}")
    use_he = method == "he_fedavg_full_ft"
    local_model, method_audit = make_full_ft_model(
        config, config.seed, device, lora_init_state
    )
    amp_dtype = resolve_full_ft_amp_dtype(config, device)
    method_audit.update(
        {
            "amp_dtype_resolved": str(amp_dtype).replace("torch.", ""),
            "federated": True,
            "aggregation": (
                "ckks_streaming_weighted_sum" if use_he else "plaintext_weighted_mean"
            ),
            "local_objective": "empirical_risk",
            "fedprox_mu": 0.0,
            "client_optimizer_state": "independent_and_reset_for_each_client_round",
        }
    )

    # The CPU FP32 global state is the authoritative round checkpoint.  It is
    # also the source used for every client reload, avoiding order-dependent
    # leakage of a preceding client's optimizer/model state.
    global_state = pilot.clone_trainable_state(local_model)
    state_manifest = state_manifest_without_flattening(global_state)
    state_total_numel = int(sum(int(entry["numel"]) for entry in state_manifest))
    if state_total_numel != EXPECTED_FULL_FT_TRAINABLE_PARAMETERS:
        raise RuntimeError(
            "Full-FT initial state manifest has an unexpected number of elements: "
            f"{state_total_numel:,}"
        )
    state_manifest_sha256 = json_hash(state_manifest)
    initial_state_fp32_sha256 = state_fp32_streaming_sha256(global_state, state_manifest)
    full_state_audit = {
        "manifest_sha256": state_manifest_sha256,
        "tensor_count": len(state_manifest),
        "total_numel": state_total_numel,
        "fp32_bytes": state_total_numel * 4,
        "initial_state_fp32_sha256": initial_state_fp32_sha256,
        "initial_state_hash_encoding": (
            "sorted parameter names; concatenated C-order little-endian FP32 values; "
            "streamed without a dense flat-vector allocation"
        ),
    }
    method_audit["full_dense_state"] = full_state_audit
    aggregator = CKKSStreamingStateAggregator(config) if use_he else None
    round_records: list[dict[str, Any]] = []

    for round_index in range(config.rounds):
        deltas: list[dict[str, torch.Tensor]] = []
        sample_counts: list[int] = []
        client_records: list[dict[str, Any]] = []
        for site_index, site in enumerate(pilot.SITE_NAMES):
            pilot.load_trainable_state(local_model, global_state)
            optimizer = torch.optim.AdamW(
                [parameter for parameter in local_model.parameters() if parameter.requires_grad],
                lr=config.full_ft_learning_rate,
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
                    config.full_ft_batch_size,
                    True,
                    paired_seed,
                    config.num_workers,
                )
                loss, seconds, detail = train_one_epoch(
                    local_model,
                    loader,
                    device,
                    optimizer,
                    gradient_accumulation_steps=config.full_ft_grad_accum_steps,
                    max_grad_norm=config.max_grad_norm,
                    amp_dtype=amp_dtype,
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
            # AdamW's two moment tensors are no longer needed after this
            # client's delta is materialized.  Releasing them here keeps GPU
            # memory bounded independently of the number of clients.
            del local_state, optimizer, loader
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        total_n = sum(sample_counts)
        if total_n <= 0:
            raise RuntimeError("Federated full-FT received no client training samples")
        weights = [count / total_n for count in sample_counts]
        crypto_stats: dict[str, Any] | None = None
        fidelity: dict[str, float] | None = None
        if use_he:
            assert aggregator is not None
            aggregate_delta, crypto_stats, fidelity, current_manifest = (
                aggregator.aggregate_states(deltas, weights)
            )
            if state_manifest != current_manifest:
                raise RuntimeError("Full-FT state manifest changed between rounds")
        else:
            aggregate_delta = pilot.weighted_average_deltas(deltas, weights)

        global_state = pilot.add_delta(global_state, aggregate_delta)
        pilot.load_trainable_state(local_model, global_state)
        dev_metrics = round_dev_metrics(
            local_model,
            datasets,
            partitions,
            frames,
            method,
            config,
            device,
            round_index,
        )
        round_records.append(
            {
                "round": round_index + 1,
                "clients": client_records,
                "weights": weights,
                "fedprox_mu": 0.0,
                "dev_pooled": dev_metrics,
                "crypto": crypto_stats,
                "fidelity": fidelity,
            }
        )
        del deltas, aggregate_delta
        gc.collect()

    information: dict[str, Any] = {
        "rounds": round_records,
        "fedprox_mu": 0.0,
        "adaptation": "full_fine_tuning",
        "full_dense_state": full_state_audit,
    }
    if aggregator is not None:
        information["ckks_context"] = {
            "poly_modulus_degree": config.ckks_poly_modulus_degree,
            "coeff_mod_bit_sizes": config.ckks_coeff_mod_bits,
            "scale_bits": config.ckks_scale_bits,
            "slots": aggregator.backend.slots,
            "keygen_seconds": aggregator.backend.keygen_seconds,
            "public_context_bytes": aggregator.backend.public_context_bytes,
            "secret_context_bytes": aggregator.backend.secret_context_bytes,
            "streaming": True,
            "state_manifest_sha256": state_manifest_sha256,
            "state_tensor_count": len(state_manifest),
            "state_total_numel": state_total_numel,
            "state_fp32_bytes": state_total_numel * 4,
            "max_client_ciphertexts_in_memory": 1,
            "full_client_ciphertext_corpus_materialized": False,
        }
        method_audit.update(
            {
                "ckks_state_manifest_sha256": state_manifest_sha256,
                "ckks_state_total_numel": state_total_numel,
                "ckks_chunk_streaming": True,
                "max_client_ciphertexts_in_memory": 1,
                "full_client_ciphertext_corpus_materialized": False,
            }
        )
    return local_model, information, method_audit


def train_federated_method(
    method: str,
    config: MainExperimentConfig,
    datasets: dict[str, pilot.TokenDataset],
    partitions: dict[str, dict[str, list[int]]],
    frames: dict[str, pd.DataFrame],
    device: torch.device,
    init_state: dict[str, torch.Tensor],
) -> tuple[nn.Module, dict[str, Any]]:
    use_he = method == "he_fedavg"
    prox_mu = config.fedprox_mu if method == "fedprox" else 0.0
    global_state = {name: value.clone() for name, value in init_state.items()}
    local_model = pilot.create_model(config, config.seed, device)
    aggregator = pilot.CKKSAdditiveAggregator(config) if use_he else None
    round_records: list[dict[str, Any]] = []
    for round_index in range(config.rounds):
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
            prox_reference = None
            if prox_mu > 0:
                prox_reference = {
                    name: tensor.to(device=device, dtype=torch.float32)
                    for name, tensor in global_state.items()
                }
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
                loss, seconds, detail = train_one_epoch(
                    local_model,
                    loader,
                    device,
                    optimizer,
                    max_grad_norm=config.max_grad_norm,
                    prox_reference=prox_reference,
                    prox_mu=prox_mu,
                )
                local_seconds += seconds
                epoch_details.append({"local_epoch": local_epoch + 1, "loss": loss, **detail})
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
            del prox_reference

        total_n = sum(sample_counts)
        weights = [count / total_n for count in sample_counts]
        plaintext_delta = pilot.weighted_average_deltas(deltas, weights)
        crypto_stats: dict[str, Any] | None = None
        fidelity: dict[str, float] | None = None
        if use_he:
            weighted_vectors: list[np.ndarray] = []
            state_manifest: list[dict[str, Any]] | None = None
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
        dev_metrics = round_dev_metrics(
            local_model,
            datasets,
            partitions,
            frames,
            method,
            config,
            device,
            round_index,
        )
        round_records.append(
            {
                "round": round_index + 1,
                "clients": client_records,
                "weights": weights,
                "fedprox_mu": prox_mu,
                "dev_pooled": dev_metrics,
                "crypto": crypto_stats,
                "fidelity": fidelity,
            }
        )
    information: dict[str, Any] = {"rounds": round_records, "fedprox_mu": prox_mu}
    if aggregator is not None:
        information["ckks_context"] = {
            "poly_modulus_degree": config.ckks_poly_modulus_degree,
            "coeff_mod_bit_sizes": config.ckks_coeff_mod_bits,
            "scale_bits": config.ckks_scale_bits,
            "slots": aggregator.slots,
            "keygen_seconds": aggregator.keygen_seconds,
            "public_context_bytes": aggregator.public_context_bytes,
            "secret_context_bytes": aggregator.secret_context_bytes,
        }
    return local_model, information


def evaluate_local_models(
    models: dict[str, nn.Module],
    split: str,
    frame: pd.DataFrame,
    dataset: pilot.TokenDataset,
    partition: dict[str, list[int]],
    config: MainExperimentConfig,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    pooled_labels: list[np.ndarray] = []
    pooled_scores: list[np.ndarray] = []
    for site_index, site in enumerate(pilot.SITE_NAMES):
        model = models[site].to(device)
        loader = pilot.make_loader(
            dataset,
            partition[site],
            config.eval_batch_size,
            False,
            config.seed + 5000 + site_index,
            config.num_workers,
        )
        labels, scores, seconds = pilot.predict(model, loader, device)
        pooled_labels.append(labels)
        pooled_scores.append(scores)
        metrics.append(
            {
                "method": "local",
                "scenario": config.scenario,
                "seed": config.seed,
                "split": split,
                "scope": f"site_{site}",
                "n": int(len(labels)),
                "eval_seconds": seconds,
                **expanded_binary_metrics(labels, scores, config.ece_bins),
            }
        )
        predictions.extend(
            prediction_records("local", split, site, partition[site], frame, labels, scores, config)
        )
        models[site] = model.cpu()
    labels = np.concatenate(pooled_labels)
    scores = np.concatenate(pooled_scores)
    metrics.append(
        {
            "method": "local",
            "scenario": config.scenario,
            "seed": config.seed,
            "split": split,
            "scope": "pooled",
            "n": int(len(labels)),
            "eval_seconds": float("nan"),
            **expanded_binary_metrics(labels, scores, config.ece_bins),
        }
    )
    return metrics, predictions


def method_artifacts_valid(method: str, method_dir: Path, config: MainExperimentConfig) -> bool:
    done_path = method_dir / "DONE.json"
    if not done_path.exists():
        return False
    done = json.loads(done_path.read_text(encoding="utf-8"))
    if done.get("status") != "complete" or done.get("method") != method:
        return False
    required = ["metrics.csv", "system_metrics.json", "method_model_audit.json"]
    if config.save_predictions:
        required.append("predictions.csv.gz")
    if config.save_model_states:
        required.append("trainable_state.pt")
    expected_hashes = done.get("artifact_sha256", {})
    for filename in required:
        path = method_dir / filename
        if not path.exists() or expected_hashes.get(filename) != file_sha256(path):
            return False
    return True


def finalize_method(
    method: str,
    method_dir: Path,
    metrics: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    system_metrics: dict[str, Any],
    method_audit: dict[str, Any],
    state: Any,
    config: MainExperimentConfig,
) -> None:
    method_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(pd.DataFrame(metrics), method_dir / "metrics.csv")
    atomic_write_json(method_dir / "system_metrics.json", system_metrics)
    atomic_write_json(method_dir / "method_model_audit.json", method_audit)
    if config.save_predictions:
        atomic_write_csv(
            pd.DataFrame(predictions), method_dir / "predictions.csv.gz", compression="gzip"
        )
    if config.save_model_states:
        if state is None:
            raise RuntimeError(f"State saving is enabled but {method} returned no state")
        atomic_torch_save(state, method_dir / "trainable_state.pt")
    filenames = ["metrics.csv", "system_metrics.json", "method_model_audit.json"]
    if config.save_predictions:
        filenames.append("predictions.csv.gz")
    if config.save_model_states:
        filenames.append("trainable_state.pt")
    hashes = {filename: file_sha256(method_dir / filename) for filename in filenames}
    sizes = {filename: int((method_dir / filename).stat().st_size) for filename in filenames}
    atomic_write_json(
        method_dir / "DONE.json",
        {
            "status": "complete",
            "method": method,
            "completed_unix": time.time(),
            "artifact_sha256": hashes,
            "artifact_bytes": sizes,
            "pooled_test_metrics": [
                record
                for record in metrics
                if record["split"] == "test" and record["scope"] == "pooled"
            ],
        },
    )


def run_method(
    method: str,
    method_dir: Path,
    config: MainExperimentConfig,
    frames: dict[str, pd.DataFrame],
    datasets: dict[str, pilot.TokenDataset],
    partitions: dict[str, dict[str, list[int]]],
    device: torch.device,
    init_state: dict[str, torch.Tensor],
) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    metric_records: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    if method in ("central", "central_full_ft"):
        model, stats, method_audit = train_central_method(
            method, config, datasets, device, init_state
        )
        for split_index, split in enumerate(config.eval_splits):
            metrics, predictions = evaluate_global_model(
                model,
                method,
                split,
                frames[split],
                datasets[split],
                partitions[split],
                config,
                device,
                1000 + split_index * 10,
            )
            metric_records.extend(metrics)
            prediction_rows.extend(predictions)
        state = pilot.clone_trainable_state(model) if config.save_model_states else None
    elif method in ("fedavg_full_ft", "he_fedavg_full_ft"):
        model, stats, method_audit = train_federated_full_ft(
            method, config, datasets, partitions, frames, device, init_state
        )
        if method_audit.get("trainable_parameters") != EXPECTED_FULL_FT_TRAINABLE_PARAMETERS:
            raise RuntimeError(
                f"{method} model audit does not match the expected full-FT parameter count"
            )
        for split_index, split in enumerate(config.eval_splits):
            metrics, predictions = evaluate_global_model(
                model,
                method,
                split,
                frames[split],
                datasets[split],
                partitions[split],
                config,
                device,
                2000 + split_index * 10,
            )
            metric_records.extend(metrics)
            prediction_rows.extend(predictions)
        state = pilot.clone_trainable_state(model) if config.save_model_states else None
    elif method == "local":
        models, stats = train_local_models(config, datasets, partitions, device, init_state)
        for split in config.eval_splits:
            metrics, predictions = evaluate_local_models(
                models,
                split,
                frames[split],
                datasets[split],
                partitions[split],
                config,
                device,
            )
            metric_records.extend(metrics)
            prediction_rows.extend(predictions)
        state = (
            {site: pilot.clone_trainable_state(model) for site, model in models.items()}
            if config.save_model_states
            else None
        )
        trainable = len(pilot.flatten_state(init_state)[0])
        method_audit = {
            "adaptation": "three_independent_lora_models",
            "trainable_parameters_per_model": trainable,
            "saved_models": list(pilot.SITE_NAMES),
        }
        del models
    elif method in ("fedavg", "fedprox", "he_fedavg"):
        model, stats = train_federated_method(
            method, config, datasets, partitions, frames, device, init_state
        )
        for split_index, split in enumerate(config.eval_splits):
            metrics, predictions = evaluate_global_model(
                model,
                method,
                split,
                frames[split],
                datasets[split],
                partitions[split],
                config,
                device,
                3000 + split_index * 10,
            )
            metric_records.extend(metrics)
            prediction_rows.extend(predictions)
        state = pilot.clone_trainable_state(model) if config.save_model_states else None
        trainable_parameters = int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
        method_audit = {
            "adaptation": "lora",
            "aggregation": "ckks_weighted_sum" if method == "he_fedavg" else "plaintext_weighted_mean",
            "local_objective": "fedprox" if method == "fedprox" else "empirical_risk",
            "fedprox_mu": config.fedprox_mu if method == "fedprox" else 0.0,
            "trainable_parameters": trainable_parameters,
        }
    else:
        raise ValueError(f"Unsupported method dispatch: {method}")
    if device.type == "cuda":
        torch.cuda.synchronize()
        stats["resource"] = {
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_cuda_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "end_cuda_memory_bytes": int(torch.cuda.memory_allocated(device)),
            "peak_process_rss_bytes": process_peak_rss_bytes(),
        }
    finalize_method(
        method,
        method_dir,
        metric_records,
        prediction_rows,
        stats,
        method_audit,
        state,
        config,
    )
    if "model" in locals():
        del model
    if state is not None:
        del state
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def consolidate_run(run_dir: Path, requested_methods: list[str], config: MainExperimentConfig) -> dict[str, Any]:
    available: list[str] = []
    metric_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    systems: dict[str, Any] = {}
    for method in METHODS:
        method_dir = run_dir / "methods" / method
        if not method_artifacts_valid(method, method_dir, config):
            continue
        available.append(method)
        metric_frames.append(pd.read_csv(method_dir / "metrics.csv"))
        systems[method] = json.loads((method_dir / "system_metrics.json").read_text(encoding="utf-8"))
        if config.save_predictions:
            prediction_frames.append(pd.read_csv(method_dir / "predictions.csv.gz"))
    if metric_frames:
        atomic_write_csv(pd.concat(metric_frames, ignore_index=True), run_dir / "metrics.csv")
    if prediction_frames:
        atomic_write_csv(
            pd.concat(prediction_frames, ignore_index=True),
            run_dir / "predictions.csv.gz",
            compression="gzip",
        )
    atomic_write_json(run_dir / "system_metrics.json", systems)
    missing = [method for method in requested_methods if method not in available]
    metrics = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    pooled = []
    if not metrics.empty:
        pooled = metrics[(metrics["split"] == "test") & (metrics["scope"] == "pooled")].to_dict("records")
    summary = {
        "status": "complete" if not missing else "partial",
        "schema_version": SCHEMA_VERSION,
        "run_dir": str(run_dir.resolve()),
        "requested_methods": requested_methods,
        "available_methods": available,
        "missing_requested_methods": missing,
        "pooled_test_metrics": pooled,
    }
    atomic_write_json(run_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    truststore.inject_into_ssl()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the main experiment")
    device = torch.device("cuda")
    pilot.set_seed(args.seed)
    config = MainExperimentConfig(
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
        eval_splits=args.eval_splits,
        round_eval_interval=args.round_eval_interval,
        max_grad_norm=args.max_grad_norm,
        full_ft_learning_rate=args.full_ft_learning_rate,
        full_ft_batch_size=args.full_ft_batch_size,
        full_ft_grad_accum_steps=args.full_ft_grad_accum_steps,
        full_ft_gradient_checkpointing=args.full_ft_gradient_checkpointing,
        full_ft_amp_dtype=args.full_ft_amp_dtype,
        save_predictions=args.save_predictions,
        save_model_states=args.save_model_states,
        enforce_equal_data_passes=args.enforce_equal_data_passes,
    )
    validate_config(config)
    run_dir = Path(config.output_dir) / config.scenario / f"seed_{config.seed}"
    prepare_run_dir(run_dir, config, args.resume)

    frames, dataset_audit = load_frames(
        Path(config.data_dir),
        (config.train_limit, config.dev_limit, config.test_limit),
        config.seed,
    )
    partitions = {
        split: pilot.allocate_indices(frame["label"].to_numpy(), config.scenario, config.seed + offset * 10)
        for offset, (split, frame) in enumerate(frames.items())
    }
    audit = {
        "dataset": dataset_audit,
        "partitions": pilot.partition_audit(frames, partitions),
        "partition_algorithm": {
            "iid": "class-stratified equal shards",
            "noniid": "controlled label skew; class0=[0.15,0.35,0.50], class1=[0.60,0.30,0.10]",
        },
    }
    atomic_write_json(run_dir / "data_audit.json", audit)
    export_partitions(run_dir, frames, partitions)

    tokenizer = pilot.AutoTokenizer.from_pretrained(pilot.MODEL_NAME, trust_remote_code=True)
    datasets = pilot.tokenize_frames(frames, tokenizer, config.max_length)
    init_model = pilot.create_model(config, config.seed, device)
    init_state = pilot.clone_trainable_state(init_model)
    init_vector, state_manifest = pilot.flatten_state(init_state)
    model_audit = {
        "base_model": pilot.MODEL_NAME,
        "adaptation_for_fl": {
            "lora_methods": ["fedavg", "fedprox", "he_fedavg"],
            "full_fine_tuning_methods": ["fedavg_full_ft", "he_fedavg_full_ft"],
        },
        "expected_full_ft_trainable_parameters": EXPECTED_FULL_FT_TRAINABLE_PARAMETERS,
        "lora_trainable_parameters": int(init_vector.size),
        "lora_trainable_fp32_bytes": int(init_vector.size * 4),
        "trainable_manifest_sha256": json_hash(state_manifest),
        "initial_trainable_state_sha256": hashlib.sha256(init_vector.tobytes()).hexdigest(),
        "trainable_manifest": state_manifest,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }
    atomic_write_json(run_dir / "model_audit.json", model_audit)
    del init_model
    torch.cuda.empty_cache()

    for method in config.methods:
        method_dir = run_dir / "methods" / method
        if method_artifacts_valid(method, method_dir, config):
            print(f"[resume] {config.scenario}/seed_{config.seed}/{method}: already complete")
            continue
        print(f"[run] {config.scenario}/seed_{config.seed}/{method}")
        run_method(method, method_dir, config, frames, datasets, partitions, device, init_state)
        # A consolidated partial summary is useful even if a later method fails.
        consolidate_run(run_dir, config.methods, config)

    summary = consolidate_run(run_dir, config.methods, config)
    if summary["status"] != "complete":
        raise RuntimeError(f"Requested methods are incomplete: {summary['missing_requested_methods']}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
