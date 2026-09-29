#!/usr/bin/env python3
"""Run one full-data GUE/GUE+ LoRA arm (central, FedAvg, or CKKS FedAvg).

This runner generalizes the earlier binary EMP pipeline to all 36 local
benchmark datasets: binary and multiclass labels, single sequences and EPI
sequence pairs, and long GUE+ inputs.  It preserves official train/dev/test
boundaries, selects the best checkpoint by pooled dev cross-entropy, and
evaluates test exactly once.
"""

from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoModel, AutoTokenizer
from transformers.models.bert.configuration_bert import BertConfig

import run_fedhe_experiment as base


SITES = ("A", "B", "C")
SNAPSHOT_REVISION = "7bce263b15377fc15361f52cfab88f8b586abda0"
DEFAULT_SNAPSHOT = (
    ROOT
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--zhihan1996--DNABERT-2-117M"
    / "snapshots"
    / SNAPSHOT_REVISION
)


@dataclass(frozen=True)
class TaskSpec:
    benchmark: str
    task: str
    dataset_id: str
    primary_metric: str
    max_length: int
    passes: int
    batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int


@dataclass
class RunConfig:
    data_root: str
    dataset_id: str
    output_root: str
    cache_root: str
    model_snapshot: str
    method: str
    seed: int
    learning_rate: float
    weight_decay: float
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    num_workers: int
    ckks_poly_modulus_degree: int
    ckks_coeff_mod_bits: list[int]
    ckks_scale_bits: int
    max_length: int
    passes: int
    batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    activation_offload: bool
    primary_metric: str
    benchmark: str
    task: str
    num_classes: int
    sequence_columns: int
    train_limit: int
    dev_limit: int
    test_limit: int


def task_spec(dataset_id: str) -> TaskSpec:
    category, dataset = dataset_id.split("/", 1)
    if category == "EMP":
        return TaskSpec("GUE", "EMP", dataset_id, "mcc", 128, 3, 8, 16, 2)
    if category == "tf":
        return TaskSpec("GUE", "TF-H", dataset_id, "mcc", 30, 3, 8, 64, 2)
    if category == "mouse":
        return TaskSpec("GUE", "TF-M", dataset_id, "mcc", 30, 5, 8, 64, 2)
    if category == "splice":
        return TaskSpec("GUE", "SSP", dataset_id, "mcc", 80, 5, 8, 16, 2)
    if category == "prom":
        is_tata = dataset.endswith("_tata")
        if dataset.startswith("prom_core_"):
            return TaskSpec("GUE", "CPD", dataset_id, "mcc", 20, 10 if is_tata else 4, 8, 16, 2)
        if dataset.startswith("prom_300_"):
            return TaskSpec("GUE", "PD", dataset_id, "mcc", 70, 10 if is_tata else 4, 8, 16, 2)
    if category == "virus" and dataset == "covid":
        return TaskSpec("GUE", "CVC", dataset_id, "f1_macro", 256, 8, 8, 16, 2)
    if category == "EPI":
        # Seven of 72,000 local EPI examples exceed 1,280 tokens; the audited
        # maximum is 2,177, so 2,200 preserves every example.
        return TaskSpec("GUE+", "EPI", dataset_id, "mcc", 2200, 3, 1, 1, 16)
    if category == "fungi" and dataset == "species_20":
        # Full local audit: untruncated token length is at most 2,192.
        return TaskSpec("GUE+", "SC-Fungi", dataset_id, "mcc", 2200, 3, 1, 1, 16)
    if category == "virus" and dataset == "species_40":
        return TaskSpec("GUE+", "SC-Virus", dataset_id, "mcc", 1280, 3, 1, 2, 16)
    raise ValueError(f"Unsupported GUE/GUE+ dataset id: {dataset_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--method", required=True, choices=("central", "fedavg", "he_fedavg"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=ROOT / "GUE_v2")
    parser.add_argument("--output-root", type=Path, default=ROOT / "experiments" / "gue_lora_macro_v1" / "runs")
    parser.add_argument("--cache-root", type=Path, default=ROOT / "experiments" / "gue_lora_macro_v1" / "token_cache")
    parser.add_argument("--model-snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=0, help="Override task default when >0")
    parser.add_argument("--passes", type=int, default=0, help="Override matched epochs/rounds when >0")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=0)
    parser.add_argument(
        "--activation-offload",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Offload saved autograd tensors to host RAM; defaults on for max_length >1536.",
    )
    parser.add_argument("--train-limit", type=int, default=0, help="Stratified smoke limit; 0 uses the full split")
    parser.add_argument("--dev-limit", type=int, default=0, help="Stratified smoke limit; 0 uses the full split")
    parser.add_argument("--test-limit", type=int, default=0, help="Stratified smoke limit; 0 uses the full split")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def snapshot_evidence(snapshot: Path) -> dict[str, Any]:
    names = (
        "config.json",
        "configuration_bert.py",
        "bert_layers.py",
        "bert_padding.py",
        "flash_attn_triton.py",
        "tokenizer.json",
        "tokenizer_config.json",
        "pytorch_model.bin",
    )
    records = []
    for name in names:
        path = snapshot / name
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append({"name": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {
        "revision": snapshot.name,
        "files": records,
        "aggregate_sha256": canonical_sha(records),
    }


def load_source_frames(
    data_dir: Path,
    limits: tuple[int, int, int],
    seed: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], int, int]:
    frames: dict[str, pd.DataFrame] = {}
    audit: dict[str, Any] = {}
    schemas: list[list[str]] = []
    label_sets: list[set[int]] = []
    for split_index, (split, limit) in enumerate(zip(("train", "dev", "test"), limits, strict=True)):
        path = data_dir / f"{split}.csv"
        frame = pd.read_csv(path)
        columns = list(frame.columns)
        if len(columns) not in (2, 3) or columns[-1] != "label":
            raise ValueError(f"Unsupported schema in {path}: {columns}")
        sequence_columns = columns[:-1]
        ambiguity_mask = np.zeros(len(frame), dtype=bool)
        ambiguity_symbols: set[str] = set()
        for column in sequence_columns:
            frame[column] = frame[column].astype(str).str.upper()
            if frame[column].str.contains(r"[^ACGTNRYWSKMBDHV]", regex=True).any():
                raise ValueError(f"Unexpected non-IUPAC DNA sequence in {path}:{column}")
            column_ambiguity = frame[column].str.contains(r"[^ACGT]", regex=True).to_numpy()
            ambiguity_mask |= column_ambiguity
            for sequence in frame.loc[column_ambiguity, column]:
                ambiguity_symbols.update(character for character in sequence if character not in "ACGT")
        frame["label"] = frame["label"].astype(int)
        frame["source_row"] = np.arange(len(frame), dtype=np.int64)
        stable_values = []
        for row in frame.itertuples(index=False):
            values = [str(getattr(row, column)) for column in sequence_columns]
            stable_values.append(hashlib.sha256(("\x1f".join(values) + f"\x1f{int(row.label)}").encode()).hexdigest())
        frame["stable_id"] = stable_values
        source_n = len(frame)
        source_stable_sha = canonical_sha(frame["stable_id"].tolist())
        if 0 < limit < len(frame):
            frame = base.stratified_subset(frame, limit, seed + split_index).reset_index(drop=True)
        frames[split] = frame
        schemas.append(columns)
        label_sets.append(set(int(value) for value in frame["label"].unique()))
        audit[split] = {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "source_n": source_n,
            "used_n": len(frame),
            "iupac_ambiguity_rows": int(ambiguity_mask.sum()),
            "iupac_ambiguity_symbols": sorted(ambiguity_symbols),
            "class_counts": {str(k): int(v) for k, v in frame["label"].value_counts().sort_index().items()},
            "source_stable_id_sha256": source_stable_sha,
            "used_stable_id_sha256": canonical_sha(frame["stable_id"].tolist()),
        }
    if schemas[1:] != schemas[:-1]:
        raise ValueError("Split schemas differ")
    if label_sets[1:] != label_sets[:-1]:
        raise ValueError("Split label sets differ")
    labels = sorted(label_sets[0])
    if labels != list(range(len(labels))):
        raise ValueError(f"Labels must be contiguous from zero: {labels}")
    return frames, audit, len(labels), len(schemas[0]) - 1


def cache_key(config: RunConfig, source_audit: dict[str, Any], snapshot: dict[str, Any]) -> str:
    return canonical_sha(
        {
            "dataset_id": config.dataset_id,
            "max_length": config.max_length,
            "sources": {
                split: {
                    "sha256": source_audit[split]["sha256"],
                    "used_n": source_audit[split]["used_n"],
                    "used_stable_id_sha256": source_audit[split]["used_stable_id_sha256"],
                }
                for split in source_audit
            },
            "tokenizer_sha256": next(item["sha256"] for item in snapshot["files"] if item["name"] == "tokenizer.json"),
        }
    )[:20]


class RaggedTokenDataset(Dataset):
    def __init__(self, npz_path: Path):
        payload = np.load(npz_path, allow_pickle=False)
        self.values = payload["input_ids_values"]
        self.offsets = payload["input_ids_offsets"]
        self.labels = payload["labels"]
        self.source_rows = payload["source_rows"]
        self.token_type_values = payload["token_type_values"] if "token_type_values" in payload.files else None

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        start = int(self.offsets[index])
        stop = int(self.offsets[index + 1])
        item: dict[str, Any] = {
            "input_ids": self.values[start:stop],
            "label": int(self.labels[index]),
            "source_row": int(self.source_rows[index]),
        }
        if self.token_type_values is not None:
            item["token_type_ids"] = self.token_type_values[start:stop]
        return item


class DynamicPadCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in items)
        batch = len(items)
        input_ids = torch.full((batch, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch, max_len), dtype=torch.long)
        include_types = "token_type_ids" in items[0]
        token_type_ids = torch.zeros((batch, max_len), dtype=torch.long) if include_types else None
        for index, item in enumerate(items):
            length = len(item["input_ids"])
            input_ids[index, :length] = torch.as_tensor(item["input_ids"], dtype=torch.long)
            attention_mask[index, :length] = 1
            if token_type_ids is not None:
                token_type_ids[index, :length] = torch.as_tensor(item["token_type_ids"], dtype=torch.long)
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor([item["label"] for item in items], dtype=torch.long),
            "source_rows": torch.tensor([item["source_row"] for item in items], dtype=torch.long),
        }
        if token_type_ids is not None:
            result["token_type_ids"] = token_type_ids
        return result


def build_token_cache(
    config: RunConfig,
    frames: dict[str, pd.DataFrame],
    source_audit: dict[str, Any],
    snapshot: dict[str, Any],
) -> tuple[dict[str, Path], dict[str, Any]]:
    cache_dir = Path(config.cache_root) / config.dataset_id.replace("/", "__") / cache_key(config, source_audit, snapshot)
    manifest_path = cache_dir / "manifest.json"
    expected = {split: cache_dir / f"{split}.npz" for split in frames}
    if manifest_path.is_file() and all(path.is_file() for path in expected.values()):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for split, path in expected.items():
            if sha256_file(path) != manifest["splits"][split]["sha256"]:
                raise RuntimeError(f"Token cache hash mismatch: {path}")
        return expected, manifest

    cache_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_snapshot,
        trust_remote_code=True,
        local_files_only=True,
        model_max_length=config.max_length,
        padding_side="right",
        use_fast=True,
    )
    split_manifest: dict[str, Any] = {}
    for split, frame in frames.items():
        sequence_columns = [column for column in frame.columns if column not in ("label", "source_row", "stable_id")]
        batch_size = 256 if config.max_length <= 256 else 16
        all_ids: list[np.ndarray] = []
        all_types: list[np.ndarray] = []
        lengths: list[int] = []
        for offset in range(0, len(frame), batch_size):
            batch = frame.iloc[offset : offset + batch_size]
            first = batch[sequence_columns[0]].tolist()
            second = batch[sequence_columns[1]].tolist() if len(sequence_columns) == 2 else None
            encoded = tokenizer(
                first,
                second,
                add_special_tokens=True,
                truncation=True,
                max_length=config.max_length,
                padding=False,
                return_attention_mask=False,
            )
            for ids in encoded["input_ids"]:
                array = np.asarray(ids, dtype=np.int32)
                all_ids.append(array)
                lengths.append(len(array))
            if "token_type_ids" in encoded:
                all_types.extend(np.asarray(values, dtype=np.int8) for values in encoded["token_type_ids"])
        offsets = np.zeros(len(all_ids) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(values) for values in all_ids], dtype=np.int64)
        values = np.concatenate(all_ids).astype(np.int32, copy=False)
        payload: dict[str, Any] = {
            "input_ids_values": values,
            "input_ids_offsets": offsets,
            "labels": frame["label"].to_numpy(dtype=np.int16),
            "source_rows": frame["source_row"].to_numpy(dtype=np.int64),
        }
        if all_types:
            payload["token_type_values"] = np.concatenate(all_types).astype(np.int8, copy=False)
        temporary = expected[split].with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez(handle, **payload)
        temporary.replace(expected[split])
        split_manifest[split] = {
            "path": expected[split].relative_to(ROOT).as_posix(),
            "bytes": expected[split].stat().st_size,
            "sha256": sha256_file(expected[split]),
            "n": len(frame),
            "encoded_token_min": min(lengths),
            "encoded_token_median": float(np.median(lengths)),
            "encoded_token_p95": float(np.quantile(lengths, 0.95)),
            "encoded_token_max": max(lengths),
            "at_max_length": sum(length == config.max_length for length in lengths),
        }
    manifest = {
        "schema": "gue-token-cache-v1",
        "dataset_id": config.dataset_id,
        "max_length": config.max_length,
        "cache_key": cache_dir.name,
        "splits": split_manifest,
    }
    atomic_json(manifest_path, manifest)
    return expected, manifest


def allocate_iid(labels: np.ndarray, seed: int) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed)
    result = {site: [] for site in SITES}
    for label in sorted(np.unique(labels)):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        for site, values in zip(SITES, np.array_split(indices, len(SITES)), strict=True):
            result[site].extend(int(value) for value in values)
    for site in SITES:
        rng.shuffle(result[site])
    combined = sorted(value for site in SITES for value in result[site])
    if combined != list(range(len(labels))):
        raise RuntimeError("IID partition is not exhaustive and disjoint")
    return result


def partition_evidence(datasets: dict[str, RaggedTokenDataset], seed: int) -> tuple[dict[str, dict[str, list[int]]], dict[str, Any]]:
    partitions: dict[str, dict[str, list[int]]] = {}
    audit: dict[str, Any] = {}
    for offset, (split, dataset) in enumerate(datasets.items()):
        site_map = allocate_iid(dataset.labels.astype(np.int64), seed + offset)
        partitions[split] = site_map
        audit[split] = {}
        for site, indices in site_map.items():
            labels = dataset.labels[np.asarray(indices, dtype=np.int64)]
            audit[split][site] = {
                "n": len(indices),
                "class_counts": {str(label): int(np.sum(labels == label)) for label in sorted(np.unique(dataset.labels))},
                "indices_sha256": canonical_sha(indices),
            }
    audit["aggregate_sha256"] = canonical_sha(audit)
    return partitions, audit


def make_loader(
    dataset: Dataset,
    collator: DynamicPadCollator,
    indices: Iterable[int] | None,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    selected: Dataset = dataset if indices is None else Subset(dataset, list(indices))
    return DataLoader(
        selected,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
    )


class GUEClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_size: int, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kwargs: dict[str, torch.Tensor] = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**kwargs)
        hidden = outputs[0]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.classifier(self.dropout(pooled))


def create_model(config: RunConfig, device: torch.device) -> GUEClassifier:
    set_seed(config.seed)
    bert_config = BertConfig.from_pretrained(config.model_snapshot, local_files_only=True)
    bert_config.attention_probs_dropout_prob = 0.1
    backbone = AutoModel.from_pretrained(
        config.model_snapshot,
        trust_remote_code=True,
        config=bert_config,
        add_pooling_layer=False,
        local_files_only=True,
    )
    lora = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=["Wqkv"],
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
    )
    encoder = get_peft_model(backbone, lora)
    set_seed(config.seed)
    model = GUEClassifier(encoder, bert_config.hidden_size, config.num_classes)
    return model.to(device)


def trainable_state_sha(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().float().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def train_one_pass(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
    accumulation_steps: int,
    activation_offload: bool,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    model.train()
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    criterion = nn.CrossEntropyLoss()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_n = 0
    start = time.perf_counter()
    steps = 0
    batches_in_group = 0
    for batch_index, batch in enumerate(loader):
        labels = batch.pop("labels").to(device, non_blocking=True)
        batch.pop("source_rows")
        inputs = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        offload_context = (
            torch.autograd.graph.save_on_cpu(pin_memory=True)
            if (
                activation_offload
                and device.type == "cuda"
                and int(inputs["input_ids"].shape[1]) > 1536
            )
            else nullcontext()
        )
        with offload_context:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(**inputs)
                loss = criterion(logits.float(), labels)
                scaled = loss / accumulation_steps
        scaled.backward()
        batches_in_group += 1
        should_step = (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(loader)
        if should_step:
            # The last accumulation group can contain fewer batches.  Correct
            # its gradient scale so every optimizer step averages the batches
            # it actually observed instead of shrinking the final update.
            if batches_in_group != accumulation_steps:
                correction = accumulation_steps / batches_in_group
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
            batches_in_group = 0
        total_loss += float(loss.detach()) * labels.shape[0]
        total_n += int(labels.shape[0])
        if not math.isfinite(float(loss.detach())):
            raise FloatingPointError("Non-finite training loss")
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {
        "classification_loss": total_loss / max(total_n, 1),
        "examples": total_n,
        "optimizer_steps": steps,
        "seconds": time.perf_counter() - start,
    }


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = np.argmax(probabilities, axis=1)
    classes = np.arange(probabilities.shape[1])
    result = {
        "classification_loss": float(log_loss(labels, probabilities, labels=classes)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
    }
    try:
        if probabilities.shape[1] == 2:
            result["auroc_macro_ovr"] = float(roc_auc_score(labels, probabilities[:, 1]))
            result["auprc_macro_ovr"] = float(average_precision_score(labels, probabilities[:, 1]))
        else:
            result["auroc_macro_ovr"] = float(
                roc_auc_score(labels, probabilities, labels=classes, multi_class="ovr", average="macro")
            )
            one_hot = label_binarize(labels, classes=classes)
            result["auprc_macro_ovr"] = float(average_precision_score(one_hot, probabilities, average="macro"))
    except ValueError:
        result["auroc_macro_ovr"] = float("nan")
        result["auprc_macro_ovr"] = float("nan")
    return result


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    model.eval()
    labels_all: list[np.ndarray] = []
    probabilities_all: list[np.ndarray] = []
    source_rows_all: list[np.ndarray] = []
    start = time.perf_counter()
    for batch in loader:
        labels = batch.pop("labels").numpy()
        source_rows = batch.pop("source_rows").numpy()
        inputs = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(**inputs)
        probabilities = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        labels_all.append(labels)
        probabilities_all.append(probabilities)
        source_rows_all.append(source_rows)
    if device.type == "cuda":
        torch.cuda.synchronize()
    labels = np.concatenate(labels_all)
    probabilities = np.concatenate(probabilities_all)
    source_rows = np.concatenate(source_rows_all)
    metrics = classification_metrics(labels, probabilities)
    metrics["n"] = int(len(labels))
    metrics["eval_seconds"] = time.perf_counter() - start
    return metrics, {"labels": labels, "probabilities": probabilities, "source_rows": source_rows}


def loader_for_full(
    dataset: RaggedTokenDataset,
    collator: DynamicPadCollator,
    config: RunConfig,
    split_seed: int,
    train: bool,
) -> DataLoader:
    return make_loader(
        dataset,
        collator,
        None,
        config.batch_size if train else config.eval_batch_size,
        train,
        split_seed,
        config.num_workers,
    )


def run_central(
    model: nn.Module,
    datasets: dict[str, RaggedTokenDataset],
    collator: DynamicPadCollator,
    config: RunConfig,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor], dict[str, np.ndarray], dict[str, Any]]:
    progress: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    train_seconds = 0.0
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    dev_loader = loader_for_full(datasets["dev"], collator, config, config.seed + 9000, False)
    for epoch in range(1, config.passes + 1):
        train_loader = loader_for_full(datasets["train"], collator, config, config.seed + epoch, True)
        train_stats = train_one_pass(
            model,
            train_loader,
            device,
            config.learning_rate,
            config.weight_decay,
            config.gradient_accumulation_steps,
            config.activation_offload,
            optimizer,
        )
        train_seconds += train_stats["seconds"]
        dev_metrics, _ = evaluate(model, dev_loader, device)
        progress.append({"progress": epoch, "train": train_stats, "dev": dev_metrics})
        if dev_metrics["classification_loss"] < best_loss:
            best_loss = dev_metrics["classification_loss"]
            best_state = base.clone_trainable_state(model)
    if best_state is None:
        raise RuntimeError("Central training did not produce a best state")
    base.load_trainable_state(model, best_state)
    test_loader = loader_for_full(datasets["test"], collator, config, config.seed + 9100, False)
    test_metrics, test_outputs = evaluate(model, test_loader, device)
    return progress, best_state, test_outputs, {"train_seconds": train_seconds, "test_metrics": test_metrics}


def run_federated(
    model: nn.Module,
    datasets: dict[str, RaggedTokenDataset],
    partitions: dict[str, dict[str, list[int]]],
    collator: DynamicPadCollator,
    config: RunConfig,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor], dict[str, np.ndarray], dict[str, Any]]:
    global_state = base.clone_trainable_state(model)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    progress: list[dict[str, Any]] = []
    train_seconds = 0.0
    he_rounds: list[dict[str, Any]] = []
    aggregator = base.CKKSAdditiveAggregator(config) if config.method == "he_fedavg" else None
    dev_loader = loader_for_full(datasets["dev"], collator, config, config.seed + 9200, False)

    for round_index in range(1, config.passes + 1):
        local_deltas: list[dict[str, torch.Tensor]] = []
        client_weights: list[float] = []
        client_stats: list[dict[str, Any]] = []
        total_n = sum(len(partitions["train"][site]) for site in SITES)
        for site_index, site in enumerate(SITES):
            base.load_trainable_state(model, global_state)
            indices = partitions["train"][site]
            loader = make_loader(
                datasets["train"],
                collator,
                indices,
                config.batch_size,
                True,
                config.seed + round_index * 100 + site_index,
                config.num_workers,
            )
            stats = train_one_pass(
                model,
                loader,
                device,
                config.learning_rate,
                config.weight_decay,
                config.gradient_accumulation_steps,
                config.activation_offload,
            )
            train_seconds += stats["seconds"]
            local_state = base.clone_trainable_state(model)
            local_deltas.append(base.state_delta(local_state, global_state))
            client_weights.append(len(indices) / total_n)
            client_stats.append({"site": site, "weight": client_weights[-1], **stats})

        plaintext_delta = base.weighted_average_deltas(local_deltas, client_weights)
        if aggregator is not None:
            vectors: list[np.ndarray] = []
            manifest: list[dict[str, Any]] | None = None
            for delta, weight in zip(local_deltas, client_weights, strict=True):
                vector, current_manifest = base.flatten_state(delta)
                manifest = manifest or current_manifest
                if current_manifest != manifest:
                    raise RuntimeError("Trainable-state manifest changed across clients")
                vectors.append(vector * weight)
            recovered, he_stats = aggregator.aggregate(vectors)
            plain_vector, plain_manifest = base.flatten_state(plaintext_delta)
            if manifest != plain_manifest:
                raise RuntimeError("Plain/HE trainable-state manifest mismatch")
            error = recovered - plain_vector
            denominator = float(np.linalg.norm(plain_vector))
            he_stats.update(
                {
                    "round": round_index,
                    "mae": float(np.mean(np.abs(error))),
                    "max_abs": float(np.max(np.abs(error))),
                    "relative_l2": float(np.linalg.norm(error) / denominator) if denominator else 0.0,
                    "cosine": float(np.dot(recovered, plain_vector) / (np.linalg.norm(recovered) * denominator))
                    if denominator and np.linalg.norm(recovered)
                    else 1.0,
                }
            )
            if he_stats["max_abs"] > 1e-6 or he_stats["cosine"] < 0.999999:
                raise RuntimeError(f"CKKS fidelity gate failed: {he_stats}")
            aggregate_delta = base.unflatten_state(recovered, manifest or [])
            he_rounds.append(he_stats)
        else:
            aggregate_delta = plaintext_delta
        global_state = base.add_delta(global_state, aggregate_delta)
        base.load_trainable_state(model, global_state)
        dev_metrics, _ = evaluate(model, dev_loader, device)
        progress.append({"progress": round_index, "clients": client_stats, "dev": dev_metrics})
        if dev_metrics["classification_loss"] < best_loss:
            best_loss = dev_metrics["classification_loss"]
            best_state = {name: tensor.clone() for name, tensor in global_state.items()}

    if best_state is None:
        raise RuntimeError("Federated training did not produce a best state")
    base.load_trainable_state(model, best_state)
    test_loader = loader_for_full(datasets["test"], collator, config, config.seed + 9300, False)
    test_metrics, test_outputs = evaluate(model, test_loader, device)
    return progress, best_state, test_outputs, {
        "train_seconds": train_seconds,
        "test_metrics": test_metrics,
        "he_rounds": he_rounds,
        "ckks_keygen_seconds": aggregator.keygen_seconds if aggregator is not None else None,
        "ckks_public_context_bytes": aggregator.public_context_bytes if aggregator is not None else None,
    }


def write_predictions(
    path: Path,
    frame: pd.DataFrame,
    outputs: dict[str, np.ndarray],
    config: RunConfig,
) -> None:
    order = np.argsort(outputs["source_rows"])
    labels = outputs["labels"][order]
    probabilities = outputs["probabilities"][order]
    source_rows = outputs["source_rows"][order]
    frame_by_source = frame.set_index("source_row", drop=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "dataset_id",
            "benchmark",
            "task",
            "method",
            "seed",
            "split",
            "source_row",
            "stable_id",
            "label",
            "prediction",
        ] + [f"score_{index}" for index in range(config.num_classes)]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, source_row in enumerate(source_rows):
            row = frame_by_source.loc[int(source_row)]
            record: dict[str, Any] = {
                "dataset_id": config.dataset_id,
                "benchmark": config.benchmark,
                "task": config.task,
                "method": config.method,
                "seed": config.seed,
                "split": "test",
                "source_row": int(source_row),
                "stable_id": row["stable_id"],
                "label": int(labels[index]),
                "prediction": int(np.argmax(probabilities[index])),
            }
            record.update({f"score_{class_index}": float(value) for class_index, value in enumerate(probabilities[index])})
            writer.writerow(record)


def valid_done(method_dir: Path, fingerprint: str) -> bool:
    done_path = method_dir / "DONE.json"
    if not done_path.is_file():
        return False
    try:
        done = json.loads(done_path.read_text(encoding="utf-8"))
        if done.get("status") != "complete" or done.get("fingerprint") != fingerprint:
            return False
        for item in done.get("artifacts", []):
            path = method_dir / item["name"]
            if not path.is_file() or path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
                return False
        return True
    except Exception:
        return False


def artifact_record(path: Path) -> dict[str, Any]:
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def main() -> None:
    args = parse_args()
    spec = task_spec(args.dataset_id)
    data_root = args.data_root.resolve()
    data_dir = data_root / Path(args.dataset_id)
    snapshot_path = args.model_snapshot.resolve()
    frames, source_audit, num_classes, sequence_columns = load_source_frames(
        data_dir,
        (args.train_limit, args.dev_limit, args.test_limit),
        args.seed,
    )
    selected_max_length = args.max_length or spec.max_length
    config = RunConfig(
        data_root=str(data_root),
        dataset_id=args.dataset_id,
        output_root=str(args.output_root.resolve()),
        cache_root=str(args.cache_root.resolve()),
        model_snapshot=str(snapshot_path),
        method=args.method,
        seed=args.seed,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        num_workers=args.num_workers,
        ckks_poly_modulus_degree=8192,
        ckks_coeff_mod_bits=[60, 40, 60],
        ckks_scale_bits=40,
        max_length=selected_max_length,
        passes=args.passes or spec.passes,
        batch_size=args.batch_size or spec.batch_size,
        eval_batch_size=args.eval_batch_size or spec.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps or spec.gradient_accumulation_steps,
        activation_offload=(
            args.activation_offload if args.activation_offload is not None else selected_max_length > 1536
        ),
        primary_metric=spec.primary_metric,
        benchmark=spec.benchmark,
        task=spec.task,
        num_classes=num_classes,
        sequence_columns=sequence_columns,
        train_limit=args.train_limit,
        dev_limit=args.dev_limit,
        test_limit=args.test_limit,
    )
    if config.max_length < 8 or config.passes < 1:
        raise ValueError("Invalid max_length or passes")
    snapshot = snapshot_evidence(snapshot_path)
    if snapshot["revision"] != SNAPSHOT_REVISION:
        raise RuntimeError(f"Unexpected model revision: {snapshot['revision']}")
    script_sha = sha256_file(Path(__file__))
    fingerprint_payload = {
        "schema": "gue-lora-macro-run-v1",
        "config": asdict(config),
        "source_audit": source_audit,
        "model_snapshot": snapshot,
        "runner_sha256": script_sha,
    }
    fingerprint = canonical_sha(fingerprint_payload)
    method_dir = Path(config.output_root) / config.dataset_id.replace("/", "__") / f"seed_{config.seed}" / config.method
    if not args.force and valid_done(method_dir, fingerprint):
        print(json.dumps({"status": "resume-skip", "path": method_dir.relative_to(ROOT).as_posix()}))
        return
    if method_dir.exists() and any(method_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite incomplete or mismatched method directory: {method_dir}")
    method_dir.mkdir(parents=True, exist_ok=True)

    cache_paths, token_cache_manifest = build_token_cache(config, frames, source_audit, snapshot)
    datasets = {split: RaggedTokenDataset(path) for split, path in cache_paths.items()}
    partitions, partition_audit = partition_evidence(datasets, config.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_snapshot,
        trust_remote_code=True,
        local_files_only=True,
        model_max_length=config.max_length,
        padding_side="right",
        use_fast=True,
    )
    collator = DynamicPadCollator(tokenizer.pad_token_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This full benchmark runner requires CUDA")
    torch.cuda.reset_peak_memory_stats()
    start_wall = time.perf_counter()
    model = create_model(config, device)
    initial_state = base.clone_trainable_state(model)
    model_audit = {
        "trainable_parameters": int(sum(tensor.numel() for tensor in initial_state.values())),
        "trainable_tensors": len(initial_state),
        "initial_trainable_state_sha256": trainable_state_sha(initial_state),
        "num_classes": config.num_classes,
        "lora_target_modules": ["Wqkv"],
        "activation_offload": config.activation_offload,
    }
    if config.method == "central":
        progress, best_state, test_outputs, result = run_central(model, datasets, collator, config, device)
    else:
        progress, best_state, test_outputs, result = run_federated(
            model, datasets, partitions, collator, config, device
        )
    wall_seconds = time.perf_counter() - start_wall
    test_metrics = result["test_metrics"]
    test_metrics["official_score"] = float(test_metrics[config.primary_metric])
    test_metrics["official_metric"] = config.primary_metric
    if any(not math.isfinite(float(value)) for key, value in test_metrics.items() if key not in ("official_metric", "auroc_macro_ovr", "auprc_macro_ovr")):
        raise FloatingPointError(f"Non-finite final metric: {test_metrics}")

    config_path = method_dir / "config.json"
    audit_path = method_dir / "audit.json"
    progress_path = method_dir / "progress.json"
    metrics_path = method_dir / "metrics.json"
    system_path = method_dir / "system.json"
    predictions_path = method_dir / "predictions.csv.gz"
    atomic_json(config_path, {"fingerprint": fingerprint, **fingerprint_payload})
    atomic_json(
        audit_path,
        {
            "source": source_audit,
            "token_cache": token_cache_manifest,
            "partition": partition_audit,
            "model": model_audit,
            "best_trainable_state_sha256": trainable_state_sha(best_state),
        },
    )
    atomic_json(progress_path, progress)
    atomic_json(
        metrics_path,
        {
            "dataset_id": config.dataset_id,
            "benchmark": config.benchmark,
            "task": config.task,
            "method": config.method,
            "seed": config.seed,
            "split": "test",
            **test_metrics,
        },
    )
    system_payload = {
        **{key: value for key, value in result.items() if key != "test_metrics"},
        "wall_seconds": wall_seconds,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device),
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    atomic_json(system_path, system_payload)
    write_predictions(predictions_path, frames["test"], test_outputs, config)
    artifacts = [artifact_record(path) for path in (config_path, audit_path, progress_path, metrics_path, system_path, predictions_path)]
    done_path = method_dir / "DONE.json"
    atomic_json(
        done_path,
        {
            "status": "complete",
            "fingerprint": fingerprint,
            "dataset_id": config.dataset_id,
            "method": config.method,
            "seed": config.seed,
            "official_metric": config.primary_metric,
            "official_score": test_metrics["official_score"],
            "test_evaluations": 1,
            "artifacts": artifacts,
        },
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "dataset_id": config.dataset_id,
                "method": config.method,
                "seed": config.seed,
                "official_metric": config.primary_metric,
                "official_score": test_metrics["official_score"],
                "wall_seconds": wall_seconds,
                "peak_cuda_reserved_gib": system_payload["peak_cuda_reserved_bytes"] / 2**30,
                "path": method_dir.relative_to(ROOT).as_posix(),
            },
            ensure_ascii=False,
        )
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
