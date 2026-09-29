from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import pandas as pd
import tenseal as ts
import torch
import torch.nn as nn
import truststore
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoModel, AutoTokenizer
from transformers.models.bert.configuration_bert import BertConfig


MODEL_NAME = "zhihan1996/DNABERT-2-117M"
SITE_NAMES = ("A", "B", "C")


@dataclass
class ExperimentConfig:
    data_dir: str
    output_dir: str
    train_limit: int
    dev_limit: int
    test_limit: int
    max_length: int
    batch_size: int
    eval_batch_size: int
    epochs: int
    rounds: int
    local_epochs: int
    learning_rate: float
    weight_decay: float
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    seed: int
    scenario: str
    methods: list[str]
    num_workers: int
    ckks_poly_modulus_degree: int
    ckks_coeff_mod_bits: list[int]
    ckks_scale_bits: int


class TokenDataset(Dataset):
    def __init__(self, encodings: dict[str, torch.Tensor], labels: torch.Tensor):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


class DNABertLoRAClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_size: int, num_labels: int = 2):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs[0]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.classifier(self.dropout(pooled))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DNABERT-2 LoRA federated/HE experiment")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "GUE_v2" / "EMP" / "H3K4me3"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "experiments" / "fedhe_h3k4me3"))
    parser.add_argument("--train-limit", type=int, default=1200)
    parser.add_argument("--dev-limit", type=int, default=300)
    parser.add_argument("--test-limit", type=int, default=600)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario", choices=("iid", "noniid"), default="iid")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("local", "central", "fedavg", "he_fedavg"),
        default=["local", "central", "fedavg", "he_fedavg"],
    )
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def stratified_subset(frame: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    if limit <= 0 or limit >= len(frame):
        return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    labels = frame["label"].to_numpy()
    unique, counts = np.unique(labels, return_counts=True)
    raw = counts / counts.sum() * limit
    take = np.floor(raw).astype(int)
    for index in np.argsort(-(raw - take))[: limit - int(take.sum())]:
        take[index] += 1
    for label, count in zip(unique, take, strict=True):
        candidates = np.flatnonzero(labels == label)
        selected.extend(rng.choice(candidates, size=int(count), replace=False).tolist())
    rng.shuffle(selected)
    return frame.iloc[selected].reset_index(drop=True)


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
        subset = stratified_subset(frame, limit, seed + offset)
        frames[split] = subset
        audit[split] = {
            "source_path": str(path),
            "source_sha256": sha256_file(path),
            "source_n": int(len(frame)),
            "used_n": int(len(subset)),
            "used_class_counts": {str(k): int(v) for k, v in subset["label"].value_counts().sort_index().items()},
            "sequence_length_min": int(subset["sequence"].str.len().min()),
            "sequence_length_median": float(subset["sequence"].str.len().median()),
            "sequence_length_max": int(subset["sequence"].str.len().max()),
            "exact_duplicates": int(subset.duplicated("sequence").sum()),
        }
    return frames, audit


def allocate_indices(labels: np.ndarray, scenario: str, seed: int) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed)
    sites = {site: [] for site in SITE_NAMES}
    if scenario == "iid":
        for label in (0, 1):
            indices = np.flatnonzero(labels == label)
            rng.shuffle(indices)
            chunks = np.array_split(indices, len(SITE_NAMES))
            for site, chunk in zip(SITE_NAMES, chunks, strict=True):
                sites[site].extend(chunk.tolist())
    else:
        # Controlled label skew: site A is positive-rich and C is negative-rich.
        class_weights = {
            0: np.asarray([0.15, 0.35, 0.50], dtype=np.float64),
            1: np.asarray([0.60, 0.30, 0.10], dtype=np.float64),
        }
        for label in (0, 1):
            indices = np.flatnonzero(labels == label)
            rng.shuffle(indices)
            weights = class_weights[label] / class_weights[label].sum()
            raw_counts = weights * len(indices)
            counts = np.floor(raw_counts).astype(int)
            for idx in np.argsort(-(raw_counts - counts))[: len(indices) - int(counts.sum())]:
                counts[idx] += 1
            boundaries = np.cumsum(counts)[:-1]
            chunks = np.split(indices, boundaries)
            for site, chunk in zip(SITE_NAMES, chunks, strict=True):
                sites[site].extend(chunk.tolist())
    for site in SITE_NAMES:
        rng.shuffle(sites[site])
    all_indices = sorted(index for values in sites.values() for index in values)
    if all_indices != list(range(len(labels))):
        raise RuntimeError("Partition does not cover every example exactly once")
    return sites


def partition_audit(frames: dict[str, pd.DataFrame], partitions: dict[str, dict[str, list[int]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split, site_map in partitions.items():
        result[split] = {}
        for site, indices in site_map.items():
            labels = frames[split].iloc[indices]["label"]
            result[split][site] = {
                "n": int(len(indices)),
                "class_0": int((labels == 0).sum()),
                "class_1": int((labels == 1).sum()),
                "positive_rate": float(labels.mean()),
            }
    return result


def tokenize_frames(frames: dict[str, pd.DataFrame], tokenizer: Any, max_length: int) -> dict[str, TokenDataset]:
    datasets: dict[str, TokenDataset] = {}
    for split, frame in frames.items():
        encoded = tokenizer(
            frame["sequence"].tolist(),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        datasets[split] = TokenDataset(
            {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]},
            torch.tensor(frame["label"].to_numpy(), dtype=torch.long),
        )
    return datasets


def make_loader(
    dataset: Dataset,
    indices: Iterable[int] | None,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    selected: Dataset = dataset if indices is None else Subset(dataset, list(indices))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        selected,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def create_model(config: ExperimentConfig, init_seed: int, device: torch.device) -> DNABertLoRAClassifier:
    bert_config = BertConfig.from_pretrained(MODEL_NAME)
    bert_config.attention_probs_dropout_prob = 0.1  # Forces the Windows-compatible PyTorch attention path.
    backbone = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        config=bert_config,
        add_pooling_layer=False,
    )
    set_seed(init_seed)
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=["Wqkv"],
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
    )
    encoder = get_peft_model(backbone, lora_config)
    model = DNABertLoRAClassifier(encoder, hidden_size=bert_config.hidden_size)
    return model.to(device)


def trainable_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    params = {name: param for name, param in model.named_parameters() if param.requires_grad}
    if not params:
        raise RuntimeError("No trainable parameters found")
    return dict(sorted(params.items()))


def clone_trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: param.detach().cpu().float().clone() for name, param in trainable_parameters(model).items()}


def load_trainable_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    params = trainable_parameters(model)
    if list(params) != list(state):
        raise ValueError("Trainable parameter manifest mismatch")
    with torch.no_grad():
        for name, param in params.items():
            param.copy_(state[name].to(device=param.device, dtype=param.dtype))


def state_delta(local_state: dict[str, torch.Tensor], global_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: local_state[name] - global_state[name] for name in global_state}


def weighted_average_deltas(deltas: list[dict[str, torch.Tensor]], weights: list[float]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name in deltas[0]:
        value = torch.zeros_like(deltas[0][name], dtype=torch.float64)
        for delta, weight in zip(deltas, weights, strict=True):
            value.add_(delta[name].double(), alpha=float(weight))
        result[name] = value.float()
    return result


def add_delta(state: dict[str, torch.Tensor], delta: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: state[name] + delta[name] for name in state}


def flatten_state(state: dict[str, torch.Tensor]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    arrays: list[np.ndarray] = []
    manifest: list[dict[str, Any]] = []
    offset = 0
    for name in sorted(state):
        tensor = state[name].detach().cpu().float().contiguous()
        flat = tensor.numpy().reshape(-1)
        arrays.append(flat)
        manifest.append({"name": name, "shape": list(tensor.shape), "offset": offset, "numel": int(flat.size)})
        offset += int(flat.size)
    return np.concatenate(arrays).astype(np.float64), manifest


def unflatten_state(vector: np.ndarray, manifest: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for entry in manifest:
        start = int(entry["offset"])
        stop = start + int(entry["numel"])
        array = vector[start:stop].reshape(entry["shape"]).astype(np.float32)
        state[entry["name"]] = torch.from_numpy(array.copy())
    return state


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
    model.train()
    criterion = nn.CrossEntropyLoss()
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start = time.perf_counter()
    total_loss = 0.0
    total_n = 0
    for batch in loader:
        labels = batch.pop("labels").to(device, non_blocking=True)
        inputs = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(**inputs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach()) * labels.shape[0]
        total_n += int(labels.shape[0])
    if device.type == "cuda":
        torch.cuda.synchronize()
    return total_loss / max(total_n, 1), time.perf_counter() - start


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    labels_all: list[np.ndarray] = []
    scores_all: list[np.ndarray] = []
    start = time.perf_counter()
    for batch in loader:
        labels = batch.pop("labels").numpy()
        inputs = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(**inputs)
        scores = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
        labels_all.append(labels)
        scores_all.append(scores)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return np.concatenate(labels_all), np.concatenate(scores_all), time.perf_counter() - start


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    predictions = (scores >= 0.5).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
    }


def evaluate_scopes(
    model: nn.Module,
    dataset: TokenDataset,
    partitions: dict[str, list[int]],
    split_name: str,
    config: ExperimentConfig,
    device: torch.device,
    seed_offset: int,
) -> tuple[list[dict[str, Any]], dict[str, tuple[np.ndarray, np.ndarray]]]:
    records: list[dict[str, Any]] = []
    outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    pooled_labels: list[np.ndarray] = []
    pooled_scores: list[np.ndarray] = []
    for index, site in enumerate(SITE_NAMES):
        loader = make_loader(
            dataset,
            partitions[site],
            config.eval_batch_size,
            False,
            config.seed + seed_offset + index,
            config.num_workers,
        )
        labels, scores, eval_seconds = predict(model, loader, device)
        outputs[site] = (labels, scores)
        pooled_labels.append(labels)
        pooled_scores.append(scores)
        records.append({"split": split_name, "scope": f"site_{site}", "n": int(len(labels)), "eval_seconds": eval_seconds, **binary_metrics(labels, scores)})
    labels = np.concatenate(pooled_labels)
    scores = np.concatenate(pooled_scores)
    outputs["pooled"] = (labels, scores)
    records.append({"split": split_name, "scope": "pooled", "n": int(len(labels)), "eval_seconds": float("nan"), **binary_metrics(labels, scores)})
    return records, outputs


class CKKSAdditiveAggregator:
    def __init__(self, config: ExperimentConfig):
        start = time.perf_counter()
        self.secret_context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=config.ckks_poly_modulus_degree,
            coeff_mod_bit_sizes=config.ckks_coeff_mod_bits,
        )
        self.secret_context.global_scale = 2**config.ckks_scale_bits
        public_blob = self.secret_context.serialize(
            save_public_key=True,
            save_secret_key=False,
            save_galois_keys=False,
            save_relin_keys=False,
        )
        self.public_context = ts.context_from(public_blob)
        self.keygen_seconds = time.perf_counter() - start
        self.public_context_bytes = len(public_blob)
        self.secret_context_bytes = len(self.secret_context.serialize(save_secret_key=True))
        self.slots = config.ckks_poly_modulus_degree // 2

    def aggregate(self, weighted_vectors: list[np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
        encrypted_clients: list[list[bytes]] = []
        encrypt_seconds: list[float] = []
        upload_bytes: list[int] = []
        for vector in weighted_vectors:
            start = time.perf_counter()
            chunks: list[bytes] = []
            for offset in range(0, len(vector), self.slots):
                encrypted = ts.ckks_vector(self.public_context, vector[offset : offset + self.slots].tolist())
                chunks.append(encrypted.serialize())
            encrypt_seconds.append(time.perf_counter() - start)
            upload_bytes.append(sum(len(chunk) for chunk in chunks))
            encrypted_clients.append(chunks)

        start = time.perf_counter()
        aggregate_chunks: list[bytes] = []
        for chunk_index in range(len(encrypted_clients[0])):
            value = ts.ckks_vector_from(self.public_context, encrypted_clients[0][chunk_index])
            for client_index in range(1, len(encrypted_clients)):
                value += ts.ckks_vector_from(self.public_context, encrypted_clients[client_index][chunk_index])
            aggregate_chunks.append(value.serialize())
        aggregate_seconds = time.perf_counter() - start

        start = time.perf_counter()
        recovered: list[float] = []
        for chunk in aggregate_chunks:
            value = ts.ckks_vector_from(self.secret_context, chunk)
            recovered.extend(value.decrypt())
        decrypt_seconds = time.perf_counter() - start
        result = np.asarray(recovered[: len(weighted_vectors[0])], dtype=np.float64)
        stats = {
            "encrypt_seconds_by_client": encrypt_seconds,
            "upload_bytes_by_client": upload_bytes,
            "server_aggregate_seconds": aggregate_seconds,
            "decrypt_seconds": decrypt_seconds,
            "aggregate_ciphertext_bytes": sum(len(chunk) for chunk in aggregate_chunks),
            "ciphertext_chunks": len(aggregate_chunks),
        }
        return result, stats


def fit_central(
    config: ExperimentConfig,
    datasets: dict[str, TokenDataset],
    partitions: dict[str, dict[str, list[int]]],
    device: torch.device,
    init_state: dict[str, torch.Tensor],
) -> tuple[nn.Module, dict[str, Any]]:
    model = create_model(config, config.seed, device)
    load_trainable_state(model, init_state)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    epoch_records = []
    total_seconds = 0.0
    for epoch in range(config.epochs):
        loader = make_loader(datasets["train"], None, config.batch_size, True, config.seed + epoch, config.num_workers)
        loss, seconds = train_one_epoch(model, loader, device, optimizer)
        total_seconds += seconds
        epoch_records.append({"epoch": epoch + 1, "loss": loss, "seconds": seconds})
    return model, {"train_seconds": total_seconds, "epochs": epoch_records}


def fit_local_models(
    config: ExperimentConfig,
    datasets: dict[str, TokenDataset],
    partitions: dict[str, dict[str, list[int]]],
    device: torch.device,
    init_state: dict[str, torch.Tensor],
) -> tuple[dict[str, nn.Module], dict[str, Any]]:
    models: dict[str, nn.Module] = {}
    site_stats: dict[str, Any] = {}
    for site_index, site in enumerate(SITE_NAMES):
        model = create_model(config, config.seed, device)
        load_trainable_state(model, init_state)
        optimizer = torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        total_seconds = 0.0
        epochs = []
        for epoch in range(config.epochs):
            set_seed(config.seed + 1000 + site_index * 100 + epoch)
            loader = make_loader(
                datasets["train"], partitions["train"][site], config.batch_size, True,
                config.seed + site_index * 100 + epoch, config.num_workers,
            )
            loss, seconds = train_one_epoch(model, loader, device, optimizer)
            total_seconds += seconds
            epochs.append({"epoch": epoch + 1, "loss": loss, "seconds": seconds})
        models[site] = model.cpu()
        site_stats[site] = {"train_seconds": total_seconds, "epochs": epochs}
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return models, site_stats


def fit_federated(
    config: ExperimentConfig,
    datasets: dict[str, TokenDataset],
    partitions: dict[str, dict[str, list[int]]],
    device: torch.device,
    init_state: dict[str, torch.Tensor],
    use_he: bool,
) -> tuple[nn.Module, dict[str, Any]]:
    global_state = {name: value.clone() for name, value in init_state.items()}
    local_model = create_model(config, config.seed, device)
    aggregator = CKKSAdditiveAggregator(config) if use_he else None
    round_records: list[dict[str, Any]] = []
    for round_index in range(config.rounds):
        deltas: list[dict[str, torch.Tensor]] = []
        sample_counts: list[int] = []
        local_stats: list[dict[str, Any]] = []
        for site_index, site in enumerate(SITE_NAMES):
            load_trainable_state(local_model, global_state)
            optimizer = torch.optim.AdamW(
                [param for param in local_model.parameters() if param.requires_grad],
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            local_seconds = 0.0
            losses = []
            for local_epoch in range(config.local_epochs):
                paired_seed = config.seed + round_index * 1000 + site_index * 100 + local_epoch
                set_seed(paired_seed)
                loader = make_loader(
                    datasets["train"], partitions["train"][site], config.batch_size, True,
                    paired_seed, config.num_workers,
                )
                loss, seconds = train_one_epoch(local_model, loader, device, optimizer)
                local_seconds += seconds
                losses.append(loss)
            local_state = clone_trainable_state(local_model)
            deltas.append(state_delta(local_state, global_state))
            n_site = len(partitions["train"][site])
            sample_counts.append(n_site)
            local_stats.append({"site": site, "n": n_site, "losses": losses, "train_seconds": local_seconds})

        total_n = sum(sample_counts)
        weights = [count / total_n for count in sample_counts]
        plaintext_delta = weighted_average_deltas(deltas, weights)
        crypto_stats: dict[str, Any] | None = None
        fidelity: dict[str, float] | None = None
        if use_he:
            weighted_vectors = []
            manifest: list[dict[str, Any]] | None = None
            for delta, weight in zip(deltas, weights, strict=True):
                vector, current_manifest = flatten_state(delta)
                if manifest is None:
                    manifest = current_manifest
                elif manifest != current_manifest:
                    raise RuntimeError("Client manifests differ")
                weighted_vectors.append(vector * weight)
            assert aggregator is not None and manifest is not None
            he_vector, crypto_stats = aggregator.aggregate(weighted_vectors)
            plain_vector, _ = flatten_state(plaintext_delta)
            error = he_vector - plain_vector
            denom = max(float(np.linalg.norm(plain_vector)), 1e-12)
            fidelity = {
                "mae": float(np.mean(np.abs(error))),
                "max_abs": float(np.max(np.abs(error))),
                "relative_l2": float(np.linalg.norm(error) / denom),
                "cosine": float(np.dot(he_vector, plain_vector) / (max(np.linalg.norm(he_vector), 1e-12) * denom)),
            }
            aggregate_delta = unflatten_state(he_vector, manifest)
        else:
            aggregate_delta = plaintext_delta
        global_state = add_delta(global_state, aggregate_delta)
        round_records.append({
            "round": round_index + 1,
            "clients": local_stats,
            "weights": weights,
            "crypto": crypto_stats,
            "fidelity": fidelity,
        })
    load_trainable_state(local_model, global_state)
    info: dict[str, Any] = {"rounds": round_records}
    if aggregator is not None:
        info["ckks_context"] = {
            "poly_modulus_degree": config.ckks_poly_modulus_degree,
            "coeff_mod_bit_sizes": config.ckks_coeff_mod_bits,
            "scale_bits": config.ckks_scale_bits,
            "slots": aggregator.slots,
            "keygen_seconds": aggregator.keygen_seconds,
            "public_context_bytes": aggregator.public_context_bytes,
            "secret_context_bytes": aggregator.secret_context_bytes,
        }
    return local_model, info


def attach_method_fields(records: list[dict[str, Any]], method: str, config: ExperimentConfig) -> list[dict[str, Any]]:
    return [{"method": method, "scenario": config.scenario, "seed": config.seed, **record} for record in records]


def main() -> None:
    args = parse_args()
    truststore.inject_into_ssl()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this experiment")
    device = torch.device("cuda")
    set_seed(args.seed)
    config = ExperimentConfig(
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
    )
    run_dir = Path(config.output_dir) / config.scenario / f"seed_{config.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    frames, dataset_audit = load_frames(
        Path(config.data_dir),
        (config.train_limit, config.dev_limit, config.test_limit),
        config.seed,
    )
    partitions = {
        split: allocate_indices(frame["label"].to_numpy(), config.scenario, config.seed + offset * 10)
        for offset, (split, frame) in enumerate(frames.items())
    }
    audit = {
        "dataset": dataset_audit,
        "partitions": partition_audit(frames, partitions),
    }
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    (run_dir / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    for split, frame in frames.items():
        export = frame.copy()
        site_by_index = {}
        for site, indices in partitions[split].items():
            site_by_index.update({index: site for index in indices})
        export.insert(0, "stable_id", [f"{split}_{index:06d}" for index in range(len(export))])
        export["site"] = [site_by_index[index] for index in range(len(export))]
        export.to_csv(run_dir / f"{split}_partition.csv", index=False)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    datasets = tokenize_frames(frames, tokenizer, config.max_length)

    init_model = create_model(config, config.seed, device)
    init_state = clone_trainable_state(init_model)
    trainable_vector, manifest = flatten_state(init_state)
    manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    model_audit = {
        "trainable_parameters": int(trainable_vector.size),
        "trainable_fp32_bytes": int(trainable_vector.size * 4),
        "manifest_sha256": manifest_hash,
        "manifest": manifest,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }
    (run_dir / "model_audit.json").write_text(json.dumps(model_audit, indent=2), encoding="utf-8")
    del init_model
    torch.cuda.empty_cache()

    metric_records: list[dict[str, Any]] = []
    system_results: dict[str, Any] = {}

    if "central" in config.methods:
        model, stats = fit_central(config, datasets, partitions, device, init_state)
        for split_index, split in enumerate(("dev", "test")):
            records, _ = evaluate_scopes(model, datasets[split], partitions[split], split, config, device, 100 + split_index * 10)
            metric_records.extend(attach_method_fields(records, "central", config))
        system_results["central"] = stats
        del model
        torch.cuda.empty_cache()

    if "local" in config.methods:
        models, stats = fit_local_models(config, datasets, partitions, device, init_state)
        local_records: list[dict[str, Any]] = []
        for split_index, split in enumerate(("dev", "test")):
            local_outputs: list[tuple[np.ndarray, np.ndarray]] = []
            for site_index, site in enumerate(SITE_NAMES):
                model = models[site].to(device)
                loader = make_loader(
                    datasets[split], partitions[split][site], config.eval_batch_size, False,
                    config.seed + 200 + split_index * 10 + site_index, config.num_workers,
                )
                labels, scores, eval_seconds = predict(model, loader, device)
                local_outputs.append((labels, scores))
                local_records.append({"split": split, "scope": f"site_{site}", "n": int(len(labels)), "eval_seconds": eval_seconds, **binary_metrics(labels, scores)})
                models[site] = model.cpu()
            pooled_labels = np.concatenate([item[0] for item in local_outputs])
            pooled_scores = np.concatenate([item[1] for item in local_outputs])
            local_records.append({"split": split, "scope": "pooled", "n": int(len(pooled_labels)), "eval_seconds": float("nan"), **binary_metrics(pooled_labels, pooled_scores)})
        metric_records.extend(attach_method_fields(local_records, "local", config))
        system_results["local"] = stats
        del models
        torch.cuda.empty_cache()

    for method, use_he in (("fedavg", False), ("he_fedavg", True)):
        if method not in config.methods:
            continue
        model, stats = fit_federated(config, datasets, partitions, device, init_state, use_he)
        for split_index, split in enumerate(("dev", "test")):
            records, _ = evaluate_scopes(
                model, datasets[split], partitions[split], split, config, device,
                (300 if not use_he else 400) + split_index * 10,
            )
            metric_records.extend(attach_method_fields(records, method, config))
        system_results[method] = stats
        torch.save(clone_trainable_state(model), run_dir / f"{method}_trainable_state.pt")
        del model
        torch.cuda.empty_cache()

    pd.DataFrame(metric_records).to_csv(run_dir / "metrics.csv", index=False)
    (run_dir / "system_metrics.json").write_text(json.dumps(system_results, indent=2), encoding="utf-8")
    summary = {
        "status": "complete",
        "run_dir": str(run_dir),
        "methods": config.methods,
        "pooled_test_metrics": [record for record in metric_records if record["scope"] == "pooled" and record["split"] == "test"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
