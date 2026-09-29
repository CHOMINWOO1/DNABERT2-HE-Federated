from __future__ import annotations

"""Read-only readiness audit for the multiparty CKKS experiment.

The default command imports no project training modules and never launches
model training.  ``--crypto-smoke`` is an explicit opt-in for a three-vector
OpenFHE arithmetic check; it still does not load DNABERT-2 or touch a GPU.
"""

import argparse
import importlib
import importlib.util
import json
import platform
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "GUE_v2" / "EMP" / "H3K4me3"
DEFAULT_REFERENCE = PROJECT_ROOT / "experiments" / "h3k4me3_fl_earlystop_v3"
DEFAULT_MODEL = (
    PROJECT_ROOT
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--zhihan1996--DNABERT-2-117M"
    / "snapshots"
    / "7bce263b15377fc15361f52cfab88f8b586abda0"
)
SEEDS = (42, 43, 44, 45, 46)
REQUIRED_MODEL_FILES = (
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "pytorch_model.bin",
    "configuration_bert.py",
    "bert_layers.py",
    "bert_padding.py",
    "flash_attn_triton.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only readiness check for H3K4me3 multiparty CKKS"
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA))
    parser.add_argument("--reference-root", default=str(DEFAULT_REFERENCE))
    parser.add_argument("--model-snapshot-dir", default=str(DEFAULT_MODEL))
    parser.add_argument(
        "--crypto-smoke",
        action="store_true",
        help="Run only a tiny OpenFHE multiparty sum; never starts model training",
    )
    return parser.parse_args()


def _module_probe(name: str) -> dict[str, Any]:
    try:
        spec = importlib.util.find_spec(name)
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"available": spec is not None, "origin": getattr(spec, "origin", None)}


def _openfhe_symbol_available(module: Any, name: str, enum_name: str | None = None) -> bool:
    if hasattr(module, name):
        return True
    enum = getattr(module, enum_name, None) if enum_name else None
    return enum is not None and hasattr(enum, name)


def _reference_probe(root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    valid = True
    for seed in SEEDS:
        method_dir = root / "noniid" / f"seed_{seed}" / "methods" / "fedavg"
        done_path = method_dir / "DONE.json"
        audit_path = method_dir / "method_model_audit.json"
        row: dict[str, Any] = {
            "seed": seed,
            "done_path": str(done_path),
            "audit_path": str(audit_path),
        }
        try:
            done = json.loads(done_path.read_text(encoding="utf-8"))
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            selection = audit["early_stopping"]
            selected_round = int(selection["selected_round"])
            row.update(
                {
                    "valid": (
                        done.get("status") == "complete"
                        and done.get("method") == "fedavg"
                        and selection.get("selection_owner") == "fedavg"
                        and selected_round >= 1
                    ),
                    "selected_round": selected_round,
                }
            )
        except Exception as exc:
            row.update({"valid": False, "error": f"{type(exc).__name__}: {exc}"})
        valid = valid and bool(row["valid"])
        rows.append(row)
    return {"valid": valid, "seeds": rows}


def collect_preflight(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = Path(args.data_dir).resolve()
    model_dir = Path(args.model_snapshot_dir).resolve()
    reference_root = Path(args.reference_root).resolve()
    modules = {name: _module_probe(name) for name in ("numpy", "torch", "openfhe")}

    data_files = {
        split: (data_dir / f"{split}.csv").is_file()
        for split in ("train", "dev", "test")
    }
    model_files = {
        name: (model_dir / name).is_file() for name in REQUIRED_MODEL_FILES
    }
    torch_probe: dict[str, Any] = {"imported": False, "cuda_available": False}
    if modules["torch"]["available"]:
        try:
            torch = importlib.import_module("torch")
            torch_probe = {
                "imported": True,
                "version": getattr(torch, "__version__", None),
                "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
                "cuda_available": bool(torch.cuda.is_available()),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        except Exception as exc:
            torch_probe = {
                "imported": False,
                "cuda_available": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    openfhe_probe: dict[str, Any] = {"imported": False, "required_api": False}
    if modules["openfhe"]["available"]:
        try:
            openfhe = importlib.import_module("openfhe")
            required = {
                "CCParamsCKKSRNS": _openfhe_symbol_available(
                    openfhe, "CCParamsCKKSRNS"
                ),
                "GenCryptoContext": _openfhe_symbol_available(
                    openfhe, "GenCryptoContext"
                ),
                "PKE": _openfhe_symbol_available(
                    openfhe, "PKE", "PKESchemeFeature"
                ),
                "KEYSWITCH": _openfhe_symbol_available(
                    openfhe, "KEYSWITCH", "PKESchemeFeature"
                ),
                "LEVELEDSHE": _openfhe_symbol_available(
                    openfhe, "LEVELEDSHE", "PKESchemeFeature"
                ),
                "ADVANCEDSHE": _openfhe_symbol_available(
                    openfhe, "ADVANCEDSHE", "PKESchemeFeature"
                ),
                "MULTIPARTY": _openfhe_symbol_available(
                    openfhe, "MULTIPARTY", "PKESchemeFeature"
                ),
                "HEStd_128_classic": _openfhe_symbol_available(
                    openfhe, "HEStd_128_classic", "SecurityLevel"
                ),
                "SerializeToFile": _openfhe_symbol_available(
                    openfhe, "SerializeToFile"
                ),
                "BINARY": _openfhe_symbol_available(openfhe, "BINARY"),
            }
            openfhe_probe = {
                "imported": True,
                "required_api": all(required.values()),
                "version": getattr(openfhe, "__version__", None),
                "missing_api": [name for name, available in required.items() if not available],
            }
        except Exception as exc:
            openfhe_probe = {
                "imported": False,
                "required_api": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    reference = _reference_probe(reference_root)
    python_ok = sys.version_info >= (3, 10)
    static_ready = (
        python_ok
        and modules["numpy"]["available"]
        and openfhe_probe["imported"]
        and openfhe_probe["required_api"]
        and all(data_files.values())
        and all(model_files.values())
        and reference["valid"]
    )
    training_ready = static_ready and torch_probe["imported"] and torch_probe["cuda_available"]
    payload: dict[str, Any] = {
        "schema_version": "threshold-ckks-preflight-v1",
        "read_only": True,
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "supported": python_ok,
            "platform": platform.platform(),
        },
        "modules": modules,
        "torch": torch_probe,
        "openfhe": openfhe_probe,
        "data": {"root": str(data_dir), "files": data_files},
        "model_snapshot": {"root": str(model_dir), "files": model_files},
        "reference_plain_earlystop": reference,
        "static_ready": static_ready,
        "full_experiment_ready": training_ready,
        "notes": [
            "The official OpenFHE wheel is distributed for supported Ubuntu LTS environments.",
            "Windows requires a source build or a Linux/WSL execution environment.",
            "The experiment runner remains locked until --execute is passed explicitly.",
        ],
    }
    if args.crypto_smoke:
        payload["crypto_smoke"] = _run_crypto_smoke() if static_ready else {
            "ran": False,
            "reason": "static preflight failed",
        }
    return payload


def _run_crypto_smoke() -> dict[str, Any]:
    import numpy as np

    from openfhe_threshold_ckks import (
        OpenFHEThresholdCKKSAggregator,
        ThresholdCKKSConfig,
    )

    vectors = [
        np.asarray([0.1, 0.2, -0.3], dtype=np.float64),
        np.asarray([0.4, -0.2, 0.1], dtype=np.float64),
        np.asarray([-0.1, 0.3, 0.2], dtype=np.float64),
    ]
    expected = np.sum(vectors, axis=0)
    aggregator = OpenFHEThresholdCKKSAggregator(
        ThresholdCKKSConfig(batch_size=8, measure_serialized_bytes=False)
    )
    recovered, stats = aggregator.aggregate(vectors)
    max_abs = float(np.max(np.abs(recovered - expected)))
    return {
        "ran": True,
        "max_abs": max_abs,
        "tolerance": 1e-6,
        "passed": max_abs <= 1e-6,
        "stats": stats,
    }


def main() -> None:
    args = parse_args()
    payload = collect_preflight(args)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not payload["full_experiment_ready"]:
        raise SystemExit(2)
    if args.crypto_smoke and not payload["crypto_smoke"].get("passed", False):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
