from __future__ import annotations

"""OpenFHE multiparty CKKS aggregation for weighted federated updates.

The implementation intentionally keeps model training out of this module.  It
accepts already sample-weighted client vectors, encrypts them under a joint
public key, adds ciphertexts, requests one partial decryption from every party,
and fuses those shares into the aggregate vector.

Security scope
--------------
This Python adapter invokes the party roles in one process so that numerical
fidelity and systems overhead can be measured before a real multi-host
deployment.  No complete secret key is generated, but every secret-key share is
still resident in this process.  Consequently, results produced by this module
must be described as an *in-process multiparty CKKS simulation*, not as physical
institution isolation or a production threshold service.
"""

import importlib
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

import numpy as np


class OpenFHEUnavailable(RuntimeError):
    """Raised when the OpenFHE Python wrapper or a required API is unavailable."""


@dataclass(frozen=True)
class ThresholdCKKSConfig:
    party_count: int = 3
    ring_dimension: int = 8192
    batch_size: int = 4096
    multiplicative_depth: int = 1
    scaling_mod_size: int = 40
    first_mod_size: int = 60
    security_level: str = "HEStd_128_classic"
    measure_serialized_bytes: bool = True

    def validate(self) -> None:
        if self.party_count < 2:
            raise ValueError("Threshold CKKS requires at least two parties")
        if self.ring_dimension < 16 or self.ring_dimension & (self.ring_dimension - 1):
            raise ValueError("ring_dimension must be a power of two")
        if not 1 <= self.batch_size <= self.ring_dimension // 2:
            raise ValueError("batch_size must be between 1 and ring_dimension/2")
        if self.multiplicative_depth < 1:
            raise ValueError("multiplicative_depth must be positive")
        if not 20 <= self.scaling_mod_size < 60:
            raise ValueError("scaling_mod_size must be in [20, 60)")
        if not self.scaling_mod_size <= self.first_mod_size <= 60:
            raise ValueError("first_mod_size must be between scaling_mod_size and 60")


def _load_openfhe() -> ModuleType:
    try:
        module = importlib.import_module("openfhe")
    except Exception as exc:  # ImportError plus native-loader failures
        raise OpenFHEUnavailable(
            "OpenFHE-Python could not be imported. Install a compatible OpenFHE "
            "1.5.1+ wrapper before running threshold CKKS."
        ) from exc
    return module


def _require_api(module: ModuleType, *names: str) -> None:
    missing = [name for name in names if not hasattr(module, name)]
    if missing:
        raise OpenFHEUnavailable(f"OpenFHE wrapper lacks required APIs: {missing}")


def _resolve_feature(module: ModuleType, name: str) -> Any:
    if hasattr(module, name):
        return getattr(module, name)
    enum = getattr(module, "PKESchemeFeature", None)
    if enum is not None and hasattr(enum, name):
        return getattr(enum, name)
    raise OpenFHEUnavailable(f"OpenFHE wrapper lacks scheme feature {name}")


def _resolve_security_level(module: ModuleType, name: str) -> Any:
    if hasattr(module, name):
        return getattr(module, name)
    enum = getattr(module, "SecurityLevel", None)
    if enum is not None and hasattr(enum, name):
        return getattr(enum, name)
    raise OpenFHEUnavailable(f"OpenFHE wrapper lacks security level {name}")


def _keypair_good(keypair: Any) -> bool:
    good = getattr(keypair, "good", None)
    return bool(good() if callable(good) else good)


def _safe_key_tag(key: Any) -> str | None:
    getter = getattr(key, "GetKeyTag", None)
    if not callable(getter):
        return None
    try:
        return str(getter())
    except Exception:
        return None


class OpenFHEThresholdCKKSAggregator:
    """All-party multiparty CKKS weighted-sum aggregator.

    OpenFHE's sequential ``MultipartyKeyGen`` protocol produces one secret-key
    share per party and a final joint public key.  The server role uses only the
    joint public key.  The current adapter keeps the role objects in one Python
    process for controlled experiments; see :meth:`security_metadata`.
    """

    def __init__(
        self,
        config: ThresholdCKKSConfig | None = None,
        *,
        openfhe_module: ModuleType | None = None,
    ) -> None:
        self.config = config or ThresholdCKKSConfig()
        self.config.validate()
        self._of = openfhe_module or _load_openfhe()
        _require_api(self._of, "CCParamsCKKSRNS", "GenCryptoContext")

        started = time.perf_counter()
        parameters = self._of.CCParamsCKKSRNS()
        parameters.SetMultiplicativeDepth(self.config.multiplicative_depth)
        parameters.SetScalingModSize(self.config.scaling_mod_size)
        parameters.SetFirstModSize(self.config.first_mod_size)
        parameters.SetBatchSize(self.config.batch_size)
        parameters.SetRingDim(self.config.ring_dimension)
        parameters.SetSecurityLevel(
            _resolve_security_level(self._of, self.config.security_level)
        )

        self.context = self._of.GenCryptoContext(parameters)
        for name in ("PKE", "KEYSWITCH", "LEVELEDSHE", "ADVANCEDSHE", "MULTIPARTY"):
            self.context.Enable(_resolve_feature(self._of, name))

        first = self.context.KeyGen()
        if not _keypair_good(first):
            raise RuntimeError("OpenFHE party 0 key generation failed")
        self._secret_key_shares = [first.secretKey]
        self.joint_public_key = first.publicKey
        for party_index in range(1, self.config.party_count):
            keypair = self.context.MultipartyKeyGen(self.joint_public_key)
            if not _keypair_good(keypair):
                raise RuntimeError(f"OpenFHE party {party_index} key generation failed")
            self._secret_key_shares.append(keypair.secretKey)
            self.joint_public_key = keypair.publicKey

        self.keygen_seconds = time.perf_counter() - started
        self.slots = self.config.batch_size
        actual_ring = getattr(self.context, "GetRingDimension", lambda: None)()
        if actual_ring is not None and int(actual_ring) != self.config.ring_dimension:
            raise RuntimeError(
                "OpenFHE generated an unexpected ring dimension: "
                f"{actual_ring} != {self.config.ring_dimension}"
            )

    def security_metadata(self) -> dict[str, Any]:
        return {
            "backend": "OpenFHE-Python",
            "scheme": "CKKS",
            "key_model": "multiparty joint public key with per-party secret-key shares",
            "decryption_policy": "all configured parties contribute one partial decryption",
            "party_count": self.config.party_count,
            "joint_public_key_tag": _safe_key_tag(self.joint_public_key),
            "complete_secret_key_created": False,
            "secret_key_shares_persisted": False,
            "role_isolation": "in_process_simulation",
            "physical_institution_isolation": False,
            "production_threshold_claim_supported": False,
            "parameters": asdict(self.config),
        }

    def _serialized_size(self, value: Any, directory: Path, filename: str) -> tuple[int, float]:
        if not self.config.measure_serialized_bytes:
            return 0, 0.0
        serializer = getattr(self._of, "SerializeToFile", None)
        binary = getattr(self._of, "BINARY", None)
        if not callable(serializer) or binary is None:
            raise OpenFHEUnavailable(
                "Serialized-byte measurement requires OpenFHE SerializeToFile and BINARY"
            )
        path = directory / filename
        started = time.perf_counter()
        ok = serializer(str(path), value, binary)
        elapsed = time.perf_counter() - started
        if not ok or not path.is_file():
            raise RuntimeError(f"OpenFHE failed to serialize {filename}")
        return int(path.stat().st_size), elapsed

    def aggregate(self, weighted_vectors: Sequence[np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
        if len(weighted_vectors) != self.config.party_count:
            raise ValueError(
                f"Expected {self.config.party_count} client vectors, got {len(weighted_vectors)}"
            )
        vectors = [np.asarray(vector, dtype=np.float64).reshape(-1) for vector in weighted_vectors]
        if not vectors or vectors[0].size == 0:
            raise ValueError("Client update vectors must be non-empty")
        length = int(vectors[0].size)
        if any(vector.size != length for vector in vectors):
            raise ValueError("All client update vectors must have identical lengths")
        if any(not np.isfinite(vector).all() for vector in vectors):
            raise ValueError("Client update vectors must contain only finite values")

        encrypt_seconds = [0.0] * self.config.party_count
        upload_bytes = [0] * self.config.party_count
        upload_serialize_seconds = [0.0] * self.config.party_count
        partial_seconds = [0.0] * self.config.party_count
        partial_bytes = [0] * self.config.party_count
        partial_serialize_seconds = [0.0] * self.config.party_count
        server_add_seconds = 0.0
        aggregate_bytes = 0
        aggregate_serialize_seconds = 0.0
        fusion_seconds = 0.0
        recovered: list[float] = []
        chunk_count = int(math.ceil(length / self.slots))

        with tempfile.TemporaryDirectory(prefix="openfhe-threshold-wire-") as temporary:
            wire_dir = Path(temporary)
            for chunk_index, offset in enumerate(range(0, length, self.slots)):
                chunk_length = min(self.slots, length - offset)
                encrypted_clients: list[Any] = []
                for party_index, vector in enumerate(vectors):
                    plaintext = self.context.MakeCKKSPackedPlaintext(
                        vector[offset : offset + chunk_length].tolist()
                    )
                    started = time.perf_counter()
                    ciphertext = self.context.Encrypt(self.joint_public_key, plaintext)
                    encrypt_seconds[party_index] += time.perf_counter() - started
                    size, elapsed = self._serialized_size(
                        ciphertext,
                        wire_dir,
                        f"client_{party_index}_chunk_{chunk_index}.bin",
                    )
                    upload_bytes[party_index] += size
                    upload_serialize_seconds[party_index] += elapsed
                    encrypted_clients.append(ciphertext)

                started = time.perf_counter()
                aggregate_ciphertext = encrypted_clients[0]
                for ciphertext in encrypted_clients[1:]:
                    aggregate_ciphertext = self.context.EvalAdd(
                        aggregate_ciphertext, ciphertext
                    )
                server_add_seconds += time.perf_counter() - started
                size, elapsed = self._serialized_size(
                    aggregate_ciphertext,
                    wire_dir,
                    f"aggregate_chunk_{chunk_index}.bin",
                )
                aggregate_bytes += size
                aggregate_serialize_seconds += elapsed

                partials: list[Any] = []
                for party_index, secret_share in enumerate(self._secret_key_shares):
                    started = time.perf_counter()
                    if party_index == 0:
                        produced = self.context.MultipartyDecryptLead(
                            [aggregate_ciphertext], secret_share
                        )
                    else:
                        produced = self.context.MultipartyDecryptMain(
                            [aggregate_ciphertext], secret_share
                        )
                    partial_seconds[party_index] += time.perf_counter() - started
                    if len(produced) != 1:
                        raise RuntimeError("OpenFHE returned an invalid partial-decryption vector")
                    partial = produced[0]
                    size, elapsed = self._serialized_size(
                        partial,
                        wire_dir,
                        f"partial_{party_index}_chunk_{chunk_index}.bin",
                    )
                    partial_bytes[party_index] += size
                    partial_serialize_seconds[party_index] += elapsed
                    partials.append(partial)

                started = time.perf_counter()
                plaintext_result = self.context.MultipartyDecryptFusion(partials)
                fusion_seconds += time.perf_counter() - started
                plaintext_result.SetLength(chunk_length)
                values = list(plaintext_result.GetRealPackedValue())
                if len(values) < chunk_length:
                    raise RuntimeError("OpenFHE threshold decryption returned too few values")
                recovered.extend(float(value) for value in values[:chunk_length])

        result = np.asarray(recovered, dtype=np.float64)
        if result.size != length or not np.isfinite(result).all():
            raise RuntimeError("Threshold CKKS aggregate is incomplete or non-finite")
        logical_network_bytes = (
            sum(upload_bytes)
            + aggregate_bytes * self.config.party_count
            + sum(partial_bytes)
        )
        return result, {
            "backend": "openfhe_threshold_ckks",
            "party_count": self.config.party_count,
            "ciphertext_chunks": chunk_count,
            "vector_length": length,
            "keygen_seconds": self.keygen_seconds,
            "encrypt_seconds_by_party": encrypt_seconds,
            "upload_bytes_by_party": upload_bytes,
            "upload_serialize_seconds_by_party": upload_serialize_seconds,
            "server_add_seconds": server_add_seconds,
            "aggregate_ciphertext_bytes": aggregate_bytes,
            "aggregate_serialize_seconds": aggregate_serialize_seconds,
            "aggregate_broadcast_bytes": aggregate_bytes * self.config.party_count,
            "partial_decrypt_seconds_by_party": partial_seconds,
            "partial_decryption_bytes_by_party": partial_bytes,
            "partial_serialize_seconds_by_party": partial_serialize_seconds,
            "fusion_seconds": fusion_seconds,
            "logical_network_bytes": logical_network_bytes,
            "network_scope": (
                "serialized logical payload: client uploads + aggregate broadcast to every "
                "party + partial-decryption uploads; transport framing/TLS excluded"
            ),
            "security": self.security_metadata(),
        }

