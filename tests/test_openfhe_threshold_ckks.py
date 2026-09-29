from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from openfhe_threshold_ckks import (  # noqa: E402
    OpenFHEThresholdCKKSAggregator,
    ThresholdCKKSConfig,
)


class _Parameters:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    def _set(self, name: str, value: object) -> None:
        self.values[name] = value

    def SetMultiplicativeDepth(self, value: int) -> None:
        self._set("depth", value)

    def SetScalingModSize(self, value: int) -> None:
        self._set("scale", value)

    def SetFirstModSize(self, value: int) -> None:
        self._set("first", value)

    def SetBatchSize(self, value: int) -> None:
        self._set("batch", value)

    def SetRingDim(self, value: int) -> None:
        self._set("ring", value)

    def SetSecurityLevel(self, value: object) -> None:
        self._set("security", value)


class _Key:
    def __init__(self, tag: str) -> None:
        self.tag = tag

    def GetKeyTag(self) -> str:
        return self.tag


class _KeyPair:
    def __init__(self, party: int) -> None:
        self.publicKey = _Key(f"joint-public-{party}")
        self.secretKey = _Key(f"secret-share-{party}")

    def good(self) -> bool:
        return True


class _Plaintext:
    def __init__(self, values: list[float]) -> None:
        self.values = np.asarray(values, dtype=np.float64)
        self.length = len(values)

    def SetLength(self, length: int) -> None:
        self.length = length

    def GetRealPackedValue(self) -> list[float]:
        return self.values[: self.length].tolist()


class _Ciphertext:
    def __init__(self, values: np.ndarray) -> None:
        self.values = np.asarray(values, dtype=np.float64)


class _Partial:
    def __init__(self, values: np.ndarray) -> None:
        self.values = np.asarray(values, dtype=np.float64)


class _Context:
    def __init__(self, parameters: _Parameters, party_count: int) -> None:
        self.parameters = parameters
        self.party_count = party_count
        self.enabled: list[object] = []
        self.generated_parties = 0

    def Enable(self, feature: object) -> None:
        self.enabled.append(feature)

    def KeyGen(self) -> _KeyPair:
        self.generated_parties = 1
        return _KeyPair(1)

    def MultipartyKeyGen(self, _prior_public_key: _Key) -> _KeyPair:
        self.generated_parties += 1
        return _KeyPair(self.generated_parties)

    def GetRingDimension(self) -> int:
        return int(self.parameters.values["ring"])

    def MakeCKKSPackedPlaintext(self, values: list[float]) -> _Plaintext:
        return _Plaintext(values)

    def Encrypt(self, _public_key: _Key, plaintext: _Plaintext) -> _Ciphertext:
        return _Ciphertext(plaintext.values)

    def EvalAdd(self, left: _Ciphertext, right: _Ciphertext) -> _Ciphertext:
        return _Ciphertext(left.values + right.values)

    def MultipartyDecryptLead(
        self, ciphertexts: list[_Ciphertext], _secret_key: _Key
    ) -> list[_Partial]:
        return [_Partial(ciphertexts[0].values / self.party_count)]

    def MultipartyDecryptMain(
        self, ciphertexts: list[_Ciphertext], _secret_key: _Key
    ) -> list[_Partial]:
        return [_Partial(ciphertexts[0].values / self.party_count)]

    def MultipartyDecryptFusion(self, partials: list[_Partial]) -> _Plaintext:
        return _Plaintext(np.sum([partial.values for partial in partials], axis=0).tolist())


class _FakeOpenFHE:
    PKE = "PKE"
    KEYSWITCH = "KEYSWITCH"
    LEVELEDSHE = "LEVELEDSHE"
    ADVANCEDSHE = "ADVANCEDSHE"
    MULTIPARTY = "MULTIPARTY"
    HEStd_128_classic = "HEStd_128_classic"
    BINARY = "BINARY"

    def __init__(self, party_count: int) -> None:
        self.party_count = party_count
        self.context: _Context | None = None

    @staticmethod
    def CCParamsCKKSRNS() -> _Parameters:
        return _Parameters()

    def GenCryptoContext(self, parameters: _Parameters) -> _Context:
        self.context = _Context(parameters, self.party_count)
        return self.context

    @staticmethod
    def SerializeToFile(path: str, value: object, _mode: object) -> bool:
        payload = np.asarray(getattr(value, "values"), dtype=np.float64).tobytes()
        Path(path).write_bytes(b"fake-openfhe:" + payload)
        return True


class ThresholdCKKSTests(unittest.TestCase):
    def make_aggregator(self, *, slots: int = 4) -> OpenFHEThresholdCKKSAggregator:
        config = ThresholdCKKSConfig(
            party_count=3,
            ring_dimension=16,
            batch_size=slots,
            multiplicative_depth=1,
            scaling_mod_size=40,
            first_mod_size=60,
        )
        return OpenFHEThresholdCKKSAggregator(
            config, openfhe_module=_FakeOpenFHE(config.party_count)
        )

    def test_three_party_weighted_sum_and_chunk_accounting(self) -> None:
        aggregator = self.make_aggregator(slots=4)
        vectors = [
            np.asarray([0.1, 0.2, 0.3, 0.4, 0.5]),
            np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]),
            np.asarray([-0.5, -0.4, -0.3, -0.2, -0.1]),
        ]

        recovered, audit = aggregator.aggregate(vectors)

        np.testing.assert_allclose(recovered, np.sum(vectors, axis=0), atol=1e-12)
        self.assertEqual(audit["party_count"], 3)
        self.assertEqual(audit["ciphertext_chunks"], 2)
        self.assertEqual(len(audit["upload_bytes_by_party"]), 3)
        self.assertTrue(all(value > 0 for value in audit["upload_bytes_by_party"]))
        self.assertTrue(
            all(value > 0 for value in audit["partial_decryption_bytes_by_party"])
        )
        self.assertEqual(
            audit["logical_network_bytes"],
            sum(audit["upload_bytes_by_party"])
            + audit["aggregate_broadcast_bytes"]
            + sum(audit["partial_decryption_bytes_by_party"]),
        )
        self.assertFalse(audit["security"]["complete_secret_key_created"])
        self.assertFalse(audit["security"]["secret_key_shares_persisted"])
        self.assertEqual(audit["security"]["role_isolation"], "in_process_simulation")

    def test_rejects_invalid_client_vectors(self) -> None:
        aggregator = self.make_aggregator()
        with self.assertRaisesRegex(ValueError, "Expected 3"):
            aggregator.aggregate([np.ones(2), np.ones(2)])
        with self.assertRaisesRegex(ValueError, "identical lengths"):
            aggregator.aggregate([np.ones(2), np.ones(3), np.ones(2)])
        with self.assertRaisesRegex(ValueError, "finite"):
            aggregator.aggregate([np.ones(2), np.asarray([1.0, np.nan]), np.ones(2)])

    def test_configuration_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two"):
            ThresholdCKKSConfig(party_count=1).validate()
        with self.assertRaisesRegex(ValueError, "power of two"):
            ThresholdCKKSConfig(ring_dimension=1000).validate()
        with self.assertRaisesRegex(ValueError, "ring_dimension/2"):
            ThresholdCKKSConfig(ring_dimension=16, batch_size=9).validate()


if __name__ == "__main__":
    unittest.main()
