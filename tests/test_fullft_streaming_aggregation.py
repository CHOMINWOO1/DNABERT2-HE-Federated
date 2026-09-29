from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
import run_fedhe_main_experiment as main  # noqa: E402


def canonical_fp64_vector(state: dict[str, torch.Tensor]) -> np.ndarray:
    return np.concatenate(
        [
            state[name].detach().cpu().double().contiguous().numpy().reshape(-1)
            for name in sorted(state)
        ]
    )


class StateChunkRoundTripTests(unittest.TestCase):
    def test_tensor_boundaries_and_final_partial_chunk_round_trip(self) -> None:
        # Insertion order is intentionally non-canonical.  With chunk_size=4,
        # canonical lengths 3|4|2 exercise two tensor-boundary crossings and a
        # final one-element partial chunk.
        state = {
            "z_last": torch.tensor([[8.0, 9.0]], dtype=torch.float32),
            "a_first": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
            "m_middle": torch.tensor([[4.0, 5.0], [6.0, 7.0]], dtype=torch.float32),
        }
        manifest = main.state_manifest_without_flattening(state)

        self.assertEqual([entry["name"] for entry in manifest], ["a_first", "m_middle", "z_last"])
        self.assertEqual([entry["offset"] for entry in manifest], [0, 3, 7])
        self.assertEqual([entry["numel"] for entry in manifest], [3, 4, 2])

        reader = main.StateChunkReader(state, manifest, chunk_size=4)
        chunks = list(reader)
        self.assertEqual([len(chunk) for chunk in chunks], [4, 4, 1])
        np.testing.assert_array_equal(
            np.concatenate(chunks),
            np.arange(1.0, 10.0, dtype=np.float64),
        )
        with self.assertRaises(StopIteration):
            next(reader)

        writer = main.StateChunkWriter(state, manifest)
        for chunk in chunks:
            writer.write(chunk)
        recovered = writer.finish()
        self.assertEqual(list(recovered), ["a_first", "m_middle", "z_last"])
        for name in recovered:
            self.assertEqual(recovered[name].dtype, torch.float32)
            torch.testing.assert_close(recovered[name], state[name], rtol=0.0, atol=0.0)


class CanonicalStateAuditTests(unittest.TestCase):
    def test_manifest_order_and_streaming_sha_determinism_and_change_detection(self) -> None:
        first = {
            "z": torch.tensor([5.0, 6.0], dtype=torch.float32),
            "a": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        }
        same_values_different_insertion_order = {
            "a": first["a"].clone(),
            "z": first["z"].clone(),
        }
        first_manifest = main.state_manifest_without_flattening(first)
        second_manifest = main.state_manifest_without_flattening(same_values_different_insertion_order)

        self.assertEqual(first_manifest, second_manifest)
        self.assertEqual([entry["name"] for entry in first_manifest], ["a", "z"])
        self.assertEqual(main.json_hash(first_manifest), main.json_hash(second_manifest))
        first_hash = main.state_fp32_streaming_sha256(first, first_manifest, chunk_elements=3)
        second_hash = main.state_fp32_streaming_sha256(
            same_values_different_insertion_order,
            second_manifest,
            chunk_elements=2,
        )
        self.assertEqual(first_hash, second_hash)

        changed = {name: tensor.clone() for name, tensor in first.items()}
        changed["a"][0, 0] += 1.0
        self.assertNotEqual(
            first_hash,
            main.state_fp32_streaming_sha256(changed, first_manifest, chunk_elements=1),
        )

        wrong_shape = {name: tensor.clone() for name, tensor in first.items()}
        wrong_shape["z"] = wrong_shape["z"].reshape(1, 2)
        with self.assertRaises(RuntimeError):
            main.validate_state_manifest(wrong_shape, first_manifest)


class CKKSStreamingAggregationTests(unittest.TestCase):
    def test_three_client_toy_state_matches_fp64_weighted_plaintext(self) -> None:
        config = SimpleNamespace(
            ckks_poly_modulus_degree=8192,
            ckks_coeff_mod_bits=[60, 40, 60],
            ckks_scale_bits=40,
        )
        base_a = torch.linspace(-0.2, 0.2, 3001, dtype=torch.float32).reshape(3001, 1)
        base_z = torch.linspace(0.15, -0.15, 1499, dtype=torch.float32)
        deltas = [
            {"z": base_z + 0.01, "a": base_a - 0.02},
            {"a": base_a * -0.5, "z": base_z * 0.25},
            {"z": base_z * -0.75 - 0.03, "a": base_a * 0.4 + 0.01},
        ]
        weights = [0.2, 0.3, 0.5]

        aggregator = main.CKKSStreamingStateAggregator(config)
        actual_state, statistics, fidelity, manifest = aggregator.aggregate_states(
            deltas, weights
        )

        expected_state = {
            name: sum(
                (delta[name].double() * weight for delta, weight in zip(deltas, weights, strict=True)),
                torch.zeros_like(deltas[0][name], dtype=torch.float64),
            )
            for name in deltas[0]
        }
        actual = canonical_fp64_vector(actual_state)
        expected = canonical_fp64_vector(expected_state)
        max_abs = float(np.max(np.abs(actual - expected)))
        cosine = float(
            np.dot(actual, expected)
            / (max(float(np.linalg.norm(actual)), 1e-12) * max(float(np.linalg.norm(expected)), 1e-12))
        )

        total_numel = sum(int(entry["numel"]) for entry in manifest)
        expected_chunks = math.ceil(total_numel / aggregator.backend.slots)
        self.assertEqual(total_numel, 4500)
        self.assertEqual(expected_chunks, 2)
        self.assertEqual(statistics["ciphertext_chunks"], expected_chunks)
        self.assertLess(max_abs, 1e-6)
        self.assertGreater(cosine, 0.999999)
        self.assertLess(fidelity["max_abs"], 1e-6)
        self.assertGreater(fidelity["cosine"], 0.999999)
        self.assertTrue(statistics["streaming"])
        self.assertEqual(statistics["max_client_ciphertexts_in_memory"], 1)
        self.assertFalse(statistics["full_client_ciphertext_corpus_materialized"])
        self.assertFalse(statistics["full_plaintext_vector_materialized"])
        self.assertEqual(len(statistics["upload_bytes_by_client"]), 3)
        self.assertTrue(all(size > 0 for size in statistics["upload_bytes_by_client"]))
        self.assertEqual([entry["name"] for entry in manifest], ["a", "z"])


if __name__ == "__main__":
    unittest.main()
