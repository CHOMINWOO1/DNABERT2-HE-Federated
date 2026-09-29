from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_fedhe_earlystop_v3 import advance_patience


class EarlyStoppingRuleTests(unittest.TestCase):
    def test_first_finite_score_sets_anchor(self) -> None:
        anchor, stale, meaningful = advance_patience(0.70, -math.inf, 8, 0.001)
        self.assertEqual(anchor, 0.70)
        self.assertEqual(stale, 0)
        self.assertTrue(meaningful)

    def test_subthreshold_gain_consumes_patience(self) -> None:
        anchor, stale, meaningful = advance_patience(0.7005, 0.70, 2, 0.001)
        self.assertEqual(anchor, 0.70)
        self.assertEqual(stale, 3)
        self.assertFalse(meaningful)

    def test_accumulated_gain_resets_patience(self) -> None:
        anchor, stale, meaningful = advance_patience(0.7011, 0.70, 4, 0.001)
        self.assertEqual(anchor, 0.7011)
        self.assertEqual(stale, 0)
        self.assertTrue(meaningful)

    def test_nonfinite_score_fails(self) -> None:
        with self.assertRaises(ValueError):
            advance_patience(float("nan"), 0.70, 0, 0.001)


if __name__ == "__main__":
    unittest.main()
