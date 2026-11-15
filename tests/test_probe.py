# tests/test_probe.py
"""Tests for src/model_ci/probe.py."""

import unittest
from statistics import median as _median
from model_ci.probe import drift, stabilized_tps, parse_compute_apps, quiet_machine, gpu_rule


class TestDrift(unittest.TestCase):
    def test_drift_up(self):
        self.assertTrue(drift(10.0, 12.0))          # +20% > 10%
        self.assertFalse(drift(10.0, 11.0))         # +10% == tolerance → no drift
        self.assertTrue(drift(10.0, 11.5))          # +15% > 10%
        # La millora TAMBÉ és derivació (spec T3, bidireccional): una flota més ràpida
        # mereix bench complet perquè les baselines millorin.

    def test_drift_down(self):
        self.assertTrue(drift(10.0, 8.0))           # -20% > 10%
        self.assertFalse(drift(10.0, 9.5))          # -5% < 10%

    def test_drift_no_baseline(self):
        self.assertFalse(drift(0.0, 10.0))
        self.assertFalse(drift(-1.0, 10.0))

    def test_drift_within_tolerance(self):
        self.assertFalse(drift(10.0, 10.5))         # exactly 5%
        self.assertFalse(drift(10.0, 9.5))


class TestStabilizedTps(unittest.TestCase):
    def setUp(self):
        self.calls: list[float] = []

    def _make_complete(self) -> callable[[int], float]:
        def complete(tokens: int) -> float:
            self.calls.append(tokens)
            return 42.0
        return complete

    def test_median_ignores_outlier(self):
        """Three generations with one outlier — median should be the middle value."""
        values = [30.0, 40.0, 50.0]
        idx = 0
        def complete(tokens: int) -> float:
            nonlocal idx
            val = values[idx]; idx += 1; return val
        result = stabilized_tps(complete, generations=3)
        self.assertEqual(result, 40.0)

    def test_single_generation(self):
        result = stabilized_tps(lambda t: 7.0, generations=1)
        self.assertEqual(result, 7.0)

    def test_tokens_passed_through(self):
        tokens_seen: list[int] = []
        def complete(tokens: int) -> float:
            tokens_seen.append(tokens); return 1.0
        stabilized_tps(complete, generations=2, tokens=256)
        self.assertEqual(tokens_seen, [256, 256])


class TestParseComputeApps(unittest.TestCase):
    def test_basic_lines(self):
        text = "1234, 392 MiB\n5678, 1024 MiB"
        out = parse_compute_apps(text)
        self.assertEqual(out, [(1234, 392), (5678, 1024)])

    def test_blank_lines_skipped(self):
        text = "\n1234, 392 MiB\n\n"
        out = parse_compute_apps(text)
        self.assertEqual(out, [(1234, 392)])

    def test_empty_string(self):
        self.assertEqual(parse_compute_apps(""), [])


class TestQuietMachine(unittest.TestCase):
    def test_load_ok_no_apps(self):
        ok, reason = quiet_machine(1.0, 4, [], allow={})
        self.assertTrue(ok)
        self.assertEqual(reason, "quiet")

    def test_load_exceeds_threshold(self):
        ok, reason = quiet_machine(3.0, 4, [], allow={}, max_load_ratio=0.5)
        self.assertFalse(ok)
        self.assertIn("load", reason.lower())

    def test_memory_below_threshold(self):
        apps = [(100, 100)]
        ok, _ = quiet_machine(0.1, 4, apps, allow=set(), min_free_mib=2000)
        self.assertTrue(ok)

    def test_memory_above_threshold_not_allowed(self):
        apps = [(100, 3000)]
        ok, reason = quiet_machine(0.1, 4, apps, allow=set(), min_free_mib=2000)
        self.assertFalse(ok)
        self.assertIn("PID 100", reason)

    def test_memory_above_threshold_but_allowed(self):
        apps = [(100, 3000)]
        ok, _ = quiet_machine(0.1, 4, apps, allow={100}, min_free_mib=2000)
        self.assertTrue(ok)


class TestGpuRule(unittest.TestCase):
    def test_both_alive_conflict(self):
        ok, reason = gpu_rule(True, True)
        self.assertFalse(ok)
        self.assertIn("both", reason.lower())

    def test_llama_only_ok(self):
        ok, _ = gpu_rule(True, False)
        self.assertTrue(ok)

    def test_strata_only_ok(self):
        ok, _ = gpu_rule(False, True)
        self.assertTrue(ok)

    def test_neither_alive_ok(self):
        ok, _ = gpu_rule(False, False)
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
