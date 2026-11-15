# tests/test_bench.py
"""Tests for src/model_ci/bench."""

import unittest
from unittest.mock import MagicMock

from model_ci.bench import plan, run


class TestPlan(unittest.TestCase):
    def test_empty_engines(self):
        self.assertEqual(plan({"engines": {}}), [])

    def test_single_engine(self):
        state = {
            "engines": {
                "llamacpp": {"baseline": {"model": "qwen2-7b"}}
            }
        }
        result = plan(state)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["engine"], "llamacpp")
        self.assertEqual(result[0]["model"], "qwen2-7b")
        self.assertEqual(result[0]["reason"], "drift")

    def test_order_llamacpp_first(self):
        """llamacpp must always come before strata (and any others)."""
        state = {
            "engines": {
                "strata":   {"baseline": {"model": "m"}},
                "llamacpp": {"baseline": {"model": "n"}},
                "vulkan":   {"baseline": {"model": "o"}},
            }
        }
        names = [s["engine"] for s in plan(state)]
        self.assertEqual(names, ["llamacpp", "strata", "vulkan"])

    def test_no_llamacpp_sorted_alphabetically(self):
        state = {
            "engines": {
                "z-model": {"baseline": {}},
                "a-model": {"baseline": {}},
            }
        }
        names = [s["engine"] for s in plan(state)]
        self.assertEqual(names, ["a-model", "z-model"])


class TestRun(unittest.TestCase):
    def setUp(self):
        self.runner = MagicMock(return_value=0)

    def test_all_pass(self):
        steps = [{"engine": "e1"}, {"engine": "e2"}]
        result = run(steps, self.runner)
        self.assertEqual(result["ran"], ["e1", "e2"])
        self.assertEqual(result["failed"], [])
        self.assertEqual(result["rc"], 0)
        # runner called once per step
        self.assertEqual(self.runner.call_count, 2)

    def test_first_fails_stops_early(self):
        self.runner.side_effect = [42, 0]
        steps = [{"engine": "fail"}, {"engine": "ok"}]
        result = run(steps, self.runner)
        self.assertEqual(result["ran"], [])
        self.assertEqual(result["failed"], ["fail"])
        self.assertEqual(result["rc"], 42)
        self.assertEqual(self.runner.call_count, 1)

    def test_second_fails(self):
        self.runner.side_effect = [0, 3]
        steps = [{"engine": "ok"}, {"engine": "bad"}]
        result = run(steps, self.runner)
        self.assertEqual(result["ran"], ["ok"])
        self.assertEqual(result["failed"], ["bad"])
        self.assertEqual(result["rc"], 3)
        self.assertEqual(self.runner.call_count, 2)


if __name__ == "__main__":
    unittest.main()
