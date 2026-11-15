# tests/test_propose.py
"""Tests for src/model_ci/propose."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from model_ci.propose import evidence_pack, write


class TestEvidencePack(unittest.TestCase):
    def test_contains_all_sections(self):
        pack = evidence_pack(
            baseline_tps=100.0,
            current_tps=95.3,
            stabilized_tps=97.1,
            engine="llamacpp",
            build="abc123",
            model="qwen2-7b",
            receipts={"load": "0.5", "flags": "--threads 4"},
        )
        self.assertIn("# Drift Evidence — llamacpp", pack)
        self.assertIn("Baseline bench", pack)
        self.assertIn("Probe", pack)
        self.assertIn("Stabilized ≥512tok", pack)
        self.assertIn("`llamacpp`", pack)
        self.assertIn("`abc123`", pack)
        self.assertIn("`qwen2-7b`", pack)
        self.assertIn("Full re-bench + routing review at the roles apply gate.", pack)
        # date is today (YYYY-MM-DD)
        self.assertIn("Date: ", pack)

    def test_no_invented_numbers(self):
        """Only the numbers passed in appear."""
        pack = evidence_pack(
            baseline_tps=42.0,
            current_tps=41.0,
            stabilized_tps=41.5,
            engine="strata",
            build="xyz",
            model="m",
            receipts={},
        )
        self.assertNotIn("invented", pack.lower())


class TestWrite(unittest.TestCase):
    def setUp(self):
        self.tmpdir = TemporaryDirectory()
        self.out_dir = Path(self.tmpdir.name) / "packs"

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_creates_file_and_returns_path(self):
        pack = evidence_pack(
            10.0, 9.0, 9.5, "llamacpp", "build", "model", {}
        )
        path = write(pack, self.out_dir, "llamacpp")
        self.assertTrue(path.exists())
        self.assertEqual(path.read_text(encoding="utf-8"), pack)

    def test_filename_format(self):
        import datetime
        date_str = datetime.date.today().strftime("%Y%m%d")
        expected_name = f"{date_str}-llamacpp-drift.md"
        pack = evidence_pack(10.0, 9.0, 9.5, "llamacpp", "b", "m", {})
        path = write(pack, self.out_dir, "llamacpp")
        self.assertEqual(path.name, expected_name)

    def test_creates_parent_dirs(self):
        deep = self.out_dir / "a" / "b"
        pack = evidence_pack(10.0, 9.0, 9.5, "e", "b", "m", {})
        path = write(pack, deep, "e")
        self.assertTrue(path.parent.exists())


if __name__ == "__main__":
    unittest.main()
