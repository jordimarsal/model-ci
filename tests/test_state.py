"""Tests for model_ci.state."""

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from model_ci.state import StateError, load, save, migrate


class TestLoad(unittest.TestCase):
    def test_missing_file_returns_skeleton(self):
        with TemporaryDirectory() as td:
            path = Path(td) / "state.json"
            self.assertFalse(path.exists())
            result = load(path)
            self.assertEqual(result["version"], 1)
            self.assertEqual(result["engines"], {})
            self.assertEqual(result["models"], {})

    def test_valid_round_trip(self):
        with TemporaryDirectory() as td:
            path = Path(td) / "state.json"
            original = {
                "version": 1,
                "engines": {"llamacpp": {"sha": "abc"}},
                "models": {},
            }
            save(path, original)
            loaded = load(path)
            self.assertEqual(loaded, original)

    def test_unknown_version_raises_state_error(self):
        with TemporaryDirectory() as td:
            path = Path(td) / "state.json"
            path.write_text(json.dumps({"version": 2}))
            with self.assertRaises(StateError) as ctx:
                load(path)
            self.assertIn("unsupported version", str(ctx.exception))

    def test_missing_version_key_raises_state_error(self):
        with TemporaryDirectory() as td:
            path = Path(td) / "state.json"
            path.write_text(json.dumps({"engines": {}}))
            with self.assertRaises(StateError) as ctx:
                load(path)
            self.assertIn('missing "version"', str(ctx.exception))

    def test_invalid_json_raises_state_error(self):
        with TemporaryDirectory() as td:
            path = Path(td) / "state.json"
            path.write_text("{not valid json")
            with self.assertRaises(StateError) as ctx:
                load(path)
            self.assertIn("invalid JSON", str(ctx.exception))


class TestSave(unittest.TestCase):
    def test_save_validates_before_writing(self):
        """Saving a state with an unsupported version must raise before any file is created."""
        with TemporaryDirectory() as td:
            path = Path(td) / "state.json"
            bad = {"version": 99}
            with self.assertRaises(StateError):
                save(path, bad)
            # No file should have been written.
            self.assertFalse(path.exists())

    def test_atomic_write_no_tmp_file_left_behind(self):
        """After a successful save there must be no .model-ci-state-*.tmp files."""
        with TemporaryDirectory() as td:
            path = Path(td) / "state.json"
            save(path, {"version": 1, "engines": {}, "models": {}})
            tmps = list(Path(td).glob(".model-ci-state-*"))
            self.assertEqual(tmps, [])

    def test_parent_dirs_created(self):
        with TemporaryDirectory() as td:
            path = Path(td) / "sub" / "deep" / "state.json"
            save(path, {"version": 1, "engines": {}, "models": {}})
            self.assertTrue(path.exists())


class TestMigrate(unittest.TestCase):
    def test_v1_migrate_is_load(self):
        with TemporaryDirectory() as td:
            path = Path(td) / "state.json"
            original = {"version": 1, "engines": {}, "models": {}}
            save(path, original)
            result = migrate(path)
            self.assertEqual(result, original)


if __name__ == "__main__":
    unittest.main()
