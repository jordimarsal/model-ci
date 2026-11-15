# tests/test_cli_seeding.py
"""Offline unit tests for model-ci init/accept/check seeding behavior."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model_ci.cli import main


class TestInitAcceptCheck(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.tmp = Path(self._tmpdir.name)
        self.state_path = self.tmp / ".model-ci-state.json"

    def test_init_creates_skeleton(self) -> None:
        rc = main(["init", "--state", str(self.state_path)])
        self.assertEqual(rc, 0)
        data = json.loads(self.state_path.read_text())
        self.assertEqual(data["version"], 1)
        self.assertEqual(data["engines"], {})
        self.assertEqual(data["models"], {})

    def test_init_existing_file_exits_65(self) -> None:
        # First create the file
        self.state_path.write_text(json.dumps({"version": 1, "engines": {}, "models": {}}))
        with self.assertRaises(SystemExit) as ctx:
            main(["init", "--state", str(self.state_path)])
        self.assertEqual(ctx.exception.code, 65)

    def test_accept_writes_baseline(self) -> None:
        # Init first
        main(["init", "--state", str(self.state_path)])
        rc = main([
            "accept",
            "--engine", "llamacpp",
            "--tps", "252.4",
            "--ms", "15",
            "--model", "llama-3.2-1b-instruct-q4_k_m.gguf",
            "--build", "v999",
            "--sha", "abc123",
            "--state", str(self.state_path),
        ])
        self.assertEqual(rc, 0)
        data = json.loads(self.state_path.read_text())
        baseline = data["engines"]["llamacpp"]["baseline"]
        self.assertAlmostEqual(baseline["tps"], 252.4)
        self.assertEqual(baseline["ms"], 15)
        self.assertEqual(baseline["model"], "llama-3.2-1b-instruct-q4_k_m.gguf")
        self.assertEqual(baseline["build"], "v999")
        self.assertEqual(baseline["sha"], "abc123")
        self.assertIn("ts", baseline)

    def test_check_without_baseline_exits_65_with_hint(self) -> None:
        # Create a state with an engine that has a repo but no baseline
        state_data: dict[str, Any] = {
            "version": 1,
            "engines": {
                "llamacpp": {"repo": "https://github.com/ggml-org/llama.cpp", "ref": "refs/heads/main"}
            },
            "models": {},
        }
        self.state_path.write_text(json.dumps(state_data))

        config_content = """\
[engines.llamacpp]
repo = "https://github.com/ggml-org/llama.cpp"
ref = "refs/heads/main"
"""
        cfg_path = self.tmp / "model-ci.toml"
        cfg_path.write_text(config_content, encoding="utf-8")

        with patch("model_ci.cli._load_config") as mock_load_cfg:
            from model_ci.config import load_config
            mock_load_cfg.return_value = load_config(cfg_path)

            rc = main(["check", "--state", str(self.state_path), "--config", str(cfg_path)])
        self.assertEqual(rc, 65)


if __name__ == "__main__":
    unittest.main()
