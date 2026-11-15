# tests/test_config.py
"""Offline unit tests for model_ci.config."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from model_ci.config import ConfigError, load_config


class TestLoadConfig(unittest.TestCase):
    def _write_toml(self, content: str) -> Path:
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        p = Path(d.name) / "model-ci.toml"
        p.write_text(content, encoding="utf-8")
        return p

    def test_valid_minimal(self) -> None:
        toml_content = """\
[engines.llamacpp]
repo = "https://github.com/ggml-org/llama.cpp"
ref = "refs/heads/main"
"""
        p = self._write_toml(toml_content)
        cfg = load_config(p)
        self.assertIn("engines", cfg)
        self.assertEqual(cfg["engines"]["llamacpp"]["repo"], "https://github.com/ggml-org/llama.cpp")
        self.assertEqual(cfg["engines"]["llamacpp"]["ref"], "refs/heads/main")
        self.assertEqual(cfg["models"], [])

    def test_with_models(self) -> None:
        toml_content = """\
[engines.strata]
repo = "https://github.com/Niko1221/Strata"
ref = "refs/heads/main"

[[models]]
repo = "unsloth/Llama-3.2-1B-Instruct-GGUF"
"""
        p = self._write_toml(toml_content)
        cfg = load_config(p)
        self.assertEqual(len(cfg["models"]), 1)
        self.assertEqual(cfg["models"][0]["repo"], "unsloth/Llama-3.2-1B-Instruct-GGUF")

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(ConfigError):
            load_config("/nonexistent/model-ci.toml")

    def test_invalid_toml_raises(self) -> None:
        p = self._write_toml("this is not valid toml {{{")
        with self.assertRaises(ConfigError):
            load_config(p)

    def test_no_engines_raises(self) -> None:
        p = self._write_toml("[other]\nkey = 'val'\n")
        with self.assertRaises(ConfigError):
            load_config(p)

    def test_engine_without_repo_raises(self) -> None:
        p = self._write_toml('[engines.foo]\nref = "main"\n')
        with self.assertRaises(ConfigError):
            load_config(p)


if __name__ == "__main__":
    unittest.main()
