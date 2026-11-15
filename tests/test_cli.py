# tests/test_cli.py
"""Tests for model-ci CLI — all external contact monkey-patched."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from model_ci.cli import main


class TestCli(unittest.TestCase):
    """End-to-end cli.main with everything stubbed via module-level hooks."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.state_path = os.path.join(self.tmpdir.name, "state.json")

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    # -- helpers to build state files ----------------------------------------

    @staticmethod
    def _write_state(path: str, engines: dict, models: dict) -> None:
        from model_ci.state import save
        save(path, {"version": 1, "engines": engines, "models": models})

    # -- monkey-patch the injectable hooks on the cli module -----------------

    def _patch_hooks(
        self,
        git_ls_remote_fn=None,
        hf_tree_fn=None,
        ggufbench_speed_fn=None,
        stabilized_fn=None,
        alive_llama_fn=None,
        alive_strata_fn=None,
        compute_apps_fn=None,
        load_fn=None,
    ) -> None:
        import model_ci.cli as cli_mod
        if git_ls_remote_fn is not None:
            cli_mod._git_ls_remote = git_ls_remote_fn
        if hf_tree_fn is not None:
            cli_mod._hf_tree = hf_tree_fn
        if ggufbench_speed_fn is not None:
            cli_mod._ggufbench_speed = ggufbench_speed_fn
        if stabilized_fn is not None:
            cli_mod._stabilized = stabilized_fn
        if alive_llama_fn is not None:
            cli_mod._alive_llama = alive_llama_fn
        if alive_strata_fn is not None:
            cli_mod._alive_strata = alive_strata_fn
        if compute_apps_fn is not None:
            cli_mod._compute_apps = compute_apps_fn
        if load_fn is not None:
            cli_mod._load = load_fn

    # -- test 1: no change → exit 0 ----------------------------------------

    def test_no_change_exit_0(self) -> None:
        self._write_state(
            self.state_path,
            engines={
                "llamacpp": {
                    "repo": "https://github.com/ggerganov/llama.cpp",
                    "baseline": {"sha": "abc123"},
                }
            },
            models={},
        )

        def fake_ls_remote(url):
            return f"abc123\trefs/heads/main\n"

        self._patch_hooks(git_ls_remote_fn=fake_ls_remote)

        rc = main(["check", "--state", self.state_path])
        self.assertEqual(rc, 0)

    # -- test 2: engine changed, probe within tolerance → exit 1 -----------

    def test_probe_within_tolerance_exit_1(self) -> None:
        self._write_state(
            self.state_path,
            engines={
                "llamacpp": {
                    "repo": "https://github.com/ggerganov/llama.cpp",
                    "baseline": {"sha": "old_sha", "model": "/models/qwen-7b.gguf", "tps": 50.0},
                }
            },
            models={},
        )

        def fake_ls_remote(url):
            return "new_sha\trefs/heads/main\n"

        def fake_speed(model):
            return 51.0  # within 10% of 50.0

        self._patch_hooks(
            git_ls_remote_fn=fake_ls_remote,
            ggufbench_speed_fn=fake_speed,
        )

        rc = main(["check", "--state", self.state_path])
        self.assertEqual(rc, 1)

    # -- test 3: drift + bench + proposal → exit 3 -------------------------

    def test_drift_bench_proposal_exit_3(self) -> None:
        self._write_state(
            self.state_path,
            engines={
                "llamacpp": {
                    "repo": "https://github.com/ggerganov/llama.cpp",
                    "baseline": {"sha": "old_sha", "model": "/models/qwen-7b.gguf", "tps": 50.0},
                },
                "vulkan": {
                    "repo": "https://github.com/user/vulkan-engine",
                    "baseline": {"model": "/models/qwen-7b.gguf"},
                }
            },
            models={},
        )

        def fake_ls_remote(url):
            if "llama" in url:
                return "new_sha\trefs/heads/main\n"
            return "other_sha\trefs/heads/main\n"

        def fake_speed(model):
            return 40.0  # drift from 50.0

        def fake_stabilized(model):
            return 42.0

        def fake_runner(step):
            return 0  # bench succeeds

        import model_ci.cli as cli_mod
        cli_mod._run = lambda steps, runner: {"ran": ["llamacpp", "vulkan"], "failed": [], "rc": 0}

        self._patch_hooks(
            git_ls_remote_fn=fake_ls_remote,
            ggufbench_speed_fn=fake_speed,
            stabilized_fn=fake_stabilized,
            alive_llama_fn=lambda: False,
            alive_strata_fn=lambda: False,
            compute_apps_fn=lambda: [(100, 100)],
        )

        rc = main(["check", "--state", self.state_path])
        self.assertEqual(rc, 3)

    # -- test 4: GPU conflict → exit 65 ------------------------------------

    def test_gpu_conflict_exit_65(self) -> None:
        self._write_state(
            self.state_path,
            engines={
                "llamacpp": {
                    "repo": "https://github.com/ggerganov/llama.cpp",
                    "baseline": {"sha": "old_sha", "model": "/models/qwen-7b.gguf", "tps": 50.0},
                }
            },
            models={},
        )

        def fake_ls_remote(url):
            return "new_sha\trefs/heads/main\n"

        def fake_speed(model):
            return 40.0

        self._patch_hooks(
            git_ls_remote_fn=fake_ls_remote,
            ggufbench_speed_fn=fake_speed,
            alive_llama_fn=lambda: True,
            alive_strata_fn=lambda: True,
        )

        rc = main(["check", "--state", self.state_path])
        self.assertEqual(rc, 65)

    # -- test 5: usage error (no subcommand) → exit 64 ---------------------

    def test_usage_no_subcommand_exit_64(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            main([])
        self.assertEqual(ctx.exception.code, 64)

    # -- test 6: quiet-machine guard → exit 65 -----------------------------

    def test_quiet_machine_guard_exit_65(self) -> None:
        self._write_state(
            self.state_path,
            engines={
                "llamacpp": {
                    "repo": "https://github.com/ggerganov/llama.cpp",
                    "baseline": {"sha": "old_sha", "model": "/models/qwen-7b.gguf", "tps": 50.0},
                }
            },
            models={},
        )

        def fake_ls_remote(url):
            return "new_sha\trefs/heads/main\n"

        def fake_speed(model):
            return 40.0

        self._patch_hooks(
            git_ls_remote_fn=fake_ls_remote,
            ggufbench_speed_fn=fake_speed,
            alive_llama_fn=lambda: False,
            alive_strata_fn=lambda: False,
            compute_apps_fn=lambda: [(100, 3000)],  # high memory usage
        )

        rc = main(["check", "--state", self.state_path])
        self.assertEqual(rc, 65)


if __name__ == "__main__":
    unittest.main()
