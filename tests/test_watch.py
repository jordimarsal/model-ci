"""Tests for src/model_ci/watch.py — pure-string, no filesystem."""

import unittest
from model_ci.watch import parse_ls_remote, engine_changed, hf_revisions, models_changed


class TestParseLsRemote(unittest.TestCase):
    def test_basic_refs(self):
        out = "abc123\trefs/heads/main\n"
        self.assertEqual(parse_ls_remote(out), {"refs/heads/main": "abc123"})

    def test_tags_and_peeled_replace_tag(self):
        out = (
            "def456\trefs/tags/v0.2.11\n"
            "789xyz\trefs/tags/v0.2.11^{}\n"
        )
        # peeled line replaces plain tag entry
        self.assertEqual(
            parse_ls_remote(out),
            {"refs/tags/v0.2.11": "789xyz"},
        )

    def test_bare_caret_lines_ignored(self):
        out = "aaa\trefs/tags/v1.0^\nbbb\trefs/heads/main\n"
        self.assertEqual(parse_ls_remote(out), {"refs/heads/main": "bbb"})

    def test_blank_lines_and_cr_tolerated(self):
        out = "\r\n\r\ndef456\trefs/heads/main\r\n"
        self.assertEqual(parse_ls_remote(out), {"refs/heads/main": "def456"})


class TestEngineChanged(unittest.TestCase):
    def test_first_run_is_change(self):
        self.assertTrue(engine_changed(None, {"refs/heads/main": "abc"}))

    def test_same_sha_no_change(self):
        self.assertFalse(engine_changed("abc", {"refs/heads/main": "abc"}))

    def test_different_sha_is_change(self):
        self.assertTrue(engine_changed("abc", {"refs/heads/main": "def"}))

    def test_missing_ref_no_change(self):
        self.assertFalse(engine_changed("abc", {}))
        self.assertFalse(
            engine_changed("abc", {"refs/heads/dev": "def"}, ref="refs/heads/main")
        )


class TestHfRevisions(unittest.TestCase):
    def test_basic_gguf_extraction(self):
        entries = [
            {
                "path": "model.gguf",
                "type": "file",
                "lfs": {"oid": "sha256:abcdef1234567890"},
            },
            {"path": "readme.md", "type": "file"},  # not .gguf → skip
        ]
        self.assertEqual(hf_revisions(entries), {"model.gguf": "abcdef1234567890"})

    def test_flat_lfs_sha256_variant(self):
        entries = [{"path": "x.gguf", "lfs_sha256": "flat-hash-xyz"}]
        self.assertEqual(hf_revisions(entries), {"x.gguf": "flat-hash-xyz"})

    def test_no_hash_skipped(self):
        entries = [{"path": "broken.gguf", "type": "file"}]
        self.assertEqual(hf_revisions(entries), {})


class TestModelsChanged(unittest.TestCase):
    def test_new_file_appears(self):
        state = {"repo/model.gguf": {"revision": None}}
        hf = {"repo/model.gguf": "new-sha"}
        self.assertEqual(models_changed(state, hf), {"repo/model.gguf": "new-sha"})

    def test_same_sha_unchanged_absent(self):
        state = {"repo/model.gguf": {"revision": "abc"}}
        hf = {"repo/model.gguf": "abc"}
        self.assertEqual(models_changed(state, hf), {})

    def test_different_sha_changes(self):
        state = {"repo/model.gguf": {"revision": "old"}}
        hf = {"repo/model.gguf": "new"}
        self.assertEqual(
            models_changed(state, hf), {"repo/model.gguf": "new"}
        )

    def test_missing_in_hf_not_reported(self):
        state = {"repo/missing.gguf": {"revision": "abc"}}
        hf = {}
        self.assertEqual(models_changed(state, hf), {})


if __name__ == "__main__":
    unittest.main()
