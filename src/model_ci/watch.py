"""Watch engines and models — pure parsing over already-fetched text/JSON."""

from __future__ import annotations


def parse_ls_remote(output: str) -> dict[str, str]:
    """Parse ``git ls-remote`` output into a mapping of ref → SHA.

    Each non-empty line has the form::

        <sha>\\t<ref>

    Peeling rules:
      - Lines ending in ``^{}`` represent the commit a tag points to; they *replace*
        the plain tag entry so that the returned sha is always the dereferenced commit.
      - Lines ending in ``^`` are ignored entirely (they carry no useful ref key).
    Blank lines and trailing CRs are tolerated.
    """
    result: dict[str, str] = {}
    for raw_line in output.splitlines():
        line = raw_line.rstrip("\r")
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        sha, ref = parts[0].strip(), parts[1].strip()
        if ref.endswith("^"):
            # bare "^" — skip
            continue
        if ref.endswith("^{}"):
            # peeled tag — replace any previous plain-tag entry
            base_ref = ref[:-3]
            result[base_ref] = sha
        else:
            result[ref] = sha
    return result


def engine_changed(
    baseline_sha: str | None, refs: dict[str, str], ref: str = "refs/heads/main"
) -> bool:
    """Return True when the engine repo has a new SHA for *ref*.

    First run (baseline is ``None``) always returns True.  If the ref does not exist
    in *refs* the repo did not change → False.
    """
    if baseline_sha is None:
        return True
    current_sha = refs.get(ref)
    if current_sha is None:
        return False
    return current_sha != baseline_sha


def hf_revisions(entries: list[dict]) -> dict[str, str]:
    """Extract SHA-256 hashes from HF tree API entries for every ``*.gguf`` file.

    Each entry may carry either::

        {"lfs": {"oid": "sha256:<hash>"}}

    or the flat variant::

        {"lfs_sha256": "<hash>"}

    Entries without a hash are silently skipped.
    """
    result: dict[str, str] = {}
    for entry in entries:
        path = entry.get("path", "")
        if not path.endswith(".gguf"):
            continue
        sha: str | None = None
        lfs = entry.get("lfs")
        if isinstance(lfs, dict):
            oid = lfs.get("oid", "")
            # strip optional prefix like "sha256:"
            sha = oid.split(":", 1)[-1] if ":" in oid else oid
        if not sha and (flat := entry.get("lfs_sha256")):
            sha = flat
        if sha:
            result[path] = sha
    return result


def models_changed(
    state_models: dict[str, dict], hf: dict[str, str]
) -> dict[str, str]:
    """Return files that are new or whose SHA changed compared with *state_models*.

    ``state_models`` maps repo paths to ``{"revision": <sha>}``.
    Only keys present in both dicts are compared; unchanged files are absent from the
    returned mapping.
    """
    changes: dict[str, str] = {}
    for path, old_info in state_models.items():
        current_sha = hf.get(path)
        if current_sha is None:
            continue
        old_sha = old_info.get("revision")
        if old_sha != current_sha:
            changes[path] = current_sha
    return changes
