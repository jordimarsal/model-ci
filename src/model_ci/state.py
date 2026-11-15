"""model_ci.state — persistent state store for model-ci.

Pure stdlib implementation (json / os / pathlib).
No network, no GPU, no external dependencies.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class StateError(Exception):
    """Raised when the state file is unreadable or has an unsupported schema."""

    def __init__(self, path: str | Path, message: str) -> None:
        self.path = path
        super().__init__(f"{path}: {message}")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CURRENT_VERSION = 1

_EMPTY_SKELETON = {"version": _CURRENT_VERSION, "engines": {}, "models": {}}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_version(state: dict) -> None:
    """Raise *StateError* if ``state`` does not declare a supported version."""
    ver = state.get("version")
    if ver is None:
        raise StateError(
            "<unknown>",
            'missing "version" key',
        )
    if ver != _CURRENT_VERSION:
        raise StateError(
            "<unknown>",
            f'unsupported version {ver} (expected {_CURRENT_VERSION})',
        )


def _load_raw(path: Path) -> dict:
    """Read and parse JSON; return raw dict.

    Returns the empty skeleton for missing files.
    Raises *StateError* on invalid / unknown versions.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return dict(_EMPTY_SKELETON)
    except OSError as exc:
        raise StateError(path, str(exc)) from exc

    try:
        state = json.loads(text)
    except json.JSONDecodeError as exc:
        raise StateError(path, f"invalid JSON ({exc})") from exc

    _validate_version(state)
    return state


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load(path: str | Path) -> dict:
    """Load the state file at *path*.

    Missing file → returns the v1 skeleton.
    Unknown or unsupported version → raises *StateError*.
    """
    return _load_raw(Path(path))


def save(path: str | Path, state: dict) -> None:
    """Atomically write *state* to *path*.

    Validates that ``version`` is present and equals 1 before writing.
    Uses a temporary file in the same directory + ``os.replace`` so a
    scheduler killed mid-write cannot corrupt the persisted state.
    """
    p = Path(path)
    _validate_version(state)

    # Create parent directories if needed (idempotent).
    p.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        dir=str(p.parent),
        prefix=".model-ci-state-",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2, ensure_ascii=False)
            fh.write("\n")          # trailing newline for POSIX convention
        os.replace(tmp_path, str(p))
    except BaseException:
        # Clean up the temp file on any failure (including KeyboardInterrupt).
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def migrate(path: str | Path) -> dict:
    """Migrate the state file at *path* to the current schema version.

    For v1 today this is a no-op — it simply loads and returns the state.
    Callers should use this function as their single entry point so future
    versions can be handled transparently.
    """
    return load(path)
