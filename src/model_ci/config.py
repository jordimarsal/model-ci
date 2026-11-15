# src/model_ci/config.py
"""config — watched-repos TOML loader for model-ci."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


class ConfigError(Exception):
    """Raised when model-ci.toml is missing or invalid."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Parse and validate a model-ci.toml file.

    Returns a dict with keys:
      - "engines": dict[str, {"repo": str, "ref": str}]
      - "models": list[{"repo": str}]

    Raises ConfigError if the file is missing, unparseable, or structurally invalid.
    """
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")

    try:
        raw = tomllib.loads(p.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {p}: {exc}") from exc

    # Validate engines
    engines_raw = raw.get("engines", {})
    if not isinstance(engines_raw, dict) or len(engines_raw) == 0:
        raise ConfigError("config must define at least one engine under [engines.*]")

    engines: dict[str, dict[str, str]] = {}
    for name, info in engines_raw.items():
        if not isinstance(info, dict):
            raise ConfigError(f"[engines.{name}] must be a table")
        repo = info.get("repo")
        if not repo or not isinstance(repo, str):
            raise ConfigError(f"[engines.{name}].repo is required and must be a string")
        ref = info.get("ref", "refs/heads/main")
        if not isinstance(ref, str):
            raise ConfigError(f"[engines.{name}].ref must be a string")
        engines[name] = {"repo": repo, "ref": ref}

    # Validate models (optional)
    models_raw = raw.get("models", [])
    if not isinstance(models_raw, list):
        raise ConfigError("'models' must be an array of tables")

    models: list[dict[str, str]] = []
    for i, m in enumerate(models_raw):
        if not isinstance(m, dict):
            raise ConfigError(f"models[{i}] must be a table")
        repo = m.get("repo")
        if not repo or not isinstance(repo, str):
            raise ConfigError(f"models[{i}].repo is required and must be a string")
        models.append({"repo": repo})

    return {"engines": engines, "models": models}
