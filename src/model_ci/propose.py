# src/model_ci/propose.py
"""propose — evidence pack for the human gate."""

from __future__ import annotations

import datetime
from pathlib import Path


def evidence_pack(
    baseline_tps: float,
    current_tps: float,
    stabilized_tps: float,
    engine: str,
    build: str,
    model: str,
    receipts: dict[str, str],
) -> str:
    """Return a single markdown string with drift evidence.

    Contains: title, three labelled numbers (baseline bench / probe /
    stabilized ≥512tok), receipts (host load, flags), proposed action line,
    and today's UTC date.  No invention beyond what is passed in.
    """
    lines: list[str] = []
    lines.append(f"# Drift Evidence — {engine}")
    lines.append("")
    lines.append("## Numbers")
    lines.append("")
    lines.append(f"- **Baseline bench**:   {baseline_tps:.2f} tps")
    lines.append(f"- **Probe**:            {current_tps:.2f} tps")
    lines.append(f"- **Stabilized ≥512tok**: {stabilized_tps:.2f} tps")
    lines.append("")
    lines.append("## Metadata")
    lines.append("")
    lines.append(f"- Engine: `{engine}`")
    lines.append(f"- Build:  `{build}`")
    lines.append(f"- Model:  `{model}`")
    lines.append("")
    lines.append("## Receipts")
    lines.append("")
    for key, value in sorted(receipts.items()):
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## Proposed Action")
    lines.append("")
    lines.append(
        "Full re-bench + routing review at the roles apply gate."
    )
    lines.append("")
    today = datetime.date.today().isoformat()
    lines.append(f"Date: {today}")
    return "\n".join(lines)


def write(pack: str, out_dir: Path, engine: str) -> Path:
    """Write *pack* to `out_dir/{date}-{engine}-drift.md`.

    Creates the directory if it does not exist. Returns the path written.
    """
    date_str = datetime.date.today().strftime("%Y%m%d")
    filename = f"{date_str}-{engine}-drift.md"
    target = out_dir / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(pack, encoding="utf-8")
    return target
