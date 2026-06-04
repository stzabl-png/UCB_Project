"""Atomic status.json updates for long-running pipeline jobs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from demo.pipeline.env import PIPELINE_VERSION


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def write_status_atomic(output_dir: Path, payload: dict[str, Any]) -> Path:
    out = output_dir / "status.json"
    tmp = output_dir / "status.json.tmp"
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(out)
    return out


def write_progress(
    output_dir: Path,
    *,
    session_id: str,
    session_root: str,
    steps: dict[str, str],
    state: str = "running",
    warnings: list[str] | None = None,
    errors: list[str] | None = None,
    started_at_iso: str | None = None,
    current_step: str | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "schema_version": "1.1",
        "session_id": session_id,
        "success": False,
        "state": state,
        "pipeline_version": PIPELINE_VERSION,
        "updated_at_iso": _now_iso(),
        "steps": steps,
        "warnings": warnings or [],
        "errors": errors or [],
        "titan": {"session_root": session_root},
        "package": {"output_package_doc": "demo/TITAN_OUTPUT.md"},
    }
    if started_at_iso:
        payload["started_at_iso"] = started_at_iso
    if current_step:
        payload["current_step"] = current_step
    return write_status_atomic(output_dir, payload)
