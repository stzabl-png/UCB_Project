"""Read Razor → Titan hints from input/session.json (V2AP phase2 contract)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Matches V2AP-demo demo/phase2/constants.DEFAULT_TITAN_MAX_CANDIDATES
DEFAULT_TITAN_MAX_CANDIDATES = 50


def read_titan_max_candidates(session_data: dict[str, Any]) -> int | None:
    """Return ``pipeline.titan.max_candidates`` if present and valid."""
    pipeline = session_data.get("pipeline")
    if not isinstance(pipeline, dict):
        return None
    titan = pipeline.get("titan")
    if not isinstance(titan, dict):
        return None
    val = titan.get("max_candidates")
    if val is None:
        return None
    n = int(val)
    if n < 1:
        raise ValueError(f"pipeline.titan.max_candidates must be >= 1, got {n}")
    return n


def load_session_json(input_dir: Path) -> dict[str, Any] | None:
    path = Path(input_dir) / "session.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_pdm_n_samples(
    session_root: Path,
    *,
    cli_n_samples: int | None = None,
    default: int = DEFAULT_TITAN_MAX_CANDIDATES,
) -> int:
    """
    PDM draw count for T6.

    Priority: CLI ``--n-samples`` > ``input/session.json`` pipeline.titan.max_candidates > default (50).
    """
    if cli_n_samples is not None:
        n = int(cli_n_samples)
        if n < 1:
            raise ValueError(f"n_samples must be >= 1, got {n}")
        return n
    sess = load_session_json(Path(session_root) / "input")
    if sess is not None:
        from_session = read_titan_max_candidates(sess)
        if from_session is not None:
            return from_session
    return int(default)
