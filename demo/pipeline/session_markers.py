"""Razor↔Titan session upload markers (see demo/SERVER_CLIENT_PLAN.md)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

UPLOAD_COMPLETE_NAME = ".upload_complete"
UPLOAD_PROCESSED_NAME = ".upload_processed"
DAEMON_LOCK_NAME = "daemon.lock"


def upload_complete_path(session_root: Path) -> Path:
    return session_root / "input" / UPLOAD_COMPLETE_NAME


def upload_processed_path(session_root: Path) -> Path:
    return session_root / "input" / UPLOAD_PROCESSED_NAME


def daemon_lock_path(session_root: Path) -> Path:
    return session_root / "output" / DAEMON_LOCK_NAME


def write_upload_complete(
    session_root: Path,
    *,
    source: str = "razor",
    extra: dict[str, Any] | None = None,
) -> Path:
    """Signal Titan daemon that rsync input/ is finished."""
    path = upload_complete_path(session_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "marked_at_iso": datetime.now(timezone.utc).astimezone().isoformat(),
        "source": source,
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def mark_upload_processed(session_root: Path) -> None:
    """Move job out of the daemon queue."""
    done = upload_complete_path(session_root)
    proc = upload_processed_path(session_root)
    proc.parent.mkdir(parents=True, exist_ok=True)
    if done.is_file():
        proc.write_text(done.read_text(encoding="utf-8"), encoding="utf-8")
        done.unlink()
    elif not proc.is_file():
        proc.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "processed_at_iso": datetime.now(timezone.utc).astimezone().isoformat(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def is_upload_pending(session_root: Path) -> bool:
    if not upload_complete_path(session_root).is_file():
        return False
    if upload_processed_path(session_root).is_file():
        return False
    return True
