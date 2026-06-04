"""Razor↔Titan session upload markers (see demo/SERVER_CLIENT_PLAN.md)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

UPLOAD_COMPLETE_NAME = ".upload_complete"
UPLOAD_PROCESSING_NAME = ".upload_processing"
UPLOAD_PROCESSED_NAME = ".upload_processed"
DAEMON_LOCK_NAME = "daemon.lock"


def upload_complete_path(session_root: Path) -> Path:
    return session_root / "input" / UPLOAD_COMPLETE_NAME


def upload_processing_path(session_root: Path) -> Path:
    return session_root / "input" / UPLOAD_PROCESSING_NAME


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


def claim_upload_for_processing(session_root: Path) -> Path | None:
    """
    Atomically dequeue: .upload_complete → .upload_processing.

    Prevents the poll loop (or a second daemon) from starting the same session again
    while T3–T7 are still running.
    """
    complete = upload_complete_path(session_root)
    processing = upload_processing_path(session_root)
    if not complete.is_file():
        return None
    processing.parent.mkdir(parents=True, exist_ok=True)
    if processing.is_file():
        processing.unlink()
    complete.rename(processing)
    return processing


def release_upload_for_retry(session_root: Path) -> None:
    """On pipeline failure, put the job back on the daemon queue."""
    complete = upload_complete_path(session_root)
    processing = upload_processing_path(session_root)
    if processing.is_file() and not complete.is_file():
        processing.rename(complete)


def is_session_locked(session_root: Path) -> bool:
    lock = daemon_lock_path(session_root)
    if not lock.is_file():
        return False
    try:
        pid = int(lock.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        lock.unlink(missing_ok=True)
        return False


def mark_upload_processed(session_root: Path) -> None:
    """Move job out of the daemon queue."""
    done = upload_complete_path(session_root)
    processing = upload_processing_path(session_root)
    proc = upload_processed_path(session_root)
    proc.parent.mkdir(parents=True, exist_ok=True)
    moved = False
    for src in (done, processing):
        if src.is_file():
            proc.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            src.unlink()
            moved = True
            break
    if not moved:
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
    if upload_processed_path(session_root).is_file():
        return False
    if upload_processing_path(session_root).is_file():
        return False
    if is_session_locked(session_root):
        return False
    return upload_complete_path(session_root).is_file()
