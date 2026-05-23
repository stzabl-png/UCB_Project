"""Tee training stdout (epoch table, config) to run_dir/log/train.log."""

from __future__ import annotations

import os
from datetime import datetime
from typing import IO

_log_fp: IO[str] | None = None


def open_training_log(log_dir: str, *, resume: bool = False) -> str:
    """Open log_dir/train.log; append when resuming."""
    global _log_fp
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "train.log")
    mode = "a" if resume else "w"
    if _log_fp is not None:
        _log_fp.close()
        _log_fp = None
    _log_fp = open(path, mode, encoding="utf-8")
    if resume:
        _log_fp.write(f"\n--- resumed {datetime.now().isoformat(timespec='seconds')} ---\n")
        _log_fp.flush()
    return path


def close_training_log() -> None:
    global _log_fp
    if _log_fp is not None:
        _log_fp.close()
        _log_fp = None


def training_log(msg: str, *, flush: bool = False) -> None:
    print(msg, flush=flush)
    if _log_fp is not None:
        _log_fp.write(msg + "\n")
        if flush:
            _log_fp.flush()
