#!/usr/bin/env python3
"""
Reset a Titan session so the segment daemon can start from scratch.

Usage:

  # Clear all output/ and re-queue for daemon (T2 web + full pipeline):
  python -m demo.pipeline.reset_session \\
    --session-dir demo/sessions/20260603_184533_chips \\
    --requeue

  # Clear output only (manual rerun):
  python -m demo.pipeline.reset_session --session-dir demo/sessions/<id>

Then restart daemon or:
  python -m demo.pipeline.segment_daemon --session-dir demo/sessions/<id> --redo
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from demo.pipeline.env import repo_root
from demo.pipeline.session_markers import (
    daemon_lock_path,
    mark_upload_processed,
    upload_complete_path,
    upload_processed_path,
    write_upload_complete,
)


def _resolve(path: Path) -> Path:
    p = path.expanduser()
    if not p.is_absolute():
        p = (repo_root() / p).resolve()
    return p.resolve()


def reset_session(
    session_root: Path,
    *,
    requeue: bool = False,
    keep_mask: bool = False,
) -> None:
    session_root = _resolve(session_root)
    inp = session_root / "input"
    out = session_root / "output"

    upload_processed_path(session_root).unlink(missing_ok=True)
    upload_complete_path(session_root).unlink(missing_ok=True)
    daemon_lock_path(session_root).unlink(missing_ok=True)

    if out.is_dir():
        if keep_mask:
            seg = out / "segment"
            kept = seg / "mask.png"
            prompt_used = seg / "prompt_used.json"
            if seg.is_dir():
                for p in seg.iterdir():
                    if p.name not in ("mask.png", "prompt_used.json"):
                        p.unlink(missing_ok=True) if p.is_file() else shutil.rmtree(p, ignore_errors=True)
            for child in list(out.iterdir()):
                if child.name == "segment":
                    continue
                if child.is_file():
                    child.unlink()
                else:
                    shutil.rmtree(child, ignore_errors=True)
            print(f"Kept mask under {seg}" if kept.is_file() else "No mask to keep")
        else:
            shutil.rmtree(out, ignore_errors=True)
            print(f"Removed {out}")

    if requeue:
        if not inp.is_dir():
            raise FileNotFoundError(f"Missing input/: {session_root}")
        write_upload_complete(session_root, source="reset_session")
        print(f"Re-queued: {upload_complete_path(session_root)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Reset session for daemon / fresh pipeline run")
    ap.add_argument("--session-dir", type=Path, required=True)
    ap.add_argument(
        "--requeue",
        action="store_true",
        help="Write input/.upload_complete so segment_daemon picks it up",
    )
    ap.add_argument(
        "--keep-mask",
        action="store_true",
        help="When clearing output/, keep output/segment/mask.png (skip T2 web)",
    )
    ap.add_argument(
        "--mark-processed",
        action="store_true",
        help="Only mark job done (remove from daemon queue); do not delete output",
    )
    args = ap.parse_args(argv)

    root = _resolve(args.session_dir)
    if args.mark_processed:
        mark_upload_processed(root)
        upload_complete_path(root).unlink(missing_ok=True)
        daemon_lock_path(root).unlink(missing_ok=True)
        print("Marked upload processed; daemon will ignore this session until new .upload_complete")
        return 0

    reset_session(root, requeue=args.requeue, keep_mask=args.keep_mask)
    print("Use: python -m demo.pipeline.segment_daemon --session-dir ... --redo")
    print("  or leave daemon running; it will pick up .upload_complete on next poll.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
