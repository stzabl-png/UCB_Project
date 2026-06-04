#!/usr/bin/env python3
"""
Titan segment daemon — watch for Razor uploads, run T2 web UI, then T3–T7.

Razor must NOT ssh-run the full pipeline. Instead:

  1. rsync input/ to demo/sessions/<id>/
  2. mark upload complete (see demo/razor/mark_upload_complete.py)
  3. Titan: keep this daemon running; open SAM2 URL via SSH tunnel
  4. Razor: poll output/status.json, rsync output/, review vis, run_auto_grasp

Usage (Titan, repo root):

  export FP_ROOT=$PWD/third_party/FoundationPose
  conda activate bundlesdf
  python -m demo.pipeline.segment_daemon

  # One-shot (process queue then exit):
  python -m demo.pipeline.segment_daemon --once

  # Process a single session immediately (debug):
  python -m demo.pipeline.segment_daemon --session-dir demo/sessions/<id>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from demo.pipeline.env import (
    PIPELINE_VERSION,
    bundlesdf_python,
    repo_root,
    sessions_root,
)
from demo.pipeline.run_pipeline import PipelineOptions, run_pipeline
from demo.pipeline.session_markers import (
    daemon_lock_path,
    is_upload_pending,
    mark_upload_processed,
    upload_complete_path,
    write_upload_complete,
)
from demo.pipeline.status import write_progress


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _resolve_session(path: Path) -> Path:
    p = path.expanduser()
    if not p.is_absolute():
        p = (repo_root() / p).resolve()
    return p.resolve()


def _resolve_dirs(session_dir: Path):
    scripts = repo_root() / "demo" / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from _session_io import resolve_session_dirs  # noqa: WPS433

    return resolve_session_dirs(session_dir=session_dir)


def _session_ready(session_root: Path) -> bool:
    inp = session_root / "input"
    return (
        inp.is_dir()
        and (inp / "session.json").is_file()
        and (inp / "rgb" / "left_rgb.png").is_file()
    )


def _status_success(session_root: Path) -> bool:
    path = session_root / "output" / "status.json"
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return bool(data.get("success"))
    except (json.JSONDecodeError, OSError):
        return False


def _acquire_lock(session_root: Path) -> bool:
    lock = daemon_lock_path(session_root)
    lock.parent.mkdir(parents=True, exist_ok=True)
    if lock.is_file():
        try:
            old = int(lock.read_text(encoding="utf-8").strip())
            if old != os.getpid():
                try:
                    os.kill(old, 0)
                    return False
                except OSError:
                    pass
        except ValueError:
            pass
    lock.write_text(f"{os.getpid()}\n", encoding="utf-8")
    return True


def _release_lock(session_root: Path) -> None:
    lock = daemon_lock_path(session_root)
    if lock.is_file():
        lock.unlink(missing_ok=True)


def _write_daemon_state(
    session_root: Path,
    *,
    session_id: str,
    state: str,
    message: str = "",
    segment_url: str | None = None,
) -> None:
    out = session_root / "output"
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "state": state,
        "message": message,
        "updated_at_iso": _now_iso(),
        "pipeline_version": PIPELINE_VERSION,
        "segment_url": segment_url,
    }
    (out / "daemon_state.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    steps = {
        "segment": "pending",
        "sam3d": "pending",
        "scale": "pending",
        "foundationpose": "pending",
        "grasp_pose": "pending",
    }
    write_progress(
        out,
        session_id=session_id,
        session_root=str(session_root),
        steps=steps,
        state=state,
        current_step="segment" if state == "waiting_segment" else None,
    )


def _run_t2_web(session_root: Path, *, host: str, port: int, redo: bool) -> int:
    """Blocking SAM2 Flask UI (subprocess in bundlesdf env)."""
    py = str(bundlesdf_python())
    script = repo_root() / "demo" / "scripts" / "T2" / "segment_web.py"
    cmd = [
        py,
        str(script),
        "--session-dir",
        str(session_root),
        "--host",
        host,
        "--port",
        str(port),
    ]
    if redo:
        cmd.append("--redo")
    print(f"\n[daemon] Starting T2 web: {' '.join(cmd)}", flush=True)
    import subprocess

    return int(
        subprocess.run(
            cmd,
            cwd=str(repo_root()),
        ).returncode
    )


def _needs_interactive_t2(session_root: Path, *, redo: bool) -> bool:
    mask = session_root / "output" / "segment" / "mask.png"
    prompt = session_root / "input" / "segment" / "prompt.json"
    if prompt.is_file():
        return False
    if mask.is_file() and not redo:
        return False
    return True


def process_session(
    session_root: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 7860,
    device: str | None = None,
    redo: bool = False,
    force: bool = False,
) -> bool:
    """
    Handle one session: T2 web (if needed) then pipeline --skip-sam for T3–T7.
    """
    session_root = _resolve_session(session_root)
    if not _session_ready(session_root):
        print(f"[daemon] skip (input incomplete): {session_root}", flush=True)
        return False

    if not force and not is_upload_pending(session_root):
        if _status_success(session_root) and not redo:
            print(f"[daemon] already done: {session_root.name}", flush=True)
            return True
        if not upload_complete_path(session_root).is_file():
            print(f"[daemon] no {upload_complete_path(session_root).name}: {session_root.name}", flush=True)
            return False

    if not _acquire_lock(session_root):
        print(f"[daemon] locked (another worker?): {session_root.name}", flush=True)
        return False

    dirs = _resolve_dirs(session_root)
    ok = False
    try:
        _write_daemon_state(
            session_root,
            session_id=dirs.session_id,
            state="waiting_segment",
            message="Waiting for SAM2 mask (open tunnel URL, Save, Done)",
            segment_url=f"http://127.0.0.1:{port}",
        )

        if _needs_interactive_t2(session_root, redo=redo):
            rc = _run_t2_web(session_root, host=host, port=port, redo=redo)
            if rc != 0:
                _write_daemon_state(
                    session_root,
                    session_id=dirs.session_id,
                    state="failed",
                    message=f"T2 web exited with code {rc}",
                )
                return False
            _write_daemon_state(
                session_root,
                session_id=dirs.session_id,
                state="segment_done",
                message="T2 mask saved; running T3–T7",
            )
        else:
            print(
                f"[daemon] T2 batch/skip: prompt.json or existing mask — {session_root.name}",
                flush=True,
            )

        _write_daemon_state(
            session_root,
            session_id=dirs.session_id,
            state="running",
            message="pipeline T1–T7",
        )

        has_mask = (session_root / "output" / "segment" / "mask.png").is_file()
        has_prompt = (session_root / "input" / "segment" / "prompt.json").is_file()
        skip_sam = has_mask and not has_prompt and not redo
        result = run_pipeline(
            PipelineOptions(
                session_dir=session_root,
                skip_sam=skip_sam,
                redo=redo,
                device=device,
            )
        )
        ok = result.ok
        if ok:
            mark_upload_processed(session_root)
            _write_daemon_state(
                session_root,
                session_id=dirs.session_id,
                state="done",
                message="Pipeline complete; Razor may rsync output/",
            )
        else:
            _write_daemon_state(
                session_root,
                session_id=dirs.session_id,
                state="failed",
                message="; ".join(result.errors) or "pipeline failed",
            )
        return ok
    finally:
        _release_lock(session_root)


def discover_pending(sessions_root: Path) -> list[Path]:
    root = _resolve_session(sessions_root)
    if not root.is_dir():
        return []
    pending: list[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if is_upload_pending(child) and _session_ready(child):
            pending.append(child)
    return pending


def run_daemon(
    *,
    sessions_root: Path | None = None,
    poll_interval: float = 5.0,
    host: str = "127.0.0.1",
    port: int = 7860,
    device: str | None = None,
    once: bool = False,
    redo: bool = False,
) -> int:
    root = _resolve_session(sessions_root if sessions_root is not None else sessions_root())
    print(f"Titan segment daemon — {PIPELINE_VERSION}")
    print(f"Watching: {root}")
    print(f"T2 bind {host}:{port}  (tunnel: ssh -L {port}:127.0.0.1:{port} user@titan)")
    print(f"Razor marks ready with: input/{upload_complete_path(Path('.')).name}")
    print("Ctrl+C to stop.\n", flush=True)

    while True:
        sessions = discover_pending(root)
        if sessions:
            for session_dir in sessions:
                print(f"[daemon] processing {session_dir.name}", flush=True)
                process_session(
                    session_dir,
                    host=host,
                    port=port,
                    device=device,
                    redo=redo,
                    force=False,
                )
        elif once:
            print("[daemon] no pending uploads.", flush=True)
            return 0
        if once:
            return 0
        time.sleep(poll_interval)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Titan daemon: Razor upload → T2 web → T3–T7 pipeline"
    )
    ap.add_argument(
        "--sessions-root",
        type=Path,
        default=None,
        help=f"Default: {sessions_root()}",
    )
    ap.add_argument("--session-dir", type=Path, help="Process one session and exit")
    ap.add_argument("--poll-interval", type=float, default=5.0)
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--device", type=str, default=None, help="T6 CUDA device")
    ap.add_argument("--once", action="store_true", help="Process queue once then exit")
    ap.add_argument("--redo", action="store_true", help="Re-run T2 web / pipeline")
    ap.add_argument(
        "--mark-complete",
        action="store_true",
        help="Only write input/.upload_complete for --session-dir",
    )
    args = ap.parse_args(argv)

    if args.mark_complete:
        if not args.session_dir:
            print("--mark-complete requires --session-dir", file=sys.stderr)
            return 2
        p = write_upload_complete(_resolve_session(args.session_dir))
        print(f"Wrote {p}")
        return 0

    if args.session_dir:
        ok = process_session(
            _resolve_session(args.session_dir),
            host=args.host,
            port=args.port,
            device=args.device,
            redo=args.redo,
            force=True,
        )
        return 0 if ok else 1

    try:
        return run_daemon(
            sessions_root=args.sessions_root,
            poll_interval=args.poll_interval,
            host=args.host,
            port=args.port,
            device=args.device,
            once=args.once,
            redo=args.redo,
        )
    except KeyboardInterrupt:
        print("\n[daemon] stopped.", flush=True)
        return 0


if __name__ == "__main__":
    sys.exit(main())
