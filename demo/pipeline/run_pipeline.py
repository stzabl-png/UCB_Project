"""Run demo/scripts T1–T7 for one Razor session."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, TextIO

from demo.pipeline.env import (
    PIPELINE_VERSION,
    bundlesdf_python,
    default_fp_root,
    pipeline_env,
    repo_root,
    sam3d_python,
)
from demo.pipeline.status import write_progress


@dataclass
class PipelineOptions:
    session_dir: Path
    skip_sam: bool = False
    skip_sam3d: bool = False
    skip_fp: bool = False
    redo: bool = False
    device: str | None = None
    log: TextIO | None = None


@dataclass
class PipelineResult:
    ok: bool
    session_id: str
    steps: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0


def _log(opts: PipelineOptions, msg: str) -> None:
    line = msg if msg.endswith("\n") else msg + "\n"
    print(msg, flush=True)
    if opts.log is not None:
        opts.log.write(line)
        opts.log.flush()


def _run_step(
    opts: PipelineOptions,
    name: str,
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> int:
    _log(opts, f"\n=== {name} ===")
    _log(opts, " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=str(cwd or repo_root()),
        env=env,
        stdout=opts.log,
        stderr=subprocess.STDOUT if opts.log else None,
    )
    return int(proc.returncode)


def _resolve_dirs(session_dir: Path):
    scripts = repo_root() / "demo" / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from _session_io import resolve_session_dirs  # noqa: WPS433

    session_dir = session_dir.resolve()
    if not (session_dir / "input").is_dir():
        raise FileNotFoundError(f"No input/ under session: {session_dir}")
    return resolve_session_dirs(session_dir=session_dir)


def _artifact_exists(session_root: Path, rel: str) -> bool:
    return (session_root / rel).is_file()


def run_pipeline(opts: PipelineOptions) -> PipelineResult:
    t0 = time.perf_counter()
    started_iso = datetime.now(timezone.utc).astimezone().isoformat()
    root = repo_root()
    scripts = root / "demo" / "scripts"
    session_dir = opts.session_dir.resolve()
    dirs = _resolve_dirs(session_dir)
    out_dir = dirs.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "process.log"

    steps: dict[str, str] = {
        "segment": "pending",
        "sam3d": "pending",
        "scale": "pending",
        "foundationpose": "pending",
        "grasp_pose": "pending",
    }
    errors: list[str] = []
    warnings: list[str] = []

    with log_path.open("w", encoding="utf-8") as log_f:
        opts.log = log_f
        _log(opts, f"Auto demo pipeline {PIPELINE_VERSION}")
        _log(opts, f"Session: {dirs.session_id}")
        _log(opts, f"Root: {session_dir}")

        write_progress(
            out_dir,
            session_id=dirs.session_id,
            session_root=str(session_dir),
            steps=steps,
            state="running",
            started_at_iso=started_iso,
            current_step="segment",
        )

        py_b = str(bundlesdf_python())
        py_s3d = str(sam3d_python())
        env_b = pipeline_env(default_fp_root())
        sd = str(session_dir)

        # T1
        rc = _run_step(
            opts,
            "T1 validate_input",
            [py_b, str(scripts / "T1" / "validate_input.py"), "--session-dir", sd, "--json"],
            env=env_b,
        )
        if rc != 0:
            errors.append("T1 validate_input failed")
            _finish_fail(out_dir, dirs, steps, errors, warnings, started_iso, log_path, t0)
            return PipelineResult(False, dirs.session_id, steps, errors, warnings, time.perf_counter() - t0)

        # T2 segment
        mask_rel = "output/segment/mask.png"
        prompt_path = dirs.input_rel("segment", "prompt.json")
        if opts.skip_sam and _artifact_exists(session_dir, mask_rel):
            steps["segment"] = "ok"
            _log(
                opts,
                "T2 ok (mask already saved via SAM2 web UI or Razor — not re-running segment_prompt)",
            )
        elif _artifact_exists(session_dir, mask_rel) and not opts.redo:
            steps["segment"] = "ok"
            _log(opts, f"T2 skipped (exists): {mask_rel}")
        elif prompt_path.is_file():
            write_progress(
                out_dir,
                session_id=dirs.session_id,
                session_root=str(session_dir),
                steps=steps,
                state="running",
                started_at_iso=started_iso,
                current_step="segment",
            )
            cmd = [
                py_b,
                str(scripts / "T2" / "segment_prompt.py"),
                "--session-dir",
                sd,
            ]
            if opts.redo:
                cmd.append("--redo")
            rc = _run_step(opts, "T2 segment_prompt", cmd, env=env_b)
            steps["segment"] = "ok" if rc == 0 else "fail"
            if rc != 0:
                errors.append("T2 segment_prompt failed")
        else:
            steps["segment"] = "fail"
            errors.append(
                "T2: no output/segment/mask.png and no input/segment/prompt.json "
                "(use --skip-sam if mask already uploaded, or add prompt.json)"
            )

        if steps["segment"] != "ok" and steps["segment"] != "skipped":
            _finish_fail(out_dir, dirs, steps, errors, warnings, started_iso, log_path, t0)
            return PipelineResult(False, dirs.session_id, steps, errors, warnings, time.perf_counter() - t0)

        # T3 SAM3D
        raw_glb = "output/mesh/object_raw.glb"
        if opts.skip_sam3d and _artifact_exists(session_dir, raw_glb):
            steps["sam3d"] = "skipped"
            _log(opts, "T3 skipped (--skip-sam3d)")
        elif _artifact_exists(session_dir, raw_glb) and not opts.redo:
            steps["sam3d"] = "ok"
            _log(opts, f"T3 skipped (exists): {raw_glb}")
        else:
            write_progress(
                out_dir,
                session_id=dirs.session_id,
                session_root=str(session_dir),
                steps=steps,
                state="running",
                started_at_iso=started_iso,
                current_step="sam3d",
            )
            cmd = [py_s3d, str(scripts / "T3" / "reconstruct.py"), "--session-dir", sd]
            if opts.redo:
                cmd.append("--redo")
            rc = _run_step(opts, "T3 reconstruct", cmd, env=env_b)
            steps["sam3d"] = "ok" if rc == 0 else "fail"
            if rc != 0:
                errors.append("T3 reconstruct failed")

        if steps["sam3d"] not in ("ok", "skipped"):
            _finish_fail(out_dir, dirs, steps, errors, warnings, started_iso, log_path, t0)
            return PipelineResult(False, dirs.session_id, steps, errors, warnings, time.perf_counter() - t0)

        # T4 scale
        scaled = "output/mesh/object_scaled.glb"
        if _artifact_exists(session_dir, scaled) and not opts.redo:
            steps["scale"] = "ok"
            _log(opts, f"T4 skipped (exists): {scaled}")
        else:
            write_progress(
                out_dir,
                session_id=dirs.session_id,
                session_root=str(session_dir),
                steps=steps,
                state="running",
                started_at_iso=started_iso,
                current_step="scale",
            )
            cmd = [py_b, str(scripts / "T4" / "scale_from_depth.py"), "--session-dir", sd]
            if opts.redo:
                cmd.append("--redo")
            rc = _run_step(opts, "T4 scale", cmd, env=env_b)
            steps["scale"] = "ok" if rc == 0 else "fail"
            if rc != 0:
                errors.append("T4 scale_from_depth failed")

        if steps["scale"] != "ok":
            _finish_fail(out_dir, dirs, steps, errors, warnings, started_iso, log_path, t0)
            return PipelineResult(False, dirs.session_id, steps, errors, warnings, time.perf_counter() - t0)

        # T5 FP
        t_cam = "output/register/T_cam_mesh.json"
        if opts.skip_fp and _artifact_exists(session_dir, t_cam):
            steps["foundationpose"] = "skipped"
            _log(opts, "T5 skipped (--skip-fp)")
        elif _artifact_exists(session_dir, t_cam) and not opts.redo:
            steps["foundationpose"] = "ok"
            _log(opts, f"T5 skipped (exists): {t_cam}")
        else:
            write_progress(
                out_dir,
                session_id=dirs.session_id,
                session_root=str(session_dir),
                steps=steps,
                state="running",
                started_at_iso=started_iso,
                current_step="foundationpose",
            )
            cmd = [
                py_b,
                str(scripts / "T5" / "register_foundationpose.py"),
                "--session-dir",
                sd,
            ]
            if opts.redo:
                cmd.append("--redo")
            rc = _run_step(opts, "T5 register_foundationpose", cmd, env=env_b)
            steps["foundationpose"] = "ok" if rc == 0 else "fail"
            if rc != 0:
                errors.append("T5 register_foundationpose failed")

        if steps["foundationpose"] not in ("ok", "skipped"):
            _finish_fail(out_dir, dirs, steps, errors, warnings, started_iso, log_path, t0)
            return PipelineResult(False, dirs.session_id, steps, errors, warnings, time.perf_counter() - t0)

        # T6 PDM
        cand = "output/inference/candidates.json"
        if _artifact_exists(session_dir, cand) and not opts.redo:
            steps["grasp_pose"] = "ok"
            _log(opts, f"T6 skipped (exists): {cand}")
        else:
            write_progress(
                out_dir,
                session_id=dirs.session_id,
                session_root=str(session_dir),
                steps=steps,
                state="running",
                started_at_iso=started_iso,
                current_step="grasp_pose",
            )
            cmd = [py_b, str(scripts / "T6" / "run_pdm_grasp.py"), "--session-dir", sd]
            if opts.redo:
                cmd.append("--redo")
            if opts.device:
                cmd.extend(["--device", opts.device])
            rc = _run_step(opts, "T6 run_pdm_grasp", cmd, env=env_b)
            steps["grasp_pose"] = "ok" if rc == 0 else "fail"
            if rc != 0:
                errors.append("T6 run_pdm_grasp failed")

        if steps["grasp_pose"] != "ok":
            _finish_fail(out_dir, dirs, steps, errors, warnings, started_iso, log_path, t0)
            return PipelineResult(False, dirs.session_id, steps, errors, warnings, time.perf_counter() - t0)

        # T7 finalize
        write_progress(
            out_dir,
            session_id=dirs.session_id,
            session_root=str(session_dir),
            steps=steps,
            state="running",
            started_at_iso=started_iso,
            current_step="finalize",
        )
        cmd = [
            py_b,
            str(scripts / "T7" / "write_status.py"),
            "--session-dir",
            sd,
            "--pipeline-version",
            PIPELINE_VERSION,
        ]
        rc = _run_step(opts, "T7 write_status", cmd, env=env_b)
        if rc != 0:
            errors.append("T7 write_status failed")
            _finish_fail(out_dir, dirs, steps, errors, warnings, started_iso, log_path, t0)
            return PipelineResult(False, dirs.session_id, steps, errors, warnings, time.perf_counter() - t0)

        # Merge log path + started_at into final status
        status_path = out_dir / "status.json"
        if status_path.is_file():
            data = json.loads(status_path.read_text(encoding="utf-8"))
            data["started_at_iso"] = started_iso
            data["log_file"] = "output/logs/process.log"
            data["state"] = "done"
            data["elapsed_s"] = round(time.perf_counter() - t0, 2)
            if "titan" in data and isinstance(data["titan"], dict):
                data["titan"]["hostname"] = socket.gethostname()
            from demo.pipeline.status import write_status_atomic

            write_status_atomic(out_dir, data)
            ok = bool(data.get("success"))
            warnings.extend(str(w) for w in data.get("warnings", []))
            errors.extend(str(e) for e in data.get("errors", []))
            elapsed = time.perf_counter() - t0
            _log(opts, f"\nPipeline {'PASS' if ok else 'FAIL'} ({elapsed:.1f}s)")
            return PipelineResult(ok, dirs.session_id, steps, errors, warnings, elapsed)

        errors.append("T7 did not write status.json")
        _finish_fail(out_dir, dirs, steps, errors, warnings, started_iso, log_path, t0)
        return PipelineResult(False, dirs.session_id, steps, errors, warnings, time.perf_counter() - t0)


def _finish_fail(
    out_dir: Path,
    dirs,
    steps: dict[str, str],
    errors: list[str],
    warnings: list[str],
    started_iso: str,
    log_path: Path,
    t0: float,
) -> None:
    write_progress(
        out_dir,
        session_id=dirs.session_id,
        session_root=str(dirs.session_root),
        steps=steps,
        state="failed",
        warnings=warnings,
        errors=errors,
        started_at_iso=started_iso,
    )
    data_path = out_dir / "status.json"
    if data_path.is_file():
        data = json.loads(data_path.read_text(encoding="utf-8"))
        data["success"] = False
        data["state"] = "failed"
        data["log_file"] = "output/logs/process.log"
        data["elapsed_s"] = round(time.perf_counter() - t0, 2)
        from demo.pipeline.status import write_status_atomic

        write_status_atomic(out_dir, data)
