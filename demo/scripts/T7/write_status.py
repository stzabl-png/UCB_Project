#!/usr/bin/env python3
"""
T7 — Finalize Titan session output for Razor.

Checks T2–T6 artifacts, aggregates warnings, writes output/status.json.

Usage (repo root):

  python demo/scripts/T7/write_status.py \\
    --session-dir demo/sessions/20260602_192346_chips

  python demo/scripts/T7/write_status.py --session-dir ... --json
  python demo/scripts/T7/write_status.py --session-dir ... --allow-partial
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

from _session_io import SessionDirs, resolve_session_dirs  # noqa: E402

PIPELINE_VERSION = "demo.scripts.T7.write_status 0.1.0"

# Paths relative to session root (Razor rsync: output/ tree).
REQUIRED_OUTPUT: dict[str, list[str]] = {
    "segment": [
        "output/segment/mask.png",
        "output/segment/prompt_used.json",
    ],
    "sam3d": [
        "output/mesh/object_raw.glb",
        "output/mesh/sam3d_meta.json",
    ],
    "scale": [
        "output/mesh/object_scaled.glb",
        "output/mesh/scale.json",
    ],
    "foundationpose": [
        "output/mesh/object_base_aligned.glb",
        "output/register/T_cam_mesh.json",
        "output/register/T_base_mesh.json",
        "output/register/T_cam_mesh_fp.json",
        "output/register/foundationpose_meta.json",
        "output/register/mesh_frame_align.json",
        "output/register/ob_in_cam/000000.txt",
    ],
    "grasp_pose": [
        "output/inference/affordance_grasp.hdf5",
        "output/inference/candidates.json",
    ],
}

RECOMMENDED_VIS = [
    "output/vis/T3_sam3d_mesh_preview.png",
    "output/vis/T4_scale_scene_preview.png",
    "output/vis/T5_foundationpose_overlay.png",
    "output/vis/T6_grasp_vis.png",
]

RSYNC_OUTPUT_PATHS: list[str] = sorted(
    {
        "output/status.json",
        *[p for paths in REQUIRED_OUTPUT.values() for p in paths],
        *RECOMMENDED_VIS,
        "output/inference/pdm_meta.json",
    }
)


@dataclass
class StepStatus:
    name: str
    state: str  # ok | fail | skip
    missing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class FinalizeReport:
    ok: bool
    session_id: str
    steps: dict[str, str]
    warnings: list[str]
    errors: list[str]
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "session_id": self.session_id,
            "steps": self.steps,
            "warnings": self.warnings,
            "errors": self.errors,
            "info": self.info,
        }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_step_files(dirs: SessionDirs, step: str, rel_paths: list[str]) -> StepStatus:
    missing: list[str] = []
    for rel in rel_paths:
        if not (dirs.session_root / rel).is_file():
            missing.append(rel)
    if missing:
        return StepStatus(step, "fail", missing=missing)
    return StepStatus(step, "ok")


def _collect_meta_warnings(dirs: SessionDirs) -> list[str]:
    warnings: list[str] = []

    scale_path = dirs.output_rel("mesh", "scale.json")
    if scale_path.is_file():
        scale = _load_json(scale_path)
        warnings.extend(str(w) for w in scale.get("warnings", []))
        if scale.get("scale_clamped"):
            warnings.append(
                f"T4: scale_factor clamped to {scale.get('scale_factor')}"
            )

    fp_meta = dirs.output_rel("register", "foundationpose_meta.json")
    if fp_meta.is_file():
        meta = _load_json(fp_meta)
        warnings.extend(str(w) for w in meta.get("warnings", []))
        if meta.get("vis_error"):
            warnings.append(f"T5 vis: {meta['vis_error']}")

    return warnings


def _validate_candidates(dirs: SessionDirs, errors: list[str], info: dict[str, Any]) -> None:
    cand_path = dirs.output_rel("inference", "candidates.json")
    if not cand_path.is_file():
        return
    cand = _load_json(cand_path)
    n = int(cand.get("n_candidates", 0))
    items = cand.get("candidates", [])
    info["n_candidates"] = n
    info["mesh_frame"] = cand.get("mesh_frame")
    info["inference_method"] = cand.get("inference_method")
    if n < 1 or not items:
        errors.append("candidates.json has zero candidates")
        return

    reg_base = np.asarray(cand.get("T_base_mesh"), dtype=np.float64).reshape(4, 4)
    t5_path = dirs.output_rel("register", "T_base_mesh.json")
    if t5_path.is_file():
        t5 = np.asarray(_load_json(t5_path)["T_base_mesh"], dtype=np.float64).reshape(4, 4)
        err = float(np.linalg.norm(reg_base - t5))
        info["T_base_mesh_candidates_vs_register_max_abs"] = err
        if err > 1e-5:
            errors.append(
                f"candidates.json T_base_mesh differs from register/T_base_mesh.json (max {err:.2e})"
            )

    conv = cand.get("conventions", {})
    info["grasp_conventions"] = {
        "grasp_point_frame": conv.get("grasp_point_frame"),
        "approach_column_index": conv.get("approach_column_index"),
        "ucb_tcp_frame": conv.get("ucb_tcp_frame"),
        "pre_grasp_offset_m": conv.get("pre_grasp_offset_m"),
    }


def _validate_input_step(dirs: SessionDirs) -> tuple[str, list[str]]:
    """Run T1 validation; return (state, warnings)."""
    import subprocess

    t1 = _SCRIPTS_ROOT / "T1" / "validate_input.py"
    if not t1.is_file():
        return "skip", []
    try:
        proc = subprocess.run(
            [
                sys.executable,
                str(t1),
                "--session-dir",
                str(dirs.session_root),
                "--json",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode not in (0, 1):
            return "fail", [f"T1 exited {proc.returncode}: {proc.stderr.strip()}"]
        data = json.loads(proc.stdout)
        warnings = list(data.get("warnings", []))
        if data.get("ok"):
            return "ok", warnings
        errors = list(data.get("errors", []))
        return "fail", errors + warnings
    except Exception as exc:
        return "fail", [f"T1 validate failed: {exc}"]


def finalize_session(
    dirs: SessionDirs,
    *,
    allow_partial: bool = False,
    skip_input_check: bool = False,
) -> FinalizeReport:
    session_id = dirs.session_id
    errors: list[str] = []
    warnings: list[str] = []
    step_results: dict[str, StepStatus] = {}
    info: dict[str, Any] = {"object_slug": None, "pipeline_version": PIPELINE_VERSION}

    sess_json = dirs.input_rel("session.json")
    if sess_json.is_file():
        sess = _load_json(sess_json)
        info["object_slug"] = sess.get("object_slug")
        info["schema_version"] = sess.get("schema_version")

    if skip_input_check:
        v_state, v_msgs = "skip", []
    else:
        v_state, v_msgs = _validate_input_step(dirs)
    if v_state == "fail":
        errors.extend(v_msgs)
    elif v_state == "ok":
        warnings.extend(v_msgs)

    for step, paths in REQUIRED_OUTPUT.items():
        st = _check_step_files(dirs, step, paths)
        step_results[step] = st
        if st.state == "fail":
            errors.append(f"{step}: missing {', '.join(st.missing)}")

    warnings.extend(_collect_meta_warnings(dirs))

    missing_vis = [p for p in RECOMMENDED_VIS if not (dirs.session_root / p).is_file()]
    if missing_vis:
        warnings.append(f"optional vis missing: {', '.join(missing_vis)}")

    _validate_candidates(dirs, errors, info)

    steps_out = {k: v.state for k, v in step_results.items()}
    # README uses validate_input implicitly via T1; expose as steps for Razor
    pipeline_steps = {
        "segment": steps_out.get("segment", "fail"),
        "sam3d": steps_out.get("sam3d", "fail"),
        "scale": steps_out.get("scale", "fail"),
        "foundationpose": steps_out.get("foundationpose", "fail"),
        "grasp_pose": steps_out.get("grasp_pose", "fail"),
    }

    core_ok = all(pipeline_steps[s] == "ok" for s in pipeline_steps)
    ok = core_ok and not errors
    if allow_partial and core_ok:
        ok = True
        if errors:
            warnings.extend([f"(allow-partial) {e}" for e in errors])
            errors = []

    return FinalizeReport(
        ok=ok,
        session_id=session_id,
        steps=pipeline_steps,
        warnings=warnings,
        errors=errors,
        info=info,
    )


def build_status_payload(
    report: FinalizeReport,
    dirs: SessionDirs,
) -> dict[str, Any]:
    return {
        "schema_version": "1.1",
        "session_id": report.session_id,
        "success": report.ok,
        "pipeline_version": PIPELINE_VERSION,
        "finished_at_iso": datetime.now(timezone.utc).astimezone().isoformat(),
        "steps": report.steps,
        "warnings": report.warnings,
        "errors": report.errors,
        "titan": {
            "session_root": str(dirs.session_root),
            "object_slug": report.info.get("object_slug"),
            "n_candidates": report.info.get("n_candidates"),
            "mesh_frame": report.info.get("mesh_frame"),
            "inference_method": report.info.get("inference_method"),
            "grasp_conventions": report.info.get("grasp_conventions"),
        },
        "package": {
            "description": "Rsync this output/ tree to V2AP-demo demo/phase2/sessions/<session_id>/",
            "rsync_output_paths": RSYNC_OUTPUT_PATHS,
            "output_package_doc": "demo/TITAN_OUTPUT.md",
            "required_for_grasp": [
                "output/status.json",
                "output/inference/candidates.json",
                "output/register/T_base_mesh.json",
                "output/mesh/object_base_aligned.glb",
            ],
        },
    }


def write_status_atomic(dirs: SessionDirs, payload: dict[str, Any]) -> Path:
    out = dirs.output_rel("status.json")
    tmp = dirs.output_rel("status.json.tmp")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(out)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Phase 2 T7: finalize output/ and write status.json for Razor"
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--session-dir", type=Path)
    g.add_argument("--input-dir", type=Path)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="Set success=true if core steps ok despite consistency errors",
    )
    ap.add_argument(
        "--skip-input-check",
        action="store_true",
        help="Do not run T1 validation (only check output artifacts)",
    )
    ap.add_argument("--json", action="store_true", help="Print status payload to stdout")
    args = ap.parse_args(argv)

    dirs = resolve_session_dirs(
        session_dir=args.session_dir,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )

    report = finalize_session(
        dirs,
        allow_partial=args.allow_partial,
        skip_input_check=args.skip_input_check,
    )
    payload = build_status_payload(report, dirs)
    status_path = write_status_atomic(dirs, payload)

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Session: {report.session_id}")
        print(f"Result:  {'PASS' if report.ok else 'FAIL'}")
        for step, state in report.steps.items():
            print(f"  {step}: {state}")
        if report.warnings:
            print("Warnings:")
            for w in report.warnings:
                print(f"  - {w}")
        if report.errors:
            print("Errors:")
            for e in report.errors:
                print(f"  - {e}")
        print(f"Saved: {status_path}")

    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
