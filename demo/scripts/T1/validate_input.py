#!/usr/bin/env python3
"""
T1 — Validate Razor session input/ package (Phase 2 schema 1.1).

Usage:
  python demo/scripts/T1/validate_input.py \\
    --session-dir demo/sessions/20260602_192346_chips

  python demo/scripts/T1/validate_input.py --input-dir path/to/input
  python demo/scripts/T1/validate_input.py --session-dir ... --write-status
  python demo/scripts/T1/validate_input.py --session-dir ... --json
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

REQUIRED_INPUT_FILES = (
    "session.json",
    "rgb/left_rgb.png",
    "depth/depth.npy",
    "calib/intrinsics.json",
    "calib/K.npy",
    "calib/extrinsics.json",
    "calib/robot_state.json",
    "scene/table.json",
)

SUPPORTED_SCHEMA_MAJOR = "1.1"
TABLE_HEIGHT_TOL_M = 0.05
K_RTOL = 1e-5
K_ATOL = 1e-6


@dataclass
class ValidationReport:
    ok: bool
    session_id: str = ""
    input_dir: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "session_id": self.session_id,
            "input_dir": self.input_dir,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
        }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _check_required_files(input_dir: Path, report: ValidationReport) -> None:
    for rel in REQUIRED_INPUT_FILES:
        if not (input_dir / rel).is_file():
            report.errors.append(f"Missing required file: {rel}")


def _validate_session_json(
    input_dir: Path, session_root: Path, report: ValidationReport
) -> dict[str, Any] | None:
    path = input_dir / "session.json"
    if not path.is_file():
        return None

    sess = _load_json(path)
    report.session_id = str(sess.get("session_id", ""))

    if session_root.name not in ("input", "") and report.session_id:
        if session_root.name != report.session_id:
            report.warnings.append(
                f"Folder name '{session_root.name}' != session_id '{report.session_id}'"
            )

    sv = str(sess.get("schema_version", ""))
    if not sv.startswith(SUPPORTED_SCHEMA_MAJOR):
        report.errors.append(
            f"Unsupported schema_version '{sv}' (need {SUPPORTED_SCHEMA_MAJOR})"
        )

    cap = sess.get("capture") or {}
    if cap.get("depth_unit") and cap["depth_unit"] != "meters":
        report.errors.append(f"capture.depth_unit must be 'meters', got {cap['depth_unit']!r}")
    if cap.get("depth_aligned_to_rgb") is False:
        report.warnings.append("capture.depth_aligned_to_rgb is false — pipeline expects aligned RGB-D")

    pipe = sess.get("pipeline") or {}
    reg = pipe.get("registration_method")
    if sv.startswith("1.1") and reg != "foundationpose":
        report.errors.append(
            f"pipeline.registration_method must be 'foundationpose' for schema 1.1, got {reg!r}"
        )
    if sv.startswith("1.1") and not pipe.get("foundationpose"):
        report.warnings.append("pipeline.foundationpose block missing (recommended for 1.1)")

    for key in ("rgb_file", "depth_file"):
        rel = cap.get(key)
        if rel and not (input_dir / rel).is_file():
            report.errors.append(f"session.json capture.{key} not found: {rel}")

    robot = sess.get("robot") or {}
    for key in ("state_file", "extrinsics_file"):
        rel = robot.get(key)
        if rel and not (input_dir / rel).is_file():
            report.errors.append(f"session.json robot.{key} not found: {rel}")

    scene = sess.get("scene") or {}
    rel = scene.get("table_file")
    if rel and not (input_dir / rel).is_file():
        report.errors.append(f"session.json scene.table_file not found: {rel}")

    report.info["object_slug"] = sess.get("object_slug")
    report.info["schema_version"] = sv
    return sess


def _validate_intrinsics(input_dir: Path, sess: dict[str, Any] | None, report: ValidationReport) -> np.ndarray | None:
    intr_path = input_dir / "calib/intrinsics.json"
    k_path = input_dir / "calib/K.npy"
    if not intr_path.is_file() or not k_path.is_file():
        return None

    intr = _load_json(intr_path)
    K_json = np.asarray(intr["K"], dtype=np.float64)
    K_npy = np.load(k_path).astype(np.float64)

    if K_json.shape != (3, 3) or K_npy.shape != (3, 3):
        report.errors.append(f"K must be 3x3; json {K_json.shape}, npy {K_npy.shape}")
        return None

    if not np.allclose(K_json, K_npy, rtol=K_RTOL, atol=K_ATOL):
        report.errors.append("calib/K.npy does not match intrinsics.json K")

    w = intr.get("width")
    h = intr.get("height")
    if sess:
        cap = sess.get("capture") or {}
        if w is not None and cap.get("rgb_width") not in (None, w):
            report.warnings.append(
                f"intrinsics width {w} != session capture rgb_width {cap.get('rgb_width')}"
            )
        if h is not None and cap.get("rgb_height") not in (None, h):
            report.warnings.append(
                f"intrinsics height {h} != session capture rgb_height {cap.get('rgb_height')}"
            )

    dist = intr.get("distortion_model", "none")
    if dist and dist != "none":
        report.warnings.append(
            f"distortion_model={dist!r}: T1 accepts pinhole K; undistort before FP if overlays are off"
        )

    report.info["K"] = K_npy.tolist()
    report.info["intrinsics_source"] = intr.get("source")
    return K_npy


def _validate_extrinsics(input_dir: Path, report: ValidationReport) -> np.ndarray | None:
    path = input_dir / "calib/extrinsics.json"
    if not path.is_file():
        return None

    ext = _load_json(path)
    T = np.asarray(ext.get("T_base_cam"), dtype=np.float64)
    if T.shape != (4, 4):
        report.errors.append(f"extrinsics T_base_cam must be 4x4, got {T.shape}")
        return None

    if abs(T[3, 3] - 1.0) > 1e-6 or np.max(np.abs(T[3, :3])) > 1e-6:
        report.warnings.append("T_base_cam bottom row is not [0,0,0,1]")

    R = T[:3, :3]
    det = float(np.linalg.det(R))
    if abs(det - 1.0) > 0.05:
        report.warnings.append(f"T_base_cam rotation det={det:.4f} (expected ~1)")

    report.info["extrinsics_method"] = ext.get("method")
    return T


def _validate_rgb_depth(
    input_dir: Path,
    sess: dict[str, Any] | None,
    K: np.ndarray | None,
    T_base_cam: np.ndarray | None,
    report: ValidationReport,
) -> None:
    rgb_path = input_dir / "rgb/left_rgb.png"
    depth_path = input_dir / "depth/depth.npy"
    if not rgb_path.is_file() or not depth_path.is_file():
        return

    try:
        import cv2
    except ImportError:
        report.errors.append("opencv-python required for RGB/depth checks (pip install opencv-python)")
        return

    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if bgr is None:
        report.errors.append("Failed to read rgb/left_rgb.png")
        return

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    report.info["rgb_shape_hw"] = [int(h), int(w)]

    try:
        from PIL import Image

        pil = np.array(Image.open(rgb_path))
        if pil.ndim == 3 and pil.shape[2] >= 3:
            diff = np.abs(pil[:, :, :3].astype(np.int16) - rgb.astype(np.int16)).max()
            if diff == 0:
                report.info["rgb_disk_layout"] = "rgb (PIL matches cv2 BGR2RGB)"
            elif np.abs(pil[:, :, :3].astype(np.int16) - bgr.astype(np.int16)).max() == 0:
                report.warnings.append(
                    "left_rgb.png on disk looks like BGR (PIL matches cv2 imread without convert); "
                    "SAM3D uses PIL — colors may be wrong unless you use BGR2RGB when loading"
                )
                report.info["rgb_disk_layout"] = "bgr_on_disk"
            else:
                report.warnings.append("RGB file channel layout ambiguous (PIL vs cv2 mismatch)")
    except ImportError:
        pass

    depth = np.load(depth_path)
    report.info["depth_shape_hw"] = list(depth.shape)
    report.info["depth_dtype"] = str(depth.dtype)

    if depth.dtype != np.float32:
        report.warnings.append(f"depth.npy dtype is {depth.dtype}, spec recommends float32")

    if depth.ndim != 2:
        report.errors.append(f"depth.npy must be 2D (H,W), got shape {depth.shape}")
        return

    if depth.shape != (h, w):
        report.errors.append(
            f"RGB/depth shape mismatch: rgb ({h},{w}) vs depth {depth.shape}"
        )

    if sess:
        cap = sess.get("capture") or {}
        for key, dim, got in (
            ("rgb_height", 0, h),
            ("rgb_width", 1, w),
            ("depth_height", 0, depth.shape[0]),
            ("depth_width", 1, depth.shape[1]),
        ):
            exp = cap.get(key)
            if exp is not None and int(exp) != int(got):
                report.errors.append(f"session.json capture.{key}={exp} but on-disk size is {got}")

    valid = np.isfinite(depth) & (depth > 0)
    frac = float(valid.mean()) if depth.size else 0.0
    report.info["depth_valid_fraction"] = round(frac, 4)
    if frac < 0.5:
        report.warnings.append(f"Only {frac*100:.1f}% valid depth pixels (>0, finite)")
    if valid.any():
        d = depth[valid]
        report.info["depth_m_range"] = [float(d.min()), float(d.max())]
        report.info["depth_median_m"] = float(np.median(d))

    if K is not None and T_base_cam is not None and valid.any():
        table_path = input_dir / "scene/table.json"
        if table_path.is_file():
            table_h = float(_load_json(table_path).get("table_height_m", np.nan))
            if np.isfinite(table_h):
                fx, fy = K[0, 0], K[1, 1]
                cx, cy = K[0, 2], K[1, 2]
                zs = []
                dh, dw = depth.shape
                for v in range(int(dh * 0.35), int(dh * 0.65), 4):
                    for u in range(int(dw * 0.35), int(dw * 0.65), 4):
                        z = depth[v, u]
                        if not (np.isfinite(z) and z > 0):
                            continue
                        p_cam = np.array([(u - cx) * z / fx, (v - cy) * z / fy, z, 1.0])
                        zs.append(float((T_base_cam @ p_cam)[2]))
                if len(zs) >= 20:
                    med_z = float(np.median(zs))
                    delta = med_z - table_h
                    report.info["table_height_m"] = table_h
                    report.info["roi_median_z_base_m"] = med_z
                    report.info["table_z_delta_m"] = round(delta, 4)
                    if abs(delta) > TABLE_HEIGHT_TOL_M:
                        report.warnings.append(
                            f"Center ROI median z in base ({med_z:.3f} m) vs table_height_m "
                            f"({table_h:.3f} m): delta {delta*100:.1f} cm (tolerance ±{TABLE_HEIGHT_TOL_M*100:.0f} cm)"
                        )


def validate_session(
    session_dir: Path | None = None,
    input_dir: Path | None = None,
    dirs: SessionDirs | None = None,
) -> ValidationReport:
    if dirs is None:
        dirs = resolve_session_dirs(session_dir=session_dir, input_dir=input_dir)
    session_root, input_dir = dirs.session_root, dirs.input_dir
    report = ValidationReport(ok=True, input_dir=str(input_dir))

    _check_required_files(input_dir, report)
    sess = _validate_session_json(input_dir, session_root, report)
    K = _validate_intrinsics(input_dir, sess, report)
    T = _validate_extrinsics(input_dir, report)
    _validate_rgb_depth(input_dir, sess, K, T, report)

    if report.errors:
        report.ok = False
    return report


def write_failure_status(session_root: Path, report: ValidationReport) -> Path:
    out_dir = session_root / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / "status.json"
    tmp_path = out_dir / "status.json.tmp"
    payload = {
        "schema_version": "1.1",
        "session_id": report.session_id,
        "success": False,
        "pipeline_version": "demo.scripts.T1.validate_input",
        "finished_at_iso": datetime.now(timezone.utc).astimezone().isoformat(),
        "steps": {},
        "warnings": report.warnings,
        "errors": report.errors,
        "validation": report.to_dict(),
    }
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(status_path)
    return status_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 2 T1: validate Razor session input/")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--session-dir",
        type=Path,
        help="Session root containing input/ (e.g. demo/sessions/<session_id>)",
    )
    g.add_argument("--input-dir", type=Path, help="Path directly to input/ folder")
    ap.add_argument(
        "--write-status",
        action="store_true",
        help="On failure, write output/status.json under session root",
    )
    ap.add_argument("--json", action="store_true", help="Print full report as JSON")
    ap.add_argument(
        "--strict-warnings",
        action="store_true",
        help="Exit 1 if there are warnings (not only errors)",
    )
    args = ap.parse_args(argv)

    try:
        report = validate_session(
            session_dir=args.session_dir,
            input_dir=args.input_dir,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(f"Session: {report.session_id or '(unknown)'}")
        print(f"Input:   {report.input_dir}")
        print(f"Result:  {'PASS' if report.ok else 'FAIL'}")
        if report.info:
            print("Info:")
            for k, v in report.info.items():
                print(f"  {k}: {v}")
        if report.warnings:
            print("Warnings:")
            for w in report.warnings:
                print(f"  - {w}")
        if report.errors:
            print("Errors:")
            for e in report.errors:
                print(f"  - {e}")

    if not report.ok and args.write_status:
        status_path = write_failure_status(Path(report.input_dir).parent, report)
        print(f"Wrote {status_path}", file=sys.stderr)

    if not report.ok:
        return 1
    if args.strict_warnings and report.warnings:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
