#!/usr/bin/env python3
"""
T2 — Interactive SAM2 segmentation for one Phase 2 session frame.

Usage:
  conda activate bundlesdf
  python demo/scripts/T2/segment.py --session-dir demo/sessions/20260602_192346_chips

Requires third_party/sam2 installed (see demo/scripts/T2/README.md).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

from _session_io import repo_root, resolve_session_dirs  # noqa: E402

_T2_DIR = Path(__file__).resolve().parent
if str(_T2_DIR) not in sys.path:
    sys.path.insert(0, str(_T2_DIR))
from sam2_client import SAM2Client  # noqa: E402
from segment_common import (  # noqa: E402
    load_rgb_pil,
    mask_coverage_pct,
    write_outputs,
)

WIN = "T2 SAM2 Segment"
DISP_H = 720
MIN_FG_POINTS = 1


def default_bundlesdf_python() -> str:
    candidates = [
        Path("/home/vision/miniconda3/envs/bundlesdf/bin/python"),
        Path.home() / "miniconda3/envs/bundlesdf/bin/python",
        Path.home() / "anaconda3/envs/bundlesdf/bin/python",
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    return sys.executable


def run_t1_validate(session_dir: Path) -> bool:
    t1 = _SCRIPTS_ROOT / "T1" / "validate_input.py"
    py = sys.executable
    r = subprocess.run(
        [py, str(t1), "--session-dir", str(session_dir)],
        cwd=str(repo_root()),
    )
    return r.returncode == 0


def draw_ui(
    frame_rgb: np.ndarray,
    mask: np.ndarray | None,
    fg: list[list[int]],
    bg: list[list[int]],
    scale: float,
    waiting: bool,
) -> np.ndarray:
    disp = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    dH, dW = int(round(frame_rgb.shape[0] * scale)), int(round(frame_rgb.shape[1] * scale))
    disp = cv2.resize(disp, (dW, dH), interpolation=cv2.INTER_AREA)

    if mask is not None:
        mx = cv2.resize(mask, (dW, dH), interpolation=cv2.INTER_NEAREST)
        overlay = disp.copy()
        overlay[mx > 0] = (0, 80, 255)
        disp = cv2.addWeighted(disp, 0.6, overlay, 0.4, 0)
        cnts, _ = cv2.findContours(mx, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cv2.drawContours(disp, cnts, -1, (0, 255, 100), 2)

    for px, py in fg:
        cv2.circle(disp, (int(px * scale), int(py * scale)), 7, (0, 255, 80), -1)
        cv2.circle(disp, (int(px * scale), int(py * scale)), 7, (255, 255, 255), 2)
    for px, py in bg:
        cv2.circle(disp, (int(px * scale), int(py * scale)), 7, (0, 0, 220), -1)
        cv2.circle(disp, (int(px * scale), int(py * scale)), 7, (255, 255, 255), 2)

    cv2.rectangle(disp, (0, 0), (dW, 50), (12, 14, 22), -1)
    cv2.putText(
        disp,
        "LMB:FG  RMB:BG  M:toggle  C:clear  ENTER:save  Q:quit",
        (6, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
    )
    info = f"FG:{len(fg)} BG:{len(bg)}"
    if mask is not None:
        info += f"  mask={mask_coverage_pct(mask):.1f}%"
    if waiting:
        info += "  [SAM2...]"
    cv2.putText(disp, info, (6, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 200, 140), 1)
    return disp


def interactive_segment(
    rgb_path: Path,
    sam2: SAM2Client,
) -> tuple[np.ndarray, list[list[int]], list[list[int]]]:
    rgb = load_rgb_pil(rgb_path)
    H, W = rgb.shape[:2]
    scale = DISP_H / H

    resp = sam2.set_image(rgb_path)
    if resp.get("status") != "ok":
        raise RuntimeError(f"SAM2 set_image failed: {resp}")

    fg: list[list[int]] = []
    bg: list[list[int]] = []
    mode = "fg"
    mask: np.ndarray | None = None
    waiting = False
    mouse = {"scale": scale}

    def on_mouse(event, x, y, flags, param):
        nonlocal waiting
        sc = param["scale"]
        if event == cv2.EVENT_LBUTTONDOWN:
            ox, oy = int(x / sc), int(y / sc)
            (fg if mode == "fg" else bg).append([ox, oy])
            waiting = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            ox, oy = int(x / sc), int(y / sc)
            bg.append([ox, oy])
            waiting = True

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, on_mouse, mouse)

    while True:
        if waiting and len(fg) >= MIN_FG_POINTS:
            waiting = False
            pred = sam2.predict(fg, bg)
            if pred.get("status") == "ok" and pred.get("mask_path"):
                m = np.array(Image.open(pred["mask_path"]))
                mask = m if m.ndim == 2 else m[:, :, 0]
            else:
                print(f"SAM2 predict failed: {pred}", file=sys.stderr)

        ui = draw_ui(rgb, mask, fg, bg, scale, waiting)
        cv2.imshow(WIN, ui)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord("q"), 27):
            cv2.destroyAllWindows()
            raise SystemExit("Cancelled by user (no mask saved)")

        if key == ord("m"):
            mode = "bg" if mode == "fg" else "fg"

        if key == ord("c"):
            fg.clear()
            bg.clear()
            mask = None
            waiting = False

        if key in (13, 10):  # Enter
            if mask is None or len(fg) < MIN_FG_POINTS:
                print("Need at least one FG point and a mask (SAM2) before saving.")
                continue
            out = mask.astype(np.uint8)
            out[out > 0] = 255
            cv2.destroyAllWindows()
            return out, fg, bg

    raise RuntimeError("unreachable")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 2 T2: SAM2 interactive segmentation")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--session-dir", type=Path)
    g.add_argument("--input-dir", type=Path)
    ap.add_argument("--output-dir", type=Path, help="Override default <session>/output/")
    ap.add_argument("--validate", action="store_true", help="Run T1 validate before segment")
    ap.add_argument("--redo", action="store_true", help="Overwrite existing output/segment/mask.png")
    ap.add_argument(
        "--python",
        type=str,
        default=None,
        help="Python for SAM2 server (default: bundlesdf)",
    )
    args = ap.parse_args(argv)

    dirs = resolve_session_dirs(
        session_dir=args.session_dir,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
    rgb_path = dirs.input_rel("rgb", "left_rgb.png")
    mask_out = dirs.output_rel("segment", "mask.png")

    if args.validate:
        if not run_t1_validate(dirs.session_root):
            print("T1 validation failed; fix input/ or drop --validate", file=sys.stderr)
            return 1

    if not rgb_path.is_file():
        print(f"Missing RGB: {rgb_path}", file=sys.stderr)
        return 1

    if mask_out.is_file() and not args.redo:
        print(f"Already exists: {mask_out}  (use --redo to replace)")
        return 0

    sam2_py = args.python or default_bundlesdf_python()
    server = repo_root() / "tools" / "sam2_server.py"
    sam2_root = repo_root() / "third_party" / "sam2"
    ckpt = sam2_root / "checkpoints" / "sam2.1_hiera_tiny.pt"
    if not sam2_root.is_dir() or not ckpt.is_file():
        print(
            "SAM2 not installed. See demo/scripts/T2/README.md\n"
            f"  expected: {sam2_root}\n"
            f"  checkpoint: {ckpt}",
            file=sys.stderr,
        )
        return 2

    sam2 = SAM2Client(sam2_py, server)
    try:
        mask, fg, bg = interactive_segment(rgb_path, sam2)
    except SystemExit as e:
        print(e, file=sys.stderr)
        return 1
    finally:
        sam2.close()

    rgb = load_rgb_pil(rgb_path)
    session_id = dirs.session_id
    sess_json = dirs.input_rel("session.json")
    if sess_json.is_file():
        session_id = json.loads(sess_json.read_text()).get("session_id", session_id)

    write_outputs(
        dirs.output_rel("segment"),
        mask,
        fg,
        bg,
        session_id,
        rgb.shape[:2],
        source="interactive",
    )

    cov = mask_coverage_pct(mask)
    print(f"Saved: {mask_out}")
    print(f"Coverage: {cov:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
