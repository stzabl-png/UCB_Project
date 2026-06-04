#!/usr/bin/env python3
"""
Blocking review of Titan output/vis PNGs (T3–T6) on Razor.

Shows each image in order; the process blocks until the user closes the window,
then continues to the next. Intended to run after rsync download, before grasp.

Usage (from Affordance2Grasp repo root or copy into V2AP-demo):

  python demo/razor/review_titan_vis.py --session-dir demo/sessions/<session_id>

  # Non-interactive / CI:
  python demo/razor/review_titan_vis.py --session-dir ... --skip
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ordered Titan review figures (relative to session root).
TITAN_VIS_SEQUENCE: tuple[tuple[str, str], ...] = (
    ("output/vis/T3_sam3d_mesh_preview.png", "T3 — SAM3D mesh"),
    ("output/vis/T4_scale_scene_preview.png", "T4 — metric scale"),
    ("output/vis/T5_foundationpose_overlay.png", "T5 — FoundationPose + base align"),
    ("output/vis/T6_grasp_vis.png", "T6 — PDM grasp candidates"),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_session(path: Path) -> Path:
    p = path.expanduser()
    if not p.is_absolute():
        p = (_repo_root() / p).resolve()
    return p.resolve()


def show_png_blocking(png_path: Path, *, title: str) -> None:
    """Display one PNG; block until the matplotlib window is closed."""
    import matplotlib.pyplot as plt

    if not png_path.is_file():
        raise FileNotFoundError(png_path)

    img = plt.imread(str(png_path))
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    ax.set_axis_off()
    fig.suptitle(f"{title}\n{png_path.name} — close window to continue", fontsize=11)
    plt.tight_layout()
    plt.show(block=True)
    plt.close(fig)


def review_titan_vis(
    session_dir: Path,
    *,
    skip: bool = False,
    strict: bool = False,
) -> list[str]:
    """
    Show T3–T6 vis PNGs in order. Returns list of missing relative paths.
    """
    session_dir = _resolve_session(session_dir)
    missing: list[str] = []

    if skip:
        return missing

    for rel, label in TITAN_VIS_SEQUENCE:
        path = session_dir / rel
        if not path.is_file():
            missing.append(rel)
            print(f"[review] skip (missing): {rel}", file=sys.stderr)
            if strict:
                raise FileNotFoundError(path)
            continue
        print(f"[review] {label}: {path}")
        show_png_blocking(path, title=label)

    return missing


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Blocking sequential review of Titan T3–T6 vis PNGs"
    )
    ap.add_argument("--session-dir", type=Path, required=True)
    ap.add_argument("--skip", action="store_true", help="Do not open any windows")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any vis PNG is missing (default: skip missing)",
    )
    args = ap.parse_args(argv)

    missing = review_titan_vis(args.session_dir, skip=args.skip, strict=args.strict)
    if missing and args.strict:
        return 1
    if missing:
        print(f"Warning: {len(missing)} vis file(s) missing (non-strict).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
