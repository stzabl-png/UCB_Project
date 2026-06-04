#!/usr/bin/env python3
"""
T2 batch — SAM2 mask from input/segment/prompt.json (no GUI).

Usage:
  python demo/scripts/T2/segment_prompt.py --session-dir <session_root>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

from _session_io import resolve_session_dirs  # noqa: E402

_T2_DIR = Path(__file__).resolve().parent
if str(_T2_DIR) not in sys.path:
    sys.path.insert(0, str(_T2_DIR))

from segment_common import (  # noqa: E402
    Sam2Predictor,
    check_sam2_installed,
    load_rgb_pil,
    mask_coverage_pct,
    session_id_from_dirs,
    write_outputs,
)


def _parse_prompt_json(path: Path) -> tuple[list[list[int]], list[list[int]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    fg: list[list[int]] = []
    bg: list[list[int]] = []
    prompts = data.get("prompts")
    if isinstance(prompts, dict):
        for item in prompts.get("fg", []):
            xy = item.get("xy", item) if isinstance(item, dict) else item
            fg.append([int(xy[0]), int(xy[1])])
        for item in prompts.get("bg", []):
            xy = item.get("xy", item) if isinstance(item, dict) else item
            bg.append([int(xy[0]), int(xy[1])])
        return fg, bg
    if isinstance(prompts, list):
        for item in prompts:
            if not isinstance(item, dict):
                continue
            xy = item.get("xy")
            if xy is None or len(xy) < 2:
                continue
            pt = [int(xy[0]), int(xy[1])]
            label = int(item.get("label", 1))
            if label == 0:
                bg.append(pt)
            else:
                fg.append(pt)
        return fg, bg
    raise ValueError(f"Unsupported prompt.json format: {path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="T2: SAM2 mask from prompt.json")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--session-dir", type=Path)
    g.add_argument("--input-dir", type=Path)
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args(argv)

    dirs = resolve_session_dirs(
        session_dir=args.session_dir,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
    prompt_path = dirs.input_rel("segment", "prompt.json")
    mask_out = dirs.output_rel("segment", "mask.png")
    rgb_path = dirs.input_rel("rgb", "left_rgb.png")

    if not prompt_path.is_file():
        print(f"Missing prompt: {prompt_path}", file=sys.stderr)
        return 1
    if not rgb_path.is_file():
        print(f"Missing RGB: {rgb_path}", file=sys.stderr)
        return 1
    if mask_out.is_file() and not args.redo:
        print(f"Already exists: {mask_out}  (use --redo)")
        return 0

    err = check_sam2_installed()
    if err:
        print(err, file=sys.stderr)
        return 2

    fg, bg = _parse_prompt_json(prompt_path)
    if not fg:
        print("prompt.json has no foreground points (label=1)", file=sys.stderr)
        return 1

    rgb = load_rgb_pil(rgb_path)
    pred = Sam2Predictor()
    pred.set_image(rgb)
    mask = pred.predict_mask(fg, bg)
    if mask is None:
        print("SAM2 returned no mask", file=sys.stderr)
        return 1

    sid = session_id_from_dirs(dirs.input_dir, dirs.session_id)
    write_outputs(
        dirs.output_rel("segment"),
        mask,
        fg,
        bg,
        sid,
        rgb.shape[:2],
        source="prompt_json",
    )
    print(f"Saved: {mask_out}  (coverage {mask_coverage_pct(mask):.2f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
