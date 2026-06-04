#!/usr/bin/env python3
"""
Phase 2 pipeline entry — run T1–T7 for one Razor session.

For interactive SAM2 in the browser, use the Titan daemon instead:
  python -m demo.pipeline.segment_daemon

Usage (from Affordance2Grasp repo root):

  export FP_ROOT=/path/to/third_party/FoundationPose
  python -m demo.pipeline.process_razor_session \\
    --session-dir demo/sessions/20260602_192346_chips

Flags:
  --skip-sam      mask already in output/segment/
  --skip-sam3d    object_raw.glb exists
  --skip-fp       register/T_cam_mesh.json exists
  --redo          force re-run steps (pass --redo to T* scripts)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from demo.pipeline.env import PIPELINE_VERSION, repo_root, sessions_root
from demo.pipeline.run_pipeline import PipelineOptions, run_pipeline


def _resolve_session_arg(path: Path) -> Path:
    p = path.expanduser()
    if not p.is_absolute():
        p = (repo_root() / p).resolve()
    else:
        p = p.resolve()
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Process one Razor session (T1–T7) on Titan"
    )
    ap.add_argument(
        "--session-dir",
        type=Path,
        required=True,
        help="Session root with input/ (e.g. demo/sessions/<session_id>)",
    )
    ap.add_argument("--skip-sam", action="store_true")
    ap.add_argument("--skip-sam3d", action="store_true")
    ap.add_argument("--skip-fp", action="store_true")
    ap.add_argument("--redo", action="store_true", help="Overwrite existing step outputs")
    ap.add_argument("--device", type=str, default=None, help="T6 CUDA device")
    ap.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Override PDM sample count (default: input/session.json pipeline.titan.max_candidates or 50)",
    )
    args = ap.parse_args(argv)

    session_dir = _resolve_session_arg(args.session_dir)
    if not session_dir.is_dir():
        print(f"Session directory not found: {session_dir}", file=sys.stderr)
        return 2

    print(f"{PIPELINE_VERSION}")
    print(f"Sessions root (default): {sessions_root()}")

    result = run_pipeline(
        PipelineOptions(
            session_dir=session_dir,
            skip_sam=args.skip_sam,
            skip_sam3d=args.skip_sam3d,
            skip_fp=args.skip_fp,
            redo=args.redo,
            device=args.device,
            n_samples=args.n_samples,
        )
    )

    if result.errors:
        for e in result.errors:
            print(f"ERROR: {e}", file=sys.stderr)
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
