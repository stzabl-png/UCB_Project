#!/usr/bin/env python3
"""
T6 — PDM affordance + grasp candidates for a Razor session.

Input:  T5 ``object_base_aligned.glb``, ``register/T_cam_mesh.json``, ``T_base_mesh.json``
Output: ``output/inference/affordance_grasp.hdf5``, ``candidates.json``, optional vis

Usage (repo root, GPU recommended):

  cd /path/to/Affordance2Grasp
  python demo/scripts/T6/run_pdm_grasp.py \\
    --session-dir demo/sessions/20260602_192346_chips
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_T6_DIR = Path(__file__).resolve().parent
_SCRIPTS_ROOT = _T6_DIR.parent
_REPO = _SCRIPTS_ROOT.parent.parent
for p in (_T6_DIR, _SCRIPTS_ROOT, _REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from _session_io import resolve_session_dirs  # noqa: E402
from demo.pipeline.titan_options import resolve_pdm_n_samples  # noqa: E402
from pdm_session import run_pdm_for_session  # noqa: E402
from rebuild_vis import rebuild_t6_vis  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="T6: PDM grasp candidates for one session")
    ap.add_argument("--session-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--redo", action="store_true", help="Overwrite existing inference outputs")
    ap.add_argument(
        "--vis-only",
        action="store_true",
        help="Only rebuild output/vis/T6_grasp_vis.png from HDF5 + affordance NPZ",
    )
    ap.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="PDM sample count (default: input/session.json pipeline.titan.max_candidates, else 50)",
    )
    ap.add_argument("--ddim-steps", type=int, default=50)
    ap.add_argument("--num-points", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gripper-width", type=float, default=0.06)
    ap.add_argument("--z-yaw-deg", type=float, default=None)
    ap.add_argument("--reject-upward", action="store_true")
    ap.add_argument(
        "--no-dexmate-approach-sector",
        action="store_true",
        help="Disable hard filter: approach XY in 225°–315° (-Y wedge, Dexmate right)",
    )
    ap.add_argument(
        "--approach-sector-min-horiz",
        type=float,
        default=0.25,
        help="Min ||approach_xy|| to apply sector filter (else too_vertical)",
    )
    ap.add_argument("--affordance-ckpt", type=Path, default=None)
    ap.add_argument("--pdm-checkpoint", type=Path, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no-vis", action="store_true")
    ap.add_argument(
        "--scene-vis-top",
        type=int,
        default=10,
        help="Max candidates on RGB (right panel)",
    )
    ap.add_argument(
        "--mesh-vis-top",
        type=int,
        default=20,
        help="Max candidates on mesh panel (left)",
    )
    ap.add_argument("--no-affordance-sidecar", action="store_true")
    args = ap.parse_args()

    dirs = resolve_session_dirs(
        session_dir=args.session_dir, output_dir=args.output_dir
    )
    if args.vis_only:
        try:
            out = rebuild_t6_vis(
                dirs,
                mesh_top=args.mesh_vis_top,
                scene_top=args.scene_vis_top,
            )
        except FileNotFoundError as e:
            print(e, file=sys.stderr)
            return 1
        print(f"Saved: {out}")
        return 0

    candidates = dirs.output_rel("inference", "candidates.json")
    if candidates.is_file() and not args.redo:
        print(f"Skip (exists): {candidates}  (use --redo)")
        return 0

    device = "cpu" if args.cpu else args.device
    n_samples = resolve_pdm_n_samples(dirs.session_root, cli_n_samples=args.n_samples)
    print(f"T6 PDM n_samples={n_samples}")
    result = run_pdm_for_session(
        dirs,
        aff_ckpt=args.affordance_ckpt,
        pdm_ckpt=args.pdm_checkpoint,
        n_samples=n_samples,
        ddim_steps=args.ddim_steps,
        num_points=args.num_points,
        seed=args.seed,
        gripper_width=args.gripper_width,
        z_yaw_deg=args.z_yaw_deg,
        reject_upward=args.reject_upward,
        dexmate_approach_sector=not args.no_dexmate_approach_sector,
        approach_sector_min_horiz=args.approach_sector_min_horiz,
        device=device,
        write_vis=not args.no_vis,
        scene_vis_top=args.scene_vis_top,
        mesh_vis_top=args.mesh_vis_top,
        write_affordance_sidecar=not args.no_affordance_sidecar,
    )
    print(f"Saved: {result.h5_path}")
    print(f"Saved: {result.candidates_path}  ({result.n_candidates} candidates)")
    print(f"Saved: {result.meta_path}")
    if result.vis_path:
        print(f"Saved: {result.vis_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
