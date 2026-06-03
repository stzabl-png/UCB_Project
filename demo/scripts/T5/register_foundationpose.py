#!/usr/bin/env python3
"""
T5 — FoundationPose mesh ↔ camera registration (single frame).

Input:  object_scaled.glb, mask, RGB-D, K, extrinsics
Output: register/T_cam_mesh.json, T_base_mesh.json, ob_in_cam/, vis overlay

Usage:
  conda activate bundlesdf
  export FP_ROOT=/path/to/third_party/FoundationPose

  python demo/scripts/T5/register_foundationpose.py \\
    --session-dir demo/sessions/20260602_192346_chips
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

_T5_DIR = Path(__file__).resolve().parent
_SCRIPTS_ROOT = _T5_DIR.parent
if str(_T5_DIR) not in sys.path:
    sys.path.insert(0, str(_T5_DIR))
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

from _session_io import repo_root, resolve_session_dirs  # noqa: E402

from fp_common import (  # noqa: E402
    DEFAULT_SHORTER_SIDE,
    check_fp_root,
    default_fp_root,
    init_fp_models,
    load_mesh_for_fp,
    load_T_base_cam,
    load_table_height_m,
    prepare_fp_scene,
    registration_done,
    run_foundationpose_register,
    sanity_warnings,
    write_register_outputs,
)


def run_t1_validate(session_dir: Path) -> bool:
    t1 = _SCRIPTS_ROOT / "T1" / "validate_input.py"
    r = subprocess.run(
        [sys.executable, str(t1), "--session-dir", str(session_dir)],
        cwd=str(repo_root()),
    )
    return r.returncode == 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Phase 2 T5: FoundationPose register (single frame)"
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--session-dir", type=Path)
    g.add_argument("--input-dir", type=Path)
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--fp-root", type=Path, default=None)
    ap.add_argument("--validate", action="store_true", help="Run T1 first")
    ap.add_argument("--redo", action="store_true", help="Overwrite register outputs")
    ap.add_argument("--no-vis", action="store_true", help="Skip FP debug overlay render")
    ap.add_argument(
        "--vis-only",
        action="store_true",
        help="Rebuild 1×3 comparison PNG from saved T_cam_mesh (no FP inference)",
    )
    ap.add_argument(
        "--shorter-side",
        type=int,
        default=DEFAULT_SHORTER_SIDE,
        help=f"FP input short side (default {DEFAULT_SHORTER_SIDE})",
    )
    ap.add_argument("--est-iter", type=int, default=5)
    ap.add_argument("--keep-fp-scene", action="store_true")
    ap.add_argument("--apply-scale-json", action="store_true",
                    help="Re-apply scale.json (default off: mesh already scaled in T4)")
    args = ap.parse_args(argv)

    dirs = resolve_session_dirs(
        session_dir=args.session_dir,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )

    if args.validate and not run_t1_validate(dirs.session_root):
        print("T1 validation failed", file=sys.stderr)
        return 1

    fp_root = Path(args.fp_root).resolve() if args.fp_root else default_fp_root()

    if args.vis_only:
        fp_json = dirs.output_rel("register", "T_cam_mesh_fp.json")
        if not fp_json.is_file() and not dirs.output_rel("register", "T_cam_mesh.json").is_file():
            print("Missing register transforms; run T5 --redo first", file=sys.stderr)
            return 1
        from fp_common import run_mesh_align_step  # noqa: E402
        from fp_visualize import comparison_from_session  # noqa: WPS433

        if fp_json.is_file():
            run_mesh_align_step(dirs)
        else:
            print(
                "Note: no T_cam_mesh_fp.json — vis uses current T_cam_mesh only; "
                "run --redo for full 2-row vis",
                file=sys.stderr,
            )

        out = comparison_from_session(dirs.session_root, fp_root=fp_root)
        print(f"Saved: {out}")
        return 0

    scaled_glb = dirs.output_rel("mesh", "object_scaled.glb")
    mask_path = dirs.output_rel("segment", "mask.png")
    t_cam_json = dirs.output_rel("register", "T_cam_mesh.json")

    for p, name in [
        (scaled_glb, "object_scaled.glb"),
        (mask_path, "mask.png"),
        (dirs.input_rel("rgb", "left_rgb.png"), "rgb"),
        (dirs.input_rel("depth", "depth.npy"), "depth"),
    ]:
        if not p.is_file():
            print(f"Missing {name}: {p}", file=sys.stderr)
            return 1

    if registration_done(dirs) and not args.redo:
        print(f"Already exists: {t_cam_json}  (use --redo)")
        vis = dirs.output_rel("vis", "T5_foundationpose_overlay.png")
        if not vis.is_file():
            print("Overlay missing; re-run with --redo to regenerate vis")
        return 0

    err = check_fp_root(fp_root)
    if err:
        print(err, file=sys.stderr)
        print("Set FP_ROOT or pass --fp-root", file=sys.stderr)
        return 1

    print(f"Session: {dirs.session_id}")
    print(f"FP_ROOT: {fp_root}")

    scene_info = prepare_fp_scene(dirs, shorter_side=args.shorter_side)
    print(
        f"fp_scene: {scene_info.W_fp}×{scene_info.H_fp}  "
        f"mask={scene_info.mask_coverage_pct:.1f}%"
    )

    mesh_pre = load_mesh_for_fp(scaled_glb, apply_scale=False)
    n_faces = len(mesh_pre.faces)

    t0 = time.perf_counter()
    scorer, refiner, glctx = init_fp_models(fp_root)
    work_dir = dirs.output_rel("register", "fp_work")
    debug = 0 if args.no_vis else 1

    T_cam_mesh, vis_path = run_foundationpose_register(
        scaled_glb,
        scene_info,
        work_dir,
        scorer,
        refiner,
        glctx,
        fp_root,
        est_iter=args.est_iter,
        debug=debug,
        apply_scale=args.apply_scale_json,
    )
    elapsed = time.perf_counter() - t0

    table_h = load_table_height_m(dirs.input_dir)
    warns = sanity_warnings(
        T_cam_mesh,
        load_T_base_cam(dirs.input_dir),
        scaled_glb,
        scene_info.K_orig,
        mask_path,
        table_h,
    )
    for w in warns:
        print(f"Warning: {w}")

    write_register_outputs(
        dirs,
        T_cam_mesh,
        scene_info,
        scaled_glb,
        est_iter=args.est_iter,
        elapsed_s=elapsed,
        fp_root=fp_root,
        warnings=warns,
        n_faces=n_faces,
        vis_src=vis_path,
        keep_fp_scene=args.keep_fp_scene,
    )

    print(f"T_cam_mesh saved ({elapsed:.1f}s)")
    print(f"  {t_cam_json}")
    print(f"  {dirs.output_rel('vis', 'T5_foundationpose_overlay.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
