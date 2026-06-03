#!/usr/bin/env python3
"""
T3 — SAM3D mesh reconstruction (unscaled) for one Phase 2 session.

Output:
  output/mesh/object_raw.glb
  output/mesh/sam3d_meta.json
  output/vis/T3_sam3d_mesh_preview.png   # unless --no-vis

Usage (sam3d-objects env recommended):
  conda activate sam3d-objects
  python demo/scripts/T3/reconstruct.py \\
    --session-dir demo/sessions/20260602_192346_chips
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

from _session_io import repo_root, resolve_session_dirs  # noqa: E402

_T3_DIR = Path(__file__).resolve().parent
if str(_T3_DIR) not in sys.path:
    sys.path.insert(0, str(_T3_DIR))

from sam3d_common import (  # noqa: E402
    Sam3dInference,
    check_sam3d_installed,
    default_sam3d_root,
    load_mask_png,
    load_rgb_pil,
    mask_coverage_pct,
    reconstruct_raw_mesh,
    sam3d_config_path,
    session_id_from_input,
    write_object_raw_glb,
    write_sam3d_meta,
)
from visualize import mesh_frame_origin, preview_from_glb, save_sam3d_mesh_preview  # noqa: E402

PIPELINE_FLAGS = {
    "with_mesh_postprocess": False,
    "with_texture_baking": False,
    "with_layout_postprocess": False,
    "use_vertex_color": True,
    "stage1_only": False,
}


def run_t1_validate(session_dir: Path) -> bool:
    t1 = _SCRIPTS_ROOT / "T1" / "validate_input.py"
    r = subprocess.run(
        [sys.executable, str(t1), "--session-dir", str(session_dir)],
        cwd=str(repo_root()),
    )
    return r.returncode == 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 2 T3: SAM3D unscaled mesh")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--session-dir", type=Path)
    g.add_argument("--input-dir", type=Path)
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--sam3d-root", type=Path, default=None, help="SAM 3D Objects repo root")
    ap.add_argument("--config", type=Path, default=None, help="pipeline.yaml (default: <sam3d>/checkpoints/hf/)")
    ap.add_argument("--validate", action="store_true", help="Run T1 validate before reconstruct")
    ap.add_argument("--redo", action="store_true", help="Overwrite existing object_raw.glb")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compile", action="store_true", help="torch.compile (slow first run)")
    ap.add_argument("--no-vis", action="store_true", help="Skip output/vis/T3_sam3d_mesh_preview.png")
    ap.add_argument(
        "--vis-only",
        action="store_true",
        help="Only regenerate preview PNG from existing object_raw.glb",
    )
    args = ap.parse_args(argv)

    sam3d_root = Path(args.sam3d_root).resolve() if args.sam3d_root else default_sam3d_root()
    err = check_sam3d_installed(sam3d_root)
    if err:
        print(err, file=sys.stderr)
        print("Install SAM3D and checkpoints; set SAM3D_ROOT if needed.", file=sys.stderr)
        return 2

    dirs = resolve_session_dirs(
        session_dir=args.session_dir,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )

    if args.validate and not run_t1_validate(dirs.session_root):
        print("T1 validation failed", file=sys.stderr)
        return 1

    rgb_path = dirs.input_rel("rgb", "left_rgb.png")
    mask_path = dirs.output_rel("segment", "mask.png")
    mesh_dir = dirs.output_rel("mesh")
    glb_out = mesh_dir / "object_raw.glb"
    meta_out = mesh_dir / "sam3d_meta.json"
    vis_out = dirs.output_rel("vis", "T3_sam3d_mesh_preview.png")

    if not rgb_path.is_file():
        print(f"Missing RGB: {rgb_path}", file=sys.stderr)
        return 1
    if not mask_path.is_file():
        print(f"Missing mask (run T2 first): {mask_path}", file=sys.stderr)
        return 1

    session_id = session_id_from_input(dirs.input_dir, dirs.session_id)

    if args.vis_only:
        if not glb_out.is_file():
            print(f"Missing mesh for --vis-only: {glb_out}", file=sys.stderr)
            return 1
        preview_from_glb(glb_out, rgb_path, mask_path, vis_out, session_id=session_id)
        if meta_out.is_file():
            import json

            meta = json.loads(meta_out.read_text(encoding="utf-8"))
            meta["preview_png"] = str(vis_out)
            meta_out.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"Preview: {vis_out}")
        return 0

    if glb_out.is_file() and not args.redo:
        import trimesh

        mesh = trimesh.load(glb_out, force="mesh")
        if not meta_out.is_file():
            rgb = load_rgb_pil(rgb_path)
            mask = load_mask_png(mask_path)
            origin = mesh_frame_origin(np.asarray(mesh.vertices))
            preview_path = vis_out if vis_out.is_file() else None
            write_sam3d_meta(
                meta_out,
                session_id=session_id,
                rgb_path=rgb_path,
                mask_path=mask_path,
                mesh_path=glb_out,
                frame_origin=origin.tolist(),
                mesh=mesh,
                mask_coverage=mask_coverage_pct(mask),
                time_s=0.0,
                seed=args.seed,
                sam3d_root=sam3d_root,
                config_path=sam3d_config_path(sam3d_root),
                pipeline_flags=PIPELINE_FLAGS,
                sam3d_output_summary={"note": "meta backfilled from existing object_raw.glb"},
                preview_path=preview_path,
            )
            print(f"Meta:  {meta_out}")
        if not args.no_vis and not vis_out.is_file():
            preview_from_glb(glb_out, rgb_path, mask_path, vis_out, session_id=session_id)
            print(f"Preview: {vis_out}")
        print(f"Already exists: {glb_out}  (use --redo to replace)")
        return 0

    config_path = Path(args.config).resolve() if args.config else sam3d_config_path(sam3d_root)
    if not config_path.is_file():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 2

    rgb = load_rgb_pil(rgb_path)
    mask = load_mask_png(mask_path)
    if mask.shape[:2] != rgb.shape[:2]:
        print(
            f"Mask shape {mask.shape[:2]} != RGB {rgb.shape[:2]}",
            file=sys.stderr,
        )
        return 1

    cov = mask_coverage_pct(mask)

    print(f"Session: {session_id}")
    print(f"RGB {rgb.shape[1]}x{rgb.shape[0]}  mask coverage {cov:.2f}%")
    print(f"Loading SAM3D from {sam3d_root} ...")

    engine = Sam3dInference(config_path, sam3d_root=sam3d_root, compile_model=args.compile)
    mesh, elapsed, out_summary = reconstruct_raw_mesh(engine, rgb, mask, seed=args.seed)

    write_object_raw_glb(mesh, glb_out)

    origin = mesh_frame_origin(np.asarray(mesh.vertices))
    preview_path: Path | None = None
    if not args.no_vis:
        preview_path = save_sam3d_mesh_preview(
            mesh,
            rgb,
            mask,
            vis_out,
            session_id=session_id,
            frame_origin=origin,
        )

    write_sam3d_meta(
        meta_out,
        session_id=session_id,
        rgb_path=rgb_path,
        mask_path=mask_path,
        mesh_path=glb_out,
        frame_origin=origin.tolist(),
        mesh=mesh,
        mask_coverage=cov,
        time_s=elapsed,
        seed=args.seed,
        sam3d_root=sam3d_root,
        config_path=config_path,
        pipeline_flags=PIPELINE_FLAGS,
        sam3d_output_summary=out_summary,
        preview_path=preview_path,
    )

    print(f"Saved: {glb_out}")
    print(f"Meta:  {meta_out}")
    if preview_path is not None:
        print(f"Preview: {preview_path}")
    print(f"Mesh:  {len(mesh.vertices)} verts / {len(mesh.faces)} faces  ({elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
