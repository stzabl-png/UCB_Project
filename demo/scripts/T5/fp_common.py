"""FoundationPose helpers for Phase 2 demo T5 (single-frame Razor session)."""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

# T4 K / depth loaders
_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))
_T4_DIR = _SCRIPTS_ROOT / "T4"
if str(_T4_DIR) not in sys.path:
    sys.path.insert(0, str(_T4_DIR))
from scale_common import load_K, load_depth_m  # noqa: E402

from _session_io import SessionDirs, repo_root  # noqa: E402

DEFAULT_SHORTER_SIDE = 480
FP_FRAME_ID = "000000"
MAX_MESH_FACES = 5000


@dataclass(frozen=True)
class FpSceneInfo:
    scene_dir: Path
    H_fp: int
    W_fp: int
    H_orig: int
    W_orig: int
    scale_fp: float
    K_fp: np.ndarray
    K_orig: np.ndarray
    mask_coverage_pct: float


def default_fp_root() -> Path:
    env = os.environ.get("FP_ROOT")
    if env:
        return Path(env).resolve()
    return repo_root() / "third_party" / "FoundationPose"


def check_fp_root(fp_root: Path) -> str | None:
    if not fp_root.is_dir():
        return f"FP_ROOT not found: {fp_root}"
    for sub in ("estimater.py", "datareader.py", "weights"):
        if not (fp_root / sub).exists():
            return f"Missing {sub} under {fp_root}"
    for folder in ("2023-10-28-18-33-37", "2024-01-11-20-02-45"):
        w = fp_root / "weights" / folder / "model_best.pth"
        if not w.is_file():
            return f"Missing weight: {w}"
    return None


def load_rgb_np(rgb_path: Path) -> np.ndarray:
    return np.array(Image.open(rgb_path).convert("RGB"))


def load_mask_uint8(mask_path: Path) -> np.ndarray:
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(mask_path)
    return (m > 0).astype(np.uint8) * 255


def load_T_base_cam(input_dir: Path) -> np.ndarray:
    ext_path = input_dir / "calib" / "extrinsics.json"
    data = json.loads(ext_path.read_text())
    return np.asarray(data["T_base_cam"], dtype=np.float64)


def load_table_height_m(input_dir: Path, default: float = 0.98) -> float:
    p = input_dir / "scene" / "table.json"
    if p.is_file():
        return float(json.loads(p.read_text()).get("table_height_m", default))
    return default


def _scale_K(K: np.ndarray, sx: float, sy: float) -> np.ndarray:
    K = K.copy().astype(np.float64)
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] *= sx
    K[1, 2] *= sy
    return K


def prepare_fp_scene(
    dirs: SessionDirs,
    *,
    shorter_side: int = DEFAULT_SHORTER_SIDE,
    frame_id: str = FP_FRAME_ID,
) -> FpSceneInfo:
    """Build output/register/fp_scene for YcbineoatReader (single frame)."""
    rgb_path = dirs.input_rel("rgb", "left_rgb.png")
    depth_path = dirs.input_rel("depth", "depth.npy")
    mask_path = dirs.output_rel("segment", "mask.png")

    rgb = load_rgb_np(rgb_path)
    depth_m = load_depth_m(depth_path)
    mask_u8 = load_mask_uint8(mask_path)
    K_orig = load_K(dirs.input_dir)

    H0, W0 = rgb.shape[:2]
    if depth_m.shape != (H0, W0):
        depth_m = cv2.resize(depth_m, (W0, H0), interpolation=cv2.INTER_NEAREST)
    if mask_u8.shape != (H0, W0):
        mask_u8 = cv2.resize(mask_u8, (W0, H0), interpolation=cv2.INTER_NEAREST)

    scale_fp = shorter_side / min(H0, W0)
    H_fp = int(round(H0 * scale_fp))
    W_fp = int(round(W0 * scale_fp))
    sx, sy = W_fp / W0, H_fp / H0

    rgb_fp = cv2.resize(rgb, (W_fp, H_fp), interpolation=cv2.INTER_AREA)
    depth_fp = cv2.resize(depth_m, (W_fp, H_fp), interpolation=cv2.INTER_NEAREST)
    depth_fp = np.nan_to_num(depth_fp, nan=0.0, posinf=0.0, neginf=0.0)
    depth_fp[depth_fp < 0] = 0.0
    mask_fp = cv2.resize(mask_u8, (W_fp, H_fp), interpolation=cv2.INTER_NEAREST)
    K_fp = _scale_K(K_orig, sx, sy)

    scene_dir = dirs.output_rel("register", "fp_scene")
    for sub in ("rgb", "depth", "masks"):
        (scene_dir / sub).mkdir(parents=True, exist_ok=True)

    out_id = frame_id
    cv2.imwrite(
        str(scene_dir / "rgb" / f"{out_id}.png"),
        cv2.cvtColor(rgb_fp, cv2.COLOR_RGB2BGR),
    )
    depth_mm = (depth_fp * 1000.0).clip(0, 65535).astype(np.uint16)
    cv2.imwrite(str(scene_dir / "depth" / f"{out_id}.png"), depth_mm)
    cv2.imwrite(str(scene_dir / "masks" / f"{out_id}.png"), mask_fp)
    np.savetxt(scene_dir / "cam_K.txt", K_fp, fmt="%.6f")

    cov = (mask_fp > 0).mean() * 100.0
    return FpSceneInfo(
        scene_dir=scene_dir,
        H_fp=H_fp,
        W_fp=W_fp,
        H_orig=H0,
        W_orig=W0,
        scale_fp=scale_fp,
        K_fp=K_fp,
        K_orig=K_orig,
        mask_coverage_pct=cov,
    )


def load_mesh_for_fp(mesh_path: Path, *, apply_scale: bool = False) -> "trimesh.Trimesh":
    import trimesh

    mesh = trimesh.load(str(mesh_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    if apply_scale:
        scale_json = mesh_path.parent / "scale.json"
        if scale_json.is_file():
            sf = float(json.loads(scale_json.read_text()).get("scale_factor", 1.0))
            if abs(sf - 1.0) > 0.01:
                mesh.vertices = mesh.vertices * sf
                print(f"    Scale applied from scale.json: ×{sf:.4f}")

    n_faces = len(mesh.faces)
    if n_faces > MAX_MESH_FACES:
        import fast_simplification

        ratio = 1.0 - min(MAX_MESH_FACES / n_faces, 0.9999)
        pts_out, faces_out = fast_simplification.simplify(
            mesh.vertices, mesh.faces, target_reduction=ratio
        )
        mesh = trimesh.Trimesh(vertices=pts_out, faces=faces_out, process=False)
        print(f"    Mesh simplified: {n_faces:,} → {len(mesh.faces):,} faces")
    return mesh


def init_fp_models(fp_root: Path):
    fp_root = fp_root.resolve()
    s = str(fp_root)
    if s not in sys.path:
        sys.path.insert(0, s)
    import nvdiffrast.torch as dr
    from estimater import PoseRefinePredictor, ScorePredictor, set_seed

    set_seed(0)
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    return scorer, refiner, glctx


def run_foundationpose_register(
    mesh_path: Path,
    scene_info: FpSceneInfo,
    work_dir: Path,
    scorer,
    refiner,
    glctx,
    fp_root: Path,
    *,
    est_iter: int = 5,
    debug: int = 1,
    apply_scale: bool = False,
) -> tuple[np.ndarray, Path | None]:
    """
    Single-frame register. Returns (T_cam_mesh 4x4, track_vis png path or None).
    """
    fp_root = fp_root.resolve()
    if str(fp_root) not in sys.path:
        sys.path.insert(0, str(fp_root))

    import imageio
    import torch
    from datareader import YcbineoatReader
    from estimater import FoundationPose, draw_posed_3d_box, draw_xyz_axis

    mesh = load_mesh_for_fp(mesh_path, apply_scale=apply_scale)
    to_origin, extents = __import__("trimesh").bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    work_dir.mkdir(parents=True, exist_ok=True)
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=str(work_dir),
        debug=debug,
        glctx=glctx,
    )

    reader = YcbineoatReader(
        video_dir=str(scene_info.scene_dir), shorter_side=None, zfar=np.inf
    )
    ob_in_cam_dir = work_dir / "ob_in_cam"
    track_vis_dir = work_dir / "track_vis"
    ob_in_cam_dir.mkdir(parents=True, exist_ok=True)
    track_vis_dir.mkdir(parents=True, exist_ok=True)

    color = reader.get_color(0)
    depth = reader.get_depth(0)
    mask = reader.get_mask(0).astype(bool)
    pose = est.register(
        K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=est_iter
    )
    pose = np.asarray(pose, dtype=np.float64).reshape(4, 4)

    np.savetxt(ob_in_cam_dir / f"{FP_FRAME_ID}.txt", pose, fmt="%.6f")

    vis_path = None
    if debug >= 1:
        center_pose = pose @ np.linalg.inv(to_origin)
        vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
        vis = draw_xyz_axis(
            color,
            ob_in_cam=center_pose,
            scale=0.1,
            K=reader.K,
            thickness=3,
            transparency=0,
            is_input_rgb=True,
        )
        vis_path = track_vis_dir / f"{FP_FRAME_ID}.png"
        imageio.imwrite(str(vis_path), vis)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pose, vis_path


def project_mesh_mask_iou(
    mesh_path: Path,
    T_cam_mesh: np.ndarray,
    K: np.ndarray,
    mask_bool: np.ndarray,
    *,
    max_verts: int = 8000,
) -> float | None:
    """Rough IoU: projected mesh vertices vs SAM mask."""
    import trimesh

    mesh = trimesh.load(str(mesh_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if len(verts) > max_verts:
        idx = np.linspace(0, len(verts) - 1, max_verts, dtype=int)
        verts = verts[idx]
    ones = np.ones((len(verts), 1), dtype=np.float64)
    v_h = np.hstack([verts, ones])
    v_cam = (T_cam_mesh @ v_h.T).T[:, :3]
    z = v_cam[:, 2]
    valid = z > 0.01
    if valid.sum() < 10:
        return None
    u = (K[0, 0] * v_cam[valid, 0] / z[valid] + K[0, 2]).astype(int)
    v = (K[1, 1] * v_cam[valid, 1] / z[valid] + K[1, 2]).astype(int)
    H, W = mask_bool.shape
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if ok.sum() < 10:
        return None
    proj = np.zeros((H, W), dtype=bool)
    proj[v[ok], u[ok]] = True
    gt = mask_bool.astype(bool)
    inter = (proj & gt).sum()
    union = (proj | gt).sum()
    return float(inter / union) if union > 0 else 0.0


def sanity_warnings(
    T_cam_mesh: np.ndarray,
    T_base_cam: np.ndarray,
    mesh_path: Path,
    K: np.ndarray,
    mask_path: Path,
    table_height_m: float,
) -> list[str]:
    warnings: list[str] = []
    T_base_mesh = T_base_cam @ T_cam_mesh

    import trimesh

    mesh = trimesh.load(str(mesh_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    ones = np.ones((len(verts), 1))
    v_base = (T_base_mesh @ np.hstack([verts, ones]).T).T[:, :3]
    z_min = float(v_base[:, 2].min())
    z_med = float(np.median(v_base[:, 2]))
    dz = abs(z_med - table_height_m)
    if dz > 0.05:
        warnings.append(
            f"mesh median Z in base={z_med:.3f}m vs table_height={table_height_m:.3f}m "
            f"(Δ={dz:.3f}m)"
        )
    if z_min < table_height_m - 0.15:
        warnings.append(f"mesh bottom Z={z_min:.3f}m far below table top")

    mask_bool = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0
    iou = project_mesh_mask_iou(mesh_path, T_cam_mesh, K, mask_bool)
    if iou is not None and iou < 0.15:
        warnings.append(f"projected mesh vs mask IoU={iou:.3f} (low)")

    return warnings


def write_register_outputs(
    dirs: SessionDirs,
    T_cam_mesh_fp: np.ndarray,
    scene_info: FpSceneInfo,
    mesh_path: Path,
    *,
    est_iter: int,
    elapsed_s: float,
    fp_root: Path,
    warnings: list[str],
    n_faces: int,
    vis_src: Path | None,
    keep_fp_scene: bool,
) -> None:
    reg = dirs.output_rel("register")
    reg.mkdir(parents=True, exist_ok=True)

    T_base_cam = load_T_base_cam(dirs.input_dir)
    T_base_mesh_fp = T_base_cam @ T_cam_mesh_fp
    rel_mesh = str(mesh_path.relative_to(dirs.session_root))

    # --- FP / SAM3D frame (preserved for debug) ---
    fp_cam_payload = {
        "camera_frame": "zed_left_camera",
        "mesh_frame": "sam3d_scaled",
        "T_cam_mesh": T_cam_mesh_fp.tolist(),
        "method": "foundationpose",
        "fp_frame": FP_FRAME_ID,
        "est_iter": est_iter,
        "mesh_file": rel_mesh,
        "note": "Raw FP output before base-axis alignment",
    }
    (reg / "T_cam_mesh_fp.json").write_text(
        json.dumps(fp_cam_payload, indent=2) + "\n"
    )
    fp_base_payload = {
        "base_frame": "base",
        "mesh_frame": "sam3d_scaled",
        "T_base_mesh": T_base_mesh_fp.tolist(),
        "method": "foundationpose",
        "T_base_cam_source": "input/calib/extrinsics.json",
        "composition": "T_base_mesh = T_base_cam @ T_cam_mesh",
    }
    (reg / "T_base_mesh_fp.json").write_text(
        json.dumps(fp_base_payload, indent=2) + "\n"
    )

    ob_fp = reg / "ob_in_cam_fp"
    ob_fp.mkdir(parents=True, exist_ok=True)
    np.savetxt(ob_fp / f"{FP_FRAME_ID}.txt", T_cam_mesh_fp, fmt="%.6f")

    # --- Base-aligned mesh frame (T6 / affordance / PDM) ---
    from mesh_align import align_mesh_to_base, write_mesh_frame_align_json  # noqa: WPS433

    aligned_glb = dirs.output_rel("mesh", "object_base_aligned.glb")
    align_result = align_mesh_to_base(
        mesh_path,
        T_cam_mesh_fp,
        T_base_cam,
        aligned_glb,
    )
    rel_aligned = str(aligned_glb.relative_to(dirs.session_root))
    write_mesh_frame_align_json(reg / "mesh_frame_align.json", align_result)

    T_cam_mesh = align_result.T_cam_mesh_aligned
    T_base_mesh = align_result.T_base_mesh_aligned

    t_cam_payload = {
        "camera_frame": "zed_left_camera",
        "mesh_frame": "base_aligned",
        "T_cam_mesh": T_cam_mesh.tolist(),
        "method": "foundationpose",
        "fp_frame": FP_FRAME_ID,
        "est_iter": est_iter,
        "mesh_file": rel_aligned,
        "source_fp": "register/T_cam_mesh_fp.json",
        "mesh_frame_align": "register/mesh_frame_align.json",
    }
    (reg / "T_cam_mesh.json").write_text(json.dumps(t_cam_payload, indent=2) + "\n")

    t_base_payload = {
        "base_frame": "base",
        "mesh_frame": "base_aligned",
        "T_base_mesh": T_base_mesh.tolist(),
        "method": "foundationpose",
        "T_base_cam_source": "input/calib/extrinsics.json",
        "composition": "T_base_mesh = T_base_cam @ T_cam_mesh",
        "note": "R should be ~identity; translation = object origin in base",
    }
    (reg / "T_base_mesh.json").write_text(
        json.dumps(t_base_payload, indent=2) + "\n"
    )

    ob_dir = reg / "ob_in_cam"
    ob_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(ob_dir / f"{FP_FRAME_ID}.txt", T_cam_mesh, fmt="%.6f")
    np.save(ob_dir / f"{FP_FRAME_ID}.npy", T_cam_mesh.astype(np.float64))

    print(
        f"Mesh align: R_residual={align_result.R_aligned_residual:.4f}  "
        f"max_base_err={align_result.max_point_err_m*1000:.2f}mm  "
        f"→ {rel_aligned}"
    )

    meta = {
        "tool": "foundationpose",
        "finished_at_iso": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": round(elapsed_s, 3),
        "fp_root": str(fp_root),
        "fp_frame": FP_FRAME_ID,
        "est_iter": est_iter,
        "H_fp": scene_info.H_fp,
        "W_fp": scene_info.W_fp,
        "H_orig": scene_info.H_orig,
        "W_orig": scene_info.W_orig,
        "scale_fp": scene_info.scale_fp,
        "shorter_side": DEFAULT_SHORTER_SIDE,
        "K_fp": scene_info.K_fp.tolist(),
        "K_orig": scene_info.K_orig.tolist(),
        "mask_coverage_pct": round(scene_info.mask_coverage_pct, 2),
        "mesh_faces": n_faces,
        "mesh_file": rel_mesh,
        "mesh_file_aligned": rel_aligned,
        "mesh_align": align_result.R_aligned_residual,
        "mesh_align_max_err_m": align_result.max_point_err_m,
        "warnings": warnings,
        "keep_fp_scene": keep_fp_scene,
    }
    (reg / "foundationpose_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )

    vis_dst = dirs.output_rel("vis", "T5_foundationpose_overlay.png")
    vis_dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        from fp_visualize import save_foundationpose_comparison  # noqa: WPS433
        from scale_common import load_depth_m, load_mask_bool  # noqa: E402

        rgb_full = load_rgb_np(dirs.input_rel("rgb", "left_rgb.png"))
        mask = load_mask_bool(dirs.output_rel("segment", "mask.png"))
        depth_m = load_depth_m(dirs.input_rel("depth", "depth.npy"))
        save_foundationpose_comparison(
            rgb_full,
            scene_info.K_orig,
            vis_dst,
            fp_root=fp_root,
            session_id=dirs.session_id,
            est_iter=est_iter,
            mask=mask,
            depth_m=depth_m,
            sam3d_T_cam_mesh=T_cam_mesh_fp,
            sam3d_mesh_path=mesh_path,
            aligned_T_cam_mesh=T_cam_mesh,
            aligned_mesh_path=aligned_glb,
            T_base_mesh_aligned=T_base_mesh,
            align_R_residual=align_result.R_aligned_residual,
            T_base_cam=T_base_cam,
        )
        meta["vis_layout"] = "2x4: row0 ①②③④ SAM3D/FP | row1 ⑤ align RGB+axes | ⑥ 3D base"
    except Exception as exc:
        meta["vis_error"] = str(exc)
        if vis_src and vis_src.is_file():
            shutil.copy2(vis_src, vis_dst)
        else:
            meta["overlay_missing"] = True

    if not keep_fp_scene and scene_info.scene_dir.is_dir():
        shutil.rmtree(scene_info.scene_dir, ignore_errors=True)


def resolve_mesh_path(dirs: SessionDirs, rel: str) -> Path:
    """Resolve mesh path from register JSON (tolerate missing output/ prefix)."""
    p = (dirs.session_root / rel).resolve()
    if p.is_file():
        return p
    name = Path(rel).name
    for cand in [
        dirs.output_rel("mesh", name),
        dirs.session_root / "mesh" / name,
    ]:
        if cand.is_file():
            return cand
    return p


def registration_done(dirs: SessionDirs) -> bool:
    return (dirs.output_rel("register", "T_cam_mesh.json")).is_file()


def run_mesh_align_step(
    dirs: SessionDirs,
    mesh_path: Path | None = None,
) -> "MeshAlignResult":
    """Apply base-axis alignment using saved FP transforms (no FP inference)."""
    from mesh_align import align_mesh_to_base, write_mesh_frame_align_json  # noqa: WPS433

    reg = dirs.output_rel("register")
    fp_path = reg / "T_cam_mesh_fp.json"
    if not fp_path.is_file():
        raise FileNotFoundError(f"Missing {fp_path}; run T5 register first")
    fp_cam = json.loads(fp_path.read_text())
    T_cam_fp = np.asarray(fp_cam["T_cam_mesh"], dtype=np.float64)
    if mesh_path is None:
        mesh_path = resolve_mesh_path(
            dirs, fp_cam.get("mesh_file", "output/mesh/object_scaled.glb")
        )
    aligned_glb = dirs.output_rel("mesh", "object_base_aligned.glb")
    result = align_mesh_to_base(
        mesh_path, T_cam_fp, load_T_base_cam(dirs.input_dir), aligned_glb
    )
    write_mesh_frame_align_json(reg / "mesh_frame_align.json", result)

    T_base_cam = load_T_base_cam(dirs.input_dir)
    rel_aligned = str(aligned_glb.relative_to(dirs.session_root))
    (reg / "T_cam_mesh.json").write_text(
        json.dumps(
            {
                "camera_frame": "zed_left_camera",
                "mesh_frame": "base_aligned",
                "T_cam_mesh": result.T_cam_mesh_aligned.tolist(),
                "method": "foundationpose",
                "fp_frame": FP_FRAME_ID,
                "mesh_file": rel_aligned,
                "source_fp": "register/T_cam_mesh_fp.json",
            },
            indent=2,
        )
        + "\n"
    )
    (reg / "T_base_mesh.json").write_text(
        json.dumps(
            {
                "base_frame": "base",
                "mesh_frame": "base_aligned",
                "T_base_mesh": result.T_base_mesh_aligned.tolist(),
                "method": "foundationpose",
                "composition": "T_base_mesh = T_base_cam @ T_cam_mesh",
            },
            indent=2,
        )
        + "\n"
    )
    return result
