"""Run PDM grasp pipeline on a Phase-2 session (T5 outputs)."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from _session_io import SessionDirs  # noqa: E402
from export_candidates import export_candidates_json  # noqa: E402
from grasp_filters import (  # noqa: E402
    filter_poses9_dexmate_approach_sector,
    sector_filter_meta,
)
from evaluation.affordance_ckpt import resolve_affordance_checkpoint  # noqa: E402
from model.inference_v6 import (  # noqa: E402
    default_threshold,
    load_model,
    normalize_affordance_pred,
    predict_heatmap_batch,
)
from model.pdm.model import PDM  # noqa: E402
from model.pdm.sample import write_candidates_hdf5  # noqa: E402
from vis_dual import save_t6_dual_vis  # noqa: E402
from tools.glb_to_pdm_grasp import (  # noqa: E402
    build_condition_tensor,
    prepare_mesh_item,
    run_pdm_sample,
    save_affordance_outputs,
)

DEFAULT_AFF_CKPT = (
    _REPO / "output" / "affordance_no_rot_executed" / "min20" / "checkpoints_v6" / "best_v6_model.pth"
)
DEFAULT_PDM_CKPT = _REPO / "output" / "pdm" / "checkpoints_yaw_v6cond" / "best_model.pth"


@dataclass(frozen=True)
class PdmSessionResult:
    h5_path: Path
    candidates_path: Path
    meta_path: Path
    vis_path: Path | None
    n_candidates: int
    affordance_npz: Path | None


def _load_registration(dirs: SessionDirs) -> tuple[np.ndarray, np.ndarray, str]:
    reg = dirs.output_rel("register")
    cam_path = reg / "T_cam_mesh.json"
    base_path = reg / "T_base_mesh.json"
    if not cam_path.is_file() or not base_path.is_file():
        raise FileNotFoundError(
            "Missing register/T_cam_mesh.json or T_base_mesh.json — run T5 first."
        )
    T_cam = np.asarray(
        json.loads(cam_path.read_text())["T_cam_mesh"], dtype=np.float64
    ).reshape(4, 4)
    T_base = np.asarray(
        json.loads(base_path.read_text())["T_base_mesh"], dtype=np.float64
    ).reshape(4, 4)
    ext_path = dirs.input_rel("calib", "extrinsics.json")
    camera_frame = "zed_left_camera"
    if ext_path.is_file():
        camera_frame = json.loads(ext_path.read_text()).get(
            "camera_frame", camera_frame
        )
    return T_cam, T_base, camera_frame


def _chain_check(dirs: SessionDirs, T_cam: np.ndarray, T_base: np.ndarray) -> float:
    ext_path = dirs.input_rel("calib", "extrinsics.json")
    if not ext_path.is_file():
        return 0.0
    T_bc = np.asarray(
        json.loads(ext_path.read_text())["T_base_cam"], dtype=np.float64
    ).reshape(4, 4)
    return float(np.max(np.abs(T_bc @ T_cam - T_base)))


def run_pdm_for_session(
    dirs: SessionDirs,
    *,
    aff_ckpt: Path | None = None,
    pdm_ckpt: Path | None = None,
    n_samples: int = 50,
    ddim_steps: int = 50,
    num_points: int = 4096,
    seed: int = 42,
    gripper_width: float = 0.06,
    z_yaw_deg: float | None = None,
    reject_upward: bool = False,
    max_approach_z: float = 0.3,
    dexmate_approach_sector: bool = True,
    approach_sector_min_horiz: float = 0.25,
    approach_sector_lo_deg: float = 225.0,
    approach_sector_hi_deg: float = 315.0,
    approach_sector_vertical_z_min: float = 0.85,
    approach_sector_allow_vertical: bool = True,
    approach_sector_oversample: float = 3.0,
    affordance_threshold: float | None = None,
    device: str | None = None,
    write_vis: bool = True,
    scene_vis_top: int = 10,
    mesh_vis_top: int = 20,
    write_affordance_sidecar: bool = True,
) -> PdmSessionResult:
    """
    PDM inference on ``object_base_aligned.glb`` without re-scaling or +X pre-rotate.

    Grasp poses are in the same mesh frame as T5 ``T_base_mesh`` / ``T_cam_mesh``.
    """
    mesh_path = dirs.output_rel("mesh", "object_base_aligned.glb")
    if not mesh_path.is_file():
        raise FileNotFoundError(
            f"Missing {mesh_path} — run T5 (base-align) before T6."
        )

    T_cam, T_base, camera_frame = _load_registration(dirs)
    chain_err = _chain_check(dirs, T_cam, T_base)
    if chain_err > 1e-3:
        raise RuntimeError(
            f"T_base_mesh != T_base_cam @ T_cam_mesh (max err {chain_err:.4e})"
        )

    aff_ckpt = Path(aff_ckpt or DEFAULT_AFF_CKPT).resolve()
    pdm_ckpt = Path(pdm_ckpt or DEFAULT_PDM_CKPT).resolve()
    if not aff_ckpt.is_file():
        raise FileNotFoundError(f"Affordance checkpoint not found: {aff_ckpt}")
    if not pdm_ckpt.is_file():
        raise FileNotFoundError(f"PDM checkpoint not found: {pdm_ckpt}")

    inf_dir = dirs.output_rel("inference")
    aff_dir = inf_dir / "affordance"
    vis_dir = dirs.output_rel("vis")
    h5_path = inf_dir / "affordance_grasp.hdf5"
    candidates_path = inf_dir / "candidates.json"
    meta_path = inf_dir / "pdm_meta.json"
    vis_path = vis_dir / "T6_grasp_vis.png" if write_vis else None

    dev = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    aff_thresh = affordance_threshold
    if aff_thresh is None:
        aff_thresh = default_threshold(str(aff_ckpt), str(aff_ckpt.parent.parent))

    t0 = time.perf_counter()
    item = prepare_mesh_item(
        mesh_path,
        obj_id_override=dirs.session_id,
        dataset="razor_session",
        sam3d_rotated_mesh=False,
        no_pre_rotate_x=True,
        pre_rotate_x_deg=90.0,
        scale_mode="never",
        ignore_scale_json=True,
        target_max_extent=0.30,
        auto_extent_lo=0.08,
        auto_extent_hi=0.38,
        min_scale_factor=0.45,
        no_center=True,
        num_points=num_points,
        seed=seed,
        index=0,
    )

    aff_model, _ = load_model(str(aff_ckpt), dev)
    pts = item["points"][np.newaxis, ...]
    nrm = item["normals"][np.newaxis, ...]
    pred = predict_heatmap_batch(aff_model, pts, nrm, dev)
    if pred.ndim == 1:
        pred = pred[np.newaxis, :]
    pred = pred[0].astype(np.float32)
    pred_norm, pred_norm_stats = normalize_affordance_pred(pred)

    aff_npz: Path | None = None
    if write_affordance_sidecar:
        aff_dir.mkdir(parents=True, exist_ok=True)
        raw_npz, _, _ = save_affordance_outputs(
            affordance_dir=aff_dir,
            item=item,
            pred=pred,
            pred_norm=pred_norm,
            pred_norm_scale=float(pred_norm_stats.get("pred_span", 0.0)),
            threshold=aff_thresh,
            no_aff_vis=True,
        )
        aff_npz = Path(raw_npz)

    pdm_model, pdm_meta = PDM.load(str(pdm_ckpt), device=dev)
    stats = pdm_meta.get("pose_stats")
    if stats is None:
        raise RuntimeError("PDM checkpoint missing pose_stats")

    condition = build_condition_tensor(item["points"], item["normals"], pred)
    n_pdm_draw = int(n_samples)
    if dexmate_approach_sector and approach_sector_oversample > 1.0:
        n_pdm_draw = max(n_pdm_draw, int(round(n_samples * approach_sector_oversample)))

    poses_np = run_pdm_sample(
        pdm_model,
        stats,
        condition,
        n_samples=n_pdm_draw,
        ddim_steps=ddim_steps,
        z_yaw_deg=z_yaw_deg,
        device=dev,
        reject_upward=reject_upward,
        max_approach_z=max_approach_z,
    )
    n_pdm_raw = int(len(poses_np))
    sector_result = None
    if dexmate_approach_sector and n_pdm_raw > 0:
        sector_result = filter_poses9_dexmate_approach_sector(
            poses_np,
            min_horiz_norm=approach_sector_min_horiz,
            lo_deg=approach_sector_lo_deg,
            hi_deg=approach_sector_hi_deg,
            vertical_z_min=approach_sector_vertical_z_min,
            allow_vertical_top_down=approach_sector_allow_vertical,
        )
        poses_np = sector_result.kept
        if len(poses_np) > n_samples:
            poses_np = poses_np[:n_samples]
    if poses_np.size == 0:
        msg = "PDM returned zero candidates after filtering"
        if sector_result is not None and sector_result.n_rejected > 0:
            msg += f" (sector filter rejected {sector_result.n_rejected}/{sector_result.n_in})"
        raise RuntimeError(msg)

    inf_dir.mkdir(parents=True, exist_ok=True)
    write_candidates_hdf5(
        str(h5_path),
        dirs.session_id,
        poses_np,
        mesh_path=str(mesh_path.resolve()),
        gripper_width=gripper_width,
        dataset="razor_session",
    )

    if write_vis and vis_path is not None:
        vis_dir.mkdir(parents=True, exist_ok=True)
        save_t6_dual_vis(
            dirs,
            h5_path,
            item["points"],
            pred_norm,
            vis_path,
            mesh_top=min(mesh_vis_top, len(poses_np)),
            scene_top=scene_vis_top,
        )

    export_candidates_json(
        h5_path,
        candidates_path,
        mesh_path=mesh_path,
        T_cam_mesh=T_cam,
        T_base_mesh=T_base,
        camera_frame=camera_frame,
        mesh_frame="base_aligned",
        affordance_npz=aff_npz,
        affordance_threshold=aff_thresh,
    )

    srep = item["scale_report"]
    meta: dict[str, Any] = {
        "session_id": dirs.session_id,
        "method": "pdm_v6",
        "mesh_file": "output/mesh/object_base_aligned.glb",
        "affordance_ckpt": str(aff_ckpt),
        "pdm_ckpt": str(pdm_ckpt),
        "n_samples": int(n_samples),
        "ddim_steps": int(ddim_steps),
        "num_points": int(num_points),
        "seed": int(seed),
        "gripper_width_m": float(gripper_width),
        "z_yaw_deg": z_yaw_deg,
        "pdm_use_yaw_condition": bool(pdm_model.config.use_yaw_condition),
        "n_pdm_sampled": n_pdm_raw,
        "n_pdm_draw": n_pdm_draw,
        "n_samples_requested": int(n_samples),
        "n_candidates": int(len(poses_np)),
        "affordance_threshold": float(aff_thresh),
        "reject_upward": bool(reject_upward),
        "max_approach_z": float(max_approach_z),
        "filters": {
            "approach_sector": (
                sector_filter_meta(
                    sector_result,
                    enabled=True,
                    min_horiz_norm=approach_sector_min_horiz,
                    lo_deg=approach_sector_lo_deg,
                    hi_deg=approach_sector_hi_deg,
                    use_arrival_direction=True,
                    vertical_z_min=approach_sector_vertical_z_min,
                    allow_vertical_top_down=approach_sector_allow_vertical,
                )
                if sector_result is not None
                else {
                    "enabled": False,
                    "name": "dexmate_right_approach_xy_sector",
                }
            ),
        },
        "mesh_prepare": {
            "no_pre_rotate_x": True,
            "scale_mode": "never",
            "no_center": True,
            "ignore_scale_json": True,
            "max_extent_m": float(srep.max_extent_after),
        },
        "register_chain_err": chain_err,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "outputs": {
            "hdf5": "output/inference/affordance_grasp.hdf5",
            "candidates_json": "output/inference/candidates.json",
            "affordance_npz": str(aff_npz.relative_to(dirs.output_dir))
            if aff_npz and aff_npz.is_relative_to(dirs.output_dir)
            else None,
            "vis_png": "output/vis/T6_grasp_vis.png" if write_vis else None,
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    return PdmSessionResult(
        h5_path=h5_path,
        candidates_path=candidates_path,
        meta_path=meta_path,
        vis_path=vis_path,
        n_candidates=int(len(poses_np)),
        affordance_npz=aff_npz,
    )
