"""Rebuild ``T6_grasp_vis.png`` from existing HDF5 + affordance NPZ."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from _session_io import SessionDirs
from model.inference_v6 import normalize_affordance_pred
from vis_dual import save_t6_dual_vis


def rebuild_t6_vis(
    dirs: SessionDirs,
    *,
    mesh_top: int = 20,
    scene_top: int = 10,
) -> Path:
    h5 = dirs.output_rel("inference", "affordance_grasp.hdf5")
    if not h5.is_file():
        raise FileNotFoundError(h5)
    npz_path = dirs.output_rel("inference", "affordance", "npz", f"{dirs.session_id}.npz")
    if not npz_path.is_file():
        raise FileNotFoundError(
            f"Missing {npz_path} — run full T6 first or pass --no-affordance-sidecar only on re-run"
        )
    z = np.load(str(npz_path), allow_pickle=False)
    points = np.asarray(z["points"], dtype=np.float64)
    if "pred_norm" in z.files:
        pred_norm = np.asarray(z["pred_norm"], dtype=np.float32).reshape(-1)
    else:
        pred_norm, _ = normalize_affordance_pred(np.asarray(z["pred"], dtype=np.float32))
    out = dirs.output_rel("vis", "T6_grasp_vis.png")
    return save_t6_dual_vis(
        dirs,
        h5,
        points,
        pred_norm,
        out,
        mesh_top=mesh_top,
        scene_top=scene_top,
    )
