"""Export PDM / grasp HDF5 to demo ``candidates.json`` for Razor."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import trimesh

from model.pdm.pose_codec import TCP_OFFSET


def _load_json_matrix(path: Path, key: str) -> np.ndarray:
    data = json.loads(path.read_text())
    return np.asarray(data[key], dtype=np.float64).reshape(4, 4)


def _mesh_span_m(mesh_path: Path) -> list[float]:
    mesh = trimesh.load(str(mesh_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    ext = np.asarray(mesh.vertices, dtype=np.float64).max(axis=0) - np.asarray(
        mesh.vertices, dtype=np.float64
    ).min(axis=0)
    return [float(x) for x in ext.tolist()]


def _affordance_summary(
    affordance_npz: Path | None, threshold: float
) -> dict[str, Any]:
    if affordance_npz is None or not affordance_npz.is_file():
        return {
            "num_points": 0,
            "contact_threshold": float(threshold),
            "n_contact_points": 0,
            "force_center": None,
        }
    z = np.load(str(affordance_npz), allow_pickle=False)
    pred = np.asarray(z["pred"], dtype=np.float32).reshape(-1)
    n = int(len(pred))
    n_contact = int((pred > threshold).sum())
    out: dict[str, Any] = {
        "num_points": n,
        "contact_threshold": float(threshold),
        "n_contact_points": n_contact,
        "force_center": None,
    }
    if "force_center" in z.files:
        fc = np.asarray(z["force_center"], dtype=np.float64).reshape(3)
        if np.isfinite(fc).all():
            out["force_center"] = [float(x) for x in fc.tolist()]
    return out


def export_candidates_json(
    h5_path: Path,
    out_path: Path,
    *,
    mesh_path: Path,
    T_cam_mesh: np.ndarray,
    T_base_mesh: np.ndarray,
    camera_frame: str = "zed_left_camera",
    mesh_frame: str = "base_aligned",
    affordance_npz: Path | None = None,
    affordance_threshold: float = 0.5,
    extrinsics_source: str = "input/calib/extrinsics.json",
) -> dict[str, Any]:
    """Write ``candidates.json`` from PDM grasp HDF5."""
    h5_path = Path(h5_path).resolve()
    out_path = Path(out_path).resolve()
    mesh_path = Path(mesh_path).resolve()

    candidates: list[dict[str, Any]] = []
    with h5py.File(h5_path, "r") as f:
        cg = f.get("candidates")
        if cg is None:
            raise ValueError(f"No candidates/ group in {h5_path}")
        n = int(cg.attrs.get("n_candidates", 0))
        keys = sorted(
            [k for k in cg.keys() if k.startswith("candidate_")],
            key=lambda k: int(k.split("_")[-1]),
        )
        for rank, key in enumerate(keys):
            gi = cg[key]
            R = np.asarray(gi["rotation"], dtype=np.float64).reshape(3, 3)
            gp = np.asarray(gi["grasp_point"], dtype=np.float64).reshape(3)
            pos = (
                np.asarray(gi["position"], dtype=np.float64).reshape(3)
                if "position" in gi
                else gp
            )
            score = float(gi.attrs.get("score", n - rank))
            candidates.append(
                {
                    "rank": rank,
                    "name": str(gi.attrs.get("name", key)),
                    "score": score,
                    "grasp_point": [float(x) for x in gp.tolist()],
                    "rotation": R.tolist(),
                    "position_panda_hand": [float(x) for x in pos.tolist()],
                    "gripper_width_m": float(gi.attrs.get("gripper_width", 0.06)),
                    "cross_section_width_m": None,
                    "approach_type": str(gi.attrs.get("approach_type", "pdm")),
                }
            )

    candidates.sort(key=lambda c: c["score"], reverse=True)
    for i, c in enumerate(candidates):
        c["rank"] = i

    T_cam_mesh = np.asarray(T_cam_mesh, dtype=np.float64).reshape(4, 4)
    T_base_mesh = np.asarray(T_base_mesh, dtype=np.float64).reshape(4, 4)

    payload: dict[str, Any] = {
        "schema_version": "1.1",
        "mesh_frame": mesh_frame,
        "base_frame": "base",
        "camera_frame": camera_frame,
        "inference_method": "pdm",
        "registration": {
            "method": "foundationpose",
            "T_cam_mesh": T_cam_mesh.tolist(),
            "T_base_mesh": T_base_mesh.tolist(),
            "T_base_cam_source": extrinsics_source,
        },
        "T_base_mesh": T_base_mesh.tolist(),
        "conventions": {
            "rotation_columns": ["finger_open", "y_body", "approach"],
            "approach_column_index": 2,
            "grasp_point_frame": mesh_frame,
            "ucb_tcp_offset_m": float(TCP_OFFSET),
            "ucb_tcp_frame": "panda_hand",
            "pre_grasp_offset_m": 0.15,
            "lift_height_m": 0.15,
            "pdm_command_frame": "mesh_tcp_center",
        },
        "mesh_span_m": _mesh_span_m(mesh_path),
        "mesh_file": "output/mesh/object_base_aligned.glb",
        "n_candidates": len(candidates),
        "candidates": candidates,
        "affordance": _affordance_summary(affordance_npz, affordance_threshold),
        "exported_at_iso": datetime.now(timezone.utc).isoformat(),
        "source_hdf5": h5_path.name,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload
