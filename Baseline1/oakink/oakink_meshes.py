"""OakInk CAD-mesh point-cloud loader.

Uses the OFFICIAL OakInk CAD meshes at
    data_hub/RawData/ThirdPersonRawData/oakink_v1/image/obj/{obj_id}.obj
which are the canonical frame for OakInk's per-frame `obj_transf` (camera-pose
of object) and `obj_anno` (world-pose of object). Sampling these and then
applying `obj_anno` gives a world-frame PC at the correct physical size.

We deliberately do NOT use the SAM3D-reconstructed mesh at
    data_hub/ProcessedData/obj_meshes/oakink/{obj_id}/mesh.ply
or the SAM3D→CAD alignment under Baseline1/assets/sam3d_align/oakink/. Reasons:
  1. The plan is to ship the v4 baseline on CAD meshes; SAM3D switch is a later
     phase.
  2. The current SAM3D loader (Baseline1.retarget_human_to_ee.get_object_points)
     applies BOTH `scale.json` AND `R_align_4x4` for OakInk objects. R_align has
     the SAM3D→CAD scale (≈0.216) folded in, and `scale.json` adds ~0.193 on top
     — double-scaling shrinks the PC ≈5× (verified on A01001 → 1.6×2.1×4.2 cm
     vs the correct CAD size 6.5×10.2×22.8 cm). Using the CAD mesh directly
     side-steps the whole alignment chain.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import trimesh

from Baseline1.oakink.oakink_paths import OAKINK_OBJ_DIR

_CACHE: dict = {}


def get_oakink_object_points(obj_id: str, n_points: int = 4096) -> Optional[np.ndarray]:
    """(n_points, 3) surface samples of OakInk's official CAD mesh in canonical
    frame (the same frame `obj_transf` / `obj_anno` are defined against).

    Args:
      obj_id   : OakInk obj_id, e.g. 'A01001'
      n_points : number of surface samples

    Returns:
      ndarray (n_points, 3) float32, or None if the mesh file is missing.
    """
    key = (obj_id, int(n_points))
    if key in _CACHE:
        return _CACHE[key]

    mesh_path = os.path.join(OAKINK_OBJ_DIR, f"{obj_id}.obj")
    if not os.path.exists(mesh_path):
        _CACHE[key] = None
        return None

    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    pts, _ = trimesh.sample.sample_surface(mesh, n_points)
    pts32 = pts.astype(np.float32)
    _CACHE[key] = pts32
    return pts32
