#!/usr/bin/env python3
"""Dataset utilities for PDM.

The dataset reads successful merged robot GT files and converts
`executed_panda_hand_at_close` wrist poses into 9D command-pose targets.
Object conditions are point clouds with channels: xyz, normal, affordance.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from typing import Iterable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .pose_codec import command_to_pose9, executed_to_command, is_valid_rotation

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_MERGED_DIR = os.path.join(PROJ, "output", "grasp_collect_no_rot", "merged")
DEFAULT_ROTATED_MESH_DIR = os.path.join(
    PROJ, "data_hub", "meshes", "SAM3DMesh", "rotated_mesh"
)
DEFAULT_OBJ_MESH_DIR = os.path.join(PROJ, "data_hub", "ProcessedData", "obj_meshes")

DATASET_DIRS = ("oakink", "ycb", "dexycb", "arctic", "egocentric", "ho3d_v3")

_PLY_DTYPE = {
    "char": ("i1", "b", 1),
    "uchar": ("u1", "B", 1),
    "int8": ("i1", "b", 1),
    "uint8": ("u1", "B", 1),
    "short": ("i2", "h", 2),
    "ushort": ("u2", "H", 2),
    "int16": ("i2", "h", 2),
    "uint16": ("u2", "H", 2),
    "int": ("i4", "i", 4),
    "uint": ("u4", "I", 4),
    "int32": ("i4", "i", 4),
    "uint32": ("u4", "I", 4),
    "float": ("f4", "f", 4),
    "float32": ("f4", "f", 4),
    "double": ("f8", "d", 8),
    "float64": ("f8", "d", 8),
}


@dataclass(frozen=True)
class PDMSampleMeta:
    obj_id: str
    merged_path: str
    grasp_key: str
    score: float
    source_file: str
    trusted_tips: bool
    yaw_deg: float


@dataclass
class ObjectCondition:
    points: np.ndarray  # (N, 7): xyz, normal, affordance
    mesh_path: str | None = None


def _decode_obj_ids(raw: np.ndarray) -> list[str]:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in raw]


def yaw_feature_from_deg(yaw_deg: float) -> np.ndarray:
    """Return yaw condition [sin(theta), cos(theta)] for degrees."""

    rad = np.deg2rad(float(yaw_deg))
    return np.array([np.sin(rad), np.cos(rad)], dtype=np.float32)


def _normalize_name(x) -> str:
    return x.decode() if isinstance(x, bytes) else str(x)


def recover_yaw_deg_from_source(
    source_file: str,
    *,
    grasp_name: str = "",
    pool_candidate_key: str = "",
) -> float:
    """Recover sim z-yaw for a merged successful grasp.

    Priority:
      1. Per-grasp `sim_z_yaw_deg` in source robot_gt (pool sim, 0/90/180/270).
      2. Root `sim_z_yaw_deg` in source robot_gt (legacy one yaw per round/object).
      3. Missing yaw fields => 0 degrees (no extra sim yaw).
    """

    if not source_file or not os.path.isfile(source_file):
        return 0.0
    try:
        with h5py.File(source_file, "r") as f:
            sg = f.get("successful_grasps")
            if sg is not None:
                for key in sg.keys():
                    g = sg[key]
                    name_match = grasp_name and _normalize_name(g.attrs.get("name", "")) == grasp_name
                    key_match = (
                        pool_candidate_key
                        and _normalize_name(g.attrs.get("pool_candidate_key", "")) == pool_candidate_key
                    )
                    if (name_match or key_match) and "sim_z_yaw_deg" in g.attrs:
                        return float(g.attrs["sim_z_yaw_deg"])
            if "sim_z_yaw_deg" in f.attrs:
                return float(f.attrs["sim_z_yaw_deg"])
    except OSError:
        return 0.0
    return 0.0


def _infer_dataset(obj_id: str) -> str:
    if obj_id.startswith("ycb_dex_"):
        return "ycb"
    if obj_id.startswith("arctic_"):
        return "arctic"
    return "oakink"


def find_mesh_path(obj_id: str, mesh_root: str = DEFAULT_ROTATED_MESH_DIR) -> str | None:
    """Find a mesh path compatible with the no-rotation merged corpus."""

    candidates = []
    inferred = _infer_dataset(obj_id)
    if obj_id.startswith("ycb_dex_"):
        candidates.append(os.path.join(mesh_root, "ycb", obj_id, "mesh.ply"))
    candidates.append(os.path.join(mesh_root, inferred, obj_id, "mesh.ply"))
    for ds in DATASET_DIRS:
        candidates.append(os.path.join(mesh_root, ds, obj_id, "mesh.ply"))
    for ds in DATASET_DIRS:
        candidates.append(os.path.join(DEFAULT_OBJ_MESH_DIR, ds, obj_id, "mesh.ply"))
    return next((p for p in candidates if os.path.isfile(p)), None)


def _mesh_scale_json_path(obj_id: str) -> str | None:
    """Find scale.json with the same convention as random_grasp_sampler."""

    if obj_id.startswith("ycb_dex_"):
        path = os.path.join(DEFAULT_OBJ_MESH_DIR, "ycb", obj_id, "scale.json")
        return path if os.path.isfile(path) else None
    for ds in DATASET_DIRS:
        path = os.path.join(DEFAULT_OBJ_MESH_DIR, ds, obj_id, "scale.json")
        if os.path.isfile(path):
            return path
    return None


def _read_scale_factor(obj_id: str) -> float:
    import json

    path = _mesh_scale_json_path(obj_id)
    if path is None:
        return 1.0
    with open(path) as f:
        return float(json.load(f).get("scale_factor", 1.0))


def _apply_metric_scale_to_mesh(obj_id: str) -> bool:
    if obj_id.startswith("arctic_"):
        return False
    return abs(_read_scale_factor(obj_id) - 1.0) > 1e-8


def _load_sampler_style_mesh(mesh_path: str):
    """Load mesh with the same robust fallback order as random_grasp_sampler."""

    import trimesh

    last_err: Exception | None = None
    for kwargs in (
        {"force": "mesh", "process": False, "skip_materials": True},
        {"force": "mesh", "process": False},
        {"process": False},
    ):
        try:
            loaded = trimesh.load(mesh_path, **kwargs)
            if isinstance(loaded, trimesh.Scene):
                geoms = [g for g in loaded.dump() if isinstance(g, trimesh.Trimesh)]
                if not geoms:
                    raise ValueError("empty scene mesh")
                loaded = trimesh.util.concatenate(geoms)
            if not isinstance(loaded, trimesh.Trimesh):
                raise TypeError(f"expected Trimesh, got {type(loaded)}")
            return loaded
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"mesh load failed: {mesh_path}") from last_err


def _safe_mesh_repair(mesh, label: str = "mesh") -> None:
    """Best-effort repair mirroring random_grasp_sampler."""

    import trimesh

    try:
        if not mesh.is_watertight:
            trimesh.repair.fill_holes(mesh)
            trimesh.repair.fix_normals(mesh)
    except (IndexError, ValueError, MemoryError, RuntimeError):
        try:
            trimesh.repair.fix_normals(mesh)
        except Exception:
            pass


def _read_ply_header(f) -> tuple[str, int, int, list, list]:
    lines = []
    while True:
        line = f.readline()
        if not line:
            raise EOFError("PLY missing end_header")
        lines.append(line.decode("ascii", errors="replace").strip())
        if lines[-1] == "end_header":
            break

    fmt = "ascii"
    n_vertices = 0
    n_faces = 0
    vertex_props = []
    face_props = []
    current = None
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            current = parts[1]
            if current == "vertex":
                n_vertices = int(parts[2])
            elif current == "face":
                n_faces = int(parts[2])
        elif parts[0] == "property":
            if current == "vertex":
                vertex_props.append(parts[1:])
            elif current == "face":
                face_props.append(parts[1:])
    return fmt, n_vertices, n_faces, vertex_props, face_props


def _load_ply_arrays(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load vertices/faces from a PLY while ignoring visual/color data.

    This avoids `trimesh.load(... force="mesh")` scene/visual conversion paths,
    which are fragile for a few large generated meshes under DataLoader workers.
    """

    with open(path, "rb") as f:
        fmt, n_vertices, n_faces, vertex_props, face_props = _read_ply_header(f)
        if fmt == "ascii":
            vertex_rows = []
            xyz_idx = [
                i
                for i, prop in enumerate(vertex_props)
                if len(prop) >= 2 and prop[-1] in ("x", "y", "z")
            ][:3]
            if len(xyz_idx) != 3:
                xyz_idx = [0, 1, 2]
            for _ in range(n_vertices):
                vals = f.readline().split()
                vertex_rows.append([float(vals[i]) for i in xyz_idx])
            faces = []
            for _ in range(n_faces):
                vals = f.readline().split()
                if not vals:
                    continue
                count = int(vals[0])
                idx = [int(v) for v in vals[1 : 1 + count]]
                if count >= 3:
                    for j in range(1, count - 1):
                        faces.append([idx[0], idx[j], idx[j + 1]])
            return np.asarray(vertex_rows, dtype=np.float32), np.asarray(faces, dtype=np.int64)

        if fmt != "binary_little_endian":
            raise ValueError(f"unsupported PLY format: {fmt}")

        dtype_fields = []
        for prop in vertex_props:
            if prop[0] == "list":
                raise ValueError("list vertex properties are not supported")
            dtype_fields.append((prop[1], "<" + _PLY_DTYPE[prop[0]][0]))
        vertex_data = np.fromfile(f, dtype=np.dtype(dtype_fields), count=n_vertices)
        vertices = np.stack(
            [
                np.asarray(vertex_data["x"], dtype=np.float32),
                np.asarray(vertex_data["y"], dtype=np.float32),
                np.asarray(vertex_data["z"], dtype=np.float32),
            ],
            axis=1,
        )

        if not face_props or face_props[0][0] != "list":
            return vertices, np.zeros((0, 3), dtype=np.int64)

        count_type = face_props[0][1]
        index_type = face_props[0][2]
        count_fmt = "<" + _PLY_DTYPE[count_type][1]
        index_fmt = "<" + _PLY_DTYPE[index_type][1]
        count_size = _PLY_DTYPE[count_type][2]
        index_size = _PLY_DTYPE[index_type][2]
        faces = []
        extra_face_props = face_props[1:]
        for _ in range(n_faces):
            raw = f.read(count_size)
            if len(raw) < count_size:
                break
            count = struct.unpack(count_fmt, raw)[0]
            raw_idx = f.read(index_size * count)
            if len(raw_idx) < index_size * count:
                break
            idx = list(struct.unpack("<" + (_PLY_DTYPE[index_type][1] * count), raw_idx))
            for prop in extra_face_props:
                if prop[0] == "list":
                    raw_count = f.read(_PLY_DTYPE[prop[1]][2])
                    if not raw_count:
                        break
                    n_extra = struct.unpack("<" + _PLY_DTYPE[prop[1]][1], raw_count)[0]
                    f.seek(_PLY_DTYPE[prop[2]][2] * n_extra, 1)
                else:
                    f.seek(_PLY_DTYPE[prop[0]][2], 1)
            if count >= 3:
                for j in range(1, count - 1):
                    faces.append([idx[0], idx[j], idx[j + 1]])
        return vertices, np.asarray(faces, dtype=np.int64)


def _sample_surface_arrays(
    vertices: np.ndarray,
    faces: np.ndarray,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample points and normals from raw mesh arrays."""

    vertices = np.ascontiguousarray(np.asarray(vertices, dtype=np.float32))
    faces = np.ascontiguousarray(np.asarray(faces, dtype=np.int64))
    if faces.size > 0:
        try:
            valid = (
                np.isfinite(faces).all(axis=1)
                & (faces >= 0).all(axis=1)
                & (faces < len(vertices)).all(axis=1)
            )
            valid_idx = np.nonzero(np.asarray(valid, dtype=bool))[0]
            faces = np.take(faces, valid_idx, axis=0)
        except Exception:
            faces = np.zeros((0, 3), dtype=np.int64)

    if len(faces) == 0:
        idx = np.random.choice(len(vertices), n_points, replace=len(vertices) < n_points)
        pts = vertices[idx]
        normals = pts - vertices.mean(axis=0, keepdims=True)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
        return pts.astype(np.float32), normals.astype(np.float32)

    try:
        tri = vertices[faces]
    except Exception:
        return _sample_surface_arrays(vertices, np.zeros((0, 3), dtype=np.int64), n_points)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area2 = np.linalg.norm(cross, axis=1)
    valid_area = area2 > 1e-12
    if not bool(valid_area.all()):
        tri = tri[valid_area]
        cross = cross[valid_area]
        area2 = area2[valid_area]
    if len(tri) == 0:
        return _sample_surface_arrays(vertices, np.zeros((0, 3), dtype=np.int64), n_points)

    prob = area2 / area2.sum()
    face_idx = np.random.choice(len(tri), size=n_points, replace=True, p=prob)
    chosen = tri[face_idx]
    u = np.random.rand(n_points, 1).astype(np.float32)
    v = np.random.rand(n_points, 1).astype(np.float32)
    flip = (u + v) > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    pts = chosen[:, 0] + u * (chosen[:, 1] - chosen[:, 0]) + v * (chosen[:, 2] - chosen[:, 0])
    normals_all = cross / (np.linalg.norm(cross, axis=1, keepdims=True) + 1e-8)
    normals = normals_all[face_idx]
    return pts.astype(np.float32), normals.astype(np.float32)


def _normals_for_sampled_faces(mesh, face_idx: np.ndarray) -> np.ndarray:
    """Compute sampled face normals without touching trimesh cached face_normals."""

    vertices = np.ascontiguousarray(np.asarray(mesh.vertices, dtype=np.float32))
    faces = np.ascontiguousarray(np.asarray(mesh.faces, dtype=np.int64))
    face_idx = np.ascontiguousarray(np.asarray(face_idx, dtype=np.int64))
    if len(faces) == 0:
        raise ValueError("mesh has no faces")
    valid_face = (
        np.isfinite(faces).all(axis=1)
        & (faces >= 0).all(axis=1)
        & (faces < len(vertices)).all(axis=1)
    )
    if not bool(valid_face.all()):
        raise ValueError("mesh has invalid faces")
    tri = vertices[faces[face_idx]]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    return normals.astype(np.float32)


class AffordanceStore:
    """Optional source of aligned point clouds and affordance values."""

    def __init__(self, h5_path: str | None):
        self.h5_path = h5_path
        self.index: dict[str, int] = {}
        if not h5_path:
            return
        with h5py.File(h5_path, "r") as f:
            if "data/obj_ids" not in f:
                raise KeyError(f"{h5_path} missing data/obj_ids")
            self.index = {oid: i for i, oid in enumerate(_decode_obj_ids(f["data/obj_ids"][:]))}

    def has(self, obj_id: str) -> bool:
        return bool(self.h5_path and obj_id in self.index)

    def load(self, obj_id: str) -> ObjectCondition:
        if not self.h5_path or obj_id not in self.index:
            raise KeyError(obj_id)
        idx = self.index[obj_id]
        with h5py.File(self.h5_path, "r") as f:
            pts = f["data/points"][idx].astype(np.float32)
            normals = f["data/normals"][idx].astype(np.float32)
            if "data/soft_labels" in f:
                aff = f["data/soft_labels"][idx].astype(np.float32)
            elif "data/labels" in f:
                aff = f["data/labels"][idx].astype(np.float32)
            elif "data/robot_gt" in f:
                aff = f["data/robot_gt"][idx].astype(np.float32)
            else:
                aff = np.zeros((pts.shape[0],), dtype=np.float32)
        channels = np.concatenate([pts, normals, aff[:, None]], axis=-1)
        return ObjectCondition(points=channels)


class PDMConditionStore:
    """Precomputed PDM condition cache.

    Expected layout mirrors the affordance HDF5 convention:
      data/points, data/normals, data/affordance, data/obj_ids
    """

    def __init__(self, h5_path: str | None):
        self.h5_path = h5_path
        self.index: dict[str, int] = {}
        if not h5_path:
            return
        with h5py.File(h5_path, "r") as f:
            if "data/obj_ids" not in f:
                raise KeyError(f"{h5_path} missing data/obj_ids")
            self.index = {oid: i for i, oid in enumerate(_decode_obj_ids(f["data/obj_ids"][:]))}

    def has(self, obj_id: str) -> bool:
        return bool(self.h5_path and obj_id in self.index)

    def load(self, obj_id: str) -> ObjectCondition:
        if not self.h5_path or obj_id not in self.index:
            raise KeyError(obj_id)
        idx = self.index[obj_id]
        with h5py.File(self.h5_path, "r") as f:
            pts = f["data/points"][idx].astype(np.float32)
            normals = f["data/normals"][idx].astype(np.float32)
            if "data/affordance" in f:
                aff = f["data/affordance"][idx].astype(np.float32)
            elif "data/soft_labels" in f:
                aff = f["data/soft_labels"][idx].astype(np.float32)
            elif "data/labels" in f:
                aff = f["data/labels"][idx].astype(np.float32)
            else:
                aff = np.zeros((pts.shape[0],), dtype=np.float32)
        channels = np.concatenate([pts, normals, aff[:, None]], axis=-1)
        return ObjectCondition(points=channels)


def sample_mesh_condition(
    obj_id: str,
    n_points: int,
    mesh_root: str = DEFAULT_ROTATED_MESH_DIR,
) -> ObjectCondition:
    """Sample mesh points and normals, with zero affordance as fallback.

    The primary path mirrors `tools/random_grasp_sampler.py`: robust trimesh
    load fallback order, metric scale application, and best-effort repair. The
    raw PLY parser remains as a last-resort fallback for assets that still trip
    trimesh visual/material code paths.
    """

    mesh_path = find_mesh_path(obj_id, mesh_root=mesh_root)
    if mesh_path is None:
        raise FileNotFoundError(f"mesh not found for {obj_id}")

    try:
        import trimesh

        mesh = _load_sampler_style_mesh(mesh_path)
        scale_factor = _read_scale_factor(obj_id)
        if _apply_metric_scale_to_mesh(obj_id):
            mesh.vertices = mesh.vertices * float(scale_factor)
        _safe_mesh_repair(mesh, "pdm_mesh")
        pts, face_idx = trimesh.sample.sample_surface(mesh, n_points)
        normals = _normals_for_sampled_faces(mesh, face_idx)
    except Exception:
        vertices, faces = _load_ply_arrays(mesh_path)
        if _apply_metric_scale_to_mesh(obj_id):
            vertices = vertices * float(_read_scale_factor(obj_id))
        pts, normals = _sample_surface_arrays(vertices, faces, n_points)

    aff = np.zeros((n_points, 1), dtype=np.float32)
    channels = np.concatenate(
        [pts.astype(np.float32), normals.astype(np.float32), aff],
        axis=-1,
    )
    return ObjectCondition(points=channels, mesh_path=mesh_path)


def _resample_rows(arr: np.ndarray, n_points: int) -> np.ndarray:
    if arr.shape[0] == n_points:
        return arr.astype(np.float32)
    idx = np.random.choice(arr.shape[0], n_points, replace=arr.shape[0] < n_points)
    return arr[idx].astype(np.float32)


class PDMMergedDataset(Dataset):
    """Merged successful-grasp dataset for PDM.

    By default this keeps the cleanest labels: rows with executed-at-close poses
    and trusted gripper tips. Set `require_trusted_tips=False` to keep more
    executed rows.
    """

    def __init__(
        self,
        merged_dir: str = DEFAULT_MERGED_DIR,
        condition_h5: str | None = None,
        affordance_h5: str | None = None,
        obj_ids: Iterable[str] | None = None,
        n_points: int = 4096,
        require_trusted_tips: bool = True,
        max_cmd_candidate_dist: float = 0.5,
        mesh_root: str = DEFAULT_ROTATED_MESH_DIR,
        cache_conditions: bool = True,
    ):
        self.merged_dir = merged_dir
        self.n_points = n_points
        self.require_trusted_tips = require_trusted_tips
        self.max_cmd_candidate_dist = max_cmd_candidate_dist
        self.mesh_root = mesh_root
        self.conditions = PDMConditionStore(condition_h5)
        self.affordance = AffordanceStore(affordance_h5)
        self.cache_conditions = cache_conditions
        self._condition_cache: dict[str, ObjectCondition] = {}

        obj_filter = set(obj_ids) if obj_ids is not None else None
        self.rows: list[tuple[PDMSampleMeta, np.ndarray]] = []
        self.skipped: dict[str, int] = {
            "missing_executed": 0,
            "untrusted_tips": 0,
            "bad_pose": 0,
            "outlier": 0,
        }
        self._scan(obj_filter)

    def _iter_merged_paths(self, obj_filter: set[str] | None) -> Iterable[tuple[str, str]]:
        for name in sorted(os.listdir(self.merged_dir)):
            if not name.endswith("_robot_gt_merged.hdf5"):
                continue
            obj_id = name.replace("_robot_gt_merged.hdf5", "")
            if obj_filter is not None and obj_id not in obj_filter:
                continue
            yield obj_id, os.path.join(self.merged_dir, name)

    def _scan(self, obj_filter: set[str] | None) -> None:
        for obj_id, path in self._iter_merged_paths(obj_filter):
            with h5py.File(path, "r") as f:
                sg = f.get("successful_grasps")
                if sg is None:
                    continue
                for key in sorted(sg.keys()):
                    g = sg[key]
                    if "executed_panda_hand_at_close" not in g:
                        self.skipped["missing_executed"] += 1
                        continue
                    trusted = bool(g.attrs.get("gripper_tips_trusted", False))
                    if self.require_trusted_tips and not trusted:
                        self.skipped["untrusted_tips"] += 1
                        continue
                    ep = g["executed_panda_hand_at_close"]
                    try:
                        command = executed_to_command(ep["position"][:], ep["rotation"][:])
                    except (KeyError, ValueError):
                        self.skipped["bad_pose"] += 1
                        continue
                    if not is_valid_rotation(command.rotation):
                        self.skipped["bad_pose"] += 1
                        continue
                    if "grasp_point" in g:
                        gp = np.asarray(g["grasp_point"][:], dtype=np.float64).reshape(3)
                        if np.linalg.norm(command.position - gp) > self.max_cmd_candidate_dist:
                            self.skipped["outlier"] += 1
                            continue
                    pose9 = command_to_pose9(command)
                    source_file = str(g.attrs.get("source_file", ""))
                    grasp_name = _normalize_name(g.attrs.get("name", key))
                    pool_candidate_key = _normalize_name(g.attrs.get("pool_candidate_key", ""))
                    yaw_deg = recover_yaw_deg_from_source(
                        source_file,
                        grasp_name=grasp_name,
                        pool_candidate_key=pool_candidate_key,
                    )
                    meta = PDMSampleMeta(
                        obj_id=obj_id,
                        merged_path=path,
                        grasp_key=key,
                        score=float(g.attrs.get("score", 0.0)),
                        source_file=source_file,
                        trusted_tips=trusted,
                        yaw_deg=yaw_deg,
                    )
                    self.rows.append((meta, pose9))

    def __len__(self) -> int:
        return len(self.rows)

    def _load_condition(self, obj_id: str) -> ObjectCondition:
        if self.cache_conditions and obj_id in self._condition_cache:
            return self._condition_cache[obj_id]
        if self.conditions.has(obj_id):
            cond = self.conditions.load(obj_id)
            cond.points = _resample_rows(cond.points, self.n_points)
        elif self.affordance.has(obj_id):
            import warnings

            warnings.warn(
                f"PDM condition for {obj_id} loaded from affordance_h5 (GT soft labels). "
                "Train with --condition-h5 from `python -m model.pdm.build_condition_cache` "
                "(v6 predictions), not raw affordance_h5.",
                stacklevel=2,
            )
            cond = self.affordance.load(obj_id)
            cond.points = _resample_rows(cond.points, self.n_points)
        else:
            cond = sample_mesh_condition(obj_id, self.n_points, mesh_root=self.mesh_root)
        if self.cache_conditions:
            self._condition_cache[obj_id] = cond
        return cond

    def __getitem__(self, idx: int) -> dict:
        meta, pose9 = self.rows[idx]
        cond = self._load_condition(meta.obj_id)
        points = _resample_rows(cond.points, self.n_points)
        return {
            "points": torch.from_numpy(points),
            "pose": torch.from_numpy(pose9.astype(np.float32)),
            "yaw": torch.from_numpy(yaw_feature_from_deg(meta.yaw_deg)),
            "yaw_deg": torch.tensor(meta.yaw_deg, dtype=torch.float32),
            "obj_id": meta.obj_id,
            "score": torch.tensor(meta.score, dtype=torch.float32),
        }


def compute_pose_stats(dataset: PDMMergedDataset) -> dict[str, torch.Tensor]:
    """Compute mean/std over packed 9D pose labels."""

    if len(dataset) == 0:
        raise ValueError("cannot compute stats for an empty dataset")
    poses = torch.stack([torch.from_numpy(pose) for _, pose in dataset.rows], dim=0)
    return {
        "pose_mean": poses.mean(dim=0),
        "pose_std": poses.std(dim=0).clamp_min(1e-4),
    }
