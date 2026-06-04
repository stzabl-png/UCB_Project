"""Server-side DP3 session builder (T6_dp3) — run on the 5090.

Consumes partner T5 perception output (object_base_aligned.glb + register/T_base_mesh.json)
and writes the DP3 session the razer driver loads:
    pc_G.npy   float32 (N,3)  — object cloud in the G/object frame
    meta.json  {"T_base_mesh": 4x4, "n_obs": 2, ...}

Frame: G frame == the base-aligned mesh frame, so the sampled glb vertices ARE pc_G
directly — no centering / normalization / scaling / pre-rotation (matches partner T6
PDM sampling and UCB_Project/deploy/DP3_DEPLOY_LOGIC.md §4). home_right_arm is NOT here:
it is a razer-side robot constant (the driver defaults to DEFAULT_JOINT_POS["right_arm"]).

Usage (5090, any env with trimesh + numpy):
    python deploy/build_dp3_session.py --session-dir <titan_session_root> [--out <dir>]
    python deploy/build_dp3_session.py --mesh a.glb --t-base-mesh-json reg.json --out <dir>

Then start the DP3 server (separate process) and rsync <out>/ to the razer:
    python Baseline1/eval/dp3_inference_server.py --ckpt <...> --host 127.0.0.1 --port 8765
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import trimesh


def load_mesh(path: pathlib.Path) -> trimesh.Trimesh:
    """Load a glb/obj/ply as a single Trimesh (scenes are concatenated)."""
    m = trimesh.load(str(path), force="mesh")
    if not isinstance(m, trimesh.Trimesh) or m.faces.shape[0] == 0:
        raise ValueError(f"{path}: not a surface mesh (got {type(m).__name__})")
    return m


def sample_pc_G(mesh: trimesh.Trimesh, num_points: int, seed: int) -> np.ndarray:
    """4096 area-weighted surface points, no transform — these ARE pc_G (mesh==G frame)."""
    pts, _ = trimesh.sample.sample_surface(mesh, num_points, seed=seed)
    return np.asarray(pts, dtype=np.float32)


def read_T_base_mesh(path: pathlib.Path) -> np.ndarray:
    """Read register/T_base_mesh.json → 4x4. Accepts {'T_base_mesh': 4x4} or a raw 4x4."""
    obj = json.loads(path.read_text())
    T = obj["T_base_mesh"] if isinstance(obj, dict) and "T_base_mesh" in obj else obj
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"{path}: T_base_mesh must be 4x4, got {T.shape}")
    return T


def snap_z_to_table(mesh: trimesh.Trimesh, T_base_mesh: np.ndarray, table_height_m: float):
    """Override FP's unreliable depth-Z: shift the object along base +Z so its LOWEST point
    rests on the table (bottom base-Z == table_height_m). Keeps FP's x,y. Returns (T', dz)."""
    V_base = (np.asarray(mesh.vertices) @ T_base_mesh[:3, :3].T) + T_base_mesh[:3, 3]
    bottom_z = float(V_base[:, 2].min())
    dz = float(table_height_m) - bottom_z
    T = T_base_mesh.copy()
    T[2, 3] += dz
    return T, dz, bottom_z


def build_session(mesh_path, t_base_mesh_json, out_dir, *, num_points=4096, seed=42, n_obs=2,
                  snap_to_table=False, table_height_m=None):
    mesh_path = pathlib.Path(mesh_path)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = load_mesh(mesh_path)
    pc_G = sample_pc_G(mesh, num_points, seed)
    T_base_mesh = read_T_base_mesh(pathlib.Path(t_base_mesh_json))

    snap_info = None
    if snap_to_table:
        if table_height_m is None:
            raise ValueError("snap_to_table requires table_height_m")
        T_base_mesh, dz, bottom_before = snap_z_to_table(mesh, T_base_mesh, table_height_m)
        snap_info = {"table_height_m": float(table_height_m), "dz_applied_m": round(dz, 4),
                     "bottom_base_z_before_m": round(bottom_before, 4)}

    np.save(out_dir / "pc_G.npy", pc_G)
    meta = {
        "T_base_mesh": T_base_mesh.tolist(),
        "n_obs": int(n_obs),
        "num_points": int(num_points),
        "seed": int(seed),
        "mesh_file": str(mesh_path),
        "pc_extent_m": (pc_G.max(0) - pc_G.min(0)).round(4).tolist(),
        "snap_to_table": snap_info,
        "note": "pc_G = base-aligned mesh surface sample (G frame); home_right_arm is razer-side",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return out_dir, pc_G, meta


def main():
    ap = argparse.ArgumentParser(description="Build a DP3 session (pc_G.npy + meta.json) for the razer.")
    ap.add_argument("--session-dir", help="Titan session root (uses output/mesh + output/register)")
    ap.add_argument("--mesh", help="object_base_aligned.glb (overrides --session-dir path)")
    ap.add_argument("--t-base-mesh-json", help="register/T_base_mesh.json (overrides --session-dir)")
    ap.add_argument("--out", help="output session dir (default: <session-dir>/dp3_session)")
    ap.add_argument("--num-points", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-obs", type=int, default=2)
    ap.add_argument("--snap-to-table", action="store_true",
                    help="Override FP depth-Z: shift object so its bottom rests on the table")
    ap.add_argument("--table-height", type=float, default=None,
                    help="Table top Z in base (m); default = input/scene/table.json table_height_m")
    args = ap.parse_args()

    root = None
    if args.session_dir:
        root = pathlib.Path(args.session_dir)
        mesh = args.mesh or (root / "output" / "mesh" / "object_base_aligned.glb")
        tbm = args.t_base_mesh_json or (root / "output" / "register" / "T_base_mesh.json")
        out = args.out or (root / "dp3_session")
    else:
        if not (args.mesh and args.t_base_mesh_json and args.out):
            ap.error("without --session-dir, give --mesh, --t-base-mesh-json, and --out")
        mesh, tbm, out = args.mesh, args.t_base_mesh_json, args.out

    table_height = args.table_height
    if args.snap_to_table and table_height is None:
        if root is None:
            ap.error("--snap-to-table needs --table-height (or --session-dir to read table.json)")
        tj = root / "input" / "scene" / "table.json"
        table_height = float(json.loads(tj.read_text())["table_height_m"])

    out_dir, pc_G, meta = build_session(mesh, tbm, out,
                                        num_points=args.num_points, seed=args.seed, n_obs=args.n_obs,
                                        snap_to_table=args.snap_to_table, table_height_m=table_height)
    print(f"wrote {out_dir}/pc_G.npy {pc_G.shape} + meta.json")
    print(f"  T_base_mesh t={np.asarray(meta['T_base_mesh'])[:3,3].round(3).tolist()} "
          f"| pc extent(cm)={[round(v*100,1) for v in meta['pc_extent_m']]} | n_obs={meta['n_obs']}")
    if meta.get("snap_to_table"):
        print(f"  snap-to-table: {meta['snap_to_table']}")
    print(f"  rsync this dir to the razer; start the DP3 server separately.")


if __name__ == "__main__":
    main()
