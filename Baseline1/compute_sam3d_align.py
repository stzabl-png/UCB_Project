#!/usr/bin/env python3
"""
Baseline1 v2 — compute the constant per-object rotation that bridges the SAM3D
mesh's canonical frame and the YCB-Video CAD model's canonical frame.

WHY
    Baseline1 (route A) builds the object point cloud by placing the SAM3D
    `ycb_dex_NN` mesh into the camera using the dataset's GT object pose `pose_y`.
    But `pose_y` is defined w.r.t. the *YCB-Video CAD model's* canonical frame,
    not the SAM3D mesh's — so the rendered object is rotated by a constant
    (per-object) amount relative to where it physically is. `R_align` fixes that:

        object point cloud in camera  =  pose_y · R_align · sam3d_mesh_points

HOW  (ICP between two complete meshes)
    The earlier algebraic method `R_align = pose_y⁻¹ · ob_in_cam` (use FP's pose
    of the SAM3D mesh) failed because FP on DexYCB is unreliable (DepthPro depth
    bias ~1.7× → FP can't register correctly, per-frame poses scatter ~uniformly).

    The robust replacement: ICP between the two *complete* meshes. Both describe
    the same physical object in metric metres but with different canonical
    orientations. We sample N points from each mesh's surface, centre at each
    centroid, then run ICP from ~24 cube-symmetry initial rotations + several
    random inits (to escape local minima). The best-cost result gives R_align.

INPUTS
    SAM3D mesh : data_hub/ProcessedData/obj_meshes/ycb/ycb_dex_NN/mesh.ply
                 + scale.json (scale_factor, ycb_name)  → metric metres, SAM3D canonical
    YCB-Video CAD mesh : data_hub/RawData/ThirdPersonRawData/dexycb/models/{ycb_name}/textured.obj
                         (already metric metres, YCB-CAD canonical — matches `pose_y`)

OUTPUT
    Baseline1/assets/sam3d_align/ycb_dex_NN.json :
        { "ycb_dex": NN, "ycb_name": ..., "R_align_4x4": [[...4x4...]],
          "icp_cost": ..., "near_optima": <#inits within 1.5× best cost>,
          "n_inits": ..., "likely_symmetric": bool }
    Then run retarget_human_to_ee.py --align-mode sam3d → v2 episodes.
"""
import os, sys, json, argparse, itertools, time
import numpy as np
import trimesh
from glob import glob
from natsort import natsorted
from scipy.spatial.transform import Rotation

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
import config

SAM3D_BASE = os.path.join(config.DATA_HUB, "ProcessedData", "obj_meshes", "ycb")
CAD_BASE   = os.path.join(config.DATA_HUB, "RawData", "ThirdPersonRawData",
                          "dexycb", "models")
OUT_DIR    = os.path.join(PROJ, "Baseline1", "assets", "sam3d_align")


def cube_rotations():
    """24 rotational symmetries of the cube — axis-aligned coverage of SO(3)
    (any rotation in SO(3) is within ~45° of one of these). All have det = +1."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            M = np.zeros((3, 3))
            for i, p in enumerate(perm):
                M[i, p] = signs[i]
            if np.linalg.det(M) > 0.5:
                out.append(M)
    return out  # 24


def load_sam3d(ycb_dex_id):
    """Returns (mesh, ycb_name) with mesh in metric metres in SAM3D canonical, or (None, None)."""
    obj_dir = os.path.join(SAM3D_BASE, f"ycb_dex_{ycb_dex_id:02d}")
    mesh_path = os.path.join(obj_dir, "mesh.ply")
    if not os.path.exists(mesh_path):
        return None, None
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    sj_path = os.path.join(obj_dir, "scale.json")
    ycb_name = "?"; sf = 1.0
    if os.path.exists(sj_path):
        sj = json.load(open(sj_path))
        sf = float(sj.get("scale_factor", 1.0))
        ycb_name = sj.get("ycb_name", "?")
    if sf != 1.0:
        mesh.vertices = mesh.vertices * sf
    return mesh, ycb_name


def load_cad(ycb_name):
    path = os.path.join(CAD_BASE, ycb_name, "textured.obj")
    if not os.path.exists(path):
        return None
    return trimesh.load(path, force="mesh", process=False)


def align_one(ycb_dex_id, n_samples, n_random_inits, max_iter):
    sam3d, ycb_name = load_sam3d(ycb_dex_id)
    if sam3d is None:
        return None, "SAM3D mesh missing"
    cad = load_cad(ycb_name)
    if cad is None:
        return None, f"CAD model missing for {ycb_name} (at {CAD_BASE}/{ycb_name}/textured.obj)"

    src_full, _ = trimesh.sample.sample_surface(sam3d, n_samples)
    dst_full, _ = trimesh.sample.sample_surface(cad,   n_samples)
    src_centroid = src_full.mean(0)
    dst_centroid = dst_full.mean(0)
    src = (src_full - src_centroid).astype(np.float64)
    dst = (dst_full - dst_centroid).astype(np.float64)

    inits = cube_rotations()
    rng = np.random.default_rng(0)
    inits += [Rotation.from_rotvec(rng.standard_normal(3) * np.pi).as_matrix()
              for _ in range(n_random_inits)]

    results = []
    for init_R in inits:
        T_init = np.eye(4); T_init[:3, :3] = init_R
        try:
            T_icp, _, cost = trimesh.registration.icp(
                src, dst, initial=T_init,
                threshold=1e-7, max_iterations=max_iter,
                reflection=False, scale=False)   # rigid only: no reflection, no scale
        except Exception:
            continue
        results.append((float(cost), T_icp))

    if not results:
        return None, "ICP failed for all inits"

    results.sort(key=lambda x: x[0])
    best_cost, best_T = results[0]
    near = sum(1 for c, _ in results if c <= best_cost * 1.5)

    # Compose full transform: cad_pt = R · sam3d_pt + (dst_centroid + best_T_t - R · src_centroid)
    R = best_T[:3, :3]
    R_align = np.eye(4)
    R_align[:3, :3] = R
    R_align[:3, 3]  = best_T[:3, 3] + dst_centroid - R @ src_centroid

    return R_align, dict(
        ycb_name=ycb_name,
        icp_cost=float(best_cost),
        near_optima=int(near),
        n_inits=len(results),
        likely_symmetric=bool(near > 4),
    )


def main():
    ap = argparse.ArgumentParser(description="Baseline1 v2 — ICP SAM3D↔YCB-CAD per-object alignment")
    ap.add_argument("--objects", nargs="*", default=None,
                    help="subset of ycb_dex_NN ids to process (default: all 20)")
    ap.add_argument("--n-samples", type=int, default=4000)
    ap.add_argument("--n-random-inits", type=int, default=8)
    ap.add_argument("--max-iter", type=int, default=80)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    obj_dirs = natsorted(glob(os.path.join(SAM3D_BASE, "ycb_dex_*")))
    ids = sorted(int(os.path.basename(d).split("_")[-1]) for d in obj_dirs)
    if args.objects:
        wanted = set(int(o.replace("ycb_dex_", "")) for o in args.objects)
        ids = [i for i in ids if i in wanted]

    print(f"{'obj':<14} {'ycb_name':<26} {'cost':>12} {'near<1.5x':>10} {'sym':>5} {'sec':>6}  status")
    print("-" * 100)
    n_ok = 0
    for dex_id in ids:
        t0 = time.time()
        T_align, res = align_one(dex_id, args.n_samples, args.n_random_inits, args.max_iter)
        dt = time.time() - t0
        if T_align is None:
            print(f"ycb_dex_{dex_id:02d}     {'?':<26} {'-':>12} {'-':>10} {'-':>5} {dt:>6.1f}  ⏭ {res}")
            continue
        name = f"ycb_dex_{dex_id:02d}"
        out = dict(ycb_dex=dex_id, R_align_4x4=T_align.tolist(), **res)
        with open(os.path.join(OUT_DIR, f"{name}.json"), "w") as f:
            json.dump(out, f, indent=2)
        flag = "  ⚠️ symmetric / multi-optima" if res["likely_symmetric"] else ""
        print(f"{name:<14} {res['ycb_name']:<26} {res['icp_cost']:>12.6f} "
              f"{res['near_optima']:>4}/{res['n_inits']:<4} {str(res['likely_symmetric']):>5} "
              f"{dt:>6.1f}  ✅ saved{flag}")
        n_ok += 1
    print("-" * 100)
    print(f"{n_ok}/{len(ids)} objects aligned → {OUT_DIR}")


if __name__ == "__main__":
    main()
