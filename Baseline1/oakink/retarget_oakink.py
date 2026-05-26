#!/usr/bin/env python3
"""OakInk → v4-format hdf5 retarget (gravity-aligned, baseline_3-v4 compatible).

Key differences vs the original Baseline1/retarget_human_to_ee.py oakink branch:

1. **Gravity alignment** — the v2 script kept everything in CAMERA frame (subtract
   PC centroid only). We transform hand joints + object PC into OakInk's WORLD
   frame (which is gravity-aligned: +Z = up, table ≈ z=0; verified in Phase 0,
   see docs/frame_conventions.md). This is what the v4 sim collector needs to
   place the object correctly in IsaacSim (sim world is +Z=up).

2. **v4-format attrs saved** — adds the per-ep attrs the v4 collector requires:
       obj_origin_G       (3,)  object position in OakInk world at frame 0
       obj_quat_G_wxyz    (4,)  object orientation at frame 0
       ycb_class_id       int   integer label (from class_id_map.json)
       origin_G_W         (3,)  alias of obj_origin_G (kept for DexYCB parity)
       table_z_G          float estimated table top z (~0 in OakInk world)
       ee_offset_m, gripper_span_m — same constants as DexYCB

3. **Per-object integer class_id** — looked up from class_id_map.json. Objects
   not in the map are skipped (so this script naturally filters to the 15
   high-priority subset until you expand the manifest).

4. **Date-tagged output dir** — defaults to
   Baseline1/data/episodes_oakink_v3_<class-id-tag>_<YYYY-MM-DD>/ so it never
   overwrites the legacy episodes_v2_oakink/.

Usage:
    python Baseline1/oakink/retarget_oakink.py            # all `use=true` objects in manifest
    python Baseline1/oakink/retarget_oakink.py --object A01001
    python Baseline1/oakink/retarget_oakink.py --object A01001 --limit-sessions 2  # smoke test
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import pickle
import sys
from collections import defaultdict

import h5py
import numpy as np
from natsort import natsorted
from scipy.spatial.transform import Rotation

# Allow `import Baseline1.retarget_human_to_ee` to find reusable helpers
_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_THIS, "..", "..")))

# Re-use the dataset-agnostic helpers (MANO retarget math, constants).
# DO NOT import get_object_points — its OakInk branch double-scales meshes
# (applies both scale.json and R_align with scale built in). We use the
# OakInk-specific loader below instead.
from Baseline1.retarget_human_to_ee import (  # noqa: E402
    mano_joints_to_ee,
    EE_OFFSET, GRIPPER_SPAN, GRASP_MARGIN, N_POINTS, MIN_FRAMES, INVALID,
)
from Baseline1.oakink.oakink_paths import (  # noqa: E402
    OAKINK_HAND_J_DIR, OAKINK_OBJ_TRANSF_DIR, OAKINK_GENERAL_INFO,
    CLASS_ID_MAP, episodes_dir,
)
from Baseline1.oakink.oakink_meshes import get_oakink_object_points  # noqa: E402


# ── manifest ─────────────────────────────────────────────────────────────────
def load_class_id_map() -> dict:
    with open(CLASS_ID_MAP) as f:
        m = json.load(f)
    return m


# ── per-frame loaders ────────────────────────────────────────────────────────
def _load_pkl(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _load_hand_j_cam(filename: str) -> np.ndarray | None:
    """(21, 3) hand joints in camera frame, metric metres."""
    v = _load_pkl(os.path.join(OAKINK_HAND_J_DIR, filename))
    if v is None:
        return None
    if hasattr(v, "numpy"):
        v = v.numpy()
    arr = np.asarray(v, dtype=np.float64)
    return arr if arr.shape == (21, 3) else None


def _load_obj_transf_cam(filename: str) -> np.ndarray | None:
    """(4, 4) object pose in CAMERA frame (T_c_o)."""
    v = _load_pkl(os.path.join(OAKINK_OBJ_TRANSF_DIR, filename))
    if v is None:
        return None
    if hasattr(v, "numpy"):
        v = v.numpy()
    arr = np.asarray(v, dtype=np.float64)
    return arr if arr.shape == (4, 4) else None


def _load_general_info(filename: str) -> dict | None:
    """{ cam_extr (T_w_c 4x4), cam_intr (3x3), obj_anno (T_w_o 4x4 — WORLD frame),
       hand_anno {hand_tsl, hand_pose, hand_shape} }."""
    v = _load_pkl(os.path.join(OAKINK_GENERAL_INFO, filename))
    if v is None:
        return None
    out = {}
    for k, val in v.items():
        if hasattr(val, "numpy"):
            out[k] = val.numpy()
        elif isinstance(val, dict):
            out[k] = {kk: (vv.numpy() if hasattr(vv, "numpy") else vv) for kk, vv in val.items()}
        else:
            out[k] = val
    return out


# ── session discovery ────────────────────────────────────────────────────────
def discover_oakink_sessions(obj_id: str | None = None):
    """Glob anno/hand_j/*.pkl → dict[(seq_id, ts, sbj, cam)] → list[(frame_idx, filename)].
    If obj_id is given, only return sessions for that object.
    """
    sessions = defaultdict(list)
    for fn in natsorted(os.listdir(OAKINK_HAND_J_DIR)):
        if not fn.endswith(".pkl"):
            continue
        parts = fn[:-4].split("__")
        if len(parts) != 5:
            continue
        seq_id, ts, sbj, frame_str, cam = parts
        if obj_id and not seq_id.startswith(obj_id):
            continue
        try:
            fi = int(frame_str)
        except ValueError:
            continue
        sessions[(seq_id, ts, sbj, cam)].append((fi, fn))
    for k in sessions:
        sessions[k].sort()
    return sessions


# ── world-frame transform ────────────────────────────────────────────────────
def _cam_to_world_pts(pts_cam: np.ndarray, T_w_c: np.ndarray) -> np.ndarray:
    """(N, 3) camera-frame points → world-frame, via T_w_c (camera pose in world).

    Convention (verified Phase 0): cam_extr = T_w_c. So
        world_pt = T_w_c @ [cam_pt; 1] / w
    But wait — if T_w_c is "pose of camera in world" (=> camera→world transform
    when treated as homogeneous), then world_pt = T_w_c @ cam_pt_hom. We verified
    this empirically (obj_anno = inv(cam_extr) @ obj_transf gives the obj_anno
    that's identical across all 4 cameras). So:
        world_pt = inv(T_w_c) @ cam_pt? NO — that gave consistent across-cam.

    Empirical Phase 0 finding:
        inv(cam_extr) @ obj_transf == obj_anno  (matches across 4 cameras)
    where obj_transf is in CAMERA frame and obj_anno is in WORLD frame.

    So inv(cam_extr) is the camera→world map. Therefore cam_extr is in fact
    T_c_w (world→camera) — despite its variable name. We name the local var
    T_w_c above to keep the calling code intuitive but actually inverse it.
    """
    T_c2w = np.linalg.inv(T_w_c)   # cam_extr name is misleading — see Phase 0 doc
    n = len(pts_cam)
    hom = np.concatenate([pts_cam, np.ones((n, 1))], axis=1)  # (N,4)
    wpts = (T_c2w @ hom.T).T[:, :3]
    return wpts


# ── episode builder ─────────────────────────────────────────────────────────
def build_episode_v4(seq_id, ts, sbj, cam, frames, n_points: int,
                     class_id: int, mass_kg: float):
    """Return (ep_dict, message) or (None, error_str)."""
    obj_id = seq_id.split("_")[0]
    # Use OakInk's OFFICIAL CAD mesh (the canonical frame for obj_transf / obj_anno).
    # See Baseline1/oakink/oakink_meshes.py for why we don't use the SAM3D path.
    obj_pts_canonical = get_oakink_object_points(obj_id, n_points)
    if obj_pts_canonical is None:
        return None, f"mesh oakink/{obj_id} not found"

    rec = []
    n_invalid = 0
    obj_anno_frame0 = None   # T_w_o at first valid frame — defines obj_origin_G

    for frame_idx, fn in frames:
        j_cam = _load_hand_j_cam(fn)
        ot_cam = _load_obj_transf_cam(fn)
        gi = _load_general_info(fn)
        if j_cam is None or ot_cam is None or gi is None:
            n_invalid += 1; continue
        if np.any(j_cam < INVALID) or not np.any(np.abs(ot_cam[:3, 3]) > 1e-6):
            n_invalid += 1; continue

        T_c_w = gi["cam_extr"]              # named cam_extr; Phase 0 showed it's T_c_w
        obj_anno = gi["obj_anno"]           # T_w_o — already in world frame

        # hand_j: camera → world. mano_joints_to_ee is frame-agnostic (just geometry).
        T_c2w = np.linalg.inv(T_c_w)
        j_world = (T_c2w @ np.concatenate([j_cam, np.ones((21, 1))], axis=1).T).T[:, :3]
        ee_out = mano_joints_to_ee(j_world)
        if ee_out is None:
            n_invalid += 1; continue
        p_ee_world, q_wxyz_world, flex = ee_out

        # object PC in world (sample at world-frame pose using obj_anno)
        R_w_o = obj_anno[:3, :3]
        t_w_o = obj_anno[:3, 3]
        pc_world = (R_w_o @ obj_pts_canonical.T).T + t_w_o

        # object centroid in world frame at this frame
        oc_world = t_w_o.copy()

        if obj_anno_frame0 is None:
            obj_anno_frame0 = obj_anno.copy()

        rec.append(dict(
            f=frame_idx,
            pc=pc_world.astype(np.float32),
            p=p_ee_world,
            oc=oc_world,
            q=q_wxyz_world,
            flex=flex,
        ))

    if len(rec) < MIN_FRAMES:
        return None, f"only {len(rec)} valid frames"

    # quat sign-continuity across frames
    for k in range(1, len(rec)):
        if np.dot(rec[k]["q"], rec[k - 1]["q"]) < 0.0:
            rec[k]["q"] = -rec[k]["q"]

    # gripper: 0 until hand reaches object (uses world-frame distances)
    d = np.array([float(np.linalg.norm(r["p"] - r["oc"])) for r in rec])
    d_min = float(d.min())
    onset = int(np.argmax(d <= d_min + GRASP_MARGIN))
    gripper = np.zeros(len(rec), dtype=np.float64); gripper[onset:] = 1.0

    # ── G-frame conversion: origin = obj_origin_world (obj pose at frame 0) ──
    # NOTE: we do NOT subtract PC centroid (the v2 oakink retarget did, but that
    # discarded the world-frame anchor). Instead we use the OakInk world frame
    # directly and define obj_origin_G := T_w_o.translation@frame0. This matches
    # what DexYCB v4 collector expects.
    obj_origin_G = obj_anno_frame0[:3, 3].astype(np.float64)
    R_w_o_frame0 = obj_anno_frame0[:3, :3]
    obj_quat_xyzw = Rotation.from_matrix(R_w_o_frame0).as_quat()
    obj_quat_G_wxyz = np.array([obj_quat_xyzw[3], *obj_quat_xyzw[:3]], dtype=np.float64)

    # state[t] in G-frame: pos relative to obj_origin_G, quat in world (retarget conv)
    pcs, states = [], []
    for k, r in enumerate(rec):
        pc_centered = r["pc"] - obj_origin_G.astype(np.float32)  # object-centric PC
        p_G = (r["p"] - obj_origin_G).astype(np.float32)
        states.append(np.concatenate([p_G, r["q"], [gripper[k]]]).astype(np.float32))
        pcs.append(pc_centered)
    pcs = np.stack(pcs)
    states = np.stack(states)
    if len(states) < 2:
        return None, "too short after processing"

    # Table z estimate: average object z over the early frames (before lift starts).
    # In OakInk world frame, table top ≈ 0 and object base z ≈ +obj_thickness/2.
    # We just report obj base z = min PC.z to give the collector a hint.
    table_z_G = float(pcs[0][:, 2].min())   # object-centric PC, min z ≈ table relative to obj origin

    ep = dict(
        point_cloud=pcs[:-1],                          # (T-1, N, 3)
        state=states[:-1],
        action=states[1:],
        n_valid=len(rec), n_invalid=n_invalid,
        grasp_onset=onset, min_hand_obj_dist_m=d_min,
        # v4-format attrs (these go into hdf5.attrs)
        obj_id=obj_id,
        ycb_class_id=int(class_id),
        obj_origin_G=obj_origin_G,
        obj_quat_G_wxyz=obj_quat_G_wxyz,
        origin_G_W=obj_origin_G,        # alias
        table_z_G=table_z_G,
        mass_kg=float(mass_kg),
    )
    return ep, (f"{len(rec)} valid frames · grasp_onset @ {onset} · "
                f"min hand-obj dist {d_min*1000:.0f}mm")


# ── save ─────────────────────────────────────────────────────────────────────
def save_v4_hdf5(out_path, ep, seq_id, ts, sbj, cam):
    """v4-format hdf5: state/action/point_cloud + the attrs the collector needs."""
    with h5py.File(out_path, "w") as h:
        h.create_dataset("point_cloud", data=ep["point_cloud"])
        h.create_dataset("state",       data=ep["state"])
        h.create_dataset("action",      data=ep["action"])
        h.attrs["dataset"]          = "oakink"
        h.attrs["baseline"]         = "baseline_3_v4_oakink"
        h.attrs["obj_id"]           = ep["obj_id"]
        h.attrs["ycb_class_id"]     = ep["ycb_class_id"]
        h.attrs["obj_origin_G"]     = ep["obj_origin_G"]
        h.attrs["obj_quat_G_wxyz"]  = ep["obj_quat_G_wxyz"]
        h.attrs["origin_G_W"]       = ep["origin_G_W"]
        h.attrs["table_z_G"]        = ep["table_z_G"]
        h.attrs["mass_kg"]          = ep["mass_kg"]
        h.attrs["ee_offset_m"]      = float(EE_OFFSET)
        h.attrs["gripper_span_m"]   = float(GRIPPER_SPAN)
        h.attrs["mano_side"]        = "right"
        h.attrs["seq_id"]           = seq_id
        h.attrs["timestamp"]        = ts
        h.attrs["subject_flag"]     = sbj
        h.attrs["camera"]           = cam
        h.attrs["n_steps"]          = int(len(ep["state"]))
        h.attrs["grasp_onset"]      = int(ep["grasp_onset"])
        h.attrs["grasp_onset_idx"]  = int(ep["grasp_onset"])
        h.attrs["min_hand_obj_dist_m"] = float(ep["min_hand_obj_dist_m"])


# ── main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", type=str, default=None,
                    help="single obj_id to process (e.g. A01001). Default: all use=true in manifest.")
    ap.add_argument("--limit-sessions", type=int, default=0,
                    help="process at most N sessions per object (0 = all). Useful for smoke tests.")
    ap.add_argument("--out-tag", type=str, default=None,
                    help="suffix for output dir: episodes_oakink_v3_<tag>_<date>. "
                         "Default: 'smoke' if --limit-sessions, else '15obj'.")
    ap.add_argument("--n-points", type=int, default=N_POINTS)
    ap.add_argument("--redo", action="store_true", help="overwrite existing hdf5")
    args = ap.parse_args()

    cmap = load_class_id_map()
    objects_use = {oid: info for oid, info in cmap["objects"].items() if info.get("use")}
    if args.object:
        if args.object not in cmap["objects"]:
            sys.exit(f"obj_id {args.object} not in manifest {CLASS_ID_MAP}")
        objects_use = {args.object: cmap["objects"][args.object]}

    today = datetime.date.today().isoformat()
    tag = args.out_tag or ("smoke" if args.limit_sessions else "15obj")
    out_dir = episodes_dir(f"oakink_v3_{tag}_{today}")
    os.makedirs(out_dir, exist_ok=True)

    # Write a per-run MANIFEST.md (matches the convention used by sim collector orchestrator)
    with open(os.path.join(out_dir, "MANIFEST.md"), "w") as f:
        f.write(f"# OakInk → v4 retarget run — {today}\n\n")
        f.write(f"- launched: {datetime.datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"- script: Baseline1/oakink/retarget_oakink.py\n")
        f.write(f"- objects ({len(objects_use)}): {sorted(objects_use)}\n")
        f.write(f"- limit-sessions: {args.limit_sessions or 'all'}\n")
        f.write(f"- n-points: {args.n_points}\n")
        f.write(f"- frame convention: OakInk world frame (Z=up, gravity-aligned). See Baseline1/oakink/docs/frame_conventions.md\n")

    print(f"=> output dir: {out_dir}")
    print(f"=> processing {len(objects_use)} objects: {sorted(objects_use)}")
    n_emit = 0; n_skip = 0; n_total = 0

    for oid, info in sorted(objects_use.items()):
        cid = info["class_id"]
        mass = info.get("mass_kg", 0.05)
        sessions = discover_oakink_sessions(obj_id=oid)
        keys = list(sessions.keys())
        if args.limit_sessions:
            keys = keys[:args.limit_sessions]
        print(f"\n=== {oid} (cid={cid}, mass={mass}kg, {len(keys)} sessions) ===")

        for key in keys:
            seq_id, ts, sbj, cam = key
            out_name = f"oakink__{seq_id}__{ts}__{sbj}__{cam}.hdf5"
            out_path = os.path.join(out_dir, out_name)
            n_total += 1
            if os.path.exists(out_path) and not args.redo:
                continue
            ep, msg = build_episode_v4(seq_id, ts, sbj, cam, sessions[key], args.n_points,
                                       class_id=cid, mass_kg=mass)
            tag2 = f"{seq_id}/{ts}/{sbj}/{cam}"
            if ep is None:
                n_skip += 1; print(f"  [skip] {tag2}: {msg}"); continue
            save_v4_hdf5(out_path, ep, seq_id, ts, sbj, cam)
            n_emit += 1
            print(f"  [emit] {tag2}: {msg}")

    print(f"\n=== DONE: emit={n_emit}  skip={n_skip}  total={n_total} ===")
    print(f"   output dir: {out_dir}")


if __name__ == "__main__":
    main()
