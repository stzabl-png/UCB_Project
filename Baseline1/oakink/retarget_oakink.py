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


# ── pinch finger selection (env-var driven for ablation) ────────────────────
# OAKINK_PINCH_FINGER = middle (default) | index | ring
# - middle (= MANO joint 12) — DEFAULT. Smoke test (2026-05-26 A02028 + A01001):
#     thumb+middle 12/28 = 42% vs thumb+index 6/28 = 21% (2× better).
#     A02028 (short container) jumped from 0% to 37% — middle finger naturally
#     wraps deeper around obj in palm-grasp, midpoint better matches Franka pinch.
# - index  (= MANO joint 8)  — original choice. Falls behind on side-grasp obj.
# - ring   (= MANO joint 16) — most "opposite" to thumb but tip-tip 8.3cm
#     exceeds Franka span 55% of the time → many retargeted poses unreachable
_PINCH_FINGER_MAP = {"index": 8, "middle": 12, "ring": 16}
PINCH_FINGER_IDX = _PINCH_FINGER_MAP.get(
    os.environ.get("OAKINK_PINCH_FINGER", "middle"), 12)


# ── onset mode (env-var driven for ablation) ─────────────────────────────────
# Default = "OLD" (d_min + 4cm, first close-approach frame).
# Smoke test on A02028 + A01001 (2026-05-26): OLD beat lift-1 / HYBRID /
# d_min+2cm both per-obj and aggregate (21% vs 4-14%) — empirically the most
# Franka-friendly because the "conservative" pose (hand 16cm from obj) gives
# cuRobo more workspace flexibility. Alternative modes kept as escape hatch
# for future ablation but DO NOT change default without re-validation.
#
# OAKINK_ONSET_MODE values:
#   OLD         (default) = argmax(d ≤ d_min+4cm) — best empirically
#   d_min+2cm   = first frame d ≤ d_min+2cm (tighter; ~5cm closer TCP)
#   lift-1      = lift_frames[0] - 1 (TCP closer but obj already rotated ~10°)
#   HYBRID      = last frame hand-close AND obj-still (mathematically clean)
ONSET_MODE = os.environ.get("OAKINK_ONSET_MODE", "OLD")


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
        ee_out = mano_joints_to_ee(j_world, pinch_finger_tip_idx=PINCH_FINGER_IDX)
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

    # ── grasp_onset — parameterised via OAKINK_ONSET_MODE env var ──
    # Computed metrics across 629 OakInk subj=0 sessions (Δrot = obj rotation
    # between frame 0 and onset; lower = T_obj_grasp more accurate):
    #   OLD     Δrot median 0.2°  hand_dist 159mm  (obj still + TCP far)
    #   lift-1  Δrot median 10.2° hand_dist 130mm  (obj rotated + TCP near)
    #   HYBRID  Δrot median 5.8°  hand_dist 131mm  (best tradeoff)
    d = np.array([float(np.linalg.norm(r["p"] - r["oc"])) for r in rec])
    d_min = float(d.min())
    oc_z = np.array([float(r["oc"][2]) for r in rec])
    oc_z_baseline = float(oc_z[:min(5, len(oc_z))].mean())
    lift_frames = np.where(oc_z > oc_z_baseline + 0.02)[0]

    if ONSET_MODE == "OLD":
        # first frame within d_min + GRASP_MARGIN (= 4cm)
        onset = int(np.argmax(d <= d_min + GRASP_MARGIN))
        onset_reason = f"OLD: d<=d_min+4cm at frame {onset}"
    elif ONSET_MODE == "d_min+2cm":
        # tighter version of OLD — first frame within 2cm of d_min
        # (rationale: 5cm closer TCP than OLD, only ~15° p95 Δrot vs 1.4° OLD)
        cand = np.where(d <= d_min + 0.02)[0]
        if len(cand) > 0:
            onset = int(cand[0])
            onset_reason = f"d_min+2cm: first frame d<=d_min+2cm at {onset}"
        else:
            onset = int(np.argmin(d))
            onset_reason = f"d_min+2cm fallback to argmin(d) at {onset}"
    elif ONSET_MODE == "lift-1" and len(lift_frames) > 0:
        # frame BEFORE +2cm obj lift
        onset = max(0, int(lift_frames[0]) - 1)
        onset_reason = f"lift-1: lift at {int(lift_frames[0])}, using frame {onset}"
    elif ONSET_MODE == "HYBRID":
        # last frame where hand close AND obj still
        hand_close = d <= d_min + 0.04
        obj_still  = oc_z <= oc_z_baseline + 0.005
        cand = np.where(hand_close & obj_still)[0]
        if len(cand) > 0:
            onset = int(cand[-1])
            onset_reason = f"HYBRID: last frame hand_close+obj_still = {onset}"
        else:
            onset = int(np.argmax(d <= d_min + GRASP_MARGIN))
            onset_reason = f"HYBRID fallback to OLD: frame {onset}"
    else:
        # Fallback path (also used when ONSET_MODE='lift-1' but no obj lift)
        flexes = np.array([float(r["flex"]) for r in rec])
        flex_base = float(flexes[:min(5, len(flexes))].mean())
        cand = np.where(flexes > flex_base + 0.10)[0]
        if len(cand) > 0:
            onset = int(cand[0]); onset_reason = f"finger flex > {flex_base+0.10:.2f} at frame {onset}"
        else:
            onset = int(np.argmin(d)); onset_reason = f"closest hand-obj approach at frame {onset}"
    onset = max(onset, MIN_FRAMES)
    onset = min(onset, len(rec) - 1)

    # Note: subj=1 handover ghost annotations (hand_dist > 30cm at onset) are
    # NOT filtered here — sim collector naturally rejects them downstream (all
    # 207 saved sim ep from full89 came from subj=0). Re-add filter if needed:
    #   d_at_onset = np.linalg.norm(rec[onset]["p"] - rec[onset]["oc"])
    #   if d_at_onset > 0.30: return None, "subj=1 non-grasp"

    # ── G-frame conversion (matches DexYCB build_gt_replay convention) ──
    # G-frame origin sits on the TABLE directly under the object at frame 0:
    #   origin_world = (obj_anno_xy@t0, 0)
    # obj_origin_G  = obj_anno_t0 - origin_world  = (0, 0, obj_z_above_table)
    # state[t]      = EE_world - origin_world
    # pc[t]         = pc_world[t] - origin_world
    obj_anno_w_t0 = obj_anno_frame0[:3, 3].astype(np.float64)
    origin_world = np.array([obj_anno_w_t0[0], obj_anno_w_t0[1], 0.0], dtype=np.float64)
    obj_origin_G = (obj_anno_w_t0 - origin_world).astype(np.float64)   # = (0, 0, obj_z)
    R_w_o_frame0 = obj_anno_frame0[:3, :3]
    obj_quat_xyzw = Rotation.from_matrix(R_w_o_frame0).as_quat()
    obj_quat_G_wxyz = np.array([obj_quat_xyzw[3], *obj_quat_xyzw[:3]], dtype=np.float64)

    # Truncate to the approach segment: frames 0..onset INCLUSIVE (matches DexYCB
    # build_gt_replay.py L457: `rec = rec[:onset + 1]`). state[-1] is then the
    # grasp moment (obj just started rising → fingers wrapped around obj, ready
    # to lift). sim collector then closes gripper + executes its own lift.
    rec_kept = rec[:onset + 1]
    gripper_kept = np.zeros(len(rec_kept), dtype=np.float64)
    gripper_kept[-1] = 1.0   # mark final frame = "arrived, would close" (DexYCB convention)

    pcs, states = [], []
    for k, r in enumerate(rec_kept):
        pc_g = (r["pc"] - origin_world.astype(np.float32))
        p_g = (r["p"] - origin_world).astype(np.float32)
        states.append(np.concatenate([p_g, r["q"], [gripper_kept[k]]]).astype(np.float32))
        pcs.append(pc_g)
    pcs = np.stack(pcs)
    states = np.stack(states)
    if len(states) < 2:
        return None, "too short after processing"

    # Table z in G-frame is 0 by construction (origin sits on the table). We
    # still report PC min-z as a sanity-check value (object base should be ~0).
    table_z_G = float(pcs[0][:, 2].min())

    # Match DexYCB build_gt_replay convention: grasp_onset == n_steps (saved trajectory
    # ends AT the grasp moment; sim replay closes/lifts past the end).
    n_steps = int(states[:-1].shape[0])
    # diagnostic: how far hand was from obj at the GRASP frame (state[-1] in saved data)
    d_at_grasp = float(np.linalg.norm(rec_kept[-1]["p"] - rec_kept[-1]["oc"]))

    ep = dict(
        point_cloud=pcs[:-1],                          # (T-1, N, 3)
        state=states[:-1],
        action=states[1:],
        n_valid=len(rec), n_invalid=n_invalid,
        grasp_onset=n_steps, min_hand_obj_dist_m=d_at_grasp,
        # v4-format attrs (these go into hdf5.attrs)
        obj_id=obj_id,
        ycb_class_id=int(class_id),
        obj_origin_G=obj_origin_G,
        obj_quat_G_wxyz=obj_quat_G_wxyz,
        origin_G_W=obj_origin_G,        # alias
        table_z_G=table_z_G,
        mass_kg=float(mass_kg),
    )
    return ep, (f"{len(rec)} valid frames → kept {len(rec_kept)} ({onset_reason}) · "
                f"hand-obj dist@grasp {d_at_grasp*1000:.0f}mm")


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
