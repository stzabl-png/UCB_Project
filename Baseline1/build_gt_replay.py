#!/usr/bin/env python3
"""
Build gt_replay-format HDF5 (G-frame, +Z up) for DexYCB grasp trajectories.

Two modes:
  (1) --single : one (session, cam, obj-class) → one HDF5 (used by sim/gt_replay_ikpd_v2.py)
  (2) --batch  : all (sessions × cams) for a subject + object class → many HDF5s
                 (replaces retarget_human_to_ee.py's DexYCB DP3-training output;
                  difference vs that script: this one emits G-frame, gravity-aligned)

Frame conventions:
  C : camera (per-camera) — raw labels live here
  W : DexYCB world = master cam (840412060917) C-frame; gravity is per-session, read
      from each extrinsics file's AprilTag (subject-09 ≈ (-0.017, 0.796, 0.605))
  G : gravity-aligned object-translated.
        +Z = -gravity, +Y_G = master cam's +Z projected onto horizontal plane,
        +X_G = +Y_G × +Z_G. Origin = object xy-centroid at frame 0, z = table.

Output HDF5 schema (compatible with Baseline1/convert_to_zarr.py):
  datasets:  state (T, 8) float32     [x, y, z, qw, qx, qy, qz, gripper]   G-frame
             action (T, 8) float32    = state shifted by one step
             point_cloud (T, N, 3) float32   object surface samples in G-frame
  attrs:     dataset, obj_id, mano_side, session, camera, ycb_class_id,
             n_steps, grasp_onset, origin_G_W, table_z_G,
             ee_offset_m, gripper_span_m

Usage:
  # single episode (Gate 3 verification target):
  python Baseline1/build_gt_replay.py --single \\
      --session 20201015_143403 --cam 840412060917 --obj-class 4 \\
      --out /tmp/gt_replay_ycb04_session143403_cam_master.hdf5

  # batch (subject-09 master_chef_can, all 5 sessions × 6 cameras = 30 episodes):
  python Baseline1/build_gt_replay.py --batch \\
      --subject 20201015-subject-09 --obj-class 1 \\
      --out-dir Baseline1/data/episodes_g
"""
import argparse, os, glob, json, sys
import numpy as np
import h5py
import yaml
import trimesh
from scipy.spatial.transform import Rotation


# ── constants ────────────────────────────────────────────────────────────────
PROJ            = "/home/accelerator/UCB_Project"
RAW             = f"{PROJ}/data_hub/RawData/ThirdPersonRawData/dexycb"
MESH_ROOT       = f"{PROJ}/data_hub/ProcessedData/obj_meshes/ycb"
SAM3D_ALIGN_DIR = f"{PROJ}/Baseline1/assets/sam3d_align"
# 6 cameras used by Phase 1A; matches retarget_human_to_ee.py:PIPELINE_CAMS
PIPELINE_CAMS = ["841412060263", "840412060917", "932122060857",
                 "836212060125", "932122061900", "932122062010"]
EE_OFFSET    = 0.10   # m — EE frame this far behind fingertip pinch midpoint
GRIPPER_SPAN = 0.08   # m — Franka 2-finger fingertip span (matches Baseline2)
GRASP_MARGIN = 0.04   # m — "grasp onset" = first frame ‖hand-obj‖ < d_min + this
MAX_PINCH_M  = 0.20   # thumb-index gap above this ⇒ bad joints ⇒ drop frame
MIN_FRAMES   = 5      # need at least this many valid frames to emit an episode
INVALID      = -0.5   # joint_3d / pose_y validity marker
# Gravity is NOT a global constant — DexYCB re-calibrates the camera rig per subject, so the
# master cam's tilt vs gravity differs between subjects (subjects 01-02 are ~13.5° off
# subjects 03-10). It is derived per session from the extrinsics AprilTag — see load_extrinsics().


class _L(yaml.SafeLoader): pass
_L.add_constructor("tag:yaml.org,2002:python/tuple",
                   lambda loader, node: tuple(loader.construct_sequence(node)))


# ── frame transforms ────────────────────────────────────────────────────────
def build_R_W_G(gravity_W):
    """Right-handed rotation: world (master cam C frame) → gravity-aligned G frame.
       +Z_G = -gravity; yaw chosen so master cam's forward (+Z_C in W) projects to +Y_G.
       gravity_W is the per-session gravity returned by load_extrinsics()."""
    g = np.asarray(gravity_W, dtype=np.float64)
    g = g / np.linalg.norm(g)
    up_G = -g
    fwd_W = np.array([0., 0., 1.])                        # master cam +Z_C expressed in W
    fwd_h = fwd_W - np.dot(fwd_W, up_G) * up_G
    y_G_in_W = fwd_h / np.linalg.norm(fwd_h)
    x_G_in_W = np.cross(y_G_in_W, up_G); x_G_in_W /= np.linalg.norm(x_G_in_W)
    y_G_in_W = np.cross(up_G, x_G_in_W); y_G_in_W /= np.linalg.norm(y_G_in_W)
    return np.column_stack([x_G_in_W, y_G_in_W, up_G]).T  # R @ v_W = v_G


def load_extrinsics(extr_id):
    """Read DexYCB calibration → (T_W←C per camera, master cam serial, gravity_W).

    gravity_W = -(AprilTag local +Z). The extrinsics file carries an 'apriltag' entry
    (T_W←AprilTag, 12-float row-major 3×4) alongside the 8 cameras; the AprilTag is fixed
    flat on the table, so its surface normal (+Z) is the anti-gravity direction. Read this
    per session — DexYCB re-calibrates the rig per subject, so gravity is NOT shared."""
    with open(f"{RAW}/calibration/extrinsics_{extr_id}/extrinsics.yml") as f:
        raw = yaml.load(f, Loader=_L)
    ext = raw["extrinsics"]
    T = {}
    for cam, rs in ext.items():
        if cam == "apriltag":
            continue
        m = np.eye(4); m[:3, :4] = np.array(rs).reshape(3, 4)
        T[cam] = m
    if "apriltag" not in ext:
        raise KeyError(f"extrinsics_{extr_id}: no 'apriltag' entry — cannot derive gravity")
    R_tag = np.array(ext["apriltag"]).reshape(3, 4)[:, :3]   # AprilTag → W rotation
    gravity_W = -R_tag[:, 2]                                 # tag local +Z = table normal = up
    gravity_W = gravity_W / np.linalg.norm(gravity_W)
    return T, raw.get("master"), gravity_W


_VALID_FRAMES_CACHE = {}
def compute_common_valid_frames(session_dir, cams_to_check, grasp_ind):
    """Return the set of raw frame indices that are valid across ALL `cams_to_check`.

       Why: DexYCB labels can be invalid (joint_3d marked with -1, pose_y all-zero) for
       the first few frames or whenever a finger is missed. Each cam has its own bad-frame
       set. If we let each cam emit episode rows from its own valid-frame set, then
       state[0] in different cams' HDF5 corresponds to different physical timestamps,
       and the (state, action) trajectories diverge by cm. Intersecting fixes this:
       every cam emits exactly the same set of physical frames, just from a different view."""
    key = (session_dir, tuple(sorted(cams_to_check)))
    if key in _VALID_FRAMES_CACHE:
        return _VALID_FRAMES_CACHE[key]
    per_cam_valid = []
    for cam in cams_to_check:
        cam_dir = f"{session_dir}/{cam}"
        if not os.path.isdir(cam_dir):
            continue
        ok = set()
        for lp in sorted(glob.glob(f"{cam_dir}/labels_*.npz")):
            try:
                lab = np.load(lp, allow_pickle=True)
                j   = np.asarray(lab["joint_3d"][0], dtype=np.float64)
                pm  = np.asarray(lab["pose_m"][0],   dtype=np.float64)
                py  = np.asarray(lab["pose_y"][grasp_ind], dtype=np.float64)
            except Exception:
                continue
            if np.any(j < INVALID) or not np.any(np.abs(pm) > 1e-6) or not np.any(np.abs(py) > 1e-6):
                continue
            ok.add(int(os.path.basename(lp).split("_")[1].split(".")[0]))
        per_cam_valid.append(ok)
    if not per_cam_valid:
        common = set()
    else:
        common = set.intersection(*per_cam_valid)
    _VALID_FRAMES_CACHE[key] = sorted(common)
    return _VALID_FRAMES_CACHE[key]


_ORIGIN_CACHE = {}
def compute_session_origin_G(session_dir, grasp_ind, pts_M, R_W_G, master_cam, T_W_C, first_frame_idx=0):
    """G-frame origin shared by all cameras within one session.
       Why: DexYCB labels are stored per-camera and the multi-view fits don't perfectly
       agree → each cam's frame-0 pose_y, transformed to W, lands at a slightly different
       centroid. If we let each cam compute its own origin, cross-cam state values drift
       by cm. Locking origin to one canonical source (master cam) fixes this.

       Returns origin_G_in_G = (centroid_x_G, centroid_y_G, table_z_G), all in G frame."""
    key = (session_dir, first_frame_idx)
    if key in _ORIGIN_CACHE:
        return _ORIGIN_CACHE[key]
    # Use the FIRST valid frame index from cross-cam intersection so the origin is
    # consistent with what every cam's HDF5 will emit as frame 0.
    lab0_path = f"{session_dir}/{master_cam}/labels_{first_frame_idx:06d}.npz"
    if not os.path.exists(lab0_path):
        cands = sorted(glob.glob(f"{session_dir}/{master_cam}/labels_*.npz"))
        if not cands:
            raise FileNotFoundError(f"no master cam labels under {session_dir}/{master_cam}")
        lab0_path = cands[0]
    lab0 = np.load(lab0_path, allow_pickle=True)
    py0 = lab0["pose_y"][grasp_ind]                                # (3,4) in master cam C frame
    T_master = T_W_C.get(master_cam, np.eye(4))
    # transform object pts to W (will be identity for master, but written generally)
    pc0_C = (py0[:, :3] @ pts_M.T).T + py0[:, 3]
    pc0_W = (T_master[:3, :3] @ pc0_C.T).T + T_master[:3, 3]
    pc0_G = (R_W_G @ pc0_W.T).T
    centroid_G = pc0_G.mean(axis=0)
    table_z_G = float(np.percentile(pc0_G[:, 2], 1))
    origin = np.array([centroid_G[0], centroid_G[1], table_z_G])
    _ORIGIN_CACHE[key] = origin
    return origin


# YCB class id → DexYCB CAD model folder. Phase 1 is the CAD-first path, so the point
# cloud is sampled from the dataset's own CAD model (textured.obj) — the SAME mesh that
# pose_y is defined against and that the sim USD is converted from. (The old code sampled
# the SAM3D neural reconstruction in obj_meshes/ycb/, whose shape is a poor approximation
# — e.g. tomato_soup_can SAM3D bbox 7.6×8.6×7.9cm vs CAD 6.8×6.8×10.2cm — and which
# doesn't even exist for some objects like foam_brick.)
YCB_CLASS_TO_CAD = {
    1: "002_master_chef_can", 2: "003_cracker_box", 3: "004_sugar_box",
    4: "005_tomato_soup_can", 5: "006_mustard_bottle", 6: "007_tuna_fish_can",
    7: "008_pudding_box", 8: "009_gelatin_box", 9: "010_potted_meat_can",
    10: "011_banana", 11: "019_pitcher_base", 12: "021_bleach_cleanser",
    13: "024_bowl", 14: "025_mug", 15: "035_power_drill", 16: "036_wood_block",
    17: "037_scissors", 18: "040_large_marker", 19: "051_large_clamp",
    20: "052_extra_large_clamp", 21: "061_foam_brick",
}

# ── object mesh: DexYCB CAD model (textured.obj), already in real metres + CAD frame ─
_MESH_CACHE = {}
def get_object_points(ycb_class_id, n_points, sample_seed=42):
    """Surface-sample the DexYCB CAD model. Returns (obj_id, pts) with pts in the CAD
       coordinate frame (= the frame pose_y is defined against). No scale.json / no
       SAM3D alignment needed: CAD textured.obj is already in metres and in CAD frame.
       Seeded sampling → byte-identical point clouds across runs."""
    key = (ycb_class_id, n_points, sample_seed)
    if key in _MESH_CACHE:
        return _MESH_CACHE[key]
    obj_id = f"ycb_dex_{ycb_class_id:02d}"
    cad_name = YCB_CLASS_TO_CAD.get(ycb_class_id)
    if cad_name is None:
        raise ValueError(f"no CAD model mapping for ycb_class_id {ycb_class_id}")
    mesh_path = f"{RAW}/models/{cad_name}/textured.obj"
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"CAD mesh missing: {mesh_path}")
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    rng = np.random.default_rng(sample_seed)
    pts, _ = trimesh.sample.sample_surface(mesh, n_points, seed=int(rng.integers(0, 2**31)))
    _MESH_CACHE[key] = (obj_id, np.asarray(pts, dtype=np.float64))
    return _MESH_CACHE[key]


# ── grasp-onset detection: finger curl (size-invariant, replaces distance heuristic) ─
# MANO non-thumb fingers: (MCP, tip) index pairs. ‖tip − MCP‖ is large when the finger
# is extended, small when curled. The old grasp_onset used EE-to-object-centroid distance,
# which fails on elongated objects (the hand passes near the centroid long before it
# actually closes). Finger curl is purely about the hand and is object-size independent.
_FLEX_PAIRS = [(5, 8), (9, 12), (13, 16), (17, 20)]   # index, middle, ring, pinky
_D_EXT, _D_FLEX = 0.085, 0.025                         # m — ‖tip−MCP‖ extended / curled
def finger_curl(joint_3d):
    """Mean curl of the 4 non-thumb fingers ∈ [0,1]  (0 = extended, 1 = fully curled)."""
    vals = []
    for mcp, tip in _FLEX_PAIRS:
        d = float(np.linalg.norm(joint_3d[tip] - joint_3d[mcp]))
        vals.append(np.clip((_D_EXT - d) / (_D_EXT - _D_FLEX), 0.0, 1.0))
    return float(np.mean(vals))


# ── retarget: MANO 21 joints (in some frame) → (p_ee, R_ee) in same frame ────
def mano_to_ee_thumb_index(j):
    """Legacy retarget: ex = thumb→index axis, ez = wrist→pinch_midpoint ⊥ ex.
       Issue: ex is noisy on in-flight frames (when thumb/index aren't anchored on object).
       Kept for back-compat; new pipelines should use mano_to_ee_rigid_body."""
    w, t, i = j[0], j[4], j[8]
    pinch = i - t; pd = np.linalg.norm(pinch)
    if pd < 1e-4 or pd > MAX_PINCH_M: return None
    ex = pinch / pd
    fwd = (t + i) * 0.5 - w
    fwd_n = np.linalg.norm(fwd)
    if fwd_n < 1e-3: return None
    fwd = fwd / fwd_n
    ez = fwd - np.dot(fwd, ex) * ex
    ez_n = np.linalg.norm(ez)
    if ez_n < 1e-3: return None
    ez = ez / ez_n
    ey = np.cross(ez, ex)
    R = np.column_stack([ex, ey, ez])
    p = (t + i) * 0.5 - EE_OFFSET * ez
    return p, R


def rigid_transform_3D(A, B):
    """Kabsch algorithm: find rigid (R, t) such that R @ A.T + t ≈ B.T.
       A, B: (N, 3) point sets, point-to-point correspondence assumed."""
    cA = A.mean(axis=0); cB = B.mean(axis=0)
    AA = A - cA; BB = B - cB
    H = AA.T @ BB
    U, _, Vt = np.linalg.svd(H)
    Rm = Vt.T @ U.T
    if np.linalg.det(Rm) < 0:                            # reflection correction
        Vt[-1, :] *= -1
        Rm = Vt.T @ U.T
    t = cB - Rm @ cA
    return Rm, t


# Initial EE orientation when retarget_mode = "rigid_body" (Point-Policy convention).
# rotvec [π, 0, 0] = rotation 180° around +X — gripper points "down" (top-down grasp ready).
# Hand rotations from frame 0 are then composed onto this fixed initial pose.
_R_EE_INITIAL_TOP_DOWN = Rotation.from_rotvec([np.pi, 0, 0]).as_matrix()


# Subset of MANO joints used by the rigid-body retarget. Point-Policy uses 9 keypoints
# (wrist + thumb-chain + index-chain) and explicitly excludes middle/ring/pinky tips —
# those curl heavily during grasp closure, and including them makes the rigid-body
# rotation estimate "explain finger curl as wrist twist" → wild EE orientation that IK
# can't reach. Limiting to (wrist + thumb + index) tracks the PALM orientation cleanly.
_RIGID_BODY_JOINT_IDS = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])  # wrist + thumb 1-4 + index 5-8

def mano_to_ee_rigid_body_factory(j_frame0):
    """Stateful Point-Policy-style retarget. Tracks the wrist+thumb+index sub-cloud's
       rigid rotation relative to frame 0, applies it to a fixed initial "top-down"
       gripper orientation. Robust to per-frame finger noise."""
    R0 = _R_EE_INITIAL_TOP_DOWN.copy()
    j0_sub = j_frame0[_RIGID_BODY_JOINT_IDS].copy()
    def _retarget(j):
        # Position: midpoint between thumb tip and index tip (same as legacy formula)
        t, i = j[4], j[8]
        pinch = i - t; pd = np.linalg.norm(pinch)
        if pd < 1e-4 or pd > MAX_PINCH_M: return None
        # Rigid rotation of the (wrist + thumb + index) sub-cloud relative to frame 0
        rot, _ = rigid_transform_3D(j0_sub, j[_RIGID_BODY_JOINT_IDS])
        R = rot @ R0
        # EE pos: pinch midpoint, then back along the gripper's approach axis (local +Z)
        p_mid = (t + i) * 0.5
        approach = R[:, 2]
        p = p_mid - EE_OFFSET * approach
        return p, R
    return _retarget


# ── single (session, cam, obj_class) → episode dict (or None + reason) ───────
def build_one_episode(session_id, cam, obj_class_id, n_points, verbose=False, retarget_mode="thumb_index"):
    """Returns dict(state, action, point_cloud, grasp_onset, origin_G_W, table_z_G, ...)
       or (None, reason_str)."""
    # locate session dir
    matches = [s for s in os.listdir(RAW)
               if os.path.isdir(f"{RAW}/{s}") and session_id in os.listdir(f"{RAW}/{s}")]
    if not matches: return None, f"session {session_id} not under any subject in {RAW}"
    subj = matches[0]
    sd = f"{RAW}/{subj}/{session_id}"
    with open(f"{sd}/meta.yml") as f: meta = yaml.load(f, Loader=_L)
    extr_id = meta["extrinsics"]
    ycb_ids = list(meta["ycb_ids"]); grasp_ind = int(meta["ycb_grasp_ind"])
    if ycb_ids[grasp_ind] != obj_class_id:
        return None, f"grasp object {ycb_ids[grasp_ind]} ≠ requested {obj_class_id}"
    side = (meta.get("mano_sides") or ["right"])[0]

    T_W_C, _master, gravity_W = load_extrinsics(extr_id)
    master_cam = _master or "840412060917"
    if cam not in T_W_C: return None, f"cam {cam} not in extrinsics {extr_id}"
    T = T_W_C[cam]
    R_W_G = build_R_W_G(gravity_W)
    if verbose:
        print(f"  extrinsics={extr_id}  gravity_W={np.round(gravity_W, 4)}")

    obj_id, pts_M = get_object_points(obj_class_id, n_points)

    if not os.path.isdir(f"{sd}/{cam}"): return None, f"no labels dir for cam {cam}"

    # ── compute common valid-frame set across all PIPELINE_CAMS (sync per-cam timing) ──
    valid_frame_ids = compute_common_valid_frames(sd, PIPELINE_CAMS, grasp_ind)
    if len(valid_frame_ids) < MIN_FRAMES:
        return None, f"only {len(valid_frame_ids)} frames valid across all {len(PIPELINE_CAMS)} cams"
    first_frame_idx = valid_frame_ids[0]

    # ── session-level G-frame origin (master cam @ first_frame_idx → cross-cam consistency) ──
    try:
        origin_G_in_G = compute_session_origin_G(sd, grasp_ind, pts_M, R_W_G, master_cam, T_W_C,
                                                  first_frame_idx=first_frame_idx)
    except FileNotFoundError as e:
        return None, str(e)
    table_z_G = float(origin_G_in_G[2])
    origin_G_W = np.linalg.inv(R_W_G) @ origin_G_in_G

    # ── object FULL pose (position + orientation) at frame 0 in G frame ──
    # The trajectory is object-centric (G-frame): EE state and object point cloud share
    # the SAME frame. For sim to be consistent, the object must be placed at EXACTLY its
    # G-frame pose (then offset by sim_origin). If sim instead lets physics settle the
    # object, it lands at an uncontrolled spot and the EE-vs-object geometry breaks
    # (verified: retarget HDF5 has fingertip 1-2cm from object surface, but physics-settled
    # sim had it 12cm off). So we store the object's exact G-frame pose here and sim
    # places it kinematically — no settling.
    lab0_master = np.load(f"{sd}/{master_cam}/labels_{first_frame_idx:06d}.npz", allow_pickle=True)
    py0_master = lab0_master["pose_y"][grasp_ind]                    # (3,4) [R|t] in master cam C = W
    R_obj_W = py0_master[:, :3]                                      # CAD→W rotation
    t_obj_W = py0_master[:, 3]                                       # object origin in W
    R_obj_G = R_W_G @ R_obj_W                                        # CAD→G rotation
    obj_origin_G = R_W_G @ t_obj_W - origin_G_in_G                   # object origin in G frame
    obj_q_xyzw = Rotation.from_matrix(R_obj_G).as_quat()
    obj_quat_G_wxyz = np.array([obj_q_xyzw[3], obj_q_xyzw[0], obj_q_xyzw[1], obj_q_xyzw[2]], dtype=np.float64)

    # ── pick retarget function based on mode ─────────────────────────────────
    # For rigid_body mode (Point-Policy style), we need the first valid frame's joints
    # to anchor the Kabsch rotation reference. Load it once.
    retarget_fn = None
    if retarget_mode == "rigid_body":
        lab_first = np.load(f"{sd}/{cam}/labels_{valid_frame_ids[0]:06d}.npz", allow_pickle=True)
        j_frame0_C = np.asarray(lab_first["joint_3d"][0], dtype=np.float64)
        # transform frame-0 joints to W (so Kabsch rotation is computed in W consistently)
        j_frame0_W = (T[:3, :3] @ j_frame0_C.T).T + T[:3, 3]
        retarget_fn = mano_to_ee_rigid_body_factory(j_frame0_W)
    elif retarget_mode == "thumb_index":
        retarget_fn = lambda j_W: mano_to_ee_thumb_index(j_W)
    else:
        return None, f"unknown retarget_mode: {retarget_mode}"

    # ── walk only common-valid frames so frame_id alignment is identical across cams ──
    rec = []
    for fid in valid_frame_ids:
        lp = f"{sd}/{cam}/labels_{fid:06d}.npz"
        try:
            lab = np.load(lp, allow_pickle=True)
            j_C  = np.asarray(lab["joint_3d"][0], dtype=np.float64)
            pm   = np.asarray(lab["pose_m"][0],   dtype=np.float64)
            py   = np.asarray(lab["pose_y"][grasp_ind], dtype=np.float64)
        except Exception:
            continue
        if np.any(j_C < INVALID) or not np.any(np.abs(pm) > 1e-6) or not np.any(np.abs(py) > 1e-6):
            continue
        # Transform joints to W frame *first* so retarget operates in W (and rigid_body's
        # Kabsch reference is in W too — comparing apples to apples across frames).
        j_W = (T[:3, :3] @ j_C.T).T + T[:3, 3]
        ret = retarget_fn(j_W)
        if ret is None: continue
        p_ee_W, R_ee_W = ret                            # retarget now works in W frame directly
        obj_t_W = (T @ np.append(py[:, 3], 1))[:3]
        p_ee_G = R_W_G @ p_ee_W - origin_G_in_G
        obj_G  = R_W_G @ obj_t_W - origin_G_in_G
        R_ee_G = R_W_G @ R_ee_W
        q_xyzw = Rotation.from_matrix(R_ee_G).as_quat()
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        pc_W = (py[:, :3] @ pts_M.T).T + py[:, 3]
        pc_G = (R_W_G @ pc_W.T).T - origin_G_in_G
        rec.append(dict(p=p_ee_G.astype(np.float32), q=q_wxyz.astype(np.float32),
                        oc=obj_G.astype(np.float32), pc=pc_G.astype(np.float32),
                        curl=finger_curl(j_C)))   # curl is distance-based → frame-invariant

    if len(rec) < MIN_FRAMES:
        return None, f"only {len(rec)} valid frames"

    # quaternion sign-continuity
    for k in range(1, len(rec)):
        if np.dot(rec[k]["q"], rec[k - 1]["q"]) < 0.0:
            rec[k]["q"] = -rec[k]["q"]

    # ── grasp-onset = the frame the OBJECT starts rising (= grasp complete, lift begins) ──
    # Baseline1 scope: DP3 learns the approach trajectory only (frame 0 → grasp_onset);
    # the trajectory is TRUNCATED at onset (no lift/carry).
    # Why object-lift, not hand-object distance: during the carry phase the hand holds the
    # object so |EE−obj| stays small → argmin(distance) lands on an arbitrary carry frame.
    # The object's GT pose_y Z rising is an unambiguous "grasp happened" signal, and it is
    # completely object-size independent (unlike finger curl, whose magnitude depends on
    # how far the fingers wrap — small for a fat can, large for a thin marker).
    oc_z = np.array([float(r["oc"][2]) for r in rec])
    oc_z_baseline = float(oc_z[:min(5, len(oc_z))].mean())       # object resting Z on the table
    lift_frames = np.where(oc_z > oc_z_baseline + 0.02)[0]       # object rose ≥ 2cm
    if len(lift_frames) > 0:
        onset = int(lift_frames[0])                              # grasp done, lift just starting
        onset_reason = f"object lift (+2cm) at frame {onset}"
    else:
        # object never lifts — fall back to finger curl, then closest hand-object approach
        curls = np.array([r["curl"] for r in rec])
        base_curl = float(curls[:min(5, len(curls))].mean())
        cand = np.where(curls > base_curl + 0.10)[0]
        if len(cand) > 0:
            onset = int(cand[0]); onset_reason = f"finger curl > {base_curl+0.10:.2f} at frame {onset}"
        else:
            d = np.array([float(np.linalg.norm(r["p"] - r["oc"])) for r in rec])
            onset = int(np.argmin(d)); onset_reason = f"closest hand-object approach at frame {onset}"
    onset = max(onset, MIN_FRAMES)                               # need a few approach frames
    onset = min(onset, len(rec) - 1)

    # Truncate to the approach segment: frames 0 .. onset (inclusive).
    rec = rec[:onset + 1]
    gripper = np.zeros(len(rec), dtype=np.float64)               # gripper stays open the whole approach
    gripper[-1] = 1.0                                            # mark final frame = "arrived, would close"

    # pack: state[t], action[t] = state[t+1]
    states_full, pcs_full = [], []
    for k, r in enumerate(rec):
        states_full.append(np.concatenate([r["p"], r["q"], [gripper[k]]]).astype(np.float32))
        pcs_full.append(r["pc"])
    states_full = np.stack(states_full)
    if len(states_full) < 2:
        return None, "too short after processing"
    state  = states_full[:-1]
    action = states_full[1:]
    pcs    = np.stack(pcs_full[:-1])

    if verbose:
        bbox = pts_M.max(0) - pts_M.min(0)
        print(f"  cam={cam}  T_emit={len(state)}  onset={onset} ({onset_reason})  "
              f"state[0]={state[0].round(3)}  mesh_bbox={bbox.round(3)}")

    # Trajectory is already truncated AT the grasp moment, so the whole emitted sequence
    # is the approach. grasp_onset = n_steps means "gripper-close would happen one frame
    # past the end" → sim replay won't trigger a close (Baseline1 scope: no close/lift).
    n_steps = int(state.shape[0])
    return dict(
        state=state, action=action, point_cloud=pcs,
        grasp_onset=n_steps, origin_G_W=origin_G_W.astype(np.float64), table_z_G=table_z_G,
        obj_quat_G_wxyz=obj_quat_G_wxyz,                          # GT object orientation in G
        obj_origin_G=obj_origin_G.astype(np.float64),             # GT object origin position in G
        n_steps=n_steps,
        obj_id=obj_id, ycb_class_id=int(obj_class_id),
        session=session_id, camera=cam, subject=subj, mano_side=side,
    ), None


def write_hdf5(out_path, ep):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with h5py.File(out_path, "w") as h:
        h.create_dataset("state",       data=ep["state"])
        h.create_dataset("action",      data=ep["action"])
        h.create_dataset("point_cloud", data=ep["point_cloud"])
        # attrs (compatible with both convert_to_zarr.py and sim/gt_replay_ikpd_v2.py)
        h.attrs["dataset"]         = "dexycb"
        h.attrs["obj_id"]          = ep["obj_id"]
        h.attrs["ycb_class_id"]    = ep["ycb_class_id"]
        h.attrs["session"]         = ep["session"]
        h.attrs["camera"]          = ep["camera"]
        h.attrs["subject"]         = ep["subject"]
        h.attrs["mano_side"]       = ep["mano_side"]
        h.attrs["n_steps"]         = ep["n_steps"]
        h.attrs["grasp_onset"]     = ep["grasp_onset"]
        h.attrs["grasp_onset_idx"] = ep["grasp_onset"]   # alias for sim/gt_replay_ikpd_v2.py
        h.attrs["origin_G_W"]      = ep["origin_G_W"]
        h.attrs["table_z_G"]       = float(ep["table_z_G"])
        h.attrs["obj_quat_G_wxyz"] = ep["obj_quat_G_wxyz"]       # GT object orientation in G (for sim spawn)
        h.attrs["obj_origin_G"]    = ep["obj_origin_G"]          # GT object origin position in G (for sim spawn)
        h.attrs["ee_offset_m"]     = EE_OFFSET
        h.attrs["gripper_span_m"]  = GRIPPER_SPAN


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--single", action="store_true", help="one (session,cam,obj) → one HDF5")
    mode.add_argument("--batch",  action="store_true", help="iterate subject's sessions × cams")
    ap.add_argument("--session",   help="single mode: full session id, e.g. 20201015_143403")
    ap.add_argument("--cam",       help="single mode: camera serial, e.g. 840412060917 (master)")
    ap.add_argument("--obj-class", type=int, required=True, help="YCB class id (1=master_chef_can, 4=tomato_soup_can …)")
    ap.add_argument("--out",       help="single mode: HDF5 output path")
    ap.add_argument("--subject",   help="batch mode: e.g. 20201015-subject-09")
    ap.add_argument("--out-dir",   help="batch mode: directory for {dataset}__{subj}__{sess}__{cam}__{obj}.hdf5")
    ap.add_argument("--cams",      nargs="+", default=None, help="batch mode: override camera list (default: PIPELINE_CAMS)")
    ap.add_argument("--n-points",  type=int, default=4096)
    ap.add_argument("--retarget-mode", choices=["thumb_index", "rigid_body"], default="thumb_index",
                    help="thumb_index = legacy ex=thumb→index heuristic (per-frame); "
                         "rigid_body = Point-Policy style (Kabsch on whole hand vs frame 0)")
    ap.add_argument("--ee-offset", type=float, default=EE_OFFSET,
                    help="m — panda_hand-to-pinch-midpoint distance (default 0.10)")
    args = ap.parse_args()
    # Override module-level EE_OFFSET so the retarget closures see the new value
    globals()["EE_OFFSET"] = args.ee_offset

    if args.single:
        if not (args.session and args.cam and args.out):
            sys.exit("single mode needs --session --cam --out")
        ep, reason = build_one_episode(args.session, args.cam, args.obj_class, args.n_points, verbose=True, retarget_mode=args.retarget_mode)
        if ep is None:
            sys.exit(f"FAIL: {reason}")
        write_hdf5(args.out, ep)
        print(f"✓ wrote {args.out}  (n_steps={ep['n_steps']}, onset={ep['grasp_onset']})")
        return

    # batch mode
    if not (args.subject and args.out_dir):
        sys.exit("batch mode needs --subject --out-dir")
    cams = args.cams or PIPELINE_CAMS
    subj_dir = f"{RAW}/{args.subject}"
    sessions = sorted(s for s in os.listdir(subj_dir) if os.path.isdir(f"{subj_dir}/{s}"))
    # filter to sessions whose grasp object matches obj_class
    keep = []
    for sess in sessions:
        try:
            with open(f"{subj_dir}/{sess}/meta.yml") as f: meta = yaml.load(f, Loader=_L)
        except Exception: continue
        ycb_ids = list(meta.get("ycb_ids", [])); gi = int(meta.get("ycb_grasp_ind", -1))
        if 0 <= gi < len(ycb_ids) and int(ycb_ids[gi]) == args.obj_class:
            keep.append(sess)
    print(f"batch: subject={args.subject}  obj_class={args.obj_class}  → {len(keep)} sessions × {len(cams)} cams")
    os.makedirs(args.out_dir, exist_ok=True)
    n_ok = n_fail = 0
    for sess in keep:
        for cam in cams:
            ep, reason = build_one_episode(sess, cam, args.obj_class, args.n_points, verbose=False, retarget_mode=args.retarget_mode)
            obj_short = f"ycb_dex_{args.obj_class:02d}"
            out = f"{args.out_dir}/dexycb__{args.subject}__{sess}__{cam}__{obj_short}.hdf5"
            if ep is None:
                print(f"  ✗ {sess}/{cam}: {reason}"); n_fail += 1
                continue
            write_hdf5(out, ep)
            print(f"  ✓ {sess}/{cam}: T={ep['n_steps']}  onset={ep['grasp_onset']}")
            n_ok += 1
    print(f"\nbatch done: {n_ok} OK, {n_fail} failed → {args.out_dir}")


if __name__ == "__main__":
    main()
