#!/usr/bin/env python3
"""Batch cuRobo IK reachability test for baseline_3, on run_grasp_sim.py's scene.

For every retargeted episode in Baseline1/data/episodes_g/, take the FINAL
(grasp-onset) retargeted EE pose = the contact grasp pose, place the object at
run_grasp_sim.py's FIXED OBJECT_POSITION xy, and test whether
  (a) the grasp pose and
  (b) a 12 cm pre-grasp pose (backed off along the gripper approach axis)
are cuRobo-IK-reachable from run_grasp_sim.py's FIXED Franka base.

Pure IK reachability (no collision world), tolerance = gt_replay defaults
(5 mm / 0.05 rad). Shells out to curobo_ik.py --solve, exactly the path
gt_replay uses; ok[i] == "pose i has >=1 IK solution within tolerance"
(the continuity chain only picks WHICH branch, never the reachable verdict).
"""
import os, sys, glob, subprocess
import numpy as np
import h5py
from scipy.spatial.transform import Rotation

SIM_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(SIM_DIR)
EPISODES = sorted(glob.glob(os.path.join(PROJ, "Baseline1/data/episodes_g/*.hdf5")))

# ── run_grasp_sim.py scene config (FIXED — current state) ────────────────────
ROBOT_POS = np.array([0.2, -0.05, 0.8])      # run_grasp_sim ROBOT_POSITION
ROBOT_ORI = np.array([0.0, 0.0, 90.0])       # run_grasp_sim ROBOT_ORIENTATION
OBJECT_XY = np.array([0.0, 0.55])            # run_grasp_sim OBJECT_POSITION xy
TABLE_TOP_Z = 0.80
PREGRASP_BACKOFF_M = 0.12
POS_TOL, ORI_TOL, NUM_SEEDS = 0.005, 0.05, 1024   # gt_replay defaults

_SWAP = Rotation.from_euler("z", -90, degrees=True)
def axis_swap(q_wxyz):
    """raw retarget quat (wxyz) -> Franka panda_hand convention (wxyz);
    identical to gt_replay.retarget_to_franka_quat (post-multiply Rz(-90deg))."""
    r = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]) * _SWAP
    o = r.as_quat()
    return np.array([o[3], o[0], o[1], o[2]])

YCB_NAME = {1:"master_chef_can",2:"cracker_box",3:"sugar_box",4:"tomato_soup_can",
    5:"mustard_bottle",6:"tuna_fish_can",7:"pudding_box",8:"gelatin_box",
    9:"potted_meat_can",10:"banana",11:"pitcher_base",12:"bleach_cleanser",
    13:"bowl",14:"mug",15:"power_drill",16:"wood_block",17:"scissors",
    18:"large_marker",19:"large_clamp",20:"extra_large_clamp",21:"foam_brick"}

# ── build the grasp + pre-grasp pose batches ─────────────────────────────────
grasp_pos, grasp_quat, pre_pos, cls = [], [], [], []
for ep in EPISODES:
    with h5py.File(ep, "r") as h:
        actions = h["action"][:]
        if "obj_origin_G" not in h.attrs:
            continue
        obj_origin_G = np.array(h.attrs["obj_origin_G"], dtype=np.float64)
        cid = int(h.attrs["ycb_class_id"])
    # gt_replay convention: world = G-frame + sim_origin_W; object placed at
    # obj_origin_G + sim_origin_W. Choose sim_origin_W so the object lands at
    # run_grasp_sim's fixed OBJECT_POSITION xy (z keeps the object on the table).
    sim_origin_W = np.array([-obj_origin_G[0], OBJECT_XY[1] - obj_origin_G[1], TABLE_TOP_Z])
    g_pos = actions[-1, :3] + sim_origin_W                 # final retargeted EE = grasp
    g_quat = axis_swap(actions[-1, 3:7])
    R = Rotation.from_quat([g_quat[1], g_quat[2], g_quat[3], g_quat[0]])
    approach = R.apply([0.0, 0.0, 1.0])                    # panda_hand +z = approach axis
    grasp_pos.append(g_pos); grasp_quat.append(g_quat)
    pre_pos.append(g_pos - PREGRASP_BACKOFF_M * approach)
    cls.append(cid)

grasp_pos = np.array(grasp_pos); grasp_quat = np.array(grasp_quat)
pre_pos = np.array(pre_pos); cls = np.array(cls)
N = len(cls)
print(f"loaded {N} episodes  |  object @ ({OBJECT_XY[0]:.2f},{OBJECT_XY[1]:.2f}), "
      f"base {ROBOT_POS.tolist()} yaw {ROBOT_ORI[2]:.0f}deg, tol {POS_TOL*1000:.0f}mm/{ORI_TOL}",
      flush=True)
# base<->object horizontal distance (info)
d = np.hypot(*(OBJECT_XY - ROBOT_POS[:2]))
print(f"base->object xy distance = {d:.3f} m", flush=True)

# ── solve each batch via curobo_ik.py --solve (same path gt_replay uses) ─────
PY = "/home/accelerator/miniforge3/envs/env_isaaclab/bin/python"
CIK = os.path.join(SIM_DIR, "curobo_ik.py")
def solve(pos, quat, tag):
    fin, fout = f"/tmp/ikreach_{tag}_in.npz", f"/tmp/ikreach_{tag}_out.npz"
    np.savez(fin, pos=pos, quat=quat, robot_pos=ROBOT_POS, robot_ori=ROBOT_ORI,
             num_seeds=NUM_SEEDS, pos_tol=POS_TOL, ori_tol=ORI_TOL)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([PY, CIK, "--solve", fin, fout], env=env,
                       capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(fout):
        print(f"FAILED {tag} (rc={r.returncode})\nSTDERR:\n{r.stderr[-3000:]}")
        sys.exit(1)
    print(f"  [{tag}] {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ''}", flush=True)
    return np.load(fout)["ok"].astype(bool)

print("solving grasp poses ...", flush=True)
ok_grasp = solve(grasp_pos, grasp_quat, "grasp")
print("solving pre-grasp poses ...", flush=True)
ok_pre = solve(pre_pos, grasp_quat, "pre")
ok_both = ok_grasp & ok_pre

# ── report ───────────────────────────────────────────────────────────────────
def pct(m): return f"{int(m.sum()):3d}/{len(m):3d} ({100.0*m.mean():5.1f}%)"
print("\n" + "=" * 70)
print(f"  cuRobo IK reachability  —  run_grasp_sim.py scene (current)   N={N}")
print("=" * 70)
print(f"  grasp pose reachable        : {pct(ok_grasp)}")
print(f"  pre-grasp (12cm) reachable  : {pct(ok_pre)}")
print(f"  BOTH reachable              : {pct(ok_both)}   <- baseline_3 usable")
print("-" * 70)
print(f"  {'object':<20}{'grasp':>15}{'pre-grasp':>15}{'both':>15}")
for cid in sorted(set(cls.tolist())):
    m = cls == cid
    print(f"  {YCB_NAME.get(cid, 'cls%d'%cid):<20}"
          f"{pct(ok_grasp[m]):>15}{pct(ok_pre[m]):>15}{pct(ok_both[m]):>15}")
print("=" * 70)
