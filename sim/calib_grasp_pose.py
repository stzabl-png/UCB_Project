#!/usr/bin/env python3
"""Calibration diagnostic for the retarget→Franka grasp pose (pure geometry, no sim).

For each source episode, take the retarget grasp pose (action[-1]) → Franka panda_hand
convention, and measure the object point cloud's extent along the gripper's three axes:
  - close-axis  (panda_hand +Y, the finger open/close direction)  ← must be < 8cm
                  (gripper max opening) for the open gripper to fit AROUND the object
  - approach    (panda_hand +Z)
  - 3rd axis    (panda_hand +X)
plus the fingertip-center → object-centroid distance.

If the FAILING episodes systematically have e.g. close-axis extent > 8cm (gripper
closing along the bottle's LONG axis) while the SUCCESS has it small → the systematic
fix is a gripper-roll correction. Run:  python calib_grasp_pose.py ycb_dex_05
"""
import glob, os, sys
import numpy as np, h5py
from scipy.spatial.transform import Rotation

EP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "Baseline1/data/episodes_g")
FINGERTIP_OFFSET = 0.105      # panda_hand origin → fingertip, along +Z
GRIPPER_MAX_OPEN = 0.08       # Franka gripper max opening (m)
_RZ_NEG90 = Rotation.from_euler("z", -90, degrees=True)

def retarget_to_franka_R(q_wxyz):
    return Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]) * _RZ_NEG90

obj = sys.argv[1] if len(sys.argv) > 1 else "ycb_dex_05"
eps = sorted(glob.glob(f"{EP_DIR}/*{obj}*.hdf5"))
print(f"{obj}: {len(eps)} episodes   (gripper max opening = {GRIPPER_MAX_OPEN*100:.0f}cm)\n")
print(f"  {'#':>3} {'session':<16} {'close-ax':>9} {'approach':>9} {'3rd-ax':>8} "
      f"{'ftip→ctr':>9} {'fits?':>6}")

rows = []
for i, ep in enumerate(eps):
    with h5py.File(ep, "r") as h:
        action = h["action"][:]
        pc = h["point_cloud"][0]                       # object surface, G-frame
    gp = action[-1, :3]
    M = retarget_to_franka_R(action[-1, 3:7]).as_matrix()
    third, close_ax, approach = M[:, 0], M[:, 1], M[:, 2]
    ftip = gp + FINGERTIP_OFFSET * approach
    centroid = pc.mean(0)
    rel = pc - centroid
    ext_close = float(np.ptp(rel @ close_ax))
    ext_appr  = float(np.ptp(rel @ approach))
    ext_third = float(np.ptp(rel @ third))
    d_ftip = float(np.linalg.norm(ftip - centroid))
    fits = ext_close < GRIPPER_MAX_OPEN                # object fits within the open gripper?
    sess = os.path.basename(ep).split("__")[2]
    rows.append((ext_close, ext_appr, ext_third, d_ftip, fits))
    print(f"  {i+1:>3} {sess:<16} {ext_close*100:>7.1f}cm {ext_appr*100:>7.1f}cm "
          f"{ext_third*100:>6.1f}cm {d_ftip*100:>7.1f}cm {'YES' if fits else 'no':>6}")

a = np.array([(r[0], r[1], r[2], r[3]) for r in rows])
nfit = sum(r[4] for r in rows)
print(f"\n  close-axis extent  : mean {a[:,0].mean()*100:.1f}cm  range "
      f"{a[:,0].min()*100:.1f}-{a[:,0].max()*100:.1f}cm")
print(f"  fits in 8cm gripper: {nfit}/{len(rows)}")
print(f"  ftip→centroid      : mean {a[:,3].mean()*100:.1f}cm  range "
      f"{a[:,3].min()*100:.1f}-{a[:,3].max()*100:.1f}cm")
