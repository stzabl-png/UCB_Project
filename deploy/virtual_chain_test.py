"""Virtual full-chain test of the DP3 real-robot deployment pipeline (5090, NO robot).

Runs the EXACT deployment loop (dp3_loop.run_closed_loop) + frame retarget (dp3_frames)
+ IK (teleop PinkLocalIK — the deployment solver) against a "virtual robot" whose joint
state we track ourselves via FK instead of real hardware. This exercises the whole chain:

  proprio (full-hand FK -> virtual_pinch -> pinch_to_proprio)  →  DP3 server /predict
  →  8 absolute EE poses (G frame)  →  action_to_base_ee  →  PinkLocalIK (z-clamp + pos/ori gate)
  →  "drive" (we set the virtual joints)  →  re-measure proprio  →  next chunk.

It also measures the per-waypoint and per-chunk-boundary joint deltas (the machine-wear
concern) so we can see how much the 300 Hz velocity-limited smoother (which the REAL robot
uses, see SmoothingAndSafetyManager) must interpolate.

Run:  /home/accelerator/miniforge3/envs/ikcheck/bin/python deploy/virtual_chain_test.py
Needs the DP3 inference server up on 127.0.0.1:8765.
"""
import sys, types
sys.path.insert(0, "/home/accelerator/V2AP-demo")
sys.path.insert(0, "/home/accelerator/V2AP-demo/demo/phase2/dp3")
sys.modules["pinocchio.casadi"] = types.ModuleType("pinocchio.casadi")  # PinkLocalIK doesn't use it

import numpy as np
import pinocchio as pin
import dp3_frames as F
from dp3_client import DP3Client
from dp3_loop import DP3Callbacks, run_closed_loop
from demo.phase1.constants import DEFAULT_JOINT_POS
from demo.phase1.config_io import load_hand_profile
from demo.phase1.grasp_geometry import se3_to_homogeneous
from teleop.ik_utils import PinkLocalIK
from teleop.robot_descriptions import build_full_robot
from demo.phase2.hand_retarget_geometry import virtual_pinch_frame_in_base, resolve_T_ee_pinch_closed
from demo.phase2.ee_retarget_io import DEFAULT_EE_RETARGET_YAML, load_ee_retarget
try:
    from teleop.arm_hand_control import DEXMATE_DEFAULT_ARM_VEL_LIMITS, DEXMATE_VEL_LIMIT_SCALE
    PER_TICK = DEXMATE_VEL_LIMIT_SCALE * np.asarray(DEXMATE_DEFAULT_ARM_VEL_LIMITS) / 300.0  # rad/tick
except Exception:
    PER_TICK = None

IK_POS_TOL_M, IK_ORI_TOL_DEG, TABLE_MARGIN_M = 0.03, 15.0, 0.02
STATE0 = np.array([0.20, -0.293, 0.59, 0.0, 1.0, 0.0, 0.0, 0.0])

# ── deployment frame: theta=0 (R=I), object bottom-center at base x=0.40 ──
ORIGIN = np.array([0.40, -0.01, 0.8845])
T_BASE_MESH = np.eye(4); T_BASE_MESH[:3, 3] = ORIGIN
PINCH_Z_FLOOR = ORIGIN[2] + TABLE_MARGIN_M

# ── models (no hardware) ──
PIK = PinkLocalIK({k: v.copy() for k, v in DEFAULT_JOINT_POS.items()})
FULL, ASM, _ = build_full_robot({k: v.copy() for k, v in DEFAULT_JOINT_POS.items()})
T_EE_PINCH = resolve_T_ee_pinch_closed(load_ee_retarget(DEFAULT_EE_RETARGET_YAML))
OPEN_Q, CLOSED_Q = load_hand_profile()
LEFT_Q = np.asarray(DEFAULT_JOINT_POS["left_arm"], dtype=np.float64)


def _se3(T):
    T = np.asarray(T); return pin.SE3(T[:3, :3].copy(), T[:3, 3].copy())

def _ree_fk(qr):
    return se3_to_homogeneous(PIK.fk(frames=["R_ee"],
        joint_pos_by_component={"left_arm": LEFT_Q, "right_arm": qr})["R_ee"])

def _pose_err(reached, target):
    pe = float(np.linalg.norm(reached[:3, 3] - target[:3, 3]))
    Re = reached[:3, :3].T @ target[:3, :3]
    oe = float(np.degrees(np.arccos(max(-1.0, min(1.0, (np.trace(Re) - 1.0) / 2.0)))))
    return pe, oe


class VirtualRobot:
    """Tracks (right_arm, right_hand) via FK; the 6 dp3_loop callbacks drive it in software."""
    def __init__(self):
        self.qr = np.asarray(DEFAULT_JOINT_POS["right_arm"], dtype=np.float64)
        self.rh = OPEN_Q.copy()
        self.waypoint_deltas, self.chunk_boundary_deltas = [], []
        self.ik_ok, self.ik_fail = 0, 0
        self._prev_chunk_last_q = None
        self._in_chunk = []

    # deployment-exact proprio: full-robot hand FK -> virtual_pinch_frame_in_base
    def measure_proprio(self, gripper):
        q = ASM({"left_arm": LEFT_Q, "right_arm": self.qr,
                 "left_hand": np.zeros(22), "right_hand": self.rh})
        pin.forwardKinematics(FULL.model, FULL.data, q); pin.updateFramePlacements(FULL.model, FULL.data)
        Tp = virtual_pinch_frame_in_base(FULL.model, FULL.data)
        return F.pinch_to_proprio(Tp, T_BASE_MESH, gripper)

    # deployment-exact per-waypoint IK: CHAINED solve_ik + table-z clamp + pos/ori gate
    def solve_ik(self, T_base_Ree, seed_q):
        T = np.asarray(T_base_Ree, dtype=np.float64).copy()
        Tp = F.base_ee_to_pinch(T, T_EE_PINCH)
        if Tp[2, 3] < PINCH_Z_FLOOR:
            Tp[2, 3] = PINCH_Z_FLOOR; T = F.pinch_to_base_ee(Tp, T_EE_PINCH)
        seed = np.asarray(seed_q, dtype=np.float64) if seed_q is not None else self.qr
        fkL = PIK.fk(frames=["L_ee"], joint_pos_by_component={"left_arm": LEFT_Q, "right_arm": seed})["L_ee"]
        q = np.asarray(seed, dtype=np.float64); pe = oe = 1e9
        for _ in range(40):                       # chain (mirrors driver WAYPOINT_MAX_CHAIN)
            ik = PIK.solve_ik(ee_target_poses={"L_ee": fkL, "R_ee": _se3(T)},
                              arm_initial_joint_pos={"left_arm": LEFT_Q, "right_arm": q})
            q = np.asarray(ik["right_arm"], dtype=np.float64)
            pe, oe = _pose_err(_ree_fk(q), T)
            if pe <= IK_POS_TOL_M and oe <= IK_ORI_TOL_DEG:
                self.ik_ok += 1; return q
        self.ik_fail += 1
        self._last_fail = (pe, oe)               # remember why (pos vs ori)
        return None

    def execute(self, q, gripper_closed):
        self.waypoint_deltas.append(float(np.max(np.abs(q - self.qr))))  # raw joint jump (no smoothing here)
        self._in_chunk.append(q.copy())
        self.qr = q  # "drive" (the real 300Hz smoother interpolates this; we teleport)

    def close_gripper(self): self.rh = CLOSED_Q.copy()
    def get_object_z(self): return float(_ree_fk(self.qr)[2, 3])
    def get_current_q(self): return self.qr.copy()

    def mark_chunk_boundary(self):
        if self._prev_chunk_last_q is not None and self._in_chunk:
            self.chunk_boundary_deltas.append(float(np.max(np.abs(self._in_chunk[0] - self._prev_chunk_last_q))))
        if self._in_chunk:
            self._prev_chunk_last_q = self._in_chunk[-1].copy()
        self._in_chunk = []

    def callbacks(self):
        return DP3Callbacks(self.measure_proprio, self.solve_ik, self.execute,
                            self.close_gripper, self.get_object_z, self.get_current_q)


def _chained_ik(T_base_Ree, seed, max_iter=60, n_seeds=8, jitter=0.5):
    """Multi-start chained IK (mirrors ik_reach_local.best_ik) → best (q, pe, oe)."""
    rng = np.random.default_rng(0)
    base = np.asarray(seed, dtype=np.float64)
    seeds = [base] + [base + rng.uniform(-jitter, jitter, base.shape) for _ in range(n_seeds - 1)]
    best = (None, 1e9, 1e9)
    for s in seeds:
        fkL = PIK.fk(frames=["L_ee"], joint_pos_by_component={"left_arm": LEFT_Q, "right_arm": s})["L_ee"]
        q = np.asarray(s, dtype=np.float64)
        for _ in range(max_iter):
            ik = PIK.solve_ik(ee_target_poses={"L_ee": fkL, "R_ee": _se3(T_base_Ree)},
                              arm_initial_joint_pos={"left_arm": LEFT_Q, "right_arm": q})
            q = np.asarray(ik["right_arm"], dtype=np.float64)
            pe, oe = _pose_err(_ree_fk(q), T_base_Ree)
            if pe <= IK_POS_TOL_M and oe <= IK_ORI_TOL_DEG:
                return q, pe, oe
        if pe + np.radians(oe) < best[1] + np.radians(best[2]):
            best = (q.copy(), pe, oe)
    return best


def main():
    pc_G = np.load("/tmp/test_pc_G.npy").astype(np.float32)
    print(f"=== VIRTUAL FULL-CHAIN TEST (no robot) ===")
    print(f"frame: theta=0, object origin(base)={ORIGIN.tolist()}, pinch_z_floor={PINCH_Z_FLOOR:.3f}")
    print(f"pc_G: {pc_G.shape}  z-range=[{pc_G[:,2].min():.3f},{pc_G[:,2].max():.3f}]\n")

    vr = VirtualRobot()
    # 1) go_home: chained IK to state[0]
    T_home, _ = F.action_to_base_ee(STATE0, T_BASE_MESH, T_EE_PINCH)
    qh, pe, oe = _chained_ik(T_home, seed=vr.qr)
    reachable = qh is not None and pe <= IK_POS_TOL_M and oe <= IK_ORI_TOL_DEG
    print(f"[go_home] IK->state[0] (multi-seed chained)  R_ee={T_home[:3,3].round(3).tolist()}  "
          f"{'OK ✅' if reachable else 'NOT within tol ❌'}  ({pe*1000:.1f}mm/{oe:.1f}d)")
    if qh is None:
        print("  aborting (no IK solution)"); return
    vr.qr = qh
    p0 = vr.measure_proprio(0.0)
    print(f"[proprio@HOME] pos={p0[:3].round(3).tolist()} quat={p0[3:7].round(3).tolist()}  "
          f"vs state0 pos[0.2,-0.293,0.59]  Δ={np.linalg.norm(p0[:3]-STATE0[:3])*1000:.1f}mm\n")

    # 2) closed loop through the real deployment loop, with chunk-boundary instrumentation
    client = DP3Client("http://127.0.0.1:8765", n_obs=2)
    print(f"[server] {client.info()}\n")
    cb = vr.callbacks()
    orig_proprio = cb.measure_proprio
    def proprio_with_boundary(g):           # mark a chunk boundary each time the loop re-measures
        vr.mark_chunk_boundary(); return orig_proprio(g)
    cb = DP3Callbacks(proprio_with_boundary, cb.solve_ik, cb.execute,
                      cb.close_gripper, cb.get_object_z, cb.get_current_q)

    res = run_closed_loop(client, cb, pc_G, T_BASE_MESH, T_EE_PINCH, max_chunks=5, success_dz_m=0.03)
    vr.mark_chunk_boundary()

    print(f"=== RESULT ===")
    print(f"  chain ran: stage={res.stage} success={res.success} chunks={res.n_chunks} "
          f"executed={res.n_executed} ik_fail={res.n_ik_fail}")
    print(f"  IK: ok={vr.ik_ok} fail={vr.ik_fail} (deployment CHAINED per waypoint, seeded from prev solved q)")
    wd = np.array(vr.waypoint_deltas); cd = np.array(vr.chunk_boundary_deltas)
    if wd.size:
        print(f"\n  joint Δ within a chunk (max-abs per waypoint, rad): mean={wd.mean():.3f} max={wd.max():.3f}")
    if cd.size:
        print(f"  joint Δ at CHUNK BOUNDARIES (max-abs, rad):          {np.round(cd,3).tolist()}  max={cd.max():.3f}")
    if PER_TICK is not None and (wd.size or cd.size):
        big = max(wd.max() if wd.size else 0, cd.max() if cd.size else 0)
        ticks = big / float(PER_TICK.min())
        print(f"\n  300Hz smoother: per-tick cap≈{PER_TICK.min()*1000:.2f} mrad/joint(min) → the largest jump "
              f"({big:.3f} rad) is interpolated over ~{ticks:.0f} ticks ≈ {ticks/300*1000:.0f} ms (NOT a sudden jump).")
    else:
        print("\n  (300Hz vel limits not importable here — but the SmoothingAndSafetyManager velocity-clips "
              "every joint to ±0.4·vel_limit/300 per tick, so any Δ above is interpolated, never a sudden jump.)")


if __name__ == "__main__":
    main()
