"""DP3 closed-loop orchestrator (hardware-agnostic).

The control-flow "brain" of run_auto_grasp_dp3: per-chunk observe -> infer ->
convert -> IK -> drive -> success-check, reproducing sim/eval_dp3_titan_protocol.py's
loop. It does NOT touch the robot directly — every robot/perception action is a
callback the caller injects (razer wires them to pinocchio FK + dexcontrol +
Sharpa + dp3_frames). This keeps the loop fully offline-unit-testable with fake
callbacks + a mock/real DP3 server.

Per chunk (mirrors the eval):
  proprio = measure_proprio(gripper)            # robot FK -> dp3_frames.pinch_to_proprio
  action8 = client.step(proprio)                # 8 absolute EE poses in the G/object frame
  for each waypoint a in action8:
      T_base_Ree, grip = dp3_frames.action_to_base_ee(a, T_base_mesh, T_ee_pinch)
      q = solve_ik(T_base_Ree, seed_q)          # PinkLocalIK seeded from prev q (continuity)
      if q is None: skip (unreachable)
      if grip>=0.5 and not closed: z0=get_object_z(); close_gripper(); closed=True
      execute(q, closed)                        # write 300Hz action_buffer
  if closed and get_object_z()-z0 > success_dz_m: return SUCCESS  (early stop)

Frame math: deploy/dp3_frames.py.  Transport/window: deploy/dp3_client.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

import dp3_frames as F

GRIP_ARRIVE_THR = 0.5
SUCCESS_DZ_M = 0.03


@dataclass
class DP3Callbacks:
    """Robot/perception hooks the caller (razer) provides.

    measure_proprio(gripper)      -> (8,) DP3 proprio in G frame (FK -> pinch_to_proprio).
    solve_ik(T_base_Ree, seed_q)  -> (J,) joints or None if unreachable. seed_q may be None.
    execute(q, gripper_closed)    -> None. Drive arm to q (+ hold gripper state) via action_buffer.
    close_gripper()               -> None. One-time Sharpa stall-close trigger.
    get_object_z()                -> float. Current object height (base/world Z), for dz success.
    get_current_q()               -> (J,) current arm joints, to seed the first IK.
    """

    measure_proprio: Callable[[float], np.ndarray]
    solve_ik: Callable[[np.ndarray, Optional[np.ndarray]], Optional[np.ndarray]]
    execute: Callable[[np.ndarray, bool], None]
    close_gripper: Callable[[], None]
    get_object_z: Callable[[], float]
    get_current_q: Callable[[], np.ndarray]


@dataclass
class LoopResult:
    success: bool
    dz: float
    n_chunks: int
    n_executed: int
    n_ik_fail: int
    gripper_closed: bool
    stage: str  # "lifted" | "never_closed" | "max_chunks"


def run_closed_loop(
    client,
    cb: DP3Callbacks,
    pc_G: np.ndarray,
    T_base_mesh: np.ndarray,
    T_ee_pinch_closed: np.ndarray,
    *,
    max_chunks: int = 8,
    success_dz_m: float = SUCCESS_DZ_M,
    grip_thr: float = GRIP_ARRIVE_THR,
    on_chunk: Optional[Callable[[int, dict], None]] = None,
) -> LoopResult:
    """Run the DP3 closed loop. `client` is a deploy.dp3_client.DP3Client (or a
    duck-typed object with reset(pc_G)/step(proprio)->(n_action,8)).

    Returns a LoopResult. `on_chunk(chunk_idx, stats)` is an optional progress hook.
    """
    client.reset(pc_G)
    gripper_closed = False
    z0: Optional[float] = None
    last_q: Optional[np.ndarray] = cb.get_current_q()
    n_executed = 0
    n_ik_fail = 0

    for chunk in range(max_chunks):
        proprio = cb.measure_proprio(1.0 if gripper_closed else 0.0)
        action = np.asarray(client.step(proprio), dtype=np.float64)  # (n_action, 8)

        n_ok = 0
        for a in action:
            T_base_Ree, grip = F.action_to_base_ee(a, T_base_mesh, T_ee_pinch_closed)
            q = cb.solve_ik(T_base_Ree, last_q)
            if q is None:
                n_ik_fail += 1
                continue
            q = np.asarray(q, dtype=np.float64)

            if grip >= grip_thr and not gripper_closed:
                z0 = float(cb.get_object_z())          # object height at grip-close instant
                cb.close_gripper()
                gripper_closed = True

            cb.execute(q, gripper_closed)
            last_q = q
            n_executed += 1
            n_ok += 1

        cur_dz = (float(cb.get_object_z()) - z0) if (gripper_closed and z0 is not None) else 0.0
        if on_chunk is not None:
            on_chunk(chunk, {"n_ok": n_ok, "gripper_closed": gripper_closed, "dz": cur_dz})

        # early stop: gripper closed + object lifted past threshold
        if gripper_closed and z0 is not None and cur_dz > success_dz_m:
            return LoopResult(True, cur_dz, chunk + 1, n_executed, n_ik_fail, True, "lifted")

    # exhausted max_chunks
    if not gripper_closed:
        return LoopResult(False, 0.0, max_chunks, n_executed, n_ik_fail, False, "never_closed")
    final_dz = (float(cb.get_object_z()) - z0) if z0 is not None else 0.0
    return LoopResult(final_dz > success_dz_m, final_dz, max_chunks, n_executed,
                      n_ik_fail, True, "max_chunks")
