#!/usr/bin/env python3
"""Out-of-process cuRobo MotionPlanner wrapper for baseline_3.

cuRobo cannot import in-process with IsaacSim (Warp version clash), so each plan
call is a fresh subprocess. Mirrors sim/curobo_ik.py.

Uses cuRobo 0.8 new API:
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types           import GoalToolPose, JointState

`plan_grasp` returns three interpolated joint trajectories (approach, grasp, lift)
in one call. The grasp segment automatically disables finger-link collisions, so the
object mesh can be a collision obstacle for the approach segment without blocking
the actual grasp contact.

Usage:
    curobo_plan.py --grasp IN.pkl OUT.pkl

IN.pkl keys (all poses in ROBOT BASE frame):
    start_qpos         (7,) float        Franka arm start joint angles (rad)
    grasp_pos_r        (3,) float        target tool-frame position
    grasp_quat_r_wxyz  (4,) float        target tool-frame orientation
    world_dict         dict              cuRobo world config (cuboid table+ground[+mesh])
    approach_offset    float  -0.12      along tool z (pre-grasp = grasp + offset * z_tool)
    lift_offset        float  -0.15      lift along tool z
    pos_tol            float   0.005
    ori_tol            float   0.05
    warmup_iters       int     2

OUT.pkl keys:
    success            bool              overall (approach AND grasp AND lift)
    status             str
    approach_qpos      (Na,7) or None    interpolated joint trajectory
    grasp_qpos         (Ng,7) or None
    lift_qpos          (Nl,7) or None
    dt                 float             interpolation timestep (s)
    phase_success      {approach,grasp,lift}: bool each
    plan_seconds       float             wall time inside subprocess
"""
from __future__ import annotations

import os
import pickle
import sys
import time
import traceback


def _parse():
    if len(sys.argv) < 4 or sys.argv[1] not in ("--grasp", "--sequence"):
        sys.stderr.write("usage: curobo_plan.py {--grasp|--sequence} IN.pkl OUT.pkl\n")
        sys.exit(2)
    return sys.argv[1], sys.argv[2], sys.argv[3]


def _build_planner(world_dict, pos_tol, ori_tol):
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

    cfg = MotionPlannerCfg.create(
        robot="franka.yml",
        scene_model=world_dict,
        max_batch_size=1,
        max_goalset=1,
        position_tolerance=pos_tol,
        orientation_tolerance=ori_tol,
        collision_cache={"obb": 8, "mesh": 4},
    )
    return MotionPlanner(cfg)


def _make_start(planner, start_qpos):
    import torch
    from curobo.types import JointState

    q = torch.tensor(start_qpos, dtype=torch.float32, device="cuda").unsqueeze(0)
    return JointState.from_position(q, joint_names=planner.joint_names)


def _make_goalset(planner, pos, quat_wxyz):
    """Build a single-candidate goalset, shape (env=1, batch=1, link=1, goal=1, 3/4)."""
    import torch
    from curobo.types import GoalToolPose

    p = torch.tensor(pos, dtype=torch.float32, device="cuda").reshape(1, 1, 1, 1, 3)
    q = torch.tensor(quat_wxyz, dtype=torch.float32, device="cuda").reshape(1, 1, 1, 1, 4)
    return GoalToolPose(tool_frames=planner.tool_frames, position=p, quaternion=q)


def _seg_to_np(traj, last_tstep=None, n_arm_dof=7):
    """JointState.position → (N, n_arm_dof) ndarray (arm joints only, padding trimmed).

    cuRobo interpolated_trajectory.position is shape (env=1, batch=1, n_steps_max, n_dof)
    where n_dof = 9 for Franka (7 arm + 2 finger) and n_steps_max is the planner's
    fixed interpolation horizon (e.g. 5000) padded with the last valid pose.
    """
    if traj is None or getattr(traj, "position", None) is None:
        return None
    arr = traj.position.detach().cpu().numpy()
    while arr.ndim > 2:
        if arr.shape[0] == 1:
            arr = arr[0]
        else:
            arr = arr.reshape(-1, arr.shape[-1])
            break
    # truncate to actual last waypoint (inclusive)
    if last_tstep is not None:
        try:
            n = int(last_tstep.view(-1)[0].item())
            if 0 < n + 1 <= arr.shape[0]:
                arr = arr[: n + 1]
        except Exception:
            pass
    return arr[:, :n_arm_dof].astype("float64")


def handle_grasp(d):
    import torch  # noqa: F401  (used by phase_success bool conversions)

    t0 = time.time()
    planner = _build_planner(
        d["world_dict"],
        pos_tol=float(d.get("pos_tol", 0.005)),
        ori_tol=float(d.get("ori_tol", 0.05)),
    )
    planner.warmup(enable_graph=True, num_warmup_iterations=int(d.get("warmup_iters", 2)))
    sys.stderr.write(f"[plan] init+warmup {time.time()-t0:.1f}s\n")
    sys.stderr.flush()

    start = _make_start(planner, d["start_qpos"])
    goal = _make_goalset(planner, d["grasp_pos_r"], d["grasp_quat_r_wxyz"])

    t0 = time.time()
    res = planner.plan_grasp(
        grasp_poses=goal,
        current_state=start,
        grasp_approach_offset=float(d.get("approach_offset", -0.12)),
        grasp_lift_offset=float(d.get("lift_offset", -0.15)),
    )
    plan_secs = time.time() - t0
    sys.stderr.write(f"[plan] plan_grasp {plan_secs:.2f}s\n")
    sys.stderr.flush()

    interp_dt = float(planner.trajopt_solver.config.interpolation_dt)

    def _ok(attr):
        t = getattr(res, attr, None)
        try:
            return bool(t.any().item())
        except Exception:
            return False

    a_arr = _seg_to_np(getattr(res, "approach_interpolated_trajectory", None),
                       getattr(res, "approach_interpolated_last_tstep", None))
    g_arr = _seg_to_np(getattr(res, "grasp_interpolated_trajectory", None),
                       getattr(res, "grasp_interpolated_last_tstep", None))
    l_arr = _seg_to_np(getattr(res, "lift_interpolated_trajectory", None),
                       getattr(res, "lift_interpolated_last_tstep", None))

    return {
        "success": _ok("success"),
        "status": str(getattr(res, "status", "") or ""),
        "approach_qpos": a_arr,
        "grasp_qpos": g_arr,
        "lift_qpos": l_arr,
        "dt": interp_dt,
        "phase_success": {
            "approach": _ok("approach_success"),
            "grasp":    _ok("grasp_success"),
            "lift":     _ok("lift_success"),
        },
        "plan_seconds": plan_secs,
    }


def handle_sequence(d):
    """3-phase plan_pose sequence — mirrors titan branch's per-phase mesh toggle.

    Input keys:
      start_qpos:  (7,) initial Franka arm joints (rad)
      phases:      list of 3 phase dicts, each with:
                     - name:           "pre-grasp" | "final" | "lift"
                     - target_pos_r:   (3,) target tool position in robot frame
                     - target_quat_r:  (4,) target tool quat wxyz in robot frame
                     - world_dict:     cuRobo world config (cuboid + optional mesh)
                                       The first phase's world is used to BUILD the
                                       planner; subsequent phases trigger update_world
                                       only if their world_dict id() differs from prev.
                     - max_attempts:   int (default 10)
      pos_tol, ori_tol, warmup_iters: planner config

    Output keys:
      phases: list of {name, success, qpos_traj (Nx7) or None, status}
      dt:     interpolation dt
    """
    from curobo._src.geom.types import SceneCfg

    t0 = time.time()
    first_world = d["phases"][0]["world_dict"]
    planner = _build_planner(
        first_world,
        pos_tol=float(d.get("pos_tol", 0.005)),
        ori_tol=float(d.get("ori_tol", 0.05)),
    )
    planner.warmup(enable_graph=True, num_warmup_iterations=int(d.get("warmup_iters", 2)))
    sys.stderr.write(f"[seq] init+warmup {time.time()-t0:.1f}s\n")
    sys.stderr.flush()

    interp_dt = float(planner.trajopt_solver.config.interpolation_dt)

    def _result_traj(res):
        traj = getattr(res, "interpolated_trajectory", None)
        if traj is None or getattr(traj, "position", None) is None:
            # fall back to non-interpolated solution
            traj = getattr(res, "solution", None)
        if traj is None:
            return None
        last = getattr(res, "interpolated_last_tstep", None)
        return _seg_to_np(traj, last)

    out_phases = []
    current_qpos = list(d["start_qpos"])
    cur_world_obj = first_world

    for phase in d["phases"]:
        # update_world if this phase has a different world_dict than current
        pw = phase["world_dict"]
        if pw is not cur_world_obj:
            try:
                planner.update_world(SceneCfg.create(pw))
                cur_world_obj = pw
                sys.stderr.write(f"[seq] [{phase['name']}] update_world (mesh={'yes' if 'mesh' in pw else 'no'})\n")
                sys.stderr.flush()
            except Exception as e:
                out_phases.append({"name": phase["name"], "success": False, "qpos_traj": None,
                                   "status": f"update_world failed: {e}"})
                break

        start = _make_start(planner, current_qpos)
        goal = _make_goalset(planner, phase["target_pos_r"], phase["target_quat_r"])
        tp0 = time.time()
        try:
            res = planner.plan_pose(goal, start, max_attempts=int(phase.get("max_attempts", 10)))
        except Exception as e:
            out_phases.append({"name": phase["name"], "success": False, "qpos_traj": None,
                               "status": f"plan_pose exception: {e}"})
            break
        plan_secs = time.time() - tp0

        try:
            ok = bool(res.success.any().item()) if res is not None else False
        except Exception:
            ok = False

        if not ok:
            status = str(getattr(res, "status", "") or "no plan")[:600]
            out_phases.append({"name": phase["name"], "success": False, "qpos_traj": None,
                               "status": status, "plan_seconds": plan_secs})
            sys.stderr.write(f"[seq] [{phase['name']}] FAILED in {plan_secs:.2f}s\n")
            sys.stderr.flush()
            break

        traj = _result_traj(res)
        if traj is None or len(traj) == 0:
            out_phases.append({"name": phase["name"], "success": False, "qpos_traj": None,
                               "status": "no interpolated trajectory", "plan_seconds": plan_secs})
            break

        # chain: next phase starts where this one ended
        current_qpos = traj[-1].tolist()
        out_phases.append({"name": phase["name"], "success": True, "qpos_traj": traj,
                           "status": "", "plan_seconds": plan_secs})
        sys.stderr.write(f"[seq] [{phase['name']}] {len(traj)} wp in {plan_secs:.2f}s\n")
        sys.stderr.flush()

    overall = all(p["success"] for p in out_phases) and len(out_phases) == len(d["phases"])
    return {"success": overall, "phases": out_phases, "dt": interp_dt}


def main():
    mode, in_pkl, out_pkl = _parse()
    with open(in_pkl, "rb") as f:
        d = pickle.load(f)
    try:
        if mode == "--grasp":
            out = handle_grasp(d)
        elif mode == "--sequence":
            out = handle_sequence(d)
        else:
            raise ValueError(f"unknown mode: {mode}")
    except Exception:
        if mode == "--sequence":
            out = {
                "success": False,
                "phases": [],
                "dt": 0.02,
                "status": "subprocess exception:\n" + traceback.format_exc(),
            }
        else:
            out = {
                "success": False,
                "status": "subprocess exception:\n" + traceback.format_exc(),
                "approach_qpos": None, "grasp_qpos": None, "lift_qpos": None,
                "dt": 0.02,
                "phase_success": {"approach": False, "grasp": False, "lift": False},
                "plan_seconds": 0.0,
            }
    with open(out_pkl, "wb") as f:
        pickle.dump(out, f)


if __name__ == "__main__":
    main()
