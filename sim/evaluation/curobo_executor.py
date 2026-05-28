"""cuRobo-backed open-loop grasp executor for evaluation."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from termcolor import cprint

from evaluation.specs import ExecutionResult, OpenLoopGraspCommand, SceneSpec
from sim.evaluation.context import SimEvaluationContext
from sim.evaluation.scene_builder import (
    ROBOT_ORIENTATION,
    ROBOT_POSITION,
    TABLE_POSITION,
    TABLE_SCALE,
    TABLE_TOP_Z,
    _euler_xyz_deg_to_wxyz,
)

SIM_DIR = Path(__file__).resolve().parents[1]
PROJ_DIR = SIM_DIR.parent
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))
for _curobo_candidate in [
    os.path.expanduser("~/Project/curobo/src"),
    os.path.expanduser("~/curobo/src"),
    "/home/vision/Project/curobo/src",
]:
    if os.path.isdir(os.path.join(_curobo_candidate, "curobo")):
        sys.path.insert(0, _curobo_candidate)
        break

from curobo_world import build_world_config_dict, object_pose_robot_frame, sync_curobo_world  # noqa: E402


LIFT_HEIGHT = 0.15
TCP_OFFSET = 0.105
PRE_GRASP_OFFSET = 0.15
_CUROBO_MG = None


def reset_motion_gen() -> None:
    """Force cuRobo MotionGen rebuild after object/yaw/world changes."""
    global _CUROBO_MG
    _CUROBO_MG = None


def make_transform(pos, quat_wxyz) -> np.ndarray:
    T = np.eye(4)
    r = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    T[:3, :3] = r.as_matrix()
    T[:3, 3] = pos
    return T


def get_robot_base_transform() -> tuple[np.ndarray, np.ndarray]:
    yaw_rad = np.deg2rad(ROBOT_ORIENTATION[2])
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    T = np.eye(4)
    T[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    T[:3, 3] = ROBOT_POSITION
    return T, np.linalg.inv(T)


def world_to_robot_pose(pos_w, quat_wxyz_w, T_robot_world):
    pos_r = (T_robot_world @ np.append(pos_w, 1.0))[:3]
    R_w = Rotation.from_quat([quat_wxyz_w[1], quat_wxyz_w[2], quat_wxyz_w[3], quat_wxyz_w[0]])
    R_rw = Rotation.from_matrix(T_robot_world[:3, :3])
    R_r = R_rw * R_w
    q = R_r.as_quat()
    return pos_r, np.array([q[3], q[0], q[1], q[2]])


def transform_grasp_to_world(grasp_pos_obj, grasp_rot_obj, T_world_obj):
    pos_w = (T_world_obj @ np.append(grasp_pos_obj, 1.0))[:3]
    rot_w = T_world_obj[:3, :3] @ grasp_rot_obj
    return pos_w, rot_w


def world_pose_to_object_mesh(pos_world, quat_wxyz_world, obj_pos_world, obj_quat_wxyz, object_scale):
    T_world_obj = make_transform(
        np.asarray(obj_pos_world, dtype=np.float64).reshape(3),
        np.asarray(obj_quat_wxyz, dtype=np.float64).reshape(4),
    )
    T_world_body = make_transform(
        np.asarray(pos_world, dtype=np.float64).reshape(3),
        np.asarray(quat_wxyz_world, dtype=np.float64).reshape(4),
    )
    T_obj_body = np.linalg.inv(T_world_obj) @ T_world_body
    scale = float(object_scale) if object_scale else 1.0
    return (T_obj_body[:3, 3] / scale).astype(np.float32), T_obj_body[:3, :3].astype(np.float32)


def world_point_to_object_mesh(point_world, obj_pos_world, obj_quat_wxyz, object_scale):
    T_world_obj = make_transform(
        np.asarray(obj_pos_world, dtype=np.float64).reshape(3),
        np.asarray(obj_quat_wxyz, dtype=np.float64).reshape(4),
    )
    p_h = np.append(np.asarray(point_world, dtype=np.float64).reshape(3), 1.0)
    scale = float(object_scale) if object_scale else 1.0
    return ((np.linalg.inv(T_world_obj) @ p_h)[:3] / scale).astype(np.float32)


def snapshot_panda_hand_object_mesh(franka, scene: SimEvaluationContext) -> dict[str, Any]:
    pos_w, quat_w = franka.get_cur_ee_pos(local_frame=False)
    obj_pos, obj_quat = scene.obj.get_obj_pos()
    pos_o, rot_o = world_pose_to_object_mesh(pos_w, quat_w, obj_pos, obj_quat, scene.spec.object_scale)
    return {
        "position": pos_o,
        "rotation": rot_o,
        "approach_dir": rot_o[:, 2].copy(),
        "finger_dir": rot_o[:, 1].copy(),
    }


def snapshot_gripper_tips_object_mesh(stage, scene: SimEvaluationContext) -> dict[str, Any]:
    from pxr import UsdGeom

    left_path = "/World/Franka/panda_leftfinger"
    right_path = "/World/Franka/panda_rightfinger"
    left_prim = stage.GetPrimAtPath(left_path)
    right_prim = stage.GetPrimAtPath(right_path)
    if not left_prim.IsValid() or not right_prim.IsValid():
        raise RuntimeError("finger prims not found")

    left_world = np.array(
        UsdGeom.Xformable(left_prim).ComputeLocalToWorldTransform(0).ExtractTranslation(),
        dtype=np.float64,
    )
    right_world = np.array(
        UsdGeom.Xformable(right_prim).ComputeLocalToWorldTransform(0).ExtractTranslation(),
        dtype=np.float64,
    )
    obj_pos, obj_quat = scene.obj.get_obj_pos()
    left_o = world_point_to_object_mesh(left_world, obj_pos, obj_quat, scene.spec.object_scale)
    right_o = world_point_to_object_mesh(right_world, obj_pos, obj_quat, scene.spec.object_scale)
    tips = np.stack([left_o, right_o]).astype(np.float32)
    return {
        "gripper_tips_loc": tips,
        "finger_width_actual": float(np.linalg.norm(tips[0] - tips[1])),
    }


def _base_result(success=False, failure_stage: str | None = None) -> ExecutionResult:
    return ExecutionResult(success=bool(success), failure_stage=failure_stage)


def init_curobo(scene: SimEvaluationContext):
    import os as _os
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

    extra = "/home/vision/isaacsim/kit/python/bin:/usr/local/cuda/bin"
    if extra not in _os.environ.get("PATH", ""):
        _os.environ["PATH"] = extra + ":" + _os.environ.get("PATH", "")

    _, T_robot_world = get_robot_base_transform()
    table_pos_r = (T_robot_world @ np.append(TABLE_POSITION, 1.0))[:3]
    ground_pos_r = (T_robot_world @ np.array([0, 0, -0.005, 1.0]))[:3]
    scene.metadata["curobo_table_pos_r"] = table_pos_r
    scene.metadata["curobo_ground_pos_r"] = ground_pos_r

    mesh_verts = mesh_faces = mesh_pose = None
    if scene.curobo_mesh_vertices is not None:
        mesh_verts = scene.curobo_mesh_vertices
        mesh_faces = scene.curobo_mesh_faces
        pos_w, quat_wxyz = scene.obj.get_obj_pos()
        mesh_pose = object_pose_robot_frame(pos_w, quat_wxyz, T_robot_world)

    world_config = build_world_config_dict(
        table_pos_r,
        ground_pos_r,
        TABLE_SCALE,
        mesh_vertices=mesh_verts,
        mesh_faces=mesh_faces,
        mesh_pose_robot=mesh_pose,
    )
    load_kwargs = {"interpolation_dt": 0.02}
    if mesh_verts is not None:
        load_kwargs["collision_cache"] = {"obb": 4, "mesh": 4}
    mg_config = MotionGenConfig.load_from_robot_config("franka.yml", world_config, **load_kwargs)
    mg = MotionGen(mg_config)
    cprint("   cuRobo warmup...", "yellow")
    mg.warmup()
    cprint("   cuRobo ready", "green")
    return mg


def plan_trajectory(
    motion_gen,
    scene: SimEvaluationContext,
    target_pos_world,
    target_quat_wxyz_world,
    *,
    label: str,
    use_object_mesh: bool,
):
    from curobo.types.math import Pose
    from curobo.types.robot import JointState as CuJointState
    from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

    _, T_robot_world = get_robot_base_transform()
    legacy_scene = scene.as_legacy_dict()
    table_pos_r = scene.metadata.get(
        "curobo_table_pos_r",
        (T_robot_world @ np.append(TABLE_POSITION, 1.0))[:3],
    )
    ground_pos_r = scene.metadata.get(
        "curobo_ground_pos_r",
        (T_robot_world @ np.array([0, 0, -0.005, 1.0]))[:3],
    )
    sync_curobo_world(
        motion_gen,
        legacy_scene,
        table_pos_r,
        ground_pos_r,
        TABLE_SCALE,
        T_robot_world,
        include_object_mesh=use_object_mesh and scene.curobo_mesh_vertices is not None,
    )
    scene.update_from_legacy_dict(legacy_scene)

    pos_r, quat_r = world_to_robot_pose(target_pos_world, target_quat_wxyz_world, T_robot_world)
    current_joints = scene.franka.get_joint_positions()[:7]
    start_state = CuJointState.from_position(
        torch.tensor(current_joints, dtype=torch.float32).unsqueeze(0).cuda(),
        joint_names=[f"panda_joint{i}" for i in range(1, 8)],
    )
    goal_pose = Pose.from_list(
        [
            float(pos_r[0]),
            float(pos_r[1]),
            float(pos_r[2]),
            float(quat_r[0]),
            float(quat_r[1]),
            float(quat_r[2]),
            float(quat_r[3]),
        ]
    )
    result = motion_gen.plan_single(
        start_state,
        goal_pose,
        MotionGenPlanConfig(max_attempts=10, enable_graph=True, enable_opt=True),
    )
    success = result.success.item() if callable(getattr(result.success, "item", None)) else result.success
    if success:
        traj = result.get_interpolated_plan()
        cprint(f"      [{label}] plan OK: {traj.position.shape[0]} steps", "green")
        return traj.position.cpu().numpy()
    cprint(f"      [{label}] plan failed", "red")
    return None


def _command_to_world_target(scene: SimEvaluationContext, command: OpenLoopGraspCommand):
    if command.frame != "object_mesh":
        raise ValueError(f"first executor only supports object_mesh commands, got {command.frame}")

    obj_pos_world, obj_quat_wxyz = scene.obj.get_obj_pos()
    T_world_obj = make_transform(obj_pos_world, obj_quat_wxyz)
    grasp_pos_scaled = np.asarray(command.position, dtype=np.float64) * scene.spec.object_scale
    grasp_rot_obj = np.asarray(command.rotation, dtype=np.float64)

    prerot = command.mesh_prerotation_euler
    if prerot and any(abs(float(e)) > 0.5 for e in prerot):
        Rp = Rotation.from_euler("xyz", prerot, degrees=True).as_matrix()
        T_eff = T_world_obj.copy()
        T_eff[:3, :3] = T_world_obj[:3, :3] @ Rp.T
    else:
        T_eff = T_world_obj

    pos_world, rot_world = transform_grasp_to_world(grasp_pos_scaled, grasp_rot_obj, T_eff)
    r_adapt = np.array(
        [
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    rot_world = rot_world @ r_adapt
    approach_dir = rot_world[:, 2]
    pos_world = pos_world - approach_dir * TCP_OFFSET
    min_grasp_z = TABLE_TOP_Z + 0.02
    if pos_world[2] < min_grasp_z:
        cprint(f"   grasp target z={pos_world[2]:.3f} below {min_grasp_z:.3f}; clamping", "yellow")
        pos_world[2] = min_grasp_z
    q_xyzw = Rotation.from_matrix(rot_world).as_quat()
    quat_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
    return pos_world, rot_world, quat_wxyz


def execute_open_loop_grasp(scene: SimEvaluationContext, command: OpenLoopGraspCommand) -> ExecutionResult:
    """Execute one open-loop object-mesh grasp command and return a common result."""
    global _CUROBO_MG

    franka = scene.franka
    world = scene.world
    render = scene.render
    planning = {
        "pregrasp_plan_success": False,
        "direct_plan_success": False,
        "final_plan_success": False,
        "lift_plan_success": False,
    }

    obj_init, _ = scene.obj.get_obj_pos()
    initial_z = float(obj_init[2])

    try:
        pos_world, rot_world, quat_wxyz = _command_to_world_target(scene, command)
    except Exception as exc:
        res = _base_result(False, "target_transform")
        res.metadata["error"] = str(exc)
        return res

    lift_pos = pos_world.copy()
    lift_pos[2] += LIFT_HEIGHT
    approach_dir = rot_world[:, 2]
    pre_grasp_pos = pos_world - approach_dir * PRE_GRASP_OFFSET

    try:
        if _CUROBO_MG is None:
            _CUROBO_MG = init_curobo(scene)
    except Exception as exc:
        res = _base_result(False, "curobo_init")
        res.metadata["error"] = str(exc)
        return res

    franka.open_gripper()
    for _ in range(30):
        world.step(render=render)

    traj = plan_trajectory(
        _CUROBO_MG,
        scene,
        pre_grasp_pos,
        quat_wxyz,
        label="pre-grasp",
        use_object_mesh=True,
    )
    if traj is not None:
        planning["pregrasp_plan_success"] = True
    else:
        traj = plan_trajectory(
            _CUROBO_MG,
            scene,
            pos_world,
            quat_wxyz,
            label="direct",
            use_object_mesh=True,
        )
        planning["direct_plan_success"] = traj is not None
    if traj is None:
        res = _base_result(False, "pregrasp_plan")
        res.planning = planning
        return res

    for joint_pos in traj:
        gripper = franka.get_joint_positions()[7:9]
        franka.set_joint_positions(np.concatenate([joint_pos, gripper]))
        world.step(render=render)
    for _ in range(10):
        world.step(render=render)

    traj_final = plan_trajectory(
        _CUROBO_MG,
        scene,
        pos_world,
        quat_wxyz,
        label="final",
        use_object_mesh=False,
    )
    planning["final_plan_success"] = traj_final is not None
    if traj_final is None:
        res = _base_result(False, "final_plan")
        res.planning = planning
        return res
    for joint_pos in traj_final:
        gripper = franka.get_joint_positions()[7:9]
        franka.set_joint_positions(np.concatenate([joint_pos, gripper]))
        for _ in range(3):
            world.step(render=render)

    franka.close_gripper()
    force_log = []
    for _ in range(80):
        world.step(render=render)
        force_log.append(franka.get_joint_positions()[7:9].copy())

    executed_at_close = None
    gripper_tips_loc = None
    finger_width_actual = None
    try:
        executed_at_close = snapshot_panda_hand_object_mesh(franka, scene)
    except Exception as exc:
        cprint(f"   panda_hand@close snapshot failed: {exc}", "yellow")
    try:
        tips = snapshot_gripper_tips_object_mesh(world.stage, scene)
        gripper_tips_loc = tips["gripper_tips_loc"]
        finger_width_actual = tips["finger_width_actual"]
    except Exception as exc:
        cprint(f"   gripper tips snapshot failed: {exc}", "yellow")

    traj_lift = plan_trajectory(
        _CUROBO_MG,
        scene,
        lift_pos,
        quat_wxyz,
        label="lift",
        use_object_mesh=False,
    )
    planning["lift_plan_success"] = traj_lift is not None
    if traj_lift is not None:
        franka.close_gripper()
        for joint_pos in traj_lift:
            from omni.isaac.core.utils.types import ArticulationAction

            action = ArticulationAction(
                joint_positions=np.concatenate([joint_pos, np.array([None, None])]),
            )
            franka.apply_action(action)
            for _ in range(2):
                world.step(render=render)

    for _ in range(80):
        world.step(render=render)

    obj_after, _ = scene.obj.get_obj_pos()
    z_delta = float(obj_after[2] - initial_z)
    success = z_delta > 0.03
    executed_post_lift = None
    try:
        executed_post_lift = snapshot_panda_hand_object_mesh(franka, scene)
    except Exception as exc:
        cprint(f"   panda_hand@post_lift snapshot failed: {exc}", "yellow")

    failure_stage = None if success else ("lift_result" if planning["lift_plan_success"] else "lift_plan")
    return ExecutionResult(
        success=success,
        failure_stage=failure_stage,
        z_delta_m=z_delta,
        initial_object_position_world=[float(x) for x in np.asarray(obj_init).reshape(3)],
        final_object_position_world=[float(x) for x in np.asarray(obj_after).reshape(3)],
        gripper_tips_loc=gripper_tips_loc,
        finger_width_actual=finger_width_actual,
        executed_at_close=executed_at_close,
        executed_post_lift=executed_post_lift,
        planning=planning,
        metadata={
            "command_name": command.name,
            "command_score": command.score,
            "finger_log_samples": len(force_log),
        },
    )


def _write_snapshot_group(parent, name: str, snapshot: dict[str, Any] | None) -> None:
    if snapshot is None:
        return
    g = parent.create_group(name)
    g.attrs["frame"] = "object_mesh"
    g.attrs["ee_frame"] = "panda_hand"
    for key in ["position", "rotation", "approach_dir", "finger_dir"]:
        if key in snapshot:
            g.create_dataset(key, data=snapshot[key])


def write_robot_gt_hdf5(
    *,
    result_dir: str,
    scene: SceneSpec,
    command: OpenLoopGraspCommand,
    execution: ExecutionResult,
    policy_name: str,
) -> str:
    os.makedirs(result_dir, exist_ok=True)
    path = os.path.join(result_dir, f"{scene.episode_id}_robot_gt.hdf5")
    with h5py.File(path, "w") as f:
        f.attrs["obj_id"] = scene.obj_id
        f.attrs["episode_id"] = scene.episode_id
        f.attrs["policy_name"] = policy_name
        f.attrs["success"] = bool(execution.success)
        f.attrs["failure_stage"] = execution.failure_stage or ""
        f.attrs["z_delta_m"] = execution.z_delta_m if execution.z_delta_m is not None else np.nan
        f.attrs["object_scale"] = scene.object_scale
        f.attrs["robot_gt_schema_version"] = 2
        f.attrs["scene_schema_version"] = 1
        f.attrs["executed_pose_frame"] = "object_mesh"
        f.attrs["executed_ee_frame"] = "panda_hand"
        f.attrs["sim_z_yaw_deg"] = scene.sim_z_yaw_deg
        if execution.video_path:
            f.attrs["video_path"] = execution.video_path

        cg = f.create_group("candidate_results")
        ci = cg.create_group("candidate_0")
        ci.attrs["name"] = command.name
        ci.attrs["score"] = command.score
        ci.attrs["success"] = bool(execution.success)
        ci.attrs["gripper_width"] = command.gripper_width
        ci.attrs["approach_type"] = command.approach_type
        ci.create_dataset("grasp_point", data=command.position.astype(np.float32))
        ci.create_dataset("rotation", data=command.rotation.astype(np.float32))
        _write_snapshot_group(ci, "executed_panda_hand_at_close", execution.executed_at_close)
        _write_snapshot_group(ci, "executed_panda_hand_post_lift", execution.executed_post_lift)
        if execution.gripper_tips_loc is not None:
            ci.create_dataset("gripper_tips_loc", data=execution.gripper_tips_loc)
            ci.attrs["finger_width_actual"] = float(execution.finger_width_actual or 0.0)

        if execution.success:
            sg = f.create_group("successful_grasps")
            sg.attrs["count"] = 1
            gi = sg.create_group("grasp_0")
            gi.attrs["name"] = command.name
            gi.attrs["score"] = command.score
            gi.attrs["gripper_width"] = command.gripper_width
            gi.attrs["approach_type"] = command.approach_type
            gi.create_dataset("grasp_point", data=command.position.astype(np.float32))
            gi.create_dataset("rotation", data=command.rotation.astype(np.float32))
            _write_snapshot_group(gi, "executed_panda_hand_at_close", execution.executed_at_close)
            _write_snapshot_group(gi, "executed_panda_hand_post_lift", execution.executed_post_lift)
    return path



# ============================================================
# Closed-loop online policy executor (DP3 / VLA / any chunked policy)
# ============================================================
# Ported from gate3-curobo-ik branch sim/eval_dp3_baseline3.py:rollout_chunked.
# Key differences from execute_open_loop_grasp:
#   - obs constructed each chunk (live PC + EE state) and sent to remote server
#   - server returns next ``n_action_steps`` EE waypoints
#   - we IK them and execute via teleport (gripper open) or PD (gripper closed)
#   - early-stop when lift > success_dz_m; retry on PhysX NaN
#   - reset_scene_pose is the CALLER's responsibility (run_eval_worker does it)
def _read_panda_hand_pose(stage):
    """Read live panda_hand world-frame pose. Returns (pos_w, quat_wxyz)."""
    from pxr import UsdGeom, Gf
    prim = stage.GetPrimAtPath("/World/Franka/panda_hand")
    xform = UsdGeom.Xformable(prim)
    M = xform.ComputeLocalToWorldTransform(0)
    pos = np.array([M[3][0], M[3][1], M[3][2]], dtype=np.float64)
    R3 = np.array([[M[i][j] for j in range(3)] for i in range(3)], dtype=np.float64)
    q_xyzw = Rotation.from_matrix(R3).as_quat()
    return pos, np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def _franka_to_retarget_quat(q_franka_wxyz):
    """Convert Franka panda_hand world quat → retarget-frame convention quat.
    See sim/eval_dp3_baseline3.py for the exact rotation; the trained DP3
    model expects this transformed quaternion."""
    R = Rotation.from_quat([q_franka_wxyz[1], q_franka_wxyz[2], q_franka_wxyz[3], q_franka_wxyz[0]])
    R_flip = Rotation.from_matrix(np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=np.float64))
    R_re = R * R_flip
    q = R_re.as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def _retarget_to_franka_quat(q_retarget_wxyz):
    R = Rotation.from_quat([q_retarget_wxyz[1], q_retarget_wxyz[2], q_retarget_wxyz[3], q_retarget_wxyz[0]])
    R_flip = Rotation.from_matrix(np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=np.float64))
    R_fr = R * R_flip.inv()
    q = R_fr.as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def _query_dp3_server(url, pc_obs, ap_obs, timeout):
    """HTTP call to DP3 inference server. Returns (n_action_steps, 8) np.array."""
    import requests
    r = requests.post(
        f"{url}/predict",
        json={"point_cloud": pc_obs.tolist(), "agent_pos": ap_obs.tolist()},
        timeout=timeout,
    ).json()
    return np.asarray(r["action"], dtype=np.float32)


# ============================================================
# cuRobo 0.8 IK helper (DP3 / online-policy path).
# We can't use partner's plan_trajectory()/init_curobo() in this branch
# because they import from curobo.wrap.reacher.motion_gen which is the
# v0.7 API; our env_isaaclab has v0.8.0 installed (required for RTX 5090
# Blackwell sm_120). This helper uses v0.8's curobo.inverse_kinematics
# directly. partner's plan_trajectory() is untouched and still used by
# execute_open_loop_grasp() for the a2g_pdm path.
# ============================================================
_IK_V08_SOLVER = None
_IK_V08_TOOL_LINK = None
_IK_V08_BATCH_SIZE = 16   # n_action_steps=8 default; pad room for n_obs steps


def _get_ik_v08_solver():
    """Lazy global init of the v0.8 IK solver. Reused across episodes."""
    global _IK_V08_SOLVER, _IK_V08_TOOL_LINK
    if _IK_V08_SOLVER is None:
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
        cfg = InverseKinematicsCfg.create(
            robot="franka.yml",          # cuRobo bundled config
            num_seeds=64,
            max_batch_size=_IK_V08_BATCH_SIZE,
            position_tolerance=0.005,
            orientation_tolerance=0.05,
            self_collision_check=True,
            success_requires_convergence=False,  # don't reject sub-tolerance solutions
        )
        _IK_V08_SOLVER = InverseKinematics(cfg)
        _IK_V08_TOOL_LINK = _IK_V08_SOLVER.tool_frames[0]
        cprint(f"  🧊 cuRobo 0.8 IK initialized — tool_link={_IK_V08_TOOL_LINK} "
               f"batch={_IK_V08_BATCH_SIZE} seeds=64", "green")
    return _IK_V08_SOLVER, _IK_V08_TOOL_LINK


def _solve_ik_chain_v08(targets_world, robot_pos_world, robot_quat_wxyz_world):
    """Batched IK in cuRobo 0.8 API.

    Args:
        targets_world: list of (pos_world(3), quat_world_wxyz(4)) — target EE
            poses in world frame.
        robot_pos_world: (3,) Franka base position in world frame
        robot_quat_wxyz_world: (4,) Franka base orientation in world frame

    Returns:
        qpos: (N, 7) np.float64, NaN-rows where IK failed
        ok:   (N,) np.bool — per-target success
    """
    import torch
    from curobo.types import GoalToolPose, Pose

    n = len(targets_world)
    ik_solver, tool_link = _get_ik_v08_solver()

    # World → robot base frame transform
    R_world_base = Rotation.from_quat([
        robot_quat_wxyz_world[1], robot_quat_wxyz_world[2],
        robot_quat_wxyz_world[3], robot_quat_wxyz_world[0],
    ]).as_matrix()
    R_base_world = R_world_base.T
    p_base_world = -R_base_world @ np.asarray(robot_pos_world, dtype=np.float64)

    positions_base = np.zeros((n, 3), dtype=np.float32)
    quats_base_wxyz = np.zeros((n, 4), dtype=np.float32)
    for i, (pos_w, q_w_wxyz) in enumerate(targets_world):
        positions_base[i] = (R_base_world @ np.asarray(pos_w, dtype=np.float64) + p_base_world).astype(np.float32)
        R_w = Rotation.from_quat([q_w_wxyz[1], q_w_wxyz[2], q_w_wxyz[3], q_w_wxyz[0]]).as_matrix()
        R_base = R_base_world @ R_w
        q_xyzw = Rotation.from_matrix(R_base).as_quat()
        quats_base_wxyz[i] = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)

    pos_t = torch.tensor(positions_base, device="cuda")
    quat_t = torch.tensor(quats_base_wxyz, device="cuda")
    goal = Pose(position=pos_t, quaternion=quat_t)
    result = ik_solver.solve_pose(
        GoalToolPose.from_poses({tool_link: goal}, num_goalset=1)
    )
    # result.js_solution.position shape: (B, 1, 9) — squeeze goalset dim,
    # slice first 7 dims (arm joints; finger joints managed separately)
    qpos_t = result.js_solution.position
    if qpos_t.ndim == 3:
        qpos_t = qpos_t.squeeze(1)        # (B, 9)
    qpos = qpos_t[:, :7].detach().cpu().numpy().astype(np.float64)   # (B, 7)
    ok = result.success.squeeze().detach().cpu().numpy().astype(bool)
    # Mark failed rows as NaN so caller's ok-mask is the source of truth
    if ok.ndim == 0:
        ok = np.array([bool(ok)])
    qpos[~ok] = np.nan
    return qpos, ok


def _get_server_info(url, timeout=10):
    import requests
    return requests.get(f"{url}/info", timeout=timeout).json()


def execute_closed_loop_actions(scene, payload: dict, pc0_world: np.ndarray | None = None,
                                origin_world: np.ndarray | None = None) -> ExecutionResult:
    """Execute a chunked receding-horizon online policy (e.g. DP3).

    Args:
        scene: titan SimEvaluationContext (has world, franka, obj, stage).
        payload: from DP3OnlinePolicy.predict().actions — dict with
            server_url, max_chunks, success_dz_m, retry_physx, n_pc_points,
            request_timeout.
        pc0_world: (N, 3) point cloud in world frame. If None, sample from the
            spawned obj's mesh. Either CALLER must provide or scene.obj must
            expose a usable mesh.
        origin_world: (3,) world position of the G-frame origin. If None,
            inferred from scene.object_placement.

    Returns:
        ExecutionResult with success, z_delta_m, planning details, metadata
        containing per-chunk debug info (chunk count, retry count, grip-close idx).
    """
    from omni.isaac.core.utils.types import ArticulationAction

    server_url      = payload["server_url"]
    max_chunks      = int(payload.get("max_chunks", 5))
    success_dz_m    = float(payload.get("success_dz_m", 0.03))
    retry_physx     = int(payload.get("retry_physx", 1))
    n_pc_points     = int(payload.get("n_pc_points", 4096))
    request_timeout = int(payload.get("request_timeout", 60))

    franka = scene.franka
    world  = scene.world
    obj    = scene.obj
    stage  = scene.stage

    # ── /info handshake to get n_obs, n_action_steps ────────────────
    try:
        info = _get_server_info(server_url, timeout=10)
    except Exception as e:
        res = ExecutionResult(success=False, failure_stage="server_info")
        res.metadata["error"] = f"server /info failed: {e}"
        return res
    n_obs    = int(info.get("n_obs_steps", 2))
    n_action = int(info.get("n_action_steps", 8))

    # ── sample PC from rotated_mesh PLY (partner's prepare_metric_point_cloud,
    # added in titan 188ff39) ───────────────────────────────────────────────
    # Priority: caller-supplied pc0_world > mesh-sampled from obj_id.
    from sim.evaluation.scene_builder import OBJECT_POSITION
    placement = scene.object_placement
    pos_w = np.array(placement.get("position", OBJECT_POSITION), dtype=np.float64)
    if origin_world is None:
        origin_world = pos_w
    if pc0_world is None:
        try:
            from model.pdm.mesh_points import prepare_metric_point_cloud
        except ImportError as e:
            res = ExecutionResult(success=False, failure_stage="pc_unavailable")
            res.metadata["error"] = f"model.pdm.mesh_points import failed: {e}"
            return res
        mesh_root = payload.get("mesh_root", "data_hub/meshes/SAM3DMesh/rotated_mesh")
        dataset   = getattr(scene.spec, "dataset", None) or "oakink"
        obj_id    = scene.spec.obj_id
        try:
            pts_canonical, _, mesh_path = prepare_metric_point_cloud(
                obj_id, mesh_root=mesh_root, dataset=dataset,
                num_points=n_pc_points, seed=0,
            )
        except Exception as e:
            res = ExecutionResult(success=False, failure_stage="mesh_sample")
            res.metadata["error"] = f"prepare_metric_point_cloud({obj_id}, {dataset}): {e}"
            return res
        # mesh canonical → world: apply obj quat (yaw aug) + spawn translation
        obj_quat_wxyz = np.array(placement.get("quat_wxyz", [1, 0, 0, 0]), dtype=np.float64)
        R_obj = Rotation.from_quat([obj_quat_wxyz[1], obj_quat_wxyz[2],
                                    obj_quat_wxyz[3], obj_quat_wxyz[0]]).as_matrix()
        pc0_world = (pts_canonical @ R_obj.T + pos_w).astype(np.float32)
        cprint(f"  📍 sampled PC from {mesh_path} ({n_pc_points} pts, dataset={dataset})", "cyan")
    pc0_G = (np.asarray(pc0_world, dtype=np.float32) - origin_world).astype(np.float32)

    initial_obj_pos, _ = obj.get_obj_pos()
    initial_z = float(initial_obj_pos[2])

    def _build_obs(gripper_state: float):
        ee_pos_w, ee_q_w = _read_panda_hand_pose(stage)
        ee_pos_G = (ee_pos_w - origin_world).astype(np.float32)
        ee_q_G_retarget = _franka_to_retarget_quat(ee_q_w).astype(np.float32)
        agent_pos = np.concatenate([ee_pos_G, ee_q_G_retarget,
                                    [np.float32(gripper_state)]]).astype(np.float32)
        return pc0_G.astype(np.float32), agent_pos

    def _qpos_corrupt() -> bool:
        try:
            qa = np.asarray(franka.get_joint_positions(), dtype=np.float64)
            return (not np.isfinite(qa).all()) or (np.max(np.abs(qa)) > 10.0)
        except Exception:
            return True

    # ── reset franka HOME + open gripper (caller already did scene reset) ──
    franka.open_gripper()
    for _ in range(5):
        world.step(render=scene.render)

    obs0 = _build_obs(gripper_state=0.0)
    obs_window = [obs0] * n_obs
    last_qpos = np.asarray(franka.get_joint_positions()[:7], dtype=np.float64)
    executed = []
    grip_signal_idx = None
    gripper_closed = False
    grip_close_initial_z = None

    for chunk in range(max_chunks):
        # update obs window's gripper-state channel to reflect current physics
        cur_obs = _build_obs(gripper_state=(1.0 if gripper_closed else 0.0))
        obs_window[-1] = cur_obs
        pc_obs = np.stack([o[0] for o in obs_window])
        ap_obs = np.stack([o[1] for o in obs_window])
        try:
            action = _query_dp3_server(server_url, pc_obs, ap_obs, request_timeout)
        except Exception as e:
            cprint(f"  ❌ DP3 server error chunk {chunk}: {e}", "red")
            res = ExecutionResult(success=False, failure_stage="server_predict")
            res.metadata["error"] = str(e); res.metadata["chunk"] = chunk
            return res

        # convert each action to world target pose
        chunk_wps = []
        chunk_grips = []
        for a in action:
            pos_w = a[:3].astype(np.float64) + origin_world
            q_franka = _retarget_to_franka_quat(a[3:7].astype(np.float64))
            chunk_wps.append((pos_w, q_franka))
            chunk_grips.append(float(a[7]))

        # ── IK chain (v0.8 batched, see _solve_ik_chain_v08) ──
        # Robot base pose in world frame, from titan's ROBOT_POSITION /
        # ROBOT_ORIENTATION constants (scene_builder.py).
        robot_quat_wxyz_world = _euler_xyz_deg_to_wxyz(list(ROBOT_ORIENTATION))
        try:
            qpos_all, ok = _solve_ik_chain_v08(
                chunk_wps,
                robot_pos_world=np.array(ROBOT_POSITION, dtype=np.float64),
                robot_quat_wxyz_world=robot_quat_wxyz_world,
            )
            qpos = np.where(np.isnan(qpos_all), last_qpos, qpos_all)
        except Exception as e:
            cprint(f"  ❌ v0.8 IK chain failed chunk {chunk}: {e}", "red")
            qpos = np.tile(last_qpos, (n_action, 1))
            ok = np.zeros(n_action, dtype=bool)
        cprint(f"  [chunk {chunk}] IK {ok.sum()}/{n_action} reachable, "
               f"grip [{min(chunk_grips):.2f}, {max(chunk_grips):.2f}], closed={gripper_closed}",
               "cyan")

        # ── execute waypoints ──
        for k in range(n_action):
            if not ok[k]:
                continue
            grip = chunk_grips[k]
            if not gripper_closed and grip > 0.5:
                grip_signal_idx = len(executed)
                grip_close_initial_z = float(obj.get_obj_pos()[0][2])
                cprint(f"  ◉ DP3 grip≥0.5 @ chunk {chunk} step {k}", "magenta")
                franka.close_gripper()
                for _ in range(80):
                    world.step(render=scene.render)
                gripper_closed = True
            if gripper_closed:
                franka.close_gripper()
                franka.apply_action(ArticulationAction(
                    joint_positions=np.concatenate([qpos[k], np.array([None, None])])))
                for _ in range(3):
                    world.step(render=scene.render)
            else:
                grip_finger = franka.get_joint_positions()[7:9]
                full_q = np.concatenate([qpos[k], grip_finger])
                franka.set_joint_positions(full_q)
                franka.apply_action(ArticulationAction(joint_positions=full_q))
                for _ in range(2):
                    world.step(render=scene.render)

            if _qpos_corrupt():
                cprint(f"  ⚠️ PhysX corrupted Franka qpos at chunk {chunk} step {k}", "red")
                res = ExecutionResult(success=False, failure_stage="physx_corrupt")
                res.metadata["chunk"] = chunk; res.metadata["step"] = k
                return res

            executed.append((chunk_wps[k][0].copy(), chunk_wps[k][1].copy()))
            last_qpos = qpos[k].copy()

        # ★ early stop: if gripper closed AND obj lifted enough → done
        if gripper_closed and grip_close_initial_z is not None:
            obj_pos_now, _ = obj.get_obj_pos()
            dz_now = float(obj_pos_now[2]) - grip_close_initial_z
            if dz_now > success_dz_m:
                cprint(f"  🎯 EARLY STOP @ chunk {chunk}: dz={dz_now*100:.1f}cm", "green")
                res = ExecutionResult(
                    success=True, z_delta_m=dz_now,
                    initial_object_position_world=list(initial_obj_pos),
                    final_object_position_world=list(obj.get_obj_pos()[0]),
                )
                res.metadata.update({
                    "policy": "dp3_online", "n_chunks": chunk + 1,
                    "grip_signal_idx": grip_signal_idx,
                    "n_executed": len(executed), "early_stop": True,
                })
                return res

        # roll obs window
        new_obs = _build_obs(gripper_state=(1.0 if gripper_closed else 0.0))
        obs_window = obs_window[1:] + [new_obs]

    # ── exited loop without early stop ──
    for _ in range(80):
        world.step(render=scene.render)
    if grip_close_initial_z is None:
        cprint(f"  ⚠️ no grip-close signal across {max_chunks} chunks", "yellow")
        res = ExecutionResult(success=False, failure_stage="no_grip_signal")
        res.metadata.update({"policy": "dp3_online", "n_chunks": max_chunks,
                             "n_executed": len(executed)})
        return res
    obj_after, _ = obj.get_obj_pos()
    dz = float(obj_after[2]) - grip_close_initial_z
    success = dz > success_dz_m
    cprint(f"  object Z Δ = {dz*100:+.1f}cm → "
           f"{'GRASPED + LIFTED' if success else 'not lifted'}",
           "green" if success else "red")
    res = ExecutionResult(
        success=bool(success), z_delta_m=dz,
        failure_stage=None if success else "not_lifted",
        initial_object_position_world=list(initial_obj_pos),
        final_object_position_world=list(obj_after),
    )
    res.metadata.update({
        "policy": "dp3_online", "n_chunks": max_chunks,
        "grip_signal_idx": grip_signal_idx,
        "n_executed": len(executed), "early_stop": False,
    })
    return res
