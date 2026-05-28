# GraspVLA closed-loop adapter on titan eval infra

This file documents the GraspVLA (PKU-EPIC) baseline integration into our
`titan-dp3-integration` branch. Parallel to DP3's adapter.

See also: `dp3_online_README.md` for general closed_loop_actions architecture.

## What's added

| file | LOC | purpose |
|---|---|---|
| `evaluation/policies/graspvla_online.py` | ~90 | GraspVLAOnlinePolicy — wraps ZMQ server endpoint into `closed_loop_actions` PolicyOutput |
| `sim/evaluation/curobo_executor.py` | TBD | needs `execute_closed_loop_actions` dispatcher to branch on `policy_name="graspvla"` → VLA-specific executor (delta-action integration, 2 RGB render, ZMQ client) |
| TBD: `sim/evaluation/cam_setup.py` | ~50 | helper to spawn 2 sim cameras at LIBERO positions, render to (256,256,3) uint8 |

## Architecture

```
DP3OnlinePolicy.predict()  →  PolicyOutput(kind=closed_loop_actions,
                                           metadata.policy_name="dp3_online")
                                          ↓
GraspVLAOnlinePolicy.predict() →  PolicyOutput(kind=closed_loop_actions,
                                              metadata.policy_name="graspvla")
                                          ↓
              sim/evaluation/curobo_executor.execute_closed_loop_actions(scene, payload)
                                          ↓
              dispatch on metadata.policy_name:
                   "dp3_online"  → existing DP3 path (HTTP, PC sampling)
                   "graspvla"    → NEW VLA path (ZMQ, 2 RGB render, delta integration)
```

## Day 2-3 TODO (executor side)

Inside `execute_closed_loop_actions`, add a `policy_name` branch:

```python
def execute_closed_loop_actions(scene, payload):
    policy_name = payload.get("_policy_name", "dp3_online")  # need to extract from metadata
    if policy_name == "graspvla":
        return _execute_vla(scene, payload)
    else:
        return _execute_dp3(scene, payload)   # current code, refactor into helper
```

### _execute_vla() requirements

1. **Camera setup** (one-time per ep):
   ```python
   front_cam = Camera(prim_path="/World/CameraFront",
                      position=payload["front_view_pos"],
                      orientation=payload["front_view_quat"])
   side_cam  = Camera(prim_path="/World/CameraSide", ...)
   ```
   (need conversion from MJCF/LIBERO frame to IsaacSim world frame —
   simplest: relative-to-robot-base shift.)

2. **Proprio history buffer** (length 4):
   - Per chunk, read EE pose → 7D `[x, y, z, roll, pitch, yaw, gripper]`
   - **IN panda_link0 (Franka base) FRAME** — compute `T_base_world` once at
     setup, transform `ee_world` via `T_base_world @ ee_world`
   - **EE point**: `panda_EE + REAL_EEF_TO_SIM_EEF`. Our standard finger →
     identity, no shift. If using extended finger, add +3cm Z (EE local).
   - rpy axes: **transforms3d 'sxyz'** (extrinsic XYZ)
   - **gripper: +1 = OPEN, -1 = CLOSE** (verified in 3 source files —
     opposite of what we initially thought!)
   - Append to ring buffer; pass list of last 4 (server uses [-4] and [-1])

3. **ZMQ client**:
   ```python
   import zmq, pickle
   ctx = zmq.Context()
   sock = ctx.socket(zmq.REQ)
   sock.connect(payload["server_addr"])
   sock.setsockopt(zmq.RCVTIMEO, payload["request_timeout_ms"])
   ```

4. **Per-chunk loop** (verified against grasp_mode.py:_run_once):
   ```python
   import transforms3d as t3d
   for chunk in range(max_chunks):
       front_img = front_cam.get_rgb()  # (256, 256, 3) uint8
       side_img  = side_cam.get_rgb()

       # Read EE pose in base frame as [x,y,z, roll,pitch,yaw, gripper]
       ee_pose_base = read_eef_pose_in_base_frame(stage, panda_link0_pose,
                                                   extended_finger=False)

       proprio_buffer.append(ee_pose_base)
       req = {
           "text": payload["instruction"],
           "front_view_image": [front_img],
           "side_view_image": [side_img],
           # NOTE: real-world-controller uses [prev_eef_pose * 3, eef_pose]
           # to fill a length-4 buffer when no history. We do same:
           "proprio_array": [proprio_buffer[-2]] * 3 + [proprio_buffer[-1]]
                            if len(proprio_buffer) >= 2 else [ee_pose_base]*4,
       }
       sock.send_pyobj(req)
       resp = sock.recv_pyobj()
       deltas = resp["result"]  # (16, 7) — 8 model × 2× interpolation

       # Integrate deltas in BASE FRAME (left-multiply for rotation!)
       # Verified against grasp_mode.py:109-115:
       current_pos = ee_pose_base[:3]
       current_rot_mat = t3d.euler.euler2mat(*ee_pose_base[3:6])  # sxyz
       for delta in deltas:
           assert delta[6] in [-1, 0, +1]
           target_pos = current_pos + delta[:3]               # pure add
           target_rot_mat = (t3d.euler.euler2mat(*delta[3:6]) @ current_rot_mat)
           # ★ LEFT-multiply (base frame composition), NOT EE-local right-mult ★
           grip = delta[6]
           if grip > 0:    franka.open_gripper()    # +1 = OPEN!
           elif grip < 0:  franka.close_gripper()   # -1 = CLOSE!
           # else (0): no change

           # Transform target back to world frame for IK
           target_world_pos = panda_link0_pose @ np.append(target_pos, 1.0)
           target_world_quat = t3d.quaternions.mat2quat(R_world_base @ target_rot_mat)
           qpos = ik_solve(target_world_pos[:3], target_world_quat, ...)
           franka.set_joint_positions(qpos)
           world.step()
           current_pos = target_pos
           current_rot_mat = target_rot_mat

       # check early-stop
       obj_z = obj.get_obj_pos()[0][2]
       if obj_z - initial_obj_z > success_dz_m: return success
   ```

## Open questions (need answers before Day 2 executor work)

1. **T_base_world for our IsaacSim**: robot at (0.2, -0.05, 0.8). Verify whether
   we want EE pose in robot-base-link frame (= isaacsim Franka base) or some
   other reference. Likely `T_base_world = inv(robot_base_pose)` —
   straightforward.

2. **euler convention**: GraspVLA uses RPY (sample shows roll=3.136≈π).
   Convention is probably extrinsic XYZ via transforms3d. Need to match
   when converting our quat → euler.

3. **Delta integration for rotation**: Adding `current_rpy + delta_rpy` is NOT
   correct in general (euler angles don't compose by addition). Need to
   convert delta_rpy → rot matrix, multiply with current rot mat, then back
   to whatever target_quat IK needs. See transforms3d.euler.euler2mat.

4. **LIBERO frame transform**: LIBERO MJCF world has robot at scene origin
   with z=0.95 floor or so. Our IsaacSim has robot at (0.2,-0.05,0.8) on
   a virtual table. Need to verify camera positions look the same way in
   the rendered output. Recommend: render once, visually compare to
   real-world.png example.

5. **Gripper sign double-convention**: our Franka uses 0=open, 1=close
   typically. GraspVLA proprio uses -1=open, +1=close. MUST convert at
   both input (our → VLA) and output (VLA action → our control).

## Smoke prerequisites

- DP3 smoke must pass first (validates closed_loop_actions infra)
- GraspVLA conda env set up (python 3.9.19 + torch 2.7.1 + safetensors)
- Pretrained ckpt downloaded: `hf download shengliangd/GraspVLA`
- Optional: visualize trial-20250507120350_data.npy first to verify ZMQ format

## ETA

Day 2-3 (after DP3 smoke passes):
- _execute_vla() implementation: ~4-6 h
- camera setup helpers: ~2 h
- frame transform utilities: ~2 h
- smoke test + frame debug: ~4-8 h
