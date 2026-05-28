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
   - **IN ROBOT BASE FRAME** — compute `T_base_world` once at setup,
     transform `ee_world` via `T_base_world @ ee_world`
   - gripper convention: **-1=open, +1=close** (negate of our usual)
   - Append to ring buffer; pass list of last 4

3. **ZMQ client**:
   ```python
   import zmq, pickle
   ctx = zmq.Context()
   sock = ctx.socket(zmq.REQ)
   sock.connect(payload["server_addr"])
   sock.setsockopt(zmq.RCVTIMEO, payload["request_timeout_ms"])
   ```

4. **Per-chunk loop**:
   ```python
   for chunk in range(max_chunks):
       front_img = front_cam.get_rgb()   # (H, W, 3) uint8
       side_img  = side_cam.get_rgb()
       ee_pose_base = read_panda_hand_in_robot_base_frame(stage)
       proprio_buffer.append(ee_pose_base)

       req = {
           "text": payload["instruction"],
           "front_view_image": [front_img],
           "side_view_image": [side_img],
           "proprio_array": proprio_buffer[-4:],
       }
       sock.send_pyobj(req)
       resp = sock.recv_pyobj()
       deltas = resp["result"]  # (16, 7) after server's 2× interpolation

       # integrate deltas in robot-base frame
       current_ee_base = ee_pose_base
       for delta in deltas:
           target_xyz = current_ee_base[:3] + delta[:3]
           target_rpy = current_ee_base[3:6] + delta[3:6]
           grip = delta[6]  # -1=open, +1=close
           if grip > 0.5: franka.close_gripper()
           elif grip < -0.5: franka.open_gripper()
           # IK + execute (transform target back to world for IK)
           target_world_xyz = T_world_base @ target_xyz
           qpos = ik_solve(target_world_xyz, target_world_quat, ...)
           franka.set_joint_positions(qpos)
           world.step()
           current_ee_base += delta  # update accumulator

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
