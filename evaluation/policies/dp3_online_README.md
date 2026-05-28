# DP3 closed-loop online policy adapter for partner's titan eval infra

This branch (`titan-dp3-integration`) adds **closed-loop online policy** support
to the partner's evaluation pipeline, with DP3 (3D Diffusion Policy) as the
first concrete adapter.

The work plugs into the **`closed_loop_actions`** slot in
`evaluation/specs.py::PolicyKind`, which the partner had reserved in the
schema but not yet implemented.

## What was added

| file | purpose | LOC |
|---|---|---|
| `evaluation/policies/dp3_online.py` | DP3OnlinePolicy adapter (`predict()` returns a `closed_loop_actions` PolicyOutput) | ~80 |
| `evaluation/policies/dp3_solution_gen.py` | Helper to emit "solution" JSONs that the task-queue infra (`eval_pool.py`) consumes | ~100 |
| `sim/evaluation/curobo_executor.py` | Appended `execute_closed_loop_actions()` — receding-horizon executor ported from `gate3-curobo-ik` branch `sim/eval_dp3_baseline3.py:rollout_chunked` | ~250 |
| `sim/evaluation/run_eval_worker.py` | Dispatch added for `closed_loop_actions` kind; skips `write_robot_gt_hdf5` for closed-loop | ~15 |

## Architecture

```
            (offline: write task JSON)
            ┌──────────────────────────────┐
            │ dp3_solution_gen.py          │
            │   makes {policy_output:      │
            │     kind=closed_loop_actions,│
            │     actions={server_url,…} } │
            └────────────────┬─────────────┘
                             │ solution_path
                             ↓
            (online: run_eval_worker)
┌────────────────────────────────────────────────────┐
│ for each task in chunk:                            │
│   solution = load_json(task.solution_path)         │
│   policy_output = _policy_output_from_solution(…)  │
│                                                    │
│   if kind == "open_loop_grasp":  ← a2g_pdm path    │
│     execute_open_loop_grasp(scene, command)        │
│                                                    │
│   elif kind == "closed_loop_actions":  ← DP3 path  │
│     execute_closed_loop_actions(scene, payload,    │
│                                  pc0_world, origin)│
│       │                                            │
│       └──> HTTP /predict to server_url             │
│            (Baseline1/eval/dp3_inference_server.py │
│             in gate3-curobo-ik branch)             │
└────────────────────────────────────────────────────┘
```

## Eval-time prerequisites

1. **DP3 inference server must be running** before launching the eval pool.
   This is a separate process (in `dp3` conda env, not `env_isaaclab`):
   ```bash
   /home/accelerator/miniforge3/envs/dp3/bin/python \
       Baseline1/eval/dp3_inference_server.py \
       --ckpt <path-to-ckpt> --port 8765
   ```
   The server file lives in the **gate3-curobo-ik** branch; verify it's
   accessible from this worktree (cross-branch shared file).

2. **Tasks must include `pc0_world` and `origin_world`** in the task dict
   (the executor cannot infer these robustly from the SceneSpec alone for
   arbitrary obj). Suggested: pre-sample the obj mesh once per (obj_id, yaw)
   and embed in the task spec.

## Per-ep policy behavior (matches gate3-curobo-ik DP3 eval)

- Pre-rollout: `franka.open_gripper()` + 5 settle steps
- For each chunk (up to `max_chunks`, default 5):
  - Build obs: PC (static, from `pc0_world`) + EE state (live read)
  - Query `/predict` (HTTP, 60s timeout)
  - Server returns `[n_action_steps, 8]` action array
  - IK each waypoint via cuRobo `plan_trajectory` (with object mesh)
  - Execute: teleport (gripper open) or PD (gripper closed)
  - When `grip > 0.5` first time: `close_gripper()` + 80 settle steps
  - **Early-stop**: if `obj.z - grip_close_initial_z > success_dz_m` (3cm)
- PhysX corruption check between waypoints; returns `physx_corrupt` stage if so
- After loop: 80 settle steps + final `dz` measurement

## What's NOT yet ported

- `--retry-physx N`: gate3 branch retries 1× on `physx_corrupt`. This adapter
  doesn't (yet) — `_execute_task` returns the failure once. Add a retry loop
  in `_execute_task` if desired (similar to gate3 `sim/eval_dp3_baseline3.py`
  around line 887).
- `--n-servers N` round-robin: currently 1 server per worker chunk. For
  multi-server scaling, modify `eval_pool.py` to distribute server ports
  across workers (already supports `pool_size`).
- Per-ep `world.reset()`: partner's `swap_scene_object` already does this
  between obj-id switches; for same-obj different-yaw, `reset_scene_pose`
  resets the franka HOME. **Cross-trial** within same (obj, yaw) might
  still have residual PhysX state — TBD whether `reset_motion_gen()` is
  enough. If `physx_corrupt` returns are observed, add explicit reset
  before re-call.

## Smoke test plan (post-merge)

1. Start DP3 server with 2800 ckpt
2. Generate 1 task JSON (single obj, 1 trial) via `dp3_solution_gen.py`
3. Run single-task eval via `evaluation/eval_single.py` (existing infra)
4. Verify: action shape (8, 8), early-stop triggers, JSON output matches
5. Compare SR with gate3 branch `sim/eval_dp3_baseline3.py` (must be within 5%)

## Origin & migration notes

- Source pipeline: `gate3-curobo-ik` branch, `sim/eval_dp3_baseline3.py`
  (850-line monolithic eval driver)
- Key lessons applied:
  - **PhysX poison fix**: per-ep `world.reset()` clears solver NaN
    (gate3 branch saved 21% of eps from being lost to NaN propagation)
  - **Multi-worker servers**: 3× /predict throughput on 5090
  - **Early-stop**: 15-20% per-ep time saved when policy lifts early
- Co-existing branches:
  - `gate3-curobo-ik`: production DP3 eval (HTTP server + per-class shell
    orchestrator). Kept stable for ckpt-comparison workflow.
  - **`titan-dp3-integration`** (this branch): adapter for partner infra.
  - `titan`: partner's main branch — PR target when this is stable.
