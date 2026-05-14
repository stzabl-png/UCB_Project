# Baseline2 — Robot DP (Sim) Baseline

A DP3 policy trained purely on **Isaac Sim robot trajectories** — no human data, no teleoperation.

## Pipeline

```
Object Mesh
    ↓ random_grasp_sampler.py     → N random candidates (no human prior)
    ↓ Isaac Sim + cuRobo           → plan + execute each candidate
    ↓ record per step:             → EE state(8D) + action(8D) + point cloud(4096,3)
    ↓ success check (z_delta>3cm) → keep only successful episodes
    ↓ convert_to_zarr.py          → DP3 training format
    ↓ VideoPolicy_internal DP3    → train policy
```

## Data Format

| Field | Shape | Description |
|---|---|---|
| `point_cloud` | `(T, 4096, 3)` | Object surface pts in **robot base frame** (mesh-sampled, same as Affordance2Grasp) |
| `state` | `(T, 8)` | EE pose `[x,y,z, qw,qx,qy,qz, gripper]` in robot base frame |
| `action` | `(T, 8)` | State at next timestep (next EE target) |

## Step-by-Step Usage

### Prerequisites
- Isaac Sim 4.5 installed (`sim45` alias)
- cuRobo installed and working inside Isaac Sim python
- Object USDs in `output/assets/` or `sim/assets/`

### Step 1 — Collect episodes

```bash
# Single object, 50 candidates (headless)
sim45 Baseline2/collect_sim_trajectories.py \
    --obj_id mug \
    --n_candidates 50 \
    --headless \
    --output_dir Baseline2/data/episodes

# All objects in config, 100 candidates each
sim45 Baseline2/collect_sim_trajectories.py \
    --all_objects \
    --n_candidates 100 \
    --headless \
    --output_dir Baseline2/data/episodes
```

**Output:** `Baseline2/data/episodes/{obj_id}_ep{N:04d}.hdf5`

### Step 2 — Convert to zarr

```bash
python Baseline2/convert_to_zarr.py \
    --input_dir  Baseline2/data/episodes \
    --output_zarr Baseline2/data/robot_dp_baseline.zarr
```

### Step 3 — Edit grasping.yaml (VideoPolicy_internal)

```yaml
# dp3/diffusion_policy_3d/config/task/grasping.yaml
dataset:
  zarr_path: /path/to/Affordance2Grasp/Baseline2/data/robot_dp_baseline.zarr

shape_meta:
  obs:
    point_cloud:
      shape: [4096, 3]    # xyz only (no color)
    state:
      shape: [8]           # xyz+quat+gripper
  action:
    shape: [8]
```

### Step 4 — Train DP3

```bash
cd /path/to/VideoPolicy_internal/dp3
bash scripts/train_policy.sh dp3 grasping baseline2_seed0 0 0
```

## Design Decisions

**Point cloud source:** Mesh surface sampling (same as Affordance2Grasp main method), NOT
depth camera. This ensures the two methods have identical observation spaces.

**No human prior:** `generate_candidates_iterative(mesh, obj_id, hp_dir=None)` → uniform
random sampling on mesh surface. This is the key difference from the main method.

**State/Action:** EE end-effector pose (7D) + gripper (1D) = 8D. Simpler than
VideoPolicy_internal's 13D (no dexterous hand here, Franka parallel gripper only).

**One pass:** Record trajectory during execution. Save only if `z_delta > 3cm` at the end.

## Comparison with Main Method

| | Affordance2Grasp (ours) | Baseline2 (Robot DP Sim) |
|---|---|---|
| Human data | ✅ (DexYCB/HO3D/OakInk/EgoDex) | ❌ |
| Sim robot | ✅ (affordance filter) | ✅ (trajectory data) |
| Learns | Affordance points | Full trajectory |
| Generalizes | ✅ new poses | ❌ pose-dependent |
| cuRobo at inference | ✅ | ✅ |
