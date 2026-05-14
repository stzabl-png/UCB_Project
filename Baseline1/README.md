# Baseline1 — Human Retarget DP

A DP3 policy trained on **human grasp trajectories retargeted to a Franka 2-finger
gripper**. Same human data as the main method (Affordance2Grasp), but it learns the
*full end-effector trajectory* instead of an object-surface contact map.

## Pipeline

```
Dataset RGB-D grasp video  (DexYCB now; OakInk / HO3D-v3 later via the same code)
    │
    ├─ GT object 6D pose (pose_y)  ──┐
    │                                 ├─→ object point cloud (T, 4096, 3)
    │  SAM3D mesh ycb_dex_NN ─────────┘     = mesh surface samples × GT object pose
    │                                       (same mesh the main method / Baseline2 use)
    │
    └─ GT MANO joints (joint_3d, camera-frame metres)
         │  retarget_human_to_ee.py   (analytic, ~1 file)
         ▼
       Franka EE trajectory:  state[t] = [x,y,z, qw,qx,qy,qz, gripper] = 8D
                              action[t] = state[t+1]
         │  + object-centric frame (subtract object centroid at frame 0)
         ▼
       episode HDF5  →  convert_to_zarr.py  →  DP3 zarr  →  train DP3
```

## Why dataset GT, not our HaPTIC / FoundationPose outputs

The Phase-1A pipeline's HaPTIC MANO (hand-size-anchored metric) and FoundationPose
object pose (DepthPro-fx-anchored metric) are **scale-inconsistent** in camera frame —
that is exactly why the main method's contact-align step works in 2D pixel space.
Baseline1 needs a *3D* EE trajectory, so the hand and object must share one metric.
DexYCB ships per-frame GT MANO (`joint_3d`, `pose_m`) and GT object 6D pose (`pose_y`)
in one consistent camera frame, so we use those directly. Bonus: `pose_m[3:48]` gives
the 45 MANO articulation angles (stored as episode metadata). Baseline1 is therefore
**independent of the running Phase-1A pipeline** — it only needs the raw dataset on disk.

## Retarget (analytic, parallel gripper)

```
t = thumb tip  (joint_3d[4])
i = index tip  (joint_3d[8])
w = wrist      (joint_3d[0])
c   = (t + i) / 2                              pinch midpoint ≈ between the two fingertips
ex  = norm(i − t)                              gripper closing/opening axis
ez  = norm( (c − w) − ((c − w)·ex) ex )        approach axis (the gripper points this way)
ey  = ez × ex                                  (right-handed by construction)
R_ee   = [ex | ey | ez]
p_ee   = c − 0.10 · ez                         EE frame ≈10 cm behind fingertips (≈ Franka panda_hand)
gripper = clip(1 − ‖i − t‖ / 0.08, 0, 1)       Franka 8 cm fingertip span; 0 = open, 1 = closed
```
plus quaternion sign-continuity along the trajectory. `gripper` uses the same convention
as `Baseline2/collect_sim_trajectories.py`'s `1 − mean(fingers)/0.04`, so the two DP
baselines have an **identical 8D action space**. Left-handed grasps need no special case
(the closing axis ±ex is symmetric for a parallel gripper; `ey = ez × ex` is always
right-handed; continuity resolves the sign).

## Data format (per episode HDF5, one per `subject__session__camera`)

| Field | Shape | Description |
|---|---|---|
| `point_cloud` | `(T, 4096, 3)` | object surface points, **object-centric frame** (object centroid at frame 0 = origin) |
| `state` | `(T, 8)` | EE pose `[x,y,z, qw,qx,qy,qz, gripper]` in the object-centric frame |
| `action` | `(T, 8)` | `state` shifted by one step (next EE target) |
| `finger_angles` | `(T, 45)` | GT MANO articulation axis-angles (metadata; **not** used by DP3) |
| `wrist_pose` | `(T, 4, 4)` | GT MANO root pose, object-centric frame (metadata) |

The zarr (`convert_to_zarr.py`) keeps only `point_cloud / state / action` + `meta/episode_ends`
— byte-compatible with Baseline2's zarr.

## Step-by-step usage

```bash
# 0. (one-time) sanity check on 2 sessions of subject-07, with a visualization
python Baseline1/retarget_human_to_ee.py --subject 20200928-subject-07 --limit 2

# 1. full route-A run — subjects 07, 08, 09, 10 (the ones with full GT on disk),
#    the 6 cameras the main pipeline uses
python Baseline1/retarget_human_to_ee.py
#    output: Baseline1/data/episodes/dexycb__{subj}__{session}__{cam}.hdf5
#    (CPU + disk only — safe to run alongside the GPU Phase-1A pipeline)

# 2. concatenate → DP3 zarr
python Baseline1/convert_to_zarr.py \
    --input_dir   Baseline1/data/episodes \
    --output_zarr Baseline1/data/human_dp_baseline.zarr

# 3. point the DP3 task config at it (VideoPolicy_internal / ~/Young_VideoPolicy/dp3)
#    dp3/diffusion_policy_3d/config/task/grasping.yaml:
#      dataset: { zarr_path: .../Baseline1/data/human_dp_baseline.zarr }
#      shape_meta: { obs: { point_cloud: {shape: [4096,3]}, state: {shape: [8]} },
#                    action: {shape: [8]} }

# 4. train DP3 (needs the GPU — run after Phase-1A frees it)
cd ~/Young_VideoPolicy/dp3 && bash scripts/train_policy.sh dp3 grasping baseline1_seed0 0 0
```

## Caveats / TODO

- **Datasets**: route A needs the dataset's GT annotations. On disk, full DexYCB GT
  (`labels_*.npz` / `meta.yml` / `pose.npz`) is present only for **subjects 07–10**
  (subjects 01–06 are color-jpg-only). Add 01–06 later by downloading the full DexYCB,
  or fall back to the pipeline's HaPTIC/FP for them. OakInk and HO3D-v3 raw data are on
  disk and have GT — supported by adding a small per-dataset GT loader (same retarget).
- **Object coverage (route A, subj 07–10)**: 19 of the 20 YCB grasp objects appear —
  `ycb_dex_19` (`051_large_clamp`) is never the grasped object in these 4 subjects'
  sessions (would appear once subjects 01–06 are added). Foam-brick grasps (`ycb_id 21`)
  are skipped — there is no `ycb_dex_21` SAM3D mesh.
- **Object mesh frame (v1)**: the object point cloud is the SAM3D `ycb_dex_NN` mesh
  (correct *size* — `scale.json` is calibrated to `real_diameter_m`) placed by the
  dataset's `pose_y`, which is defined w.r.t. the *YCB CAD model* frame. The SAM3D
  mesh's canonical frame differs from the YCB CAD frame by a constant per-object rigid
  offset, so the rendered object *orientation* carries that constant offset (location is
  GT-correct). It is consistent within every sequence of an object. A clean v2 would
  ICP-align each `ycb_dex_NN` mesh to the YCB CAD model and compose, removing the offset.
- EgoDex has no GT 6D object pose / no real depth → route A is impossible there; if
  needed it would go through the pipeline's HaWoR + a re-alignment step (route B).

## Comparison

| | Affordance2Grasp (ours) | **Baseline1 (Human Retarget DP)** | Baseline2 (Robot DP Sim) |
|---|---|---|---|
| Human data | ✅ | ✅ same human data | ❌ |
| Learns | object-surface affordance points | full EE trajectory (retargeted from human hand) | full EE trajectory (sim-collected) |
| Model | PointNet++ segmentation | DP3 | DP3 |
| Observation | object point cloud (mesh) | object point cloud (mesh), object-centric | object point cloud (mesh), robot-base |
| Action space | — | 8D `[xyz, quat, gripper]` | 8D `[xyz, quat, gripper]` (identical) |
| cuRobo at inference | ✅ | ❌ (policy outputs the trajectory directly) | ❌ |
| Expected generalisation | best (object-relative contacts) | worse (pose-dependent trajectory) | worst (no human prior) |
