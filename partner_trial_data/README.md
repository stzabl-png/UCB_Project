# Partner Trial Data — 2 successfully-grasped human-retarget trajectories

Two reference trajectories from the DexYCB human-grasp dataset that have been
**retargeted to a Franka Emika Panda (panda_hand)** end-effector and **verified
to lift the object in IsaacSim physics** (object Z-displacement > 3 cm after
gripper close + lift).

Use these to (a) sanity-check your downstream consumer (DP3 dataset loader,
controller, replay tool), and (b) compare retarget conventions / coordinate
frames before integrating the full dataset.

---

## Layout

```
partner_trial_data/
├── README.md                                  (this file)
├── sugar_box_ycb_dex_03/
│   ├── trajectory.hdf5                        retargeted EE+point-cloud
│   └── sim_grasp_success.mp4                  IsaacSim replay (grasp+lift)
└── tomato_soup_can_ycb_dex_04/
    ├── trajectory.hdf5
    └── sim_grasp_success.mp4
```

Source DexYCB sessions (camera 840412060917 = master cam):

| folder | object | DexYCB subject | DexYCB session | n_steps |
|---|---|---|---|---|
| `sugar_box_ycb_dex_03/`        | YCB 004 sugar box      | `20200709-subject-01` | `20200709_142517` | 34 |
| `tomato_soup_can_ycb_dex_04/`  | YCB 005 tomato soup can | `20201015-subject-09` | `20201015_143403` | 22 |

---

## trajectory.hdf5 — exact schema

Each file has 3 datasets and ~13 attributes:

```
state         (T, 8)   float32   robot state at every timestep
action        (T, 8)   float32   robot command at every timestep  ( = state shifted by 1)
point_cloud   (T, 4096, 3) float32  object surface point cloud at every timestep
                                    (constant across T — object is static during approach)

attrs:
  dataset            'dexycb'                    source dataset name
  subject            '20200709-subject-01'       DexYCB capture session
  session            '20200709_142517'           DexYCB capture timestamp
  camera             '840412060917'              master-cam serial (3rd-person POV)
  obj_id             'ycb_dex_03'                ycb_dex index (CAD mesh tag)
  ycb_class_id       3                            YCB official class (3=sugar, 4=tomato)
  mano_side          'right'                     which hand was used in capture
  n_steps            T                            == state.shape[0]
  ee_offset_m        0.10                        pinch_midpoint → panda_hand origin (m)
  gripper_span_m     0.08                        Franka 2-finger max opening (m)
  obj_origin_G       (3,)                        object centroid xy + bottom z, G-frame
  obj_quat_G_wxyz    (4,)                        object orientation, G-frame, wxyz
  origin_G_W         (3,)                        the G-frame's origin in WORLD coords
                                                 (= AprilTag table-plane origin)
  table_z_G          float                       table surface z in G-frame
  grasp_onset        T                            (subsampled-frame index where gripper closes;
  grasp_onset_idx    T                            equal here because trajectory ends at grasp.
                                                 In v2 datasets this is < T to mark "close NOW")
```

### state / action 8-D breakdown

```
[ 0 ]   x   ┐  panda_hand POSITION in G-frame (meters)
[ 1 ]   y   │
[ 2 ]   z   ┘
[ 3 ]   qw  ┐  panda_hand ORIENTATION as wxyz quaternion
[ 4 ]   qx  │  in RETARGET convention (apply Rz(-90°) to get Franka panda_hand convention)
[ 5 ]   qy  │
[ 6 ]   qz  ┘
[ 7 ]  grip    gripper state in [0, 1]   (0 = open, 1 = closed)
```

The `action` channel is the **next-step state** (i.e. `action[t] == state[t+1]`).
The final-frame gripper flips to 1.0 to mark "arrived → close now".

---

## Coordinate frame: G-frame

Both `state.position` and `point_cloud` are in the **G-frame** — a
gravity-aligned, object-centered frame derived from each capture session's
AprilTag-calibrated table plane:

* **+Z** = up (against gravity); read once per session from the table's
  apriltag normal (per-subject correction; subjects 01-02 have ~13° tilt
  away from world-Z).
* **xy origin** = the object's xy-centroid at frame 0 of the capture.
* **z origin** = the object's bottom plane (1st-percentile z of CAD surface
  points placed at the frame-0 object pose).

To go from G-frame to a sim WORLD frame:

```python
sim_origin_W = np.array([SIM_OBJECT_XY[0] - obj_origin_G[0],
                         SIM_OBJECT_XY[1] - obj_origin_G[1],
                         TABLE_TOP_Z])
panda_hand_pos_W = state[t, :3]          + sim_origin_W   # G → world translation
panda_hand_pos_G = panda_hand_pos_W      - sim_origin_W   # inverse
```

Orientation is unchanged across the G↔world translation (G is gravity-aligned
and sim's gravity is exactly -Z_world).

### Retarget → Franka panda_hand convention

The quaternion stored above is in **retarget convention** (pose of an
abstract gripper frame). To map to Franka's `panda_hand` link, apply a
post-multiplication by `Rz(-90°)`:

```python
from scipy.spatial.transform import Rotation as R
def retarget_to_franka(q_wxyz):
    r = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]) \
        * R.from_euler('z', -90, degrees=True)
    o = r.as_quat()                                # xyzw
    return [o[3], o[0], o[1], o[2]]                # back to wxyz
```

---

## How the data was generated (retarget pipeline)

```
DexYCB capture (RGBD + AprilTag + MANO ground-truth)
    │
    ▼
Baseline1/build_gt_replay.py
    1. read MANO right-hand pose for each frame
    2. compute pinch midpoint = (thumb_tip + index_tip) / 2
    3. compute approach axis from MANO grip
    4. panda_hand pose = (pinch_midpoint − ee_offset_m × approach_axis,
                          frame from approach + thumb-up)
    5. transform to G-frame (gravity from AprilTag, origin from object centroid)
    6. sample 4096 CAD-mesh surface points at the object's frame-0 pose
    │
    ▼
Baseline1/data/episodes_g/dexycb__SUBJECT__SESSION__CAM__OBJID.hdf5
    (= the `trajectory.hdf5` in this folder, byte-identical copy)
```

The grasp success was verified by:

```
sim/run_dp3_twophase.sh
    ├── Phase 1: gt_replay_ikpd_v2.py --dp3       (DP3 closed-loop in sim → /tmp/dp3_traj_*.hdf5)
    └── Phase 2: gt_replay_ikpd_v2.py --drive pd --grasp-lift
                                                  (replay + Franka PD execution + close+lift)
                  → ffmpeg → sim_grasp_success.mp4
```

The sim setup: Franka at world `(0.20, -0.05, 0.80)` yaw `90°`; table top
at world z `0.80`; object spawned with `obj_origin_G + sim_origin_W`
translation and `obj_quat_G_wxyz` orientation; pinned during approach,
released for close+lift.

---

## How to load + use

### 1. Inspect (Python)

```python
import h5py, numpy as np

with h5py.File("partner_trial_data/sugar_box_ycb_dex_03/trajectory.hdf5") as h:
    state  = h["state"][:]            # (T, 8)
    action = h["action"][:]           # (T, 8)
    pc     = h["point_cloud"][:]      # (T, 4096, 3)
    attrs  = dict(h.attrs)

print(f"T={len(state)}  obj_quat_G={attrs['obj_quat_G_wxyz']}")
print(f"first EE pose (G-frame, retarget conv): {state[0, :7]}")
print(f"final  EE pose (= grasp pose):          {state[-1, :7]}")
```

### 2. Train DP3 / any diffusion policy

Concatenate multiple `trajectory.hdf5` into a zarr (see
`Baseline1/convert_to_zarr.py` for the layout):

```
data/point_cloud  (N_total, 4096, 3)  float32
data/state        (N_total, 8)        float32
data/action       (N_total, 8)        float32
meta/episode_ends (n_episodes,)        int64
```

The 2 trajectories here are too few for training a useful policy on their
own — they're meant as **reference samples** to verify your loader matches
the schema.

### 3. Replay in IsaacSim (smoke-test the EE poses are reachable)

```bash
# requires IsaacSim 5.1 + curobo 0.8 (env_isaaclab in our setup)
python sim/gt_replay_ikpd_v2.py \
    --session 20200709_142517 \
    --object ycb_dex_03 \
    --traj partner_trial_data/sugar_box_ycb_dex_03/trajectory.hdf5 \
    --drive pd --grasp-lift --grasp-collision \
    --headless --video /tmp/replay_sugar_frames
ffmpeg -framerate 20 -i /tmp/replay_sugar_frames/f_%05d.png -c:v libx264 -pix_fmt yuv420p /tmp/replay_sugar.mp4
```

---

## Origin in the project repo

* Source episode files:
  `Baseline1/data/episodes_g/dexycb__20200709-subject-01__20200709_142517__840412060917__ycb_dex_03.hdf5`
  `Baseline1/data/episodes_g/dexycb__20201015-subject-09__20201015_143403__840412060917__ycb_dex_04.hdf5`
* Sim videos:
  `replay_video_check/dp3_twophase_sugar.mp4`
  `replay_video_check/dp3_twophase_tomato.mp4`
* Retarget pipeline entry: `Baseline1/build_gt_replay.py`
* Sim eval driver:        `sim/gt_replay_ikpd_v2.py`
* Two-phase orchestrator: `sim/run_dp3_twophase.sh`
