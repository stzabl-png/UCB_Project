# Phase 2 — Perception on Titan, execution on Razor

Phase 2 extends the [Phase 1](../phase1/README.md) pick-and-lift demo: **grasp poses come from the V2AP affordance pipeline** (SAM segmentation → SAM3D mesh → metric scale → affordance + grasp candidates) instead of manual `pose_tuner` joint YAMLs.

**Split of responsibility**

| Machine | Role |
|---------|------|
| **Razor** (lab laptop, Dexmate Vega + Sharpa HA4) | Capture RGB-D + robot/camera calibration → pack **input session** → rsync to Titan → receive **output session** → transform poses to `R_ee` → Phase 1 motion planning + grasp |
| **Titan** (GPU server, UCB_Project + SAM3D + FoundationPose) | Human SAM2/3 mask → SAM3D mesh → depth-based scale → **FoundationPose** 6D pose → `inference/grasp_pose.py` → pack **output session** → rsync back |

**Transport (v0): manual `rsync`** — no automated server yet. See **[SERVER_CLIENT_PLAN.md](SERVER_CLIENT_PLAN.md)** for the Titan↔Razor automation spec (SSH keys, job flow, Titan-side scripts).

**Upstream method code:** [UCB_Project `titan`](https://github.com/stzabl-png/UCB_Project/tree/titan) — especially `inference/grasp_pose.py`, `tools/batch_obj_pose_ego.py` / `run_fp()` (FoundationPose), scale patterns in `data/estimate_obj_scale_ego.py`.

**Execution code (Razor):** this repo — reuse `demo/phase1/executor.py`, `grasp_geometry.py`, `hand_close.py`, `right_hand_profile.yaml`.

---

## Implementation status (Razor, 2026-06)

| Component | Status | Entry point |
|-----------|--------|-------------|
| RGB-D capture + `input/` pack | ✅ | `capture_session.py` |
| Capture robot pose (start → j3 spread → return) | ✅ | `capture_pose.py` |
| Input validation | ✅ | `validate_input.py` / `pack_session.py` |
| Load Titan `output/` | ✅ | `session_io.py`, `retarget.py` |
| **Open-grip retarget IK** (thumb/index DP + thumb MC, hand **open**) | ✅ | `open_grip_retarget_geometry.py`, `pinch_ik.py` |
| Candidate filter (pre_grasp + grasp IK, random try order) | ✅ | `run_auto_grasp.py` |
| Auto grasp + OMPL + stall-close | ✅ | `run_auto_grasp.py` → Phase 1 `executor.py` |
| Pre-grasp object collision box (Titan mesh AABB) | ✅ | `object_obstacle.py` |
| EE retarget calib YAML | ✅ | `calib/ee_retarget.yaml`, `calibrate_ee_retarget.py` |
| Titan pipeline orchestrator (T1–T7) | ✅ | `python -m demo.pipeline` — [pipeline/README.md](pipeline/README.md) |
| Razor→Titan automation client | ✅ (Razor) | `run_s2r_pipeline.py` on V2AP-demo |

**Primary test session:** `sessions/20260602_192346_chips/` (chips on lab table).

---

## End-to-end workflow

```text
Razor                                              Titan
─────                                              ─────

1. Place object on table
2. python demo/phase2/capture_session.py --object-name chips
   → start pose → arm_j3 spread → RGB-D → arms back to start
   → writes sessions/<id>/input/
3. Human rsync input/ ──────────────────────────►  demo/sessions/<id>/input/
                                                   4. python -m demo.pipeline.process_razor_session …
                                                      (SAM → SAM3D → scale → FoundationPose → grasp_pose)
                                                   5. writes …/<id>/output/status.json + candidates.json
6. Human rsync output/ ◄────────────────────────  …/<id>/output/
7. python demo/phase2/run_auto_grasp.py --session-id <id>
   (open-grip IK filter → retarget → OMPL → stall-close → lift)
```

**Future (v1):** Razor orchestrator replaces steps 3–6 via SSH + rsync — see [SERVER_CLIENT_PLAN.md](SERVER_CLIENT_PLAN.md).

---

## Session ID and directory layout

### Session ID format

```text
<YYYYMMDD>_<HHMMSS>_<object_slug>
```

Example: `20260601_143022_chips`

- `object_slug`: lowercase `[a-z0-9_]+`, no spaces (e.g. `chips`, `cracker_box`).
- One session = **one capture** (one RGB-D frame set + robot state at capture time).

### Mirror paths (Razor ↔ Titan)

| Side | Root |
|------|------|
| **Razor** (V2AP-demo repo) | `demo/phase2/sessions/<session_id>/` |
| **Titan** (UCB_Project) | `$UCB_ROOT/demo/sessions/<session_id>/` |

Both sides use the same subtree:

```text
<session_id>/
├── input/          # Razor writes → Titan reads
└── output/         # Titan writes → Razor reads
```

Create Titan root once:

```bash
mkdir -p "$UCB_ROOT/demo/sessions"
```

---

## Rsync commands (manual)

Replace placeholders:

| Variable | Example |
|----------|---------|
| `RAZOR_HOST` | laptop hostname or IP |
| `RAZOR_REPO` | `/path/to/V2AP-demo` |
| `TITAN_HOST` | Titan SSH host |
| `UCB_ROOT` | `/path/to/UCB_Project` on Titan |
| `SESSION` | `20260601_143022_chips` |

**Razor → Titan (after capture)**

```bash
rsync -avz --progress \
  "${RAZOR_REPO}/demo/phase2/sessions/${SESSION}/input/" \
  "${TITAN_HOST}:${UCB_ROOT}/demo/sessions/${SESSION}/input/"
```

**Titan → Razor (after processing)**

```bash
rsync -avz --progress \
  "${TITAN_HOST}:${UCB_ROOT}/demo/sessions/${SESSION}/output/" \
  "${RAZOR_REPO}/demo/phase2/sessions/${SESSION}/output/"
```

Optional: rsync the whole session folder both ways for debugging.

---

## Coordinate frames and conventions

All 4×4 matrices are **homogeneous transforms** `T_dst_src`: column-vector convention `p_dst = T_dst_src @ p_src`.

| Frame | Name in files | Description |
|-------|---------------|-------------|
| **Camera (optical)** | `zed_left_camera` | ZED left RGB / depth optical frame. +X right, +Y down, +Z forward (standard computer vision). Depth `depth[v,u]` is range in meters along +Z. |
| **Robot base** | `base` | Dexmate Vega floating base / root used by Pink IK and Phase 1 (`R_ee` targets). |
| **Object / mesh** | `mesh` | SAM3D mesh canonical frame. Metric scale on Titan **before** FoundationPose. Same frame as `object_scaled.obj` and grasp outputs. |
| **Mesh in camera (FP)** | `T_cam_mesh` | FoundationPose `ob_in_cam`: **`p_cam = T_cam_mesh @ p_mesh`**. |
| **Pinch (virtual gripper)** | `pinch` | Midpoint between thumb and index contact; UCB `grasp_point`. |
| **Right EE (execution)** | `R_ee` | Pinocchio frame `R_ee` on Razor URDF — Phase 1 IK target. |

**Grasp rotation convention (UCB `inference/grasp_pose.py`)** — MUST match Phase 1 `demo/phase1/grasp_geometry.py`:

- `rotation` is 3×3 proper rotation matrix **columns** = `[finger_open, y_body, approach]`.
- **`approach`** = column 2 (0-based index 2) = third column = gripper advance direction (into object).
- **Pre-grasp** on Razor: translate **−0.15 m** along `approach` from grasp TCP (see Phase 1).
- **Lift**: **+0.15 m** world +Z from grasp pose.

**UCB Franka-specific fields (Titan output, Razor must NOT use blindly)**

- `position` in HDF5 = `panda_hand` TCP = `grasp_point - approach * 0.105` (Franka `TCP_OFFSET = 0.105` m).
- Razor retarget uses **`grasp_point` + `rotation`**, not `position`, then applies `T_pinch_to_ee` (see [Razor retarget](#razor-side-retarget-and-execution)).

---

## INPUT package (`input/`) — Razor → Titan

Razor **`capture_session.py`** produces **all** of the following before rsync.

### File tree

```text
input/
├── session.json              # required — metadata + schema version
├── rgb/
│   └── left_rgb.png          # required — uint8 RGB, H×W×3
├── depth/
│   ├── depth.npy             # required — float32 (H,W), meters; NaN/0 = invalid
│   └── depth_colormap.png    # optional preview for humans (Titan uses depth.npy)
├── calib/
│   ├── intrinsics.json       # required — camera K
│   ├── K.npy                 # required — 3×3 float64 duplicate of K (UCB / FoundationPose)
│   ├── extrinsics.json       # required — T_base_cam at capture
│   └── robot_state.json      # required — joint positions at capture
├── scene/
│   └── table.json            # required — table height for collision / FP sanity check
└── segment/                  # optional — SAM prompts on Razor (mask still produced on Titan)
    └── prompt.json           # optional — SAM point/box prompts (see below)
```

### `input/session.json`

```json
{
  "schema_version": "1.1",
  "session_id": "20260601_143022_chips",
  "object_slug": "chips",
  "created_at_iso": "2026-06-01T14:30:22-07:00",
  "capture": {
    "rgb_file": "rgb/left_rgb.png",
    "depth_file": "depth/depth.npy",
    "depth_unit": "meters",
    "depth_invalid_values": [0.0, null],
    "camera_frame": "zed_left_camera",
    "rgb_width": 640,
    "rgb_height": 360,
    "depth_width": 640,
    "depth_height": 360,
    "depth_aligned_to_rgb": true
  },
  "robot": {
    "model": "vega_1",
    "base_frame": "base",
    "ee_frame": "R_ee",
    "state_file": "calib/robot_state.json",
    "extrinsics_file": "calib/extrinsics.json"
  },
  "scene": {
    "table_file": "scene/table.json"
  },
  "pipeline": {
    "registration_method": "foundationpose",
    "foundationpose": {
      "fp_scene_layout": "ycbineoat_reader",
      "frame_index": "000000",
      "depth_storage_input": "depth/depth.npy float32 meters",
      "depth_storage_fp_scene": "uint16 PNG millimeters",
      "depth_mm_scale": 1000.0,
      "K_files": ["calib/intrinsics.json", "calib/K.npy"],
      "mask_required": true,
      "mask_source": "output/segment/mask.png on Titan",
      "mesh_file_on_titan": "output/mesh/object_scaled.obj",
      "ucb_reference": "tools/batch_obj_pose_ego.py run_fp()"
    }
  },
  "notes": ""
}
```

**Rules**

- If RGB and depth resolutions differ, Razor must **resize depth to RGB** (nearest-neighbor) before save, and set `depth_aligned_to_rgb: true`.
- `schema_version` must be **`"1.1"`** for FoundationPose pipeline. Titan rejects unknown major versions.
- Legacy `1.0` sessions (ICP-era) may still validate on Razor but must be re-captured for FP metadata.

### `input/rgb/left_rgb.png`

- PNG, 8-bit RGB (not BGR).
- Same resolution as used for SAM / SAM3D on Titan.

### `input/depth/depth.npy`

- `numpy.save`, `dtype=float32`, shape `(H, W)`.
- Values: metric distance in meters (ZED SDK / dexcontrol convention).
- Invalid pixels: `0`, `NaN`, or `Inf` — Titan must mask these out before backprojection.

### `input/calib/intrinsics.json`

```json
{
  "camera_frame": "zed_left_camera",
  "width": 640,
  "height": 360,
  "K": [
    [fx, 0.0, cx],
    [0.0, fy, cy],
    [0.0, 0.0, 1.0]
  ],
  "distortion_model": "none",
  "dist_coeffs": []
}
```

- Source: ZED factory calib via `robot.sensors.head_camera.get_camera_info()` on Razor, or offline chessboard calibration.
- Titan uses **K** from `intrinsics.json` and/or `K.npy` when building the FoundationPose scene dir (`cam_K.txt`).

### `input/calib/K.npy`

- `numpy.save`, shape `(3, 3)`, `float64` — **must match** `intrinsics.json` → `K` exactly.
- Duplicate for UCB / FoundationPose scripts that expect `K.npy` (see `batch_obj_pose_ego.py`).

### `input/calib/extrinsics.json`

```json
{
  "base_frame": "base",
  "camera_frame": "zed_left_camera",
  "T_base_cam": [
    [r00, r01, r02, tx],
    [r10, r11, r12, ty],
    [r20, r21, r22, tz],
    [0.0, 0.0, 0.0, 1.0]
  ],
  "method": "urdf_fk",
  "notes": "Computed at capture from head/torso joints + URDF zed_left_camera mount"
}
```

- **`T_base_cam`**: transforms a point from **camera frame** to **base frame**: `p_base = T_base_cam @ p_cam`.
- Razor computes at capture time: FK(base ← head chain) × fixed URDF mount (`zed_left_camera` on `head_l3`).
- Titan uses this with FoundationPose output: **`T_base_mesh = T_base_cam @ T_cam_mesh`** (see [Step T5](#step-t5--mesh--scene-registration-foundationpose)).

**Validation on Razor (before rsync):** backproject table ROI; median `z` in base should be ≈ `scene/table.json` `table_height_m` ± 5 cm.

### `input/calib/robot_state.json`

Joint positions **at the same instant** as the RGB-D frame (radians unless noted).

```json
{
  "timestamp_iso": "2026-06-01T14:30:22-07:00",
  "joints": {
    "torso": [j1, j2, j3],
    "head": [j1, j2, j3],
    "left_arm": [j1, j2, j3, j4, j5, j6, j7],
    "right_arm": [j1, j2, j3, j4, j5, j6, j7],
    "left_hand": [],
    "right_hand": []
  },
  "head_pitch_down_deg": 20.0
}
```

- Hand joints optional for capture (can be empty arrays); Phase 1 demo uses fixed head pitch ≈ 20° on `head_j1`.
- Titan: informational + optional FK cross-check; primary extrinsic is `T_base_cam` in `extrinsics.json`.

### `input/scene/table.json`

```json
{
  "table_height_m": 0.98,
  "table_frame_note": "World Z up; table top is plane z = table_height_m in base frame",
  "collision_box": {
    "size_xyz_m": [2.0, 4.0, 0.08],
    "center_xyz_m": [1.1, 0.0, 0.845]
  }
}
```

- Defaults match Phase 1 (`demo/phase1/constants.py`: `DEFAULT_TABLE_HEIGHT_M = 0.85`).
- Titan uses `table_height_m` for **sanity checks** after FoundationPose (mesh bottom vs table), not as the primary registration method.

### `input/segment/prompt.json` (optional)

If the operator already clicked on Razor, pass prompts so Titan SAM can skip re-clicking:

```json
{
  "tool": "sam2",
  "prompts": [
    {"type": "point", "xy": [320, 180], "label": 1}
  ]
}
```

- `xy` in **pixel coordinates** on `left_rgb.png`, origin top-left, `(x, y)`.
- If absent, Titan **must** run an interactive SAM2/3 step (human in the loop on Titan).

---

## TITAN processing pipeline

Titan implements a single entry point (suggested):

```bash
cd "$UCB_ROOT"
conda activate bundlesdf   # FoundationPose + PointNet++ env on Titan

python -m demo.pipeline.process_razor_session \
  --session-dir demo/sessions/20260601_143022_chips \
  [--skip-sam]            # if output/segment/mask.png already exists
  [--skip-sam3d]          # if mesh already exists
  [--skip-fp]             # if register/T_cam_mesh.json already exists
  [--device cuda]
```

**Pipeline order (fixed):** T2 SAM mask → T3 SAM3D → T4 scale → **T5 FoundationPose** → T6 grasp_pose.  
FoundationPose requires **mask + metric mesh + RGB-D + K**; it runs **after** scale on `object_scaled.obj`.

### Step T1 — Validate input

- Check `schema_version`, required files, RGB/depth shape match `session.json`.
- Fail early with `output/status.json` `success: false` if invalid.

### Step T2 — Object segmentation (human + SAM2 or SAM3)

**Input:** `rgb/left_rgb.png`, optional `segment/prompt.json`.

**Output:**

- `output/segment/mask.png` — uint8 PNG, 0 = background, 255 = object, same size as RGB.
- `output/segment/prompt_used.json` — copy of prompts actually used (for reproducibility).

**Notes**

- SAM3 on Titan (if used for 2D mask) is fine; document `tool` in `prompt_used.json`.
- Morphological cleanup optional; save raw SAM output + cleaned version if both exist (`mask_raw.png`, `mask.png`).

### Step T3 — SAM3D mesh reconstruction

**Input:** `left_rgb.png` + `mask.png`.

**Output:**

- `output/mesh/object_raw.obj` — SAM3D mesh **before metric scale** (arbitrary units / orientation).
- `output/mesh/sam3d_meta.json` — runtime, commit/hash if available, vertex/face counts.

SAM3D runs on Titan only (not in UCB GitHub). Wrapper: `demo/scripts/T3/reconstruct.py`.

### Step T4 — Metric scale (RGB-D depth vs mesh)

Align with UCB `estimate_obj_scale_ego.py` **logic**, but use **real Dexmate depth** from input instead of MegaSAM.

**Input:** `depth.npy`, `mask.png`, `K`, `object_raw.obj` (coarse pose optional).

**Output:**

- `output/mesh/scale.json`
- `output/mesh/object_scaled.obj` — **uniform scale** applied; units = meters.

**`output/mesh/scale.json` schema:**

```json
{
  "scale_factor": 1.23,
  "method": "depth_median_ratio",
  "Z_real_m": 0.85,
  "Z_mesh_pre_scale": 0.69,
  "notes": "Median depth over mask ∩ valid depth pixels in camera frame"
}
```

Suggested algorithm (v0):

1. Backproject masked valid depth → camera point cloud; take median `Z` → `Z_real_m`.
2. Render or raycast mesh (pre-scale) into mask; median vertex `Z` in front-facing region → `Z_mesh_pre_scale`.
3. `scale_factor = Z_real_m / Z_mesh_pre_scale`; multiply all vertex coordinates.
4. Sanity: clamp `scale_factor` to `[0.3, 3.0]` (UCB guard band); warn in `status.json` if clamped.

### Step T5 — Mesh ↔ scene registration (FoundationPose)

**Scale fixes metric size; FoundationPose estimates 6D pose** (rotation + translation) of the **scaled mesh** in the camera frame. This replaces ICP as the primary registration method.

**UCB reference:** `tools/batch_obj_pose_ego.py` → `prepare_scene_ego()` + `run_fp()` (single-frame `register` on frame `000000` is enough for Razor capture).

#### Transform conventions (critical — all interfaces must match)

| Symbol | Meaning | Relation |
|--------|---------|----------|
| `T_cam_mesh` | mesh → camera | **`p_cam = T_cam_mesh @ p_mesh`** (UCB `ob_in_cam`) |
| `T_base_cam` | camera → base | `p_base = T_base_cam @ p_cam` (from Razor `extrinsics.json`) |
| `T_base_mesh` | mesh → base | **`T_base_mesh = T_base_cam @ T_cam_mesh`** |

Grasp geometry from `grasp_pose.py` is in **mesh frame** (same vertices as `object_scaled.obj`). Razor executes:

```text
p_base = T_base_mesh @ p_mesh_grasp
```

#### T5.1 — Build FoundationPose scene directory

Under `output/register/fp_scene/` (temp; may delete after success):

```text
fp_scene/
├── rgb/000000.png       # BGR uint8, same H×W as capture (or FP-resized; document in meta)
├── depth/000000.png     # uint16 depth in **millimeters** (depth_m × 1000)
├── masks/000000.png     # uint8 0/255 object mask (from output/segment/mask.png)
└── cam_K.txt            # 3×3 K (same as input; if resized, scale fx,fy,cx,cy accordingly)
```

Conversion from Razor `input/`:

```python
depth_mm = (depth_m * 1000.0).clip(0, 65535).astype(np.uint16)
# cv2.imwrite(..., depth_mm)  # FP YcbineoatReader expects PNG mm
K = np.load("input/calib/K.npy")  # or intrinsics.json["K"]
```

If Titan downscales for FP (UCB uses `SHORTER_SIDE=480`), **scale K** exactly as in `prepare_scene_ego()` and record final `(H_fp, W_fp)` in `foundationpose_meta.json`.

#### T5.2 — Run FoundationPose register (single frame)

**Input:** `object_scaled.obj`, `fp_scene/`, `FP_ROOT` models (see UCB setup).

**Call pattern** (mirror `run_fp()` frame 0 only):

```python
pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=est_iter)
# pose is 4×4 → save as T_cam_mesh
```

- Simplify mesh to ≤5000 faces if needed (`fast_simplification`, same as UCB).
- **`object_scaled.obj` is already scaled** — do **not** re-apply `scale.json` inside FP unless you intentionally keep scale only in JSON.

**Output pose** → `T_cam_mesh` (identical to UCB `ob_in_cam/000000.txt`).

#### T5.3 — Compose base frame + sanity check

```python
T_base_cam = np.array(input["calib/extrinsics.json"]["T_base_cam"])
T_cam_mesh = pose   # from FP
T_base_mesh = T_base_cam @ T_cam_mesh
```

Sanity (warn in `status.json`, do not fail unless gross error):

- Project mesh bbox with `T_cam_mesh` onto RGB; overlap mask IoU should be reasonable.
- Transform mesh bottom / center to base; Z should be near `table_height_m` ± 5 cm.

#### T5.4 — Register output files

**`output/register/T_cam_mesh.json`** (FoundationPose primary):

```json
{
  "camera_frame": "zed_left_camera",
  "mesh_frame": "mesh",
  "T_cam_mesh": [[...4x4...]],
  "method": "foundationpose",
  "fp_frame": "000000",
  "est_iter": 5,
  "mesh_file": "mesh/object_scaled.obj"
}
```

**`output/register/T_base_mesh.json`** (derived, used by Razor):

```json
{
  "base_frame": "base",
  "mesh_frame": "mesh",
  "T_base_mesh": [[...4x4...]],
  "method": "foundationpose",
  "T_base_cam_source": "input/calib/extrinsics.json",
  "composition": "T_base_mesh = T_base_cam @ T_cam_mesh"
}
```

**Also write (UCB-compatible):**

- `output/register/ob_in_cam/000000.txt` — same 4×4 as `T_cam_mesh`, `.txt` format like UCB
- `output/register/ob_in_cam/000000.npy` — optional `float64` (4,4) duplicate
- `output/register/foundationpose_meta.json` — H/W, K used, mesh face count, timing, FP version
- `output/vis/foundationpose_overlay.png` — copy FP `track_vis/000000.png` (posed bbox on RGB)

**Debug optional:** `output/register/fp_scene/` (keep with `--keep-fp-scene`).

**Do not use ICP** in the default pipeline. If FP fails, set `success: false` and report in `status.json`; optional manual retry with adjusted mask/mesh is a human ops step, not a silent fallback.

### Step T6 — Affordance + grasp candidates

Call existing UCB code on **`object_scaled.obj`**:

```bash
python -m inference.grasp_pose \
  --mesh output/mesh/object_scaled.obj \
  --output output/inference/affordance_grasp.hdf5 \
  --device cuda
```

**Output files:**

- `output/inference/affordance_grasp.hdf5` — native UCB format (see below).
- `output/inference/candidates.json` — **portable summary for Razor** (required even if HDF5 present).
- `output/inference/affordance_vis.png` — optional contact heatmap on mesh or RGB projection.

### Step T7 — Write `output/status.json`

Always write status last (atomic: write to `status.json.tmp` then rename).

```json
{
  "schema_version": "1.1",
  "session_id": "20260601_143022_chips",
  "success": true,
  "pipeline_version": "demo.pipeline.process_razor_session 0.1.0",
  "finished_at_iso": "2026-06-01T14:45:00-07:00",
  "steps": {
    "segment": "ok",
    "sam3d": "ok",
    "scale": "ok",
    "foundationpose": "ok",
    "grasp_pose": "ok"
  },
  "warnings": [],
  "errors": []
}
```

On failure: `success: false`, `errors: ["..."]`, partial outputs allowed but Razor must not execute grasp.

---

## OUTPUT package (`output/`) — Titan → Razor

### Required file tree (success path)

```text
output/
├── status.json
├── segment/
│   ├── mask.png
│   └── prompt_used.json
├── mesh/
│   ├── object_raw.obj
│   ├── object_scaled.obj
│   ├── scale.json
│   └── sam3d_meta.json
├── register/
│   ├── T_cam_mesh.json             # FoundationPose ob_in_cam (primary)
│   ├── T_base_mesh.json            # T_base_cam @ T_cam_mesh (for Razor)
│   ├── foundationpose_meta.json
│   ├── ob_in_cam/
│   │   ├── 000000.txt              # UCB-compatible 4×4
│   │   └── 000000.npy              # optional duplicate
│   └── fp_scene/                   # optional debug (--keep-fp-scene)
├── inference/
│   ├── affordance_grasp.hdf5
│   ├── candidates.json             # required for Razor
│   └── affordance_vis.png          # optional
└── vis/
    └── foundationpose_overlay.png  # FP track_vis frame 0
```

---

## `output/inference/candidates.json` schema

Portable grasp list — **all geometry in mesh frame** plus global registration.

```json
{
  "schema_version": "1.1",
  "mesh_frame": "mesh",
  "base_frame": "base",
  "camera_frame": "zed_left_camera",
  "registration": {
    "method": "foundationpose",
    "T_cam_mesh": [[...4x4...]],
    "T_base_mesh": [[...4x4...]],
    "T_base_cam_source": "input/calib/extrinsics.json"
  },
  "T_base_mesh": [[...4x4...]],
  "conventions": {
    "rotation_columns": ["finger_open", "y_body", "approach"],
    "approach_column_index": 2,
    "grasp_point_frame": "mesh",
    "ucb_tcp_offset_m": 0.105,
    "ucb_tcp_frame": "panda_hand",
    "pre_grasp_offset_m": 0.15,
    "lift_height_m": 0.15
  },
  "mesh_span_m": [0.12, 0.08, 0.05],
  "n_candidates": 3,
  "candidates": [
    {
      "rank": 0,
      "name": "dynamic_front_y0",
      "score": 78.5,
      "grasp_point": [0.01, 0.02, 0.03],
      "rotation": [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
      ],
      "position_panda_hand": [0.01, 0.02, -0.075],
      "gripper_width_m": 0.045,
      "cross_section_width_m": 0.04,
      "approach_type": "horizontal"
    }
  ],
  "affordance": {
    "num_points": 4096,
    "contact_threshold": 0.5,
    "n_contact_points": 812,
    "force_center": [0.0, 0.0, 0.02]
  }
}
```

**Field notes**

| Field | Titan | Razor |
|-------|-------|-------|
| `grasp_point` | (3,) mesh frame, virtual two-finger center | Transform with `T_base_mesh` |
| `rotation` | 3×3 mesh frame | `R_base = R_base_mesh @ R_mesh` |
| `position_panda_hand` | UCB Franka TCP | **Do not use for IK**; use `grasp_point` + retarget |
| `gripper_width_m` | informational | Optional; Phase 2 v0 uses fixed hand profile + stall close |
| `rank` | 0 = best by UCB score | Try IK in rank order |

**`T_base_mesh` duplication:** must match `register/T_base_mesh.json`.  
**`registration.T_cam_mesh`:** must match `register/T_cam_mesh.json`.  
Razor uses **`T_base_mesh` only** for execution; `T_cam_mesh` is for debugging / recomposition checks:

```text
T_base_mesh  ≟  T_base_cam @ T_cam_mesh
```

---

## `output/inference/affordance_grasp.hdf5` (UCB native)

Same layout as `inference/grasp_pose.py` output:

| Path | Shape | Description |
|------|-------|-------------|
| `grasp/position` | (3,) | Best Franka TCP, mesh frame |
| `grasp/grasp_point` | (3,) | Best pinch point, mesh frame |
| `grasp/rotation` | (3,3) | Best rotation |
| `grasp/quaternion_wxyz` | (4,) | |
| `candidates/candidate_i/...` | | All candidates |
| `affordance/points` | (N,3) | Sampled surface points |
| `affordance/contact_prob` | (N,) | Predicted affordance |

Razor **may** read HDF5 or prefer `candidates.json` only (no h5py dependency on laptop).

---

## Razor-side retarget and execution

Implemented in V2AP-demo (`demo/phase2/run_auto_grasp.py`, not on Titan).  
Titan output contract for Razor: **[TITAN_OUTPUT.md](TITAN_OUTPUT.md)**.

### Step R1 — Load session output

- Require `output/status.json` with `success: true`.
- Load `inference/candidates.json`, registration (`T_base_mesh`), Phase 1 `start.yaml` for arm/home joints.

### Step R2 — Mesh frame → base frame (`T_base_pinch`)

For each candidate:

```python
T_base_mesh = np.array(candidates_json["T_base_mesh"])  # 4×4
R_mesh = np.array(c["rotation"])                         # 3×3, columns = [finger_open, y_body, approach]
t_mesh = np.array(c["grasp_point"])                      # (3,) virtual pinch origin in mesh frame

T_mesh_pinch = np.eye(4)
T_mesh_pinch[:3, :3] = R_mesh
T_mesh_pinch[:3, 3] = t_mesh

T_base_pinch = T_base_mesh @ T_mesh_pinch   # stored on GraspObjectConfig.titan_T_base_pinch
```

Pre-grasp / lift offsets use **Titan `approach`** (column 2), not `R_ee` Z.

### Step R3 — Open-grip arm IK (replaces closed `T_ee_pinch` bridge)

Arm IK uses **open-hand** FK on thumb/index (`right_thumb_DP`, `right_index_DP`, `right_thumb_MC`) with **right arm only** Gauss–Newton IK (`pinch_ik.py`):

| Constraint | Role |
|------------|------|
| **Translation** | Tip-line point at **2/3** along thumb→index (from index: 1/3 segment inward) equals `p_titan + open_pinch_forward_offset_m × approach` |
| **Parallel** | `(index_tip − thumb_tip) ∥ Titan finger_open` (column 0) |
| **Plane (soft)** | Plane (thumb tip, thumb MC, index tip) contains `approach` |

Defaults (`calib/ee_retarget.yaml`):

- `open_pinch_forward_offset_m: 0.015` (1.5 cm along approach; open hand → stall-close closes the gap)
- Override: `--open-pinch-forward-offset 0.02`

After IK, `grasp_pose` (`R_ee`) is synced from FK for logging/OMPL seeds. Legacy `T_ee_pinch_closed` in yaml is kept for debug/calibration tools only.

### Step R4 — Candidate selection + Phase 1 executor

**Candidate filter (no OMPL):** for each Titan rank (random try order by default):

1. **pre_grasp IK** — Titan pinch retreated along `−approach` by `pre_grasp_offset_m` (0.15 m); seed = `start.yaml` arms  
2. **grasp IK** — full `T_base_pinch`; seed = pre_grasp solution  

Both must converge (`mid_err ≤ 5 mm`). First passing candidate wins.

**Execution:**

```bash
source setup.sh
python demo/phase2/run_auto_grasp.py --session-id 20260602_192346_chips
python demo/phase2/run_auto_grasp.py --session-id 20260602_192346_chips --debug
python demo/phase2/run_auto_grasp.py --session-id 20260602_192346_chips --rank 3   # force rank
python demo/phase2/run_auto_grasp.py --session-id 20260602_192346_chips --no-random-candidate
```

| Flag | Purpose |
|------|---------|
| `--object-obstacle` | Titan mesh AABB in pre-grasp OMPL only |
| `--max-candidates N` | IK try pool size (default 10) |
| `--random-candidate` / `--seed` | Shuffle try order each run (default on) |
| `--no-visualize` | Skip Open3D grasp preview |
| `--debug` | Verbose logs + no Enter prompts |

Sequence: home/start → **pre_grasp** (OMPL + optional object box) → **grasp approach** (OMPL, no box) → **stall-close** (closed profile) → **lift**.

Logs: Titan pinch in base, open-grip IK errors, live thumb/index vs Titan after close.

**Scoring:** UCB rank is a hint only — Razor re-ranks by IK feasibility + random shuffle.

---

## Suggested Titan repo layout (`Affordance2Grasp/demo/`)

Phase 2 code lives under **`demo/`** — **full automation spec:** [SERVER_CLIENT_PLAN.md](SERVER_CLIENT_PLAN.md).

```text
demo/
├── pipeline/                      # python -m demo.pipeline (T1–T7 orchestrator)
├── scripts/T1 … T7/               # per-step scripts
├── sessions/                      # rsync target (gitignored)
│   └── <session_id>/
│       ├── input/                 # from Razor
│       └── output/                # for Razor
├── README.md                      # session schema (this file)
├── TITAN_OUTPUT.md                # Razor consumer guide
└── SERVER_CLIENT_PLAN.md          # SSH/rsync automation
```

---

## Razor repo layout (`V2AP-demo/demo/phase2/`)

```text
demo/phase2/
├── README.md                      # this file
├── SERVER_CLIENT_PLAN.md          # Titan↔Razor automation (give to Titan team)
├── TITAN_OUTPUT.md                # Titan output package (Razor consumer doc)
├── capture_session.py             # RGB-D capture → input/
├── capture_pose.py                # start → j3 spread → return (robot motion)
├── run_auto_grasp.py              # Titan output → open-grip IK → executor
├── open_grip_retarget_geometry.py # thumb/index FK constraints
├── pinch_ik.py                    # open-grip Gauss–Newton IK
├── retarget.py                    # candidates.json → TitanGraspPoses
├── hand_retarget_geometry.py      # legacy closed pinch / live fingertip FK
├── ee_retarget_io.py
├── calibrate_ee_retarget.py
├── object_obstacle.py             # mesh AABB for pre-grasp OMPL
├── session_output.py              # grasp_config_from_titan()
├── pack_session.py / validate_input.py / session_io.py
├── robot_capture.py / extrinsics.py / intrinsics.py
├── visualize_grasp.py
├── constants.py
├── calib/
│   ├── ee_retarget.yaml
│   └── head_zed_left_intrinsics.json
└── sessions/                      # gitignored
    └── <session_id>/
        ├── input/
        └── output/
```

---

## Razor capture (`capture_session.py`)

**Robot motion** (`capture_pose.py`) — default sequence:

| Step | Action |
|------|--------|
| 1 | Torso / head / **both arms** → `start.yaml` (half nominal arm speed, `max_vel=0.25` rad/s) |
| 2 | **Both hands** → start hand pose (left = `start.yaml`, right = open grip profile) — **once only** |
| 3 | **arm_j3 ± 90°** (left −, right +) for camera clearance; hands **not** commanded |
| 4 | Grab RGB-D + pack `input/` |
| 5 | Arms/torso/head **back to start**; hands **unchanged** |

Head: `head_j1` **−20° pitch** (`DEMO_HEAD_JOINT_POS`). Prerequisite: `pose_tuner.py --object-name start`.

```bash
source setup.sh   # or setup_local.sh on dev laptop
python demo/phase2/capture_session.py --object-name chips
python demo/phase2/capture_session.py --object-name chips --preview
python demo/phase2/capture_session.py --object-name chips --skip-move-to-start
python demo/phase2/capture_session.py --validate-only --session-id 20260601_143022_chips
python demo/phase2/capture_session.py --object-name chips --dry-run
```

| Flag | Purpose |
|------|---------|
| `--capture-arm-j3-deg 90` | Outward j3 spread (default 90°) |
| `--arm-wait-s 10` | Wait per arm move at half speed |
| `--slow-capture-move` | Stepped arm motion (Ctrl+C abort) |
| `--skip-return-to-start` | Do not move arms back after capture |
| `--camera-source zed_stream` | TCP from dexmate-nano (default) |
| `--camera-source zenoh` | dexsensor head_camera |

**Head camera (default: `zed_stream` on dexmate-nano `.22:30000`):**

On **dexmate-nano** (`ssh dexmate-nano`), start the streamer **with depth** (omit `--no-depth`):

```bash
cd zed_stream/
sudo ./build/zed_streamer --clean --jpeg-quality 100 --max-fps 30 \
  --resolution HD1080 --no-right --no-pc --no-imu
```

Then on Razor (`pip install lz4` if needed):

```bash
# Live RGB + depth preview (before capture)
python camera/view_zed_stream_rgbd.py

python demo/phase2/capture_session.py --object-name chips
```

After capture, inspect saved session:

```bash
# RGB + depth preview PNGs are written under input/rgb/ and input/depth/
python demo/phase2/capture_session.py --validate-only --session-id 20260601_143022_chips
```

RGB + depth arrive over TCP port **30000** (ZS01 protocol). Intrinsics fall back to nominal ZED values unless you pass `--fx/--fy` or `--intrinsics-json`.

If `dexsensor` is running instead, pass `--camera-source zenoh`.

**What gets written:** full `input/` tree; `robot_state.json` includes **left_hand / right_hand** (22-DOF) and `capture_pose_source`.

**Post-capture checks:** `validate_input.py` runs automatically. Table-height warnings → check `T_base_cam` or `--table-height`.

**Preview images (written at capture time):**

| Path | Purpose |
|------|---------|
| `input/rgb/left_rgb.png` | RGB (Titan pipeline) |
| `input/depth/depth.npy` | Metric depth (Titan pipeline) |
| `input/depth/depth_colormap.png` | Human-readable depth (TURBO colormap) |

Optional: `--preview` on `capture_session.py` shows RGB/depth in OpenCV **before** writing files.

Add to `.gitignore`:

```gitignore
demo/phase2/sessions/
```

---

## Milestones

| ID | Owner | Deliverable | Status |
|----|-------|-------------|--------|
| **2.0** | Razor | `capture_session.py` + valid `input/` | ✅ |
| **2.1** | Titan | T2–T4: SAM + SAM3D + scale | Titan |
| **2.2** | Titan | T5: FoundationPose | Titan |
| **2.3** | Titan | T6: `grasp_pose` + `candidates.json` | Titan |
| **2.4** | Razor | Open-grip retarget + `run_auto_grasp.py` | ✅ |
| **2.5** | Both | End-to-end rsync loop | 🔄 manual rsync works; automation in [SERVER_CLIENT_PLAN.md](SERVER_CLIENT_PLAN.md) |

---

## Troubleshooting

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| Point cloud floating above table | Wrong `T_base_cam` | Re-FK on Razor; check head joints; hand-eye calib |
| Mesh too large/small | Bad scale | Check mask; median depth; clamp warnings in `scale.json` |
| FP bbox misaligned on RGB | Bad mask, wrong K, or unscaled mesh | Fix SAM mask; verify `K.npy` matches image size; confirm scale applied before FP |
| FP pose 180° flipped | Symmetric object / bad mask | Re-segment; try `--est-iter`; pick reachable IK candidate on Razor |
| `T_base_mesh` table Z wrong but FP OK in camera | Bad `T_base_cam` | Fix Razor extrinsics; verify `T_base_mesh ≈ T_base_cam @ T_cam_mesh` |
| All IK fail on Razor | Grasp behind robot / wrong frame | Visualize `T_base_pinch`; check `T_ee_pinch` calib |
| Affordance on wrong side | Mesh vs partial view mismatch | Check FP overlay; verify same `object_scaled.obj` used for FP and grasp_pose |

---

## References

- Phase 1 execution: [demo/phase1/README.md](../phase1/README.md)
- Titan output (Razor): [TITAN_OUTPUT.md](TITAN_OUTPUT.md)
- Titan↔Razor automation: [SERVER_CLIENT_PLAN.md](SERVER_CLIENT_PLAN.md)
- UCB inference: `inference/grasp_pose.py`, `inference/predictor.py`
- UCB FoundationPose: `tools/batch_obj_pose_ego.py`, `FP_ROOT`
- UCB scale: `data/estimate_obj_scale_ego.py`

---

## Changelog

| Version | Date | Notes |
|---------|------|-------|
| 1.2 | 2026-06-03 | Razor: open-grip retarget IK, capture j3 spread pose, `SERVER_CLIENT_PLAN.md`; README sync |
| 1.1 | 2026-06-01 | **FoundationPose** registration (replaces ICP); schema 1.1 |
| 1.0 | 2026-06-01 | Initial Phase 2 spec: rsync sessions |
