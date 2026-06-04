# Phase 2 — Perception on Titan, execution on Razor

Phase 2 extends the [Phase 1](../phase1/README.md) pick-and-lift demo: **grasp poses come from the V2AP affordance pipeline** (SAM segmentation → SAM3D mesh → metric scale → affordance + grasp candidates) instead of manual `pose_tuner` joint YAMLs.

**Split of responsibility**

| Machine | Role |
|---------|------|
| **Razor** (lab laptop, Dexmate Vega + Sharpa HA4) | Capture RGB-D + robot/camera calibration → pack **input session** → rsync to Titan → receive **output session** → transform poses to `R_ee` → Phase 1 motion planning + grasp |
| **Titan** (GPU server, UCB_Project + SAM3D + FoundationPose) | SAM2 mask → SAM3D mesh → depth scale → **FoundationPose** + base align → **PDM** (`run_pdm_grasp.py`) → pack **output session** → rsync back |

**Transport:** rsync + **[Titan segment daemon](SERVER_CLIENT_PLAN.md#56-titan-segment-daemon-recommended)** (interactive T2 in browser via SSH tunnel). See **[SERVER_CLIENT_PLAN.md](SERVER_CLIENT_PLAN.md)** for Razor client (`run_server_client_pipeline.py` on [V2AP-demo](https://github.com/jiaka1chen/V2AP-demo)).

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
| Titan **segment daemon** (T2 web + T3–T7) | ✅ | `python -m demo.pipeline.segment_daemon` — [pipeline/README.md](pipeline/README.md) |
| Titan one-shot pipeline (batch T2 only) | ✅ | `python -m demo.pipeline.process_razor_session` (needs `prompt.json` or mask) |
| Razor→Titan **auto demo pipeline** client | ✅ (Razor) | `run_server_client_pipeline.py` on V2AP-demo — see [SERVER_CLIENT_PLAN.md](SERVER_CLIENT_PLAN.md) |

**Primary test session:** `sessions/20260602_192346_chips/` (chips on lab table).

---

## End-to-end workflow

```text
Razor                                              Titan
─────                                              ─────

0. (once) Titan: python -m demo.pipeline.segment_daemon
1. Place object; capture_session.py → sessions/<id>/input/
2. rsync input/ ─────────────────────────────────► demo/sessions/<id>/input/
3. mark input/.upload_complete (Razor script)     4. daemon: T2 SAM2 web (tunnel :7860)
   poll status.json                                   Save mask → Done → T3–T7
5. rsync output/ ◄────────────────────────────────  output/status.json success
6. review_titan_vis.py (T3–T6 PNGs, blocking)
7. run_auto_grasp.py (Open3D pose preview + Enter, blocking, then motion)
```

Details: [Titan segment daemon (recommended)](#titan-segment-daemon-recommended) · [SERVER_CLIENT_PLAN.md](SERVER_CLIENT_PLAN.md).

## Titan segment daemon (recommended)

Interactive T2 does **not** work when Razor only runs `ssh … python -m demo.pipeline` — the SAM2 browser UI runs on Titan and must be opened via **SSH port forward**.

### Titan (keep running)

```bash
cd /home/vision/Project/Affordance2Grasp
conda activate bundlesdf
export FP_ROOT="$PWD/third_party/FoundationPose"

python -m demo.pipeline.segment_daemon
# watches demo/sessions/*/input/.upload_complete
```

### Razor (per capture)

```bash
SESSION=20260602_192346_chips
rsync -avz "${RAZOR_REPO}/demo/phase2/sessions/${SESSION}/input/" \
  "${TITAN}:${UCB_ROOT}/demo/sessions/${SESSION}/input/"

# On Titan (or from Razor if Affordance2Grasp on PATH):
python demo/razor/mark_upload_complete.py --session-dir demo/sessions/${SESSION}
```

### Operator (SAM2 in browser)

```bash
ssh -L 7860:127.0.0.1:7860 vision@<titan-host>
# open http://127.0.0.1:7860
# click FG/BG points → Save mask → Done (closes server; daemon continues T3–T7)
```

**Blocking rule:** daemon waits until **`output/segment/mask.png`** exists and operator clicked **Done** (not merely closing the tab without save).

**Skip web UI** if you rsync `input/segment/prompt.json` (batch SAM2) or pre-upload `output/segment/mask.png`.

**Status polling (Razor):** read `output/status.json` — `state`: `waiting_segment` | `running` | `done` | `failed`; also `output/daemon_state.json`.

### One-shot pipeline (no daemon)

Only when T2 is non-interactive:

```bash
python -m demo.pipeline.process_razor_session --session-dir demo/sessions/<id>
```

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
| **SAM3D scaled** | `sam3d_scaled` | `object_scaled.glb` after T4 (meters). FoundationPose runs in this frame (`T_cam_mesh_fp`). |
| **Base-aligned mesh** | `base_aligned` | `object_base_aligned.glb` after T5 align; **T6 / `candidates.json` geometry**. `T_base_mesh` rotation ≈ identity. |
| **Mesh in camera** | `T_cam_mesh` | Aligned mesh → camera: **`p_cam = T_cam_mesh @ p_mesh`** (post-align; used with `base_aligned`). |
| **Pinch (virtual gripper)** | `pinch` | Midpoint between thumb and index contact; UCB `grasp_point`. |
| **Right EE (execution)** | `R_ee` | Pinocchio frame `R_ee` on Razor URDF — Phase 1 IK target. |

**Grasp rotation convention (T6 PDM export, same as UCB `grasp_pose.py`)** — MUST match Phase 1 `demo/phase1/grasp_geometry.py`:

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
└── segment/                  # required for unattended Titan pipeline (unless mask pre-uploaded)
    └── prompt.json           # SAM point/box prompts for T2 batch (see T2)
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
      "mesh_file_on_titan": "output/mesh/object_scaled.glb",
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

Titan entry point (orchestrator switches conda per step):

```bash
cd "$UCB_ROOT"
export FP_ROOT="$PWD/third_party/FoundationPose"

python -m demo.pipeline.process_razor_session \
  --session-dir demo/sessions/20260601_143022_chips \
  [--skip-sam]            # if output/segment/mask.png already exists
  [--skip-sam3d]          # if object_raw.glb already exists
  [--skip-fp]             # if register/T_cam_mesh.json already exists
  [--device cuda]
```

**Conda environments (orchestrator uses both automatically):**

| Steps | Conda env | Notes |
|-------|-----------|--------|
| T1, T2, T4, T5, T6, T7 | `bundlesdf` | FoundationPose, SAM2 batch, PDM |
| T3 | `sam3d-objects` | SAM3D mesh only |

Manual single-step example:

```bash
conda activate sam3d-objects
python demo/scripts/T3/reconstruct.py --session-dir demo/sessions/<id>

conda activate bundlesdf
export FP_ROOT="$PWD/third_party/FoundationPose"
python demo/scripts/T5/register_foundationpose.py --session-dir demo/sessions/<id>
```

**Pipeline order (fixed):** T1 validate → T2 mask → T3 SAM3D → T4 scale → T5 FoundationPose + base-axis align → T6 PDM grasp → T7 status.  
FoundationPose runs on **`object_scaled.glb`**; T6 uses **`object_base_aligned.glb`**.

### Step T1 — Validate input

- Check `schema_version`, required files, RGB/depth shape match `session.json`.
- Fail early with `output/status.json` `success: false` if invalid.

### Step T2 — Object segmentation (SAM2)

**Input:** `rgb/left_rgb.png`.

**Output:**

- `output/segment/mask.png` — uint8 PNG, 0 = background, 255 = object, same size as RGB.
- `output/segment/prompt_used.json` — copy of prompts actually used (for reproducibility).

**Unattended `demo.pipeline` — mask is required before T3.** `run_pipeline.py` fails T2 unless **one of**:

1. **`input/segment/prompt.json`** — batch SAM2 via `demo/scripts/T2/segment_prompt.py` (recommended for Razor→Titan automation), or  
2. **`output/segment/mask.png` already present** (pre-uploaded or `--skip-sam` after manual/interactive segmentation).

Interactive Gradio (`segment_web.py`) is for debugging only; it is **not** invoked by the orchestrator.

**Notes**

- Document `tool` in `prompt_used.json` (e.g. `sam2`, `sam3`).
- Morphological cleanup optional; save raw SAM output + cleaned version if both exist (`mask_raw.png`, `mask.png`).

### Step T3 — SAM3D mesh reconstruction

**Env:** `sam3d-objects`. **Script:** `demo/scripts/T3/reconstruct.py`.

**Input:** `left_rgb.png` + `output/segment/mask.png`.

**Output:**

- `output/mesh/object_raw.glb` — SAM3D mesh **before metric scale**.
- `output/mesh/sam3d_meta.json` — runtime, vertex/face counts, `mesh_frame_origin`.
- `output/vis/T3_sam3d_mesh_preview.png` — mesh + frame axes (unless `--no-vis`).

### Step T4 — Metric scale (RGB-D depth vs mesh)

**Env:** `bundlesdf`. **Script:** `demo/scripts/T4/scale_from_depth.py`.

Align with UCB `estimate_obj_scale_ego.py` **logic**, but use **real Dexmate depth** from input instead of MegaSAM.

**Input:** `depth.npy`, `mask.png`, `K`, `object_raw.glb`.

**Output:**

- `output/mesh/scale.json`
- `output/mesh/object_scaled.glb` — **uniform scale** applied; units = meters.
- `output/vis/T4_scale_scene_preview.png`

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

### Step T5 — FoundationPose + base-axis align

**Env:** `bundlesdf`. **Script:** `demo/scripts/T5/register_foundationpose.py`.

**Scale fixes metric size; FoundationPose estimates 6D pose** of **`object_scaled.glb`** in the camera frame, then **base-axis alignment** produces the mesh frame used by T6 and Razor.

**UCB reference:** `tools/batch_obj_pose_ego.py` → `prepare_scene_ego()` + `run_fp()` (frame `000000`).

#### Transform conventions (critical)

| Symbol | Meaning | Relation |
|--------|---------|----------|
| `T_cam_mesh_fp` | scaled SAM3D mesh → camera | Raw FP output on `object_scaled.glb` |
| `T_base_mesh_fp` | scaled mesh → base | `T_base_cam @ T_cam_mesh_fp` |
| `T_fix` | align rotation to robot base | In `mesh_frame_align.json`; vertices `v' = R_base @ (v - c) + c` |
| `T_cam_mesh` | **base_aligned** mesh → camera | **`p_cam = T_cam_mesh @ p_mesh`** (T6 / candidates) |
| `T_base_mesh` | **base_aligned** mesh → base | **`T_base_mesh = T_base_cam @ T_cam_mesh`**; **R ≈ I** |

Razor grasp uses **`base_aligned`** frame (`candidates.json` + `object_base_aligned.glb`):

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

**Input:** `object_scaled.glb`, `fp_scene/`, `FP_ROOT` models (see UCB setup).

**Call pattern** (mirror `run_fp()` frame 0 only):

```python
pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=est_iter)
# pose is 4×4 → save as T_cam_mesh
```

- Simplify mesh to ≤5000 faces if needed (`fast_simplification`, same as UCB).
- **`object_scaled.glb` is already scaled** — do **not** re-apply `scale.json` inside FP by default.

**Raw FP pose** → `T_cam_mesh_fp.json`, `T_base_mesh_fp.json`, `ob_in_cam_fp/000000.txt`.

#### T5.3 — Base-axis align + aligned registration

After FP, `demo/scripts/T5/mesh_align.py` rotates vertices so mesh axes align with robot base (`object_base_aligned.glb`). Aligned poses:

```python
T_base_cam = np.array(input["calib/extrinsics.json"]["T_base_cam"])
# T_cam_mesh, T_base_mesh refer to base_aligned mesh (see mesh_frame_align.json)
T_base_mesh = T_base_cam @ T_cam_mesh
```

Sanity (warn in `status.json`): mask IoU on overlay; table height check in base frame.

#### T5.4 — Register output files

| File | Role |
|------|------|
| `register/T_cam_mesh_fp.json` | Raw FP, `sam3d_scaled` / `object_scaled.glb` |
| `register/T_base_mesh_fp.json` | `T_base_cam @ T_cam_mesh_fp` |
| `register/mesh_frame_align.json` | `T_fix`, residuals, `mesh_frame_dst: base_aligned` |
| `register/T_cam_mesh.json` | Aligned mesh → camera (**T6 / Razor**) |
| `register/T_base_mesh.json` | Aligned mesh → base |
| `register/ob_in_cam/000000.txt` | Aligned 4×4 (UCB-compatible) |
| `register/ob_in_cam_fp/000000.txt` | Raw FP 4×4 |
| `register/foundationpose_meta.json` | H/W, K, timing |
| `mesh/object_base_aligned.glb` | Mesh for T6 + collision on Razor |
| `vis/T5_foundationpose_overlay.png` | 2×4 review figure (SAM3D + base-aligned panels) |

**Debug optional:** `output/register/fp_scene/` (`--keep-fp-scene`).

**Do not use ICP** in the default pipeline. If FP fails, set `success: false` in `status.json`.

### Step T6 — PDM grasp candidates

**Env:** `bundlesdf`. **Script:** `demo/scripts/T6/run_pdm_grasp.py`.

**Input:** `object_base_aligned.glb`, `register/T_cam_mesh.json`, `register/T_base_mesh.json`.

**Output:**

- `output/inference/affordance_grasp.hdf5` — PDM / grasp group (HDF5 layout compatible with UCB candidates).
- `output/inference/candidates.json` — **required for Razor** (`mesh_frame: "base_aligned"`).
- `output/inference/pdm_meta.json` — run metadata.
- `output/vis/T6_grasp_vis.png` — mesh PDM overlay + candidates on session RGB.

### Step T7 — Write `output/status.json`

**Script:** `demo/scripts/T7/write_status.py` (also called by orchestrator). Writes last (atomic: `status.json.tmp` → rename).

**`steps` keys (T7 only — no `validate_input`):** `segment`, `sam3d`, `scale`, `foundationpose`, `grasp_pose`.

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
  "errors": [],
  "package": {
    "required_for_grasp": [
      "output/status.json",
      "output/inference/candidates.json",
      "output/register/T_base_mesh.json",
      "output/mesh/object_base_aligned.glb"
    ]
  }
}
```

**`pipeline_version`:** Use **`demo.pipeline.process_razor_session 0.1.0`** when the full orchestrator ran. Running T7 alone sets `demo.scripts.T7.write_status 0.1.0` — Razor automation should treat orchestrator version as authoritative.

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
│   ├── object_raw.glb
│   ├── object_scaled.glb
│   ├── object_base_aligned.glb     # T6 + Razor obstacle
│   ├── scale.json
│   └── sam3d_meta.json
├── register/
│   ├── T_cam_mesh_fp.json          # raw FP (scaled mesh)
│   ├── T_base_mesh_fp.json
│   ├── mesh_frame_align.json
│   ├── T_cam_mesh.json             # base_aligned → camera
│   ├── T_base_mesh.json            # for Razor
│   ├── foundationpose_meta.json
│   ├── ob_in_cam/000000.txt        # aligned pose
│   ├── ob_in_cam_fp/000000.txt     # raw FP pose
│   └── fp_scene/                   # optional (--keep-fp-scene)
├── inference/
│   ├── affordance_grasp.hdf5
│   ├── candidates.json             # required; mesh_frame base_aligned
│   └── pdm_meta.json
└── vis/
    ├── T3_sam3d_mesh_preview.png
    ├── T4_scale_scene_preview.png
    ├── T5_foundationpose_overlay.png
    └── T6_grasp_vis.png
```

---

## `output/inference/candidates.json` schema

Portable grasp list — **geometry in `base_aligned` mesh frame** (same as `object_base_aligned.glb`) plus registration.

```json
{
  "schema_version": "1.1",
  "mesh_frame": "base_aligned",
  "inference_method": "pdm",
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
    "grasp_point_frame": "base_aligned",
    "ucb_tcp_offset_m": 0.105,
    "ucb_tcp_frame": "panda_hand",
    "pre_grasp_offset_m": 0.15,
    "lift_height_m": 0.15
  },
  "mesh_file": "output/mesh/object_base_aligned.glb",
  "mesh_span_m": [0.12, 0.08, 0.05],
  "mesh_aabb_min_m": [-0.06, -0.04, 0.01],
  "mesh_aabb_max_m": [0.06, 0.04, 0.06],
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
| `grasp_point` | (3,) **base_aligned** frame | `T_base_pinch = T_base_mesh @ T_mesh_pinch` |
| `rotation` | 3×3 **base_aligned** frame | columns `[finger_open, y_body, approach]` |
| `mesh_aabb_min_m` / `mesh_aabb_max_m` | AABB of `object_base_aligned.glb` | Razor `object_obstacle.py` (optional; can recompute from GLB) |
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

## `output/inference/affordance_grasp.hdf5` (PDM export)

HDF5 layout follows UCB `grasp_pose.py` candidate groups (written by `run_pdm_grasp.py`):

| Path | Shape | Description |
|------|-------|-------------|
| `grasp/position` | (3,) | Best Franka TCP, base_aligned frame |
| `grasp/grasp_point` | (3,) | Best pinch point, base_aligned frame |
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

### Step R0 — Review Titan output (blocking, after rsync)

After `output/` is on Razor and `status.json` has `success: true`, show **T3→T6** vis PNGs **one at a time**. Each window **blocks** until closed; then show the next.

| # | File |
|---|------|
| 1 | `output/vis/T3_sam3d_mesh_preview.png` |
| 2 | `output/vis/T4_scale_scene_preview.png` |
| 3 | `output/vis/T5_foundationpose_overlay.png` |
| 4 | `output/vis/T6_grasp_vis.png` |

Helper (Affordance2Grasp, callable from `run_server_client_pipeline.py`):

```bash
python demo/razor/review_titan_vis.py --session-dir demo/phase2/sessions/<session_id>
```

Use `--skip` only for headless runs.

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

### Step R3b — Grasp pose preview (blocking, before motion)

**Default** `run_auto_grasp.py` (V2AP-demo) — **yes, still pops up the selected pose**:

1. **Open3D preview** (`visualize_grasp.py`) of the IK-selected candidate — window **blocks until closed**.
2. **Enter / confirm** in terminal — **blocks** before OMPL + stall-close + lift.

Skip only with `--no-visualize` (no Open3D) and/or `--debug` (no Enter prompts). Normal lab runs use neither flag.

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
| `--no-visualize` | Skip Open3D grasp preview (non-default; breaks blocking pose review) |
| `--debug` | Verbose logs + **no Enter prompts** (non-default; motion starts right after IK) |

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
| **2.3** | Titan | T6: PDM (`run_pdm_grasp.py`) + `candidates.json` | Titan |
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
| Affordance on wrong side | Mesh vs partial view mismatch | Check `T5_foundationpose_overlay.png`; T6 must use `object_base_aligned.glb` + aligned `T_cam_mesh` |

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
