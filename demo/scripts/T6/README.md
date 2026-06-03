# T6 — PDM grasp + `candidates.json`

Runs **affordance v6 + PDM** on the T5 **base-aligned** mesh (same 3D pose as FoundationPose).

## Prerequisites

- T5: `output/mesh/object_base_aligned.glb`, `register/T_cam_mesh.json`, `register/T_base_mesh.json`
- Checkpoints (repo defaults):
  - `output/affordance_no_rot_executed/min20/checkpoints_v6/best_v6_model.pth`
  - `output/pdm/checkpoints_yaw_v6cond/best_model.pth`

## Hard filter (default on): Dexmate right-side approach sector

In **base-aligned / robot base** XY (+Z up), keep grasps whose **gripper arrival direction** (**−approach**, where the hand comes from) lies in **225°–315°**:

- Lower bound: 45° from **-X** toward **-Y**
- Upper bound: 45° from **+X** toward **-Y**

90° wedge centered on **-Y** → **Dexmate right arm** side (if base **+Y = robot left**). PDM’s into-object **approach** is roughly the opposite (+Y wedge); filtering **−approach** matches “从 −X/−Y 到 +X/−Y 夹角之间”.

**Top-down grasps** (approach ≈ **−base Z**, gripper above): when `||arrival_xy||` is small, the XY sector is skipped and poses with **arrival +Z ≥ 0.85** are kept (`vertical_top_down`). The old `too_vertical` reject only applied to ambiguous tilts, not true top-down.

PDM draws `n_samples × 3` poses by default, then filters and keeps up to `n_samples`. Disable: `--no-dexmate-approach-sector`. Stats: `pdm_meta.json` → `filters.approach_sector`.

## Mesh handling (session mode)

Unlike `tools/glb_to_pdm_grasp.py` defaults, T6 **does not** apply +90° X pre-rotate, auto-scale, or centering — vertices stay in the T5 base-aligned frame so `T_base_mesh` remains valid.

## Usage

```bash
cd /path/to/Affordance2Grasp

python demo/scripts/T6/run_pdm_grasp.py \
  --session-dir demo/sessions/20260602_192346_chips

# Faster debug
python demo/scripts/T6/run_pdm_grasp.py \
  --session-dir demo/sessions/20260602_192346_chips \
  --n-samples 20 --redo
```

## Outputs

```text
output/inference/
├── affordance_grasp.hdf5   # PDM candidates (method=pdm)
├── candidates.json         # Razor portable summary + registration
├── pdm_meta.json           # ckpts, timing, mesh_prepare flags
└── affordance/
    └── npz/<session_id>.npz

output/vis/
└── T6_grasp_vis.png   # left: mesh PDM overlay | right: candidates on session RGB
```

## Razor

Load `candidates.json` + `T_base_mesh`; grasp `rotation` / `grasp_point` are in **base_aligned** mesh frame. See [demo/README.md](../../README.md) § candidates schema.
