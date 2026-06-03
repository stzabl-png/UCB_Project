# T5 — FoundationPose registration

Estimates **6D pose** of the metric mesh (`object_scaled.glb`) in the camera frame. Runs **after T4**; does not re-apply `scale.json` by default.

## Prerequisites

- `bundlesdf` conda env (Python 3.10, torch 2.1.1+cu121, pytorch3d, nvdiffrast, **warp-lang**, compiled **mycpp**)
- `export FP_ROOT=/path/to/third_party/FoundationPose` (weights under `weights/*/model_best.pth`)
- T2 `output/segment/mask.png`, T4 `output/mesh/object_scaled.glb`

## Usage

```bash
conda activate bundlesdf
export FP_ROOT=/home/vision/Project/Affordance2Grasp/third_party/FoundationPose

cd /home/vision/Project/Affordance2Grasp
python demo/scripts/T5/register_foundationpose.py \
  --session-dir demo/sessions/20260602_192346_chips \
  [--validate] [--redo] [--keep-fp-scene]
```

## Outputs

```text
output/mesh/
├── object_scaled.glb           # SAM3D + T4 scale (unchanged)
└── object_base_aligned.glb     # base-axis aligned (T6 / affordance / PDM)

output/register/
├── T_cam_mesh_fp.json          # raw FP (SAM3D mesh frame)
├── T_base_mesh_fp.json
├── T_cam_mesh.json             # aligned mesh frame → camera
├── T_base_mesh.json            # R ≈ identity; translation only in base
├── mesh_frame_align.json       # T_fix, checks, residuals
├── foundationpose_meta.json
├── ob_in_cam/000000.txt        # aligned pose
└── ob_in_cam_fp/000000.txt     # raw FP pose

output/vis/T5_foundationpose_overlay.png   # single 2×4 figure (6 panels, see below)

**How to read the overlay (6 panels)**

| Row | Panels | Meaning |
|-----|--------|---------|
| **Top** | ①–④ | **SAM3D / FP mesh frame**: mask, bbox, z-colored mesh, 3D camera + depth |
| **Bottom** | ⑤–⑥ | **Base-aligned** (T6/PDM): RGB + mesh + base axes; 3D base (depth + mesh, robot axes) |

Aligned row reuses the same camera pose as ②–③. Mesh local axes ∥ robot base (`T_base_mesh` R ≈ I).

Alignment: `v' = R_base @ (v - c) + c`, `T_base_mesh' = T_base_mesh_fp @ T_fix`, checks in `mesh_frame_align.json`.

**T6** should use `object_base_aligned.glb` + `register/T_cam_mesh.json` / `T_base_mesh.json`.
```

Rebuild vis without re-running FP:

```bash
python demo/scripts/T5/register_foundationpose.py --session-dir ... --vis-only
```

**Transform:** `p_cam = T_cam_mesh @ p_mesh`; `T_base_mesh = T_base_cam @ T_cam_mesh`.

## Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--shorter-side` | 480 | FP input resolution (UCB default) |
| `--est-iter` | 5 | `register()` iterations |
| `--apply-scale-json` | off | Only if mesh is **not** pre-scaled |
| `--no-vis` | off | Skip overlay PNG |
| `--keep-fp-scene` | off | Keep `register/fp_scene/` after success |

## See also

- [demo/README.md](../../README.md) — T5 spec, milestone 2.2
- `tools/batch_obj_pose_ego.py` — UCB reference (`prepare_scene_ego` + `run_fp`)
