# T3 — SAM3D mesh (unscaled)

Reconstructs **`output/mesh/object_raw.glb`** from session RGB + T2 mask.  
Does **not** apply metric scale (that is **T4**).

## Prerequisites

- T2: `output/segment/mask.png` (0/255, same size as `input/rgb/left_rgb.png`)
- [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) with `checkpoints/hf/pipeline.yaml`
- Conda env **`sam3d-objects`** (separate from `bundlesdf`)

Default repo search order:

1. `$SAM3D_ROOT`
2. `../sam-3d-objects` (sibling of Affordance2Grasp)
3. `third_party/sam-3d-objects`

## Usage

```bash
conda activate sam3d-objects
cd /home/vision/Project/Affordance2Grasp

python demo/scripts/T3/reconstruct.py \
  --session-dir demo/sessions/20260602_192346_chips

# optional
python demo/scripts/T3/reconstruct.py --session-dir ... --validate --redo --seed 42
python demo/scripts/T3/reconstruct.py --session-dir ... --vis-only
export SAM3D_ROOT=/path/to/sam-3d-objects
```

## Outputs

```text
output/mesh/
├── object_raw.glb      # SAM3D mesh (native GLB on disk)
└── sam3d_meta.json     # verts/faces, frame origin, timing

output/vis/
└── T3_sam3d_mesh_preview.png   # triangle mesh + mesh-frame XYZ axes
```

## SAM3D raw output

Pipeline `run()` returns:

| Key | Format | Saved in T3? |
|-----|--------|----------------|
| **`glb`** | `trimesh.Trimesh` | **Yes** → `object_raw.glb` |
| **`gs`** | Gaussian splat | No (optional later) |

T3 writes the **`glb` mesh as-is** via `mesh.export(..., file_type="glb")`. Downstream T4/T5 load GLB with `trimesh.load()`.

## Visualization

**Not a point cloud** — preview uses **shaded triangle faces** (`Poly3DCollection`), subsampled only when face count > 24k (for speed).

`output/vis/T3_sam3d_mesh_preview.png` (1×3):

1. RGB + mask  
2–3. **Triangle mesh + object frame** (RGB arrows = X/Y/Z), two views

Frame origin: world `(0,0,0)` if inside mesh AABB, else vertex centroid (recorded in `sam3d_meta.json` as `mesh_frame_origin`).

Mesh is **not** overlaid on RGB (no `T_cam_mesh` until T5).

Disable preview: `--no-vis`.

## Notes

- RGB loaded with **PIL** (`RGB` order).
- UCB +X rotation is **not** applied here (T4/T5 decide alignment).

Spec: [demo/README.md](../../README.md) (Step T3).
