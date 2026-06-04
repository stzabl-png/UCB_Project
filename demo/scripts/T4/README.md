# T4 — Metric scale (depth + mask + K)

Scales T3 mesh to **metres**. **Does not use FoundationPose** (T5).

## General applicability (v3)

Designed for **Phase 2 rigid tabletop objects** with:

- Metric `depth.npy` + `K`
- SAM2 mask (largest connected component used)
- SAM3D mesh `object_raw.glb`

**Works across shapes** (cans, boxes, bottles, tools, packages) because:

| Mechanism | Why general |
|-----------|-------------|
| Largest mask CC | Removes stray SAM/table pixels on any session |
| Depth 3D cues (`pca_max`, `core`) | Rotation-aware size in camera frame |
| Mask lateral (optional) | Face-on extent; only used if consistent with 3D |
| Trimmed median fusion | Rejects one outlier cue, not tuned to one SKU |
| Mesh `pca_max` span | Same statistic family as depth cloud (UCB-style) |

**Not intended for:** transparent objects, heavy occlusion, deformable objects, mask missing most of depth.

**T5 FP** still required for 6D pose; T4 only fixes **metric uniform scale**.

## Outputs

```text
output/mesh/object_scaled.glb
output/mesh/scale.json          # fusion_cues_used, d_mesh_pca_max, warnings
output/vis/T4_scale_scene_preview.png
```

## Usage

```bash
python demo/scripts/T4/scale_from_depth.py \
  --session-dir demo/sessions/<session_id>
```

## Algorithm (depth_mask_adaptive_v3)

1. Mask → largest connected component.
2. Back-project masked depth (±15% Z band) → point cloud (m).
3. Cues: `pca_max`, `core`, `lateral` (if within 0.65–1.35× median of 3D cues).
4. `d_real` = trimmed median of cues.
5. `d_mesh` = PCA max span on mesh vertices (fallback AABB).
6. `scale_factor_depth = d_real / d_mesh`, clamp `[0.05, 3.0]`; then `scale_factor = scale_factor_depth × 0.95` (post shrink); uniform on vertices.

Spec: [demo/README.md](../../README.md) (Step T4).
