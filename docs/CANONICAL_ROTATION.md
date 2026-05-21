# Object Canonical Rotation System
## `R_align` — Coordinate Alignment for Sim-Ready Grasps

> **Handover note for colleagues**  
> Author: Interactive annotation session, May 2026  
> Status: ✅ All 100 OakInk objects annotated

---

## Background & Problem

SAM3D reconstructs 3D meshes from hand-object videos. The mesh orientation is determined by **the camera coordinate frame during recording** — there is no semantic "upright" convention. When these PLY files are converted to USD for Isaac Sim, `omni.kit.asset_converter` automatically applies a **Y-up → Z-up** axis swap (OpenGL/OBJ convention → USD/Omniverse convention).

The result: objects in Sim can be sideways or upside-down, because neither SAM3D nor the converter applies any semantic alignment.

**Impact without correction:**
- Objects placed at `OBJECT_ORIENTATION=[0,0,0]` tumble to random stable poses in physics
- Grasp approach-direction filter (`v[2] <= 0.3`, rejecting below-table approaches) is applied in the mesh frame — meaningless if the object is upside-down
- Grasp candidates generated on an inverted mesh map to **below-table positions** in world frame after physics settling → near-zero success rate for those objects

---

## Solution: `R_align` — Human-Annotated Canonical Rotation

Each object has a **`R_align`** rotation matrix stored in its `_meta.json` file. This rotation transforms the USD mesh into an **upright, Sim-ready pose**, where:

- **Z-axis = object's "up" direction** (matches Isaac Sim world Z = up)
- **XY plane = table surface**
- The object's stable resting orientation is preserved

### How it was created

Used the interactive tool `tools/interactive_align.py`:
1. Load USD mesh in Open3D viewer (initial camera: Z↑, XY horizontal)
2. Rotate with keyboard (`X`/`Y`/`Z` + arrow keys) until object stands upright on XY plane
3. Press `Enter` → saves rotation to `_meta.json`

**100 OakInk objects** were manually annotated in one session.

---

## Data Format: `_meta.json`

Each USD file `output/obj_usd/oakink/{obj_id}.usd` has a companion:  
`output/obj_usd/oakink/{obj_id}_meta.json`

```json
{
  "obj_id": "A01027",
  "R_align_euler":  [180.0, 0.0, 180.0],
  "R_align_matrix": [[...3x3 float matrix...]],
  "z_offset_m":     0.1404,
  "T_ply_to_sim":   [[...4x4 float matrix...]],
  "source":         "interactive_align",
  "note":           "z_offset recomputed after R_align"
}
```

| Field | Description |
|-------|-------------|
| `R_align_euler` | Rotation in degrees [rx, ry, rz], XYZ convention. Human-readable. |
| `R_align_matrix` | 3×3 rotation matrix (precise). **Use this in code.** |
| `z_offset_m` | Distance (meters) to raise object above table so bottom face sits at `Z=TABLE_TOP_Z`. Computed from USD vertices **after** R_align is applied. |
| `T_ply_to_sim` | 4×4 homogeneous transform: PLY coordinates → Sim-upright coordinates. Equals `R_align @ T_YZ` where `T_YZ` is the Y→Z axis swap applied by AssetConverter. **Use this for HumanPrior alignment.** |
| `source` | `"interactive_align"` = human annotated; `"usd_mesh"` = auto only (no R_align) |

---

## Where `R_align` is Applied (4 locations)

### 1. `tools/random_grasp_sampler.py` — Grasp Generation
**When:** After loading USD mesh, before sampling grasp candidates.

```python
# After load_mesh_from_usd():
if 'R_align_matrix' in meta:
    R = np.array(meta['R_align_matrix'])
    mesh.vertices = (R @ mesh.vertices.T).T
```

**Effect:** Grasp candidates are generated on the correctly-oriented mesh.  
The `v[2] <= 0.3` approach-direction filter now correctly rejects below-table approaches.

---

### 2. `sim/run_grasp_sim.py` — Isaac Sim Placement
**When:** When placing the object in the Sim scene.

```python
# In setup_scene():
if 'R_align_euler' in meta:
    obj_orientation = meta['R_align_euler']   # [rx, ry, rz] degrees
# Applied as:
obj = RigidObject(..., ori=np.array(obj_orientation), ...)
```

**Effect:** Object spawns in the correct upright orientation. Physics settling makes only minor adjustments. `T_world_obj` from `get_obj_pos()` is close to identity rotation → grasp coordinate transforms are accurate.

---

### 3. `tools/batch_write_meta.py` — z_offset Computation
**When:** When computing the `z_offset_m` for each object.

```python
# Apply R_align before computing min Z:
if R_align is not None:
    vertices = (R_align @ vertices.T).T
z_offset = max(-vertices[:, 2].min(), 0.005)
```

**Effect:** `z_offset` reflects the true bottom of the upright object, not the raw USD orientation.

---

### 4. `tools/batch_vis_sanity.py` — 2-Panel Visualization
**When:** When rendering the mesh point cloud in Panel 2.

```python
# In load_usd_mesh():
if 'R_align_matrix' in meta:
    R = np.array(meta['R_align_matrix'])
    verts = (R @ verts.T).T
```

**Effect:** The visualization mesh matches exactly what grasp candidates were generated on — no visual mismatch.

---

## For HumanPrior (HP) Alignment

OakInk provides hand-object contact annotations in the **PLY/SAM3D coordinate frame**.  
Grasp candidates are in the **Sim-upright frame** (USD + R_align).

To compare HP contacts with grasp positions, transform HP contacts using `T_ply_to_sim`:

```python
import json, numpy as np

meta = json.load(open(f'output/obj_usd/oakink/{obj_id}_meta.json'))
T = np.array(meta['T_ply_to_sim'])   # 4x4

# hp_contacts: (N, 3) in PLY frame
hp_contacts_h = np.hstack([hp_contacts, np.ones((len(hp_contacts), 1))])
hp_contacts_sim = (T @ hp_contacts_h.T).T[:, :3]

# Now hp_contacts_sim is in the same frame as grasp candidates
```

`T_ply_to_sim = R_align @ T_YZ`  
where `T_YZ` is the Y-up→Z-up rotation automatically applied by `omni.kit.asset_converter` (equivalent to rotating -90° around X-axis).

---

## Quick Reference: How to Run

```bash
# Re-annotate / correct individual objects
python3 tools/interactive_align.py --obj A01027

# Continue annotation from a checkpoint
python3 tools/interactive_align.py --start-from A02011 --skip-done

# Recompute z_offset for all objects (preserves R_align)
python3 tools/batch_write_meta.py --ds oakink --force

# Regenerate grasp candidates (uses R_align automatically)
python3 tools/random_grasp_sampler.py --all --dataset oakink --force

# Generate 2-panel visualizations
python3 tools/batch_vis_sanity.py --only-with-grasp --out output/vis_batch_aligned

# Run Sim for one object
sim45 sim/run_grasp_sim.py --hdf5 output/grasps_candidate/A01026_grasp.hdf5
```

---

## Annotation Summary (OakInk, 100 objects)

| Category | Objects | Notes |
|----------|---------|-------|
| No rotation needed (identity) | ~60% | Already upright after Y→Z conversion |
| Flipped 180° (inverted) | ~30% | `R_align_euler ≈ [180, 0, 180]` |
| Partial correction (<45°) | ~10% | Fine-tuning after axis swap |

All rotations stored in `output/obj_usd/oakink/*_meta.json`.
