# OakInk frame conventions — verified findings (Phase 0)

This document captures what was empirically verified before the Phase 1
retargeter was built. It is the authoritative source for the math in
`retarget_oakink.py` and `oakink_meshes.py`.

## Where the data lives

```
data_hub/RawData/ThirdPersonRawData/oakink_v1/
├── image/
│   ├── obj/{obj_id}.obj                 ← OFFICIAL CAD meshes, canonical frame
│   └── anno/
│       ├── hand_j/<name>.pkl             ← (21,3) hand joints, CAMERA frame
│       ├── hand_v/<name>.pkl             ← MANO mesh verts, CAMERA frame
│       ├── obj_transf/<name>.pkl         ← (4,4) T_c_o, object in CAMERA frame
│       ├── cam_intr/<name>.pkl           ← (3,3) K
│       └── general_info/<name>.pkl       ← cam_extr + cam_intr + obj_anno + hand_anno
```

The general_info pkl is the key — it contains the WORLD-frame versions of the
poses, eliminating the need to combine cam_extr with the camera-frame anno.

## The 4 keys inside general_info

| key | shape | meaning |
|---|---|---|
| `cam_extr` | (4,4) | T_c_w by name, but EMPIRICALLY = T_w_c — see "naming caveat" below |
| `cam_intr` | (3,3) | camera K |
| `obj_anno` | (4,4) | **T_w_o** — object's SE(3) pose in OakInk WORLD frame |
| `hand_anno` | dict   | `hand_tsl` (3,) world-frame wrist position, `hand_pose` (16,4) quat-encoded MANO joint rotations, `hand_shape` (10,) MANO betas |

## Verification: cross-camera world consistency

For the same physical frame observed by 4 cameras simultaneously, world-frame
quantities MUST be identical across cameras. We verified this on session
`A01001_0001_0000` frame 16:

| cam | obj_transf[:3,3] (cam frame) | obj_anno[:3,3] (world) | hand_tsl (world) |
|---|---|---|---|
| 0 | `[-0.096, +0.072, +0.760]` | **`[-0.022, +0.100, +0.106]`** | **`[+0.180, -0.273, +0.142]`** |
| 1 | `[+0.035, -0.012, +0.938]` | same | same |
| 2 | `[+0.092, +0.069, +0.842]` | same | same |
| 3 | `[-0.119, -0.031, +0.864]` | same | same |

`obj_transf` varies per camera (different views of the same object), but
`obj_anno` and `hand_tsl` are identical across all 4 cameras — confirming both
are in a single shared world frame.

## Naming caveat: `cam_extr` is T_w_c, not T_c_w

The variable is named `cam_extr` (often = T_c_w in convention) but empirically
we verified:

```python
inv(cam_extr) @ obj_transf == obj_anno   # for every frame, every camera
```

This means `cam_extr` maps WORLD points to CAMERA frame when applied as
`cam_pt = cam_extr @ world_pt_hom`. Equivalently it IS the pose of the camera
in world (T_w_c). For cam→world transform use `inv(cam_extr)`.

In `retarget_oakink.py` we use:
```python
T_c2w = np.linalg.inv(gi["cam_extr"])
world_pt = T_c2w @ cam_pt_hom
```
to bring per-frame `hand_j` (camera) into world. For object PC we prefer to
sample the CAD mesh and apply `obj_anno` directly (already world).

## OakInk world frame is gravity-aligned (+Z = up)

Across the 8 sampled sessions/cameras, world-frame quantities at frame 0:
- camera Z: ≈ +0.87 to +0.93 m  (cameras mounted high looking down)
- object Z: ≈ +0.05 to +0.15 m  (object on table, ground ≈ z=0)
- hand Z:   ≈ +0.14 to +0.30 m  (hand above object)

All consistently positive Z for things-above-ground → the world frame's +Z axis
points opposite to gravity. This matches MoCap conventions. **No per-subject
calibration needed**, unlike DexYCB where we had to read AprilTag extrinsics
per session (see Baseline1/docs/gravity_W.md).

## How retarget_oakink.py uses these conventions

1. **Object PC** (per-ep, frame 0 only) — sample 4096 surface points from
   `image/obj/{obj_id}.obj` (CAD canonical frame), then transform by
   `obj_anno` to world frame. NOT subtracted by camera transform.

2. **Per-frame EE pose** — load `hand_j` (camera frame), transform each joint
   by `inv(cam_extr)` to world, feed to `mano_joints_to_ee()` which is frame-
   agnostic. Result is a world-frame `(p_ee, q_wxyz)`.

3. **G-frame ("object-centric")** — defined as: same axes as OakInk world, with
   origin shifted to `obj_origin_G := obj_anno[:3,3]` at frame 0. PC and EE
   positions are stored as `(world_pos - obj_origin_G)`. Quaternions stay in
   world frame (rotation-only, axis-aligned with G).

4. **For the sim collector** — the v4 collector reads `obj_origin_G` and
   `obj_quat_G_wxyz` from hdf5.attrs and places the object at that pose in
   IsaacSim world (which is also +Z=up). State[t] EE waypoints are added to
   `obj_origin_G` to recover world-frame positions in sim.

## CAD-mesh choice (NOT SAM3D)

We deliberately bypass the SAM3D reconstructions at
`data_hub/ProcessedData/obj_meshes/oakink/`. Reasons documented in
`Baseline1/oakink/oakink_meshes.py` docstring:

- The plan is to ship the v4 baseline on CAD; SAM3D switch is a later phase.
- The upstream `get_object_points()` double-scales OakInk SAM3D meshes
  (applies both a per-object `scale.json` AND a rotation+scale `R_align_4x4`),
  producing PCs ~5× too small.

Using the official CAD directly side-steps the entire alignment chain and was
verified on A01001: PC bbox = 6.3×11.6×22.1 cm vs CAD's 6.5×10.2×22.8 cm (the
small differences are from `obj_anno` rotation permuting axes — physically
correct, sim-collector-compatible).
