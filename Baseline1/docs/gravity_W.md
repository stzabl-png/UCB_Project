# G-frame gravity (`gravity_W` / `R_W→G`) — how it is determined

> Technical note for Baseline1 data generation. Records *why* gravity is per-subject and
> *how* `build_gt_replay.py` derives it, so the method is not lost. Written 2026-05-20.

## 1. Why this matters

The training-data frame is the **G-frame** (gravity-aligned, object-translated). Its rotation
`R_W→G` is built entirely from one input: the **gravity direction in the W frame**
(`W` = master camera 840412060917's optical frame). If that gravity vector is wrong, the whole
G-frame is tilted — every EE trajectory and every object point cloud is silently rotated.

## 2. Gravity is **per-subject**, not a global constant

DexYCB re-calibrated its 8-camera rig **once per subject** → 10 distinct `extrinsics_*`
calibrations. The master camera's tilt relative to gravity is therefore *not guaranteed* equal
across subjects. Measured (see §3):

| subject | extrinsics id      | gravity_W (W frame)              | Δ vs subj-09 |
|---------|--------------------|----------------------------------|--------------|
| 01      | 20200702_151821    | (+0.0587, +0.6412, +0.7651)      | **13.52°**   |
| 02      | 20200813_100608    | (+0.0598, +0.6419, +0.7645)      | **13.49°**   |
| 03      | 20200820_091149    | (−0.0044, +0.7940, +0.6079)      | 0.77°        |
| 04      | 20200903_072753    | (+0.0058, +0.7950, +0.6066)      | 1.32°        |
| 05      | 20200907_105926    | (+0.0096, +0.7976, +0.6031)      | 1.54°        |
| 06      | 20200918_092020    | (+0.0023, +0.8015, +0.5979)      | 1.22°        |
| 07      | 20200928_082347    | (+0.0072, +0.8013, +0.5983)      | 1.47°        |
| 08      | 20201001_200551    | (+0.0066, +0.8018, +0.5976)      | 1.46°        |
| 09      | 20201014_215638    | (−0.0172, +0.7962, +0.6048)      | 0.00°        |
| 10      | 20201022_091549    | (−0.0138, +0.7966, +0.6044)      | 0.20°        |

**Subjects 03–10 agree within ~1.5°; subjects 01 & 02 are ~13.5° off** (the rig was
repositioned between subject-02 and subject-03). → A single hardcoded gravity is **wrong by
13.5°** for subjects 01–02. Gravity must be read per session.

## 3. Source of truth: the AprilTag in `extrinsics.yml`

Each `calibration/extrinsics_<id>/extrinsics.yml` contains, alongside the 8 camera matrices,
an extra **`apriltag`** entry — a 3×4 row-major `[R|t]` = `T_W←AprilTag`.

The AprilTag is physically fixed **flat on the table**, so its local **+Z axis is the table
surface normal = the anti-gravity (up) direction**. Hence:

```python
R_tag     = np.array(ext['apriltag']).reshape(3, 4)[:, :3]   # AprilTag → W rotation
gravity_W = -R_tag[:, 2]                                     # tag +Z = up  →  gravity = -up
gravity_W = gravity_W / np.linalg.norm(gravity_W)
```

Notes on provenance:
- DexYCB ships **no physical tag in the RGB frames** and **no gravity/table-plane file** — only
  this calibrated tag pose inside each `extrinsics.yml`.
- An earlier attempt to recover gravity from resting-object poses (fit a table plane to objects'
  lowest CAD points) was **abandoned**: ~60 mm plane residual, because DexYCB's clutter-object
  pose annotations have a few degrees of rotational error that is amplified by object height.
  The AprilTag entry is exact and is the method of record.

## 4. How `build_gt_replay.py` uses it

`GRAVITY_W` is **no longer hardcoded**. Flow, per session:

1. `meta.yml` of the session gives its `extrinsics` id.
2. `load_extrinsics(extr_id)` reads `extrinsics.yml`, returns `(T_W←C per cam, master, gravity_W)`
   — `gravity_W` computed from the `apriltag` entry as in §3.
3. `build_R_W_G(gravity_W)` builds the per-session rotation.

## 5. `R_W→G` construction — and the "horizontal plane"

```python
def build_R_W_G(gravity_W):
    g    = gravity_W / ‖gravity_W‖
    up_G = -g                                       # +Z_G = anti-gravity
    fwd_W = (0, 0, 1)                               # master cam +Z (optical axis) in W
    fwd_h = fwd_W - (fwd_W · up_G) · up_G           # ← project onto the horizontal plane
    y_G   = fwd_h / ‖fwd_h‖                         # +Y_G
    x_G   = normalize(y_G × up_G)                   # +X_G
    y_G   = normalize(up_G × x_G)                   # re-orthogonalise
    R_W→G = [x_G | y_G | up_G]ᵀ
```

The **"horizontal plane" is not stored or hardcoded anywhere** — it is, by definition, the plane
perpendicular to gravity. It appears only as the projection `fwd_h = fwd_W − (fwd_W·up_G)·up_G`,
where `up_G = −gravity_W`. Because `build_R_W_G` now receives the **per-session** `gravity_W`,
the horizontal plane (and therefore `+Y_G`, `+X_G`, and all of `R_W→G`) is automatically
re-derived per session from the correct gravity. No stale constant remains.

## 6. Calibration files

All 10 `extrinsics_*` sets are installed under
`data_hub/RawData/ThirdPersonRawData/dexycb/calibration/`.

Source of the archive (`calibration.tar.gz`, ~15 KB, contains all 10 extrinsics + intrinsics
+ MANO calib):
- **HF dataset** `BenXu123456/dexycb_complement` → `calibration.tar.gz` (team mirror).
- Official: Google Drive `https://drive.google.com/file/d/1UAwVKT4Rgb1fLcFoa1o71_-0NtSvvLAQ`.
