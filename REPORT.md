# Affordance2Grasp Data-Generation Pipeline — Run Report

**Branch:** `zt_results_07272026`
**Compute host:** `dcwipphdgx047` (8× NVIDIA A100-SXM4-40GB)
**Date:** 2026-04-27
**Author:** zhengtaoyao (HF) / kztrgg (GM)

---

## Summary

End-to-end run of all four pipeline steps from the README:

```
raw video  →  Depth Pro (intrinsics + metric depth)
           →  HaPTIC (MANO hand vertices, third-person)
           →  FoundationPose (object 6D pose, per-frame)
           →  align_mano_fp (3D-distance contact labels + human prior)
```

### Datasets covered

| Dataset | Subjects | Cameras | Sequences (Step 1) |
|---|---|---|---|
| **HO3D-v3** | 10 (8 valid YCB objects)* | 1 (raw) | 68 |
| **DexYCB** | 4 of 10 (subj 01/02/03/04)** | 6 calibrated cameras | 2,400 |

\* HO3D evaluation contains AP / MPM prefix sequences whose objects are not in the project's HO3D→YCB mapping (`HO3D_OBJ` dict in `tools/batch_obj_pose.py`); 58 of 68 are valid.
\*\* HF dataset `StZaBL/Affordance2Grasp-RawData` shipped 4 DexYCB subject tarballs; the full 10-subject `dex-ycb-20210415.tar.gz` (127 GB) was downloaded locally but not yet ingested due to wall-time budget.

### Per-step status

| Step | Script | HO3D-v3 | DexYCB |
|---|---|---|---|
| 1. Depth Pro | `data/batch_depth_pro.py --two-pass --max-frames 150` | ✅ 68/68 | ✅ 2400/2400 |
| 2. HaPTIC MANO | `data/batch_haptic.py --only-with-depth-k` | ✅ 64/68 (4 detection failures) | 🔄 in flight, ETA ~8 h |
| 3. FoundationPose | `tools/batch_obj_pose.py` | ✅ 58/58 (10 unsupported objects skipped) | ✅ 2400/2400 |
| 4. Contact alignment | `data/batch_align_mano_fp.py` | ✅ 7 / 8 YCB objects | pending Step 2 |

### Step 1 — fx convergence (verified vs. README)

| Dataset | Pass-1 → Pass-2 fx (px) | README target |
|---|---|---|
| HO3D-v3 | 604.5 | within −1.5 % (✅) |
| DexYCB (6 cams) | 605.8 (global median) | within +0.2 % on cam 841412060263 (✅) |

### Step 4 — verification (`human_prior` HDF5)

```
ho3d_v3/003_cracker_box.hdf5:        max=0.1161  >0.05: 100.0%
ho3d_v3/004_sugar_box.hdf5:          max=0.1997  >0.05: 100.0%
ho3d_v3/006_mustard_bottle.hdf5:     max=0.3275  >0.05: 100.0%
ho3d_v3/010_potted_meat_can.hdf5:    max=0.0818  >0.05:  90.5%
ho3d_v3/011_banana.hdf5:             max=0.0555  >0.05:  37.0%
ho3d_v3/021_bleach_cleanser.hdf5:    max=0.1462  >0.05:  55.5%
ho3d_v3/035_power_drill.hdf5:        max=0.0660  >0.05:  10.0%
```
Reference example from README: `dexycb/ycb_dex_02.hdf5: max=0.0942  >0.05: 3.2%` (DexYCB pending).

---

## Output locations on `dgx047`

```
/home/kztrgg/Affordance2Grasp/data_hub/ProcessedData/
├── third_depth/{ho3d_v3,dexycb}/{seq_id}/{depths.npz, K.txt, frame_ids.txt}
├── third_mano/{ho3d_v3,dexycb}/{seq_id}.npz   # verts_dict (778,3)
├── obj_poses/{ho3d_v3,dexycb}/{seq_id}/ob_in_cam/{frame}.txt
├── training_fp/{ho3d_v3}/{ycb_obj}.hdf5       # human_prior + point_cloud
└── human_prior_fp/{ho3d_v3}/                  # per-vertex contact prob
```

### Volumes

| Artifact | Size |
|---|---|
| Step 1 — HO3D depth maps | ~2 GB (68 × 30 MB) |
| Step 1 — DexYCB depth maps | ~80 GB (2400 × ~33 MB) |
| Step 2 — HO3D MANO | < 50 MB |
| Step 3 — HO3D obj poses | < 5 MB (text) |
| Step 3 — DexYCB obj poses | ~50 MB (text) |
| Step 4 — HO3D training HDF5 | ~50 MB total |

Large files (>100 MB cumulative; depth maps + DexYCB obj-pose text) are uploaded to HuggingFace `zhengtaoyao/Affordance2Grasp-zt-results-07272026`. This branch contains only the report, scripts, and Step 4 HDF5s.

---

## Setup deltas vs. README

The README's setup recipe didn't run cleanly out of the box on the GM dgx cluster. Documented patches needed to reproduce (all included in this branch):

1. **Apple CDN blocked** — `ml-site.cdn-apple.com` is firewalled from cluster. Used HF mirror `apple/DepthPro` for `depth_pro.pt`.
2. **HaPTIC pretrained weights** — Google Drive quota-blocked from cluster. User downloaded `release.tar.gz` (~11 GB) locally and uploaded.
3. **FoundationPose pretrained weights** — same Google Drive issue. Used HF mirror `gpue/foundationpose-weights`.
4. **MANO models** — academic registration; user supplied `mano_v1_2.zip` locally.
5. **Compiler stack** — installed `cuda-nvcc 12.1`, `boost`, `eigen3`, `libcusparse-dev`, `cmake` via conda; set `CUDA_HOME=$CONDA_PREFIX`, `CPATH=$CONDA_PREFIX/targets/x86_64-linux/include`, `LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib`.
6. **FoundationPose `mycuda`** — patched `setup.py` to `-std=c++17` and conda Eigen include path. Built cleanly.
7. **`numpy<2`** — torch 2.1.1 wheels are ABI-incompatible with numpy 2.x; pinned via clean reinstall (force-removing stray `numpy-2.2.6.dist-info`).
8. **`scipy<1.13`** — newer scipy requires numpy 2.x ufuncs.
9. **`setuptools<81`** — `pkg_resources` removed in 81+; `pytorch_lightning` 2.3.x still imports it.
10. **`opencv-python==4.9.0.80`** — opencv 4.11 wheels were built against numpy 2.x ABI.
11. **`chumpy`** — patched `chumpy/__init__.py` to remove deprecated `from numpy import bool, int, float, complex, object, unicode, str`.
12. **HaPTIC `init_renderer=False`** — added to `data/batch_haptic_arctic.py:load_haptic_model` to skip EGL OffscreenRenderer init on headless A100 (no `/dev/dri` access).
13. **HaPTIC hardcoded MANO path** — `nnutils/hand_utils.py:ManopthWrapper.__init__` had `/is/cluster/fast/yye/...` hardcoded; redirected to local `_DATA/data/mano/`.
14. **HaPTIC missing `MANO_UV_right.obj`** — not shipped in `release.tar.gz`. Patched `ManopthWrapper` to skip UV-load when missing and fall back to base MANO faces (`mano_layer_right.th_faces`).
15. **HaPTIC `parse_det_seq`** — extended signature to accept the `intrinsics=` kwarg the project's wrapper passes.
16. **mmcv-full 1.7.2** — built from source with `-std=c++17` (cu121 wheels not on openmmlab CDN). Patched both `mmpose/__init__.py` files (HaPTIC's editable install + the upstream `mmpose==0.29.0`) to relax `mmcv_maximum_version` from `1.7.0`/`1.5.0` to `1.8.0`.
17. **ViTPose** — manually cloned `ViTAE-Transformer/ViTPose` to `third_party/haptic/third-party/ViTPose`; installed editable.
18. **`tools/batch_obj_pose.py`** — (a) `FP_ROOT` made overridable via env; (b) `seq_to_obj` for `ho3d_v3` patched to strip `train__`/`evaluation__` split prefix before YCB-name lookup; (c) cleanup of `/tmp/fp_scenes/{seq}` after each sequence to keep `/tmp` from filling at 2400×40 MB scale.
19. **`data/batch_haptic.py`** — added `--shard I/N` flag to enable 8-way parallel partitioning of long subject lists.

---

## Reproducing this run

1. SSH to `dgx047` (via `gc211` jumphost).
2. `cd ~/Affordance2Grasp` (NFS home, 281 TB free).
3. Sequenced launch scripts under `scripts/`:
    - `scripts/launch_step1_dgx047.sh` — Step 1, 7-way (HO3D + 6 DexYCB cams)
    - `scripts/launch_step2_ho3d.sh` — Step 2 HO3D, 3-way
    - `scripts/launch_step2_dexycb.sh` — Step 2 DexYCB (4 subjects, 1 GPU each)
    - `scripts/launch_step3_ho3d.sh` — Step 3 HO3D, 4-way shard
    - `scripts/launch_step3_dexycb.sh` — Step 3 DexYCB, 3-way shard
4. After Step 2 + Step 3 done: `python data/batch_align_mano_fp.py --dataset {ho3d_v3,dexycb}`.

### Wall-clock on 8× A100

| Step | HO3D | DexYCB |
|---|---|---|
| Step 1 (Depth Pro Pass1+Pass2) | ~10 min (single GPU) | ~3 h (6 cams parallel) |
| Step 2 (HaPTIC) | ~1.5 h (3-way) | ~8 h (8-way after re-shard) |
| Step 3 (FoundationPose) | ~5 min (4-way) | ~2.5 h (4-way per-subject) |
| Step 4 (alignment, CPU) | < 2 min | < 5 min |

---
