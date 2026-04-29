# Affordance2Grasp Data-Generation Pipeline — Final Run Report

**Branch:** `zt_results_07272026`
**Compute host:** `dcwipphdgx047` (8× NVIDIA A100-SXM4-40GB)
**Window:** 2026-04-27 → 2026-04-29 (~46 h wall-clock)
**Owner:** zhengtaoyao (HF) / kztrgg (GM)

---

## TL;DR

Full DexYCB + HO3D-v3 pipeline ran end-to-end. Final artifacts on the `zt_results_07272026` GitHub branch (report + scripts + patches) and on HF dataset `zhengtaoyao/Affordance2Grasp-zt-results-07272026` (the bulky outputs). README's Step 4 verification on `dexycb/ycb_dex_02.hdf5` returns `max=0.0942` — **exact match** to the README's expected value.

---

## Final per-step inventory

| Step | DexYCB | HO3D-v3 |
|---|---|---|
| **1. Depth Pro** (intrinsics + metric depth) | **6000 / 6000** ✅ | **68 / 68** ✅ |
| **2. HaPTIC** (MANO 778-vert hand) | 5392 / 6000 (89.9 %) | 64 / 68 (94.1 %) |
| **3. FoundationPose** (object 6D pose) | **6000 / 6000** ✅ | **58 / 58 valid** ✅ |
| **4. Contact alignment** (training HDF5) | **20 / 20 YCB objects** ✅ | 7 / 8 YCB objects |

DexYCB scope: all 10 subjects × 6 calibrated cameras × 100 sessions = 6000 sequences. The 608 sequences "missing" from Step 2 hit ViTPose detection failures (no hand bbox) — same failure regime as the original 4-subject run (~10 % per subject).

HO3D-v3 scope: 68 raw sequences. 10 of them (AP / MPM prefixes) have objects not in the project's `HO3D_OBJ` mesh-mapping dict, so Step 3 + Step 4 correctly skip them; this is a data-side ceiling, not a pipeline regression. Step 4 produced 7 of the 8 reachable YCB objects (`052_extra_large_clamp` failed because its only sequence `train__SiS1` had no MANO cache from Step 2).

---

## Step-1 fx convergence (Pass-1 → Pass-2 self-calibration)

| Dataset | Global median fx (px) | README target |
|---|---|---|
| HO3D-v3 | 604.5 | within −1.5 % (✅) |
| DexYCB (6 cams) | 605.8 | within +0.2 % on cam 841412060263 (✅) |

---

## Step-4 verification (README's exact snippet)

```
dexycb/ycb_dex_01.hdf5: max=0.2218  >0.05: 91.5%
dexycb/ycb_dex_02.hdf5: max=0.0942  >0.05: 33.1%   ← max matches README's expected value
dexycb/ycb_dex_03.hdf5: max=0.0188  >0.05:  0.0%
dexycb/ycb_dex_04.hdf5: max=0.0310  >0.05:  0.0%
dexycb/ycb_dex_05.hdf5: max=0.0626  >0.05: 56.7%
dexycb/ycb_dex_06.hdf5: max=0.0171  >0.05:  0.0%
dexycb/ycb_dex_07.hdf5: max=0.0159  >0.05:  0.0%
dexycb/ycb_dex_08.hdf5: max=0.0607  >0.05: 61.7%
dexycb/ycb_dex_09.hdf5: max=0.0258  >0.05:  0.0%
dexycb/ycb_dex_10.hdf5: max=0.0221  >0.05:  0.0%
dexycb/ycb_dex_11.hdf5: max=0.0825  >0.05: 43.0%
dexycb/ycb_dex_12.hdf5: max=0.0980  >0.05: 54.2%
dexycb/ycb_dex_13.hdf5: max=0.0294  >0.05:  0.0%
dexycb/ycb_dex_14.hdf5: max=0.0406  >0.05:  0.0%
dexycb/ycb_dex_15.hdf5: max=0.0722  >0.05: 21.5%
dexycb/ycb_dex_16.hdf5: max=0.0576  >0.05: 15.2%
dexycb/ycb_dex_17.hdf5: max=0.0195  >0.05:  0.0%
dexycb/ycb_dex_18.hdf5: max=0.0114  >0.05:  0.0%
dexycb/ycb_dex_19.hdf5: max=0.0298  >0.05:  0.0%
dexycb/ycb_dex_20.hdf5: max=0.0193  >0.05:  0.0%
ho3d_v3/003_cracker_box.hdf5:        max=0.1161  >0.05: 100.0%
ho3d_v3/004_sugar_box.hdf5:          max=0.1997  >0.05: 100.0%
ho3d_v3/006_mustard_bottle.hdf5:     max=0.3275  >0.05: 100.0%
ho3d_v3/010_potted_meat_can.hdf5:    max=0.0818  >0.05:  90.5%
ho3d_v3/011_banana.hdf5:             max=0.0555  >0.05:  37.0%
ho3d_v3/021_bleach_cleanser.hdf5:    max=0.1462  >0.05:  55.5%
ho3d_v3/035_power_drill.hdf5:        max=0.0660  >0.05:  10.0%
```

Step 4 also reported `coverage=100.0%, diverged=0` for all 20 DexYCB objects ⇒ FP tracking success rate **100 %** on DexYCB (matches README's "Verified Configuration" table).

---

## Output locations on `dgx047`

```
/home/kztrgg/Affordance2Grasp/data_hub/ProcessedData/
├── third_depth/{ho3d_v3,dexycb}/{seq_id}/{depths.npz, K.txt, frame_ids.txt}
├── third_mano/{ho3d_v3,dexycb}/{seq_id}.npz       # verts_dict (778, 3) per frame
├── obj_poses/{ho3d_v3,dexycb}/{seq_id}/ob_in_cam/{frame}.txt
├── training_fp/{ho3d_v3,dexycb}/{ycb_obj}.hdf5    # human_prior + point_cloud
└── human_prior_fp/                                 # per-vertex contact prob
```

| Artifact | Volume |
|---|---|
| Step 1 — DexYCB depth maps | ~ 200 GB (6000 × 33 MB) |
| Step 1 — HO3D depth maps | ~ 2 GB (68 × 30 MB) |
| Step 2 — DexYCB MANO | ~ 800 MB (5392 × ~150 KB) |
| Step 2 — HO3D MANO | ~ 30 MB |
| Step 3 — DexYCB obj poses | ~ 60 MB text |
| Step 3 — HO3D obj poses | ~ 5 MB text |
| Step 4 — DexYCB training HDF5 | ~ 1 GB total |
| Step 4 — HO3D training HDF5 | ~ 50 MB total |

The depth maps stay on the dgx047 NFS home (regeneratable from raw RGB). All Step 2/3/4 outputs are mirrored to HF.

---

## Wall-clock on 8× A100

| Step | DexYCB | HO3D-v3 |
|---|---|---|
| Step 1 (Depth Pro Pass 1 + Pass 2) | ~ 7 h (8-way per camera + 1 GPU subj-10 recovery) | ~ 10 min (1 GPU) |
| Step 2 (HaPTIC) | ~ 24 h (8-way sharded across subjects) | ~ 1.5 h (3-way) |
| Step 3 (FoundationPose) | ~ 5 h (4-way per subject) | ~ 5 min (4-way shard) |
| Step 4 (alignment, CPU) | ~ 8 min | < 2 min |

Total wall-clock: **~ 36 h** end-to-end, down from the README's single-GPU "~80 h" estimate.

---

## README compliance — exactly the four commands

```bash
# Step 1
python data/batch_depth_pro.py --dataset dexycb  --cam $CAM --two-pass --max-frames 150  # × 6 cams
python data/batch_depth_pro.py --dataset ho3d_v3 --two-pass --max-frames 150

# Step 2
python data/batch_haptic.py --dataset dexycb  --only-with-depth-k
python data/batch_haptic.py --dataset ho3d_v3 --only-with-depth-k

# Step 3
python tools/batch_obj_pose.py --dataset dexycb
python tools/batch_obj_pose.py --dataset ho3d_v3

# Step 4
python data/batch_align_mano_fp.py --dataset dexycb
python data/batch_align_mano_fp.py --dataset ho3d_v3
```

Plus the README's verification snippet, run unmodified.

The only flag not in the literal README is `--shard I/N` on `batch_haptic.py`, added to satisfy the README's mandate "WE NEED TO UTILIZE ALL THE GPUS TO DO PARALLELIZATION" / "Step 2 HaPTIC is the main bottleneck — parallelization is recommended". The flag is inert when not passed; per-sequence outputs are bit-identical to a single-GPU run.

---

## Setup deltas vs README — patches applied

All in this branch, all required to make the README pipeline run on the GM dgx cluster (none change algorithmic behavior).

1. **Apple CDN blocked** — used HF mirror `apple/DepthPro` for `depth_pro.pt`.
2. **HaPTIC pretrained weights** — Google Drive quota-blocked from cluster, user supplied `release.tar.gz` locally.
3. **FoundationPose pretrained weights** — same Google Drive issue, used HF mirror `gpue/foundationpose-weights`.
4. **MANO models** — academic registration, user supplied `mano_v1_2.zip` locally (extracted into `third_party/haptic/_DATA/data/mano/`).
5. **Compiler stack via conda** — `cuda-nvcc 12.1`, `boost`, `eigen3`, `libcusparse-dev`, `cmake`. Required env: `CUDA_HOME=$CONDA_PREFIX`, `CPATH=$CONDA_PREFIX/targets/x86_64-linux/include`, `LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib`.
6. **FoundationPose `bundlesdf/mycuda/setup.py`** — `-std=c++14` → `-std=c++17` (modern torch ATen requires 17), conda Eigen include path. Built cleanly.
7. **`numpy<2`, `scipy<1.13`, `setuptools<81`** — torch 2.1.1 wheel ABI requires numpy 1.x; scipy ≥ 1.13 needs numpy 2; pkg_resources removed in setuptools 81+ but still imported by lightning_fabric.
8. **`opencv-python==4.9.0.80`** — opencv 4.11 wheels were built against numpy 2.x ABI.
9. **`chumpy/__init__.py`** patched to remove deprecated `from numpy import bool, int, float, complex, object, unicode, str`.
10. **`mmcv-full==1.7.2`** built from source; `mmpose==0.29.0` (HaPTIC's pinned old API); patched `mmpose/__init__.py` to relax `mmcv_maximum_version` from `1.5.0` / `1.7.0` to `1.8.0`.
11. **ViTPose** manually cloned to `third_party/haptic/third-party/ViTPose/`, installed editable.
12. **HaPTIC `init_renderer=False`** — `data/batch_haptic_arctic.py:load_haptic_model` patched to skip EGL OffscreenRenderer init on headless A100 (no `/dev/dri/` access). Renderer is for training-time vis only; Step 2 inference output `verts_dict` is unchanged.
13. **Hardcoded path purge** in `third_party/haptic/nnutils/hand_utils.py` (`/is/cluster/fast/yye/...` → local `_DATA/data/mano/`).
14. **HaPTIC missing `MANO_UV_right.obj`** — `ManopthWrapper` patched to skip UV load when missing and fall back to base MANO faces from `mano_layer_right.th_faces`. Vertices unchanged.
15. **HaPTIC `parse_det_seq`** signature extended to accept the `intrinsics=` kwarg the project's wrapper passes.
16. **`tools/batch_obj_pose.py`** — (a) `FP_ROOT` made overridable via env; (b) `seq_to_obj` for `ho3d_v3` patched to strip `train__` / `evaluation__` split prefix before YCB-name lookup (without this, **0** of 68 HO3D would have produced obj-pose outputs); (c) cleanup of `/tmp/fp_scenes/{seq}` after each sequence to keep `/tmp` from filling at 6000 × 40 MB scale.
17. **`data/batch_haptic.py`** — `--shard I/N` added (see README compliance section).

---

## Reproducing this run

1. SSH to `dgx047` via `gc211` jumphost.
2. `cd ~/Affordance2Grasp` (NFS home, 281 TB free).
3. Run the four sequenced launchers under `scripts/`:
    - `scripts/launch_step1_dgx047.sh` — Step 1, 7-way (HO3D + 6 DexYCB cams).
    - `scripts/launch_step2_ho3d.sh` — Step 2 HO3D, 3-way.
    - `scripts/launch_step2_dexycb.sh` — Step 2 DexYCB (per subject; for full 10 subjects use `scripts/launch_phase2_subj05_10.sh`).
    - `scripts/launch_step3_ho3d.sh` — Step 3 HO3D, 4-way shard.
    - `scripts/launch_step3_dexycb.sh` — Step 3 DexYCB.
4. After Step 2 + Step 3 finish: `python data/batch_align_mano_fp.py --dataset {dexycb,ho3d_v3}`.
5. Verification: README's exact h5py snippet — output reproduced above.

---

## Deliverables

- **GitHub:** branch [`zt_results_07272026`](https://github.com/stzabl-png/UCB_Project/tree/zt_results_07272026) — REPORT.md, parallel launch scripts, code patches.
- **HuggingFace:** dataset [`zhengtaoyao/Affordance2Grasp-zt-results-07272026`](https://huggingface.co/datasets/zhengtaoyao/Affordance2Grasp-zt-results-07272026) — `training_fp/`, `human_prior_fp/`, `third_mano/`, `obj_poses/` for both datasets.
