# Affordance2Grasp — Project Handover

> Written 2026-05-05 for the incoming team.
> This document covers details, rules, and known gotchas not in README.md.

---

## 0. Quick Navigation

| What you need to do | Where to look |
|---|---|
| Environment setup | README §3b / §3d / §3e |
| Pipeline commands | README §Phase 1A / 1B / Egocentric |
| Troubleshooting | README §Troubleshooting (T1–T16+) |
| Dataset download | README §4 Data layout + §2 below |
| Rules not in README | **This document** |

---

## 1. What This Project Does (One Line)

> Extract **human contact priors** from hand-object interaction videos (DexYCB / HO3D / OakInk / TACO / EgoDex), then train PointNet++ to predict robot grasp locations.

This is NOT end-to-end imitation learning or RL.
Core data chain: **RGB-D → hand pose → object pose → contact points → PointNet++ classifier**

---

## 2. Data Overview

### Datasets (lab machine, 2026-05-05)

| Dataset | Path | Status | Scale |
|---|---|---|---|
| DexYCB | `data_hub/RawData/ThirdPersonRawData/dexycb/` | ✅ Complete | subject-01, 100 seqs, 20 objects |
| HO3D v3 | `data_hub/RawData/ThirdPersonRawData/ho3d_v3/` | ✅ Complete | train+eval |
| OakInk v1 | `data_hub/RawData/ThirdPersonRawData/oakink_v1/` | ✅ Complete | 778 sequences |
| TACO Allocentric | `data_hub/RawData/ThirdPersonRawData/taco/Allocentric_RGB_Videos/` | ⚠️ Partial | 10/151 triplets extracted |
| TACO Egocentric | `data_hub/RawData/EgoRawData/taco/Egocentric_RGB_Videos/` | ✅ Complete | 151 triplets |
| EgoDex | `data_hub/RawData/EgoRawData/egodex/test/` | ✅ Complete | 101 task categories |

### Processed Outputs

| Output directory | Contents | Status |
|---|---|---|
| `ProcessedData/third_depth/` | Depth Pro depth maps | DexYCB ✅ |
| `ProcessedData/third_mano/` | HaPTIC MANO params | DexYCB ✅ |
| `ProcessedData/obj_poses/` | FoundationPose object poses | DexYCB ✅ |
| `ProcessedData/training_fp/` | Aligned training HDF5 | DexYCB ✅ (20 objects) |
| `ProcessedData/human_prior_fp/` | Contact prior HDF5 | DexYCB ✅ (20 objects) |
| `ProcessedData/obj_recon_input/` | SAM2 seed masks | 384 total (egocentric 245 + ycb etc.) |
| `ProcessedData/egocentric_depth/` | MegaSAM depth + camera | Partial EgoDex |
| `ProcessedData/ego_mano/` | HaWoR MANO params | Partial EgoDex |

### HuggingFace Repos (UCBProject org)

| Repo | Contents | Access |
|---|---|---|
| `Affordance2Grasp-Weights` | FP weights + HaPTIC + MegaSAM + DepthPro + HaWoR | Public |
| `Affordance2Grasp-Mesh` | DexYCB 20 object meshes + init masks | Public |
| `EgoDataMask` | 245 egocentric task init masks | Public |
| `ThirdDataMask` | 278 third-person init masks (YCB/OakInk/TACO) | Public |
| `Affordance2Grasp-EgoDex` | EgoDex raw videos | Public |
| `Affordance2Grasp-OakInk` | OakInk raw data | Public |
| `Affordance2Grasp-TACO` | TACO Allocentric + Ego videos | Public |
| `ARCTIC-Archive` | ARCTIC data backup | Public |

> **HF Token**: Not stored in the codebase. Generate a write-access token from HuggingFace Settings → Access Tokens. Never commit tokens to git.

---

## 3. Conda Environments

> **Critical rule**: Do NOT follow any submodule's own README for installation.
> All upstream READMEs use PyTorch versions incompatible with our CUDA 12.1 stack.

| Env | Python | PyTorch | CUDA | Used for |
|---|---|---|---|---|
| `depth-pro` | 3.9 | — | — | Phase 1A Step 1 (Depth Pro) |
| `haptic` | 3.10 | 2.1.1+cu121 | 12.1 | Phase 1A Step 2 (HaPTIC) |
| `bundlesdf` | **3.10** | 2.1.1+cu121 | 12.1 | FP / Align / Training (main env) |
| `mega_sam` | 3.10 | **2.2.0+cu121** | 12.1 | MegaSAM depth (Ego Step 1) |
| `hawor` | 3.10 | **2.1.0+cu121** | 12.1 | HaWoR hand (Ego Step 2) |

### Pinned Versions (must be exact)

```bash
# haptic env
mmpose==0.24.0  mmcv-full==1.3.9   # HaPTIC is version-sensitive

# bundlesdf env
python=3.10                         # NOT 3.9 — nvdiffrast ABI incompatible
fast-simplification>=0.1.6          # Without this, align takes 23h instead of 15min

# hawor env
torch==2.1.0+cu121                  # NOT 1.13 — HaWoR official README is wrong
torch-scatter==2.1.2+pt21cu121      # Must use pre-built wheel, not source build
pytorch3d==0.7.5                    # Pre-built wheel: py310 + cu121 + torch2.1
```

---

## 4. Required Environment Variables

```bash
export PATH=/usr/local/cuda/bin:$PATH      # nvcc must be accessible
export FP_ROOT=/path/to/FoundationPose     # Critical — FP looks for weights here
export SAM2_DIR=/path/to/sam2             # SAM2 annotation tool
```

Add to `~/.bashrc` to persist across sessions.

---

## 5. Pipeline Architecture

### 5.1 Overview

```
Phase 1A (third-person): Depth Pro → HaPTIC → FoundationPose → Align
    Output: training_fp/{dataset}/{obj}.hdf5

Phase 1B (egocentric):   MegaSAM → HaWoR → FoundationPose (ego) → Align
    Output: training_fp_ego/{dataset}/{obj}.hdf5

Phase 2 (aggregate):     training_fp/ + training_fp_ego/ → human_prior/{obj}.hdf5

Phase 3 (train):         build_dataset.py → model/train.py
```

### 5.2 Mask File Layout

| Purpose | Path | Count |
|---|---|---|
| Phase 1A FP init mask | `obj_recon_input/{dataset}/{obj}/0.png` | 1 per object |
| Phase 1B FP init mask | `obj_recon_input/egocentric/{task}/0.png` | 1 per task |

EgoDex/TACO ego masks: HF `EgoDataMask` (245 tasks).
DexYCB/OakInk/TACO third-person masks: HF `ThirdDataMask` (278 objects).

### 5.3 HaPTIC Quirks

1. **mmpose must be exactly 0.24.0** — 1.x has a completely different API.
2. Must `os.chdir(HAPTIC_DIR)` before running — weights paths are relative to CWD.
3. All `torch.load` calls must include `weights_only=False` (Python 3.10 + PyTorch 2.1).

### 5.4 FoundationPose Quirks

1. Requires `nvdiffrast` and `mycpp` — must be compiled from source.
2. Compile with system gcc, not conda gcc — avoids glibc conflicts.
3. ABI issue in `bundlesdf` env:
   ```bash
   export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
   ```
4. `cmake` must be < 4.0 — cmake ≥ 4.0 removed `FindBoost`.

### 5.5 Depth Pro Quirks

1. Checkpoint path is relative to CWD — always `cd third_party/ml-depth-pro/` first.
2. DexYCB has no GT intrinsics — use two-pass self-calibration: pass 1 estimates `fx`, pass 2 fixes `fx` and re-runs.
3. Validated DexYCB cam `841412060263` focal length: `fx ≈ 591.4`.

### 5.6 TACO Allocentric — Special Handling

TACO Allocentric is MP4 video (4096×3000, 30fps) — **not pre-extracted frames**.
Must extract frames first:

```bash
# In order:
# 1. Extract frames (~89 GB, ~2 hours)
python tools/extract_taco_frames.py --mode pipeline --cam 22139905

# 2. Manually annotate seed masks for 151 triplets (~2–3 hours)
python tools/sam2_annotate_by_object.py --dataset taco_allocentric

# 3. Auto-generate masks for all sequences
python data/batch_prepare_frame3.py --dataset taco_allocentric
```

---

## 6. Current TODO

### Immediate

| Task | Command |
|---|---|
| Extract all TACO Allocentric frames | `extract_taco_frames.py --mode pipeline` |
| Annotate 151 TACO Allocentric seed masks | `sam2_annotate_by_object.py --dataset taco_allocentric` |
| Rebuild `mega_sam` env | torch 2.2.0+cu121, compile lietorch |

### Near-term

| Task | Command |
|---|---|
| Full EgoDex MegaSAM | `batch_megasam.py --dataset egodex` (~50h) |
| Full EgoDex HaWoR | `batch_hawor.py --dataset egodex` (~40h) |
| Download remaining TACO Allocentric cameras | 11 more zips (~400 GB total) |

### Long-term

| Task | Notes |
|---|---|
| Phase 4 retrain with TACO data | After TACO pipeline completes |
| Baseline experiments | GraspNet-AnyGrasp / DexGraspVLA / DP |

---

## 7. Lab Machine Status (RTX 3090)

**Machine spec**: Ubuntu 22.04, CUDA 12.1, RTX 3090.
**Data location**: `/media/msc/5TB1/` (5 TB HDD).
**FoundationPose build**: `/media/msc/5TB1/FoundationPose/`.

**Completed (DexYCB subject-01, cam 841412060263):**
```
Phase 1A Step 1 (Depth Pro)  ✅  100/100 seqs, fx=591.4
Phase 1A Step 2 (HaPTIC)     ✅  100/100 seqs
Phase 1A Step 3 (FP)         ✅  100/100 seqs, ~6s/seq
Phase 1A Step 4 (Align)      ✅  20 objects, diverged=0
```

**Blocked:**
```
hawor env:    pytorch-lightning version mismatch → fix in README T7.3
mega_sam env: droid_backends not compiled → fix in README Section 3d
```

---

## 8. Training Data Quality

### What `diverged=0` Means

`batch_align_mano_fp.py` reports `diverged=0` when:
- FoundationPose tracking did not drift (depth sanity check passed)
- MANO hand / object scale ratio is within valid range
- Contact points land correctly on the mesh surface

Frames with `diverged > 0` are excluded from training. All 20 DexYCB objects have `diverged=0`.

### Contact Coverage Metrics

```
contact_verts=4978/4980  → fraction of mesh vertices with contact (higher = better)
cov(>0.01)=76%           → mesh surface with any weak contact
cov(>0.5)=16%            → mesh surface with strong contact (direct hand touch)
hp_max=0.840             → max contact probability in one frame (>0.5 = clear contact)
```

---

## 9. Code Conventions

### 9.1 Discoverer Function Signature

All `discover_*` functions must follow this interface:

```python
def discover_xxx(input_dir: str) -> Generator:
    # For third-person pipeline (batch_prepare_frame3, sam2_annotate):
    yield seq_id: str, img_paths: List[str]

    # For sam2_annotate_by_object:
    yield ds_out: str, obj_name: str, frames: List[str]
```

### 9.2 All `torch.load` Must Use `weights_only=False`

```python
# Correct
torch.load(path, weights_only=False)["state_dict"]

# Wrong (crashes in PyTorch 2.6+)
torch.load(path)["state_dict"]
```

Fixed globally in commit `c9456de`. All new code must follow this.

### 9.3 Custom Scripts in Submodules

If you write a script inside a submodule (e.g. `third_party/hawor/`), **always place a copy in the main project's `data/` directory** and implement auto-sync in the corresponding batch script. Submodule files are not tracked by the main repo git and will be lost after a fresh clone.

Reference: `data/run_hawor_seq.py` + `_sync_runner()` in `batch_hawor.py`.

### 9.4 Checklist for Adding a New Dataset

- [ ] `data/batch_depth_pro.py` or `batch_megasam.py` (depth)
- [ ] `data/batch_haptic_*.py` or `batch_hawor.py` (hand pose)
- [ ] `data/batch_fp.py` or `batch_obj_pose_ego.py` (object pose)
- [ ] `data/batch_align_*.py` (contact alignment)
- [ ] `tools/sam2_annotate_by_object.py` (seed mask annotation)
- [ ] `data/batch_prepare_frame3.py` (auto mask generation)
- [ ] `config.py` (path config)

---

## 10. PointNet++ Training Summary

| Parameter | Value | Reason |
|---|---|---|
| Architecture | PointNet++ v5 Multi-Task | Segmentation head + force-center regression head |
| Input channels | xyz(3) + normal(3) + human_prior(1) = 7 | human_prior is an input feature, not the label |
| Supervision | robot_gt (sim-verified contact) | Not directly supervised by human_prior |
| Segmentation loss | Focal(α=0.75,γ=2.0) + Tversky(α=0.3,β=0.7) | Label imbalance (~16% contact points) |
| Regression loss weight λ | 10.0 | Compensates for MSE magnitude |
| Optimal threshold | 0.65–0.75 | Peak F1 not at 0.5 due to class imbalance |
| Validation split | Object-level 20% (seed=42) | Prevents data leakage |
| GT-free mode | ✅ No Isaac Sim needed | Current v2 checkpoint F1=0.642 |

---

## 11. Top 5 Gotchas

1. **`bundlesdf python=3.9` → use `python=3.10`**
   nvdiffrast and pytorch3d wheels are compiled for py310.

2. **HaWoR README says torch 1.13 → actually use torch 2.1.0**
   Following the official HaWoR docs will always fail.

3. **MegaSAM README says torch 2.0.1 → actually use torch 2.2.0**
   `droid_backends` is never pip-installed — must be compiled from source.

4. **TACO Allocentric is MP4 → not a frame directory**
   All discover functions expect JPEG frames — run `extract_taco_frames.py` first.

5. **`FP_ROOT` not set → FoundationPose cannot find nvdiffrast**
   Add `export FP_ROOT="/path/to/FoundationPose"` to `~/.bashrc`.

---

## 12. Quick File Reference

| Looking for... | Location |
|---|---|
| Depth estimation scripts | `data/batch_depth_pro.py`, `data/batch_megasam.py` |
| Hand pose scripts | `data/batch_haptic_*.py`, `data/batch_hawor.py` |
| Object pose scripts | `data/batch_fp.py`, `data/batch_obj_pose_ego.py` |
| Contact alignment | `data/batch_align_mano_fp.py`, `data/batch_align_ego.py` |
| SAM2 annotation tool | `tools/sam2_annotate_by_object.py` |
| TACO frame extraction | `tools/extract_taco_frames.py` |
| Training script | `model/train.py` |
| Weight download | `setup_weights.py` |
| HaWoR core worker | `data/run_hawor_seq.py` (auto-synced to hawor submodule) |
| Known issues | `README.md` §Troubleshooting |
