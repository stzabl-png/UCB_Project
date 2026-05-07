# Deployment Issues & Fixes

> ⚠️ **Sections 1–8 are based on the early ARCTIC-only stage and are now outdated.**
> For the current pipeline, refer to **`README.md`** and **`HANDOVER.md`**.
> This document is kept as a historical reference. Appendix A (Blackwell GPU) is current.

---

## 1. Overview (Historical — ARCTIC Pipeline)

> **Goal:** Generate object contact heatmaps (Human Prior) from ARCTIC third-person videos,
> collect Robot GT data via Isaac Sim, and train the M5 PointNet++ model.

> **Paths:** `$PROJ` = project root. `SAM3D_USER` = cloud server username.

---

## 2. Full Pipeline Flow (Historical)

```
[ARCTIC third-person videos]
    ↓ Step 0a  annotate_trim.py        → data/trimmed/{seq}/
    ↓ Step 0b  annotate_obj_mask.py    → mask_bbox.png + bbox.json
    ↓ Step 0c  SAM3D (cloud)           → output/sam3d_obj_cache/{obj}/splat.ply
    ↓ Step 1   batch_depthpro.py       → output/depthpro_batch/{seq}/
    ↓ Step 2   batch_fp_register.py    → output/fp_register_batch/{seq}_T_obj_cam1.npy
    ↓ Step 3   batch_contact.py        → output/affordance_batch/{seq}/vert_contact_count.npy
    ↓ Step 4   export_arctic_prior.py  → data_hub/human_prior/{obj}.hdf5  ← key interface
    ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │  Step 5  random_grasp_sampler.py  (50% HP + 50% random)        │
    │          → output/grasps_random/{obj}_grasp.hdf5               │
    │  Step 6  Isaac Sim  batch_random_sim.sh                         │
    │          → output/robot_gt_v4_physics/{obj}_robot_gt.hdf5      │
    │  Step 7  aggregate_robot_gt.py                                  │
    │          → data_hub/training/{obj}.hdf5 (human_prior + robot_gt│
    │  Step 8  model/train.py → output/checkpoints_m5/best_m5_model  │
    └─────────────────────────────────────────────────────────────────┘
    ↓ (inference)
    inference/grasp_pose.py → candidate grasp poses HDF5 (13 candidates, 100% HP-guided)
    sim/run_grasp.py        → Isaac Sim verification → posterior
```

---

## 3. Environments (Historical)

| Env | Purpose | Activate |
|---|---|---|
| `base` | Video annotation, mask annotation (Qt GUI), random_grasp_sampler | `conda activate base` |
| `hawor` | Contact detection, prior export, visualization | `conda activate hawor` |
| `depth-pro` | Depth Pro depth estimation | `conda activate depth-pro` |
| `bundlesdf` | FoundationPose pose registration | `conda activate bundlesdf` |
| **IsaacSim** | Robot GT collection (no conda) | `$ISAAC_SIM_PATH/python.sh` |
| Cloud server | SAM3D mesh generation | `ssh sam3d-gpu` |
| `sam3d-objects` | SAM3D inference (on cloud) | `conda activate sam3d-objects` |

---

## 4. Known Issues Fixed (Historical — ARCTIC stage)

These are recorded for reference. All fixes are already applied in the current codebase.

| Issue | Fix applied |
|---|---|
| HaPTIC `--seq` filter does not match DexYCB format | Patched `batch_haptic.py` |
| HaPTIC `MANO_RIGHT.pkl` hardcoded path | Use `config.MANO_RIGHT` |
| HaPTIC `parse_det_seq()` does not accept `intrinsics` arg | Added `intrinsics` param |
| HaWoR `environment.yml` pip section fails (build isolation) | Added `--no-build-isolation` |
| HaWoR `chumpy` install fails | Pin `setuptools<70` first |
| HaWoR DROID-SLAM `setup.py` path confusion | Run from inside `DROID-SLAM/` |
| MegaSAM `droid_backends` not compiled | Compile from `mega-sam/base/` |
| MegaSAM Depth-Anything ViT-L weights missing | Download from HF |
| All `torch.load` calls crash on PyTorch 2.6+ | Added `weights_only=False` globally |

---

## Appendix A — Blackwell GPU (RTX 4090 Ti / RTX 5090, sm_120) Fixes

> This appendix is **current and applies to sm_120 GPUs**.

### A.1 `bundlesdf` + `haptic` envs: upgrade to cu128

```bash
# A.1: upgrade to Blackwell-compatible torch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install xformers --index-url https://download.pytorch.org/whl/cu128
# Re-pin torchvision after xformers possibly upgrades torch:
pip install torchvision --index-url https://download.pytorch.org/whl/cu128 --upgrade

# Source-build libs without cu128 wheels (with sm_120 arch flag)
export CUDA_HOME=$CONDA_PREFIX
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="12.0"
pip install --no-build-isolation 'mmcv-full>=1.7.0,<1.8.0'
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2'
pip install --no-build-isolation 'git+https://github.com/facebookresearch/pytorch3d.git'
```

### A.2 Same upgrade for `hawor` and `mega_sam` envs

Apply the same steps. Additionally:
- **DROID-SLAM** (`third_party/hawor/thirdparty/DROID-SLAM/setup.py`): add
  `'-gencode=arch=compute_120,code=sm_120'` to nvcc args, then `python setup.py install`.
- **mega-sam droid_slam**: `mega-sam/base/setup.py` similarly needs the sm_120 line.
  After build, **delete the stale `.so` file** committed at older ABI so import resolves
  to the freshly built one in site-packages.
- **`torch_scatter`** (mega_sam env):
  ```bash
  pip install --force-reinstall torch_scatter \
    -f https://data.pyg.org/whl/torch-2.11.0+cu128.html
  ```

### A.3 xformers `memory_efficient_attention` lacks sm_120 kernels

Even with cu128 torch + latest xformers, `memory_efficient_attention` may error:
```
`fa3F@0.0.0` ... requires device with capability <= (9, 0) but your GPU has capability (12, 0) (too new)
```
Wrap calls in `try/except` and fall back to `torch.nn.functional.scaled_dot_product_attention`.

Three call sites patched on RTX 5090:
- `third_party/haptic/haptic/models/components/pose_transformer.py:145`
- `mega-sam/UniDepth/unidepth/models/backbones/metadinov2/attention.py:79`
- `~/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/attention.py:94`

For UniDepth: force `XFORMERS_AVAILABLE = False` — UniDepth has clean PyTorch fallbacks.
For `NystromAttention` (removed in xformers 0.0.28): provide a small SDPA-based stub class.

### A.4 PyTorch 2.6+ `weights_only=True` rejects HaWoR / HaPTIC checkpoints

Patch globally at script entry:
```python
import torch as _t
_orig = _t.load
_t.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})
```

### A.5 SAM3D `kaolin` / `spconv` workarounds

- **kaolin**: No cu128 wheel. Build CPU-only from source:
  ```bash
  git clone --recurse-submodules https://github.com/NVIDIAGameWorks/kaolin
  cd kaolin
  FORCE_CUDA=0 IGNORE_TORCH_VER=1 pip install --no-build-isolation -e .
  ```
- **spconv**: Use `spconv-cu126` — PTX forward-compat works on sm_120:
  ```bash
  pip install spconv-cu126
  ```
- **gsplat**: Source build from SAM3D-pinned commit + `TORCH_CUDA_ARCH_LIST="12.0"`.
- **flash_attn**: Skip entirely. Set `ATTN_BACKEND=sdpa` — SAM3D has clean fallbacks.

### A.6 Kernel mismatch after OS update

If `nvidia-smi` fails with "couldn't communicate with NVIDIA driver":
```bash
# Boot into previous kernel via GRUB:
sudo grub-reboot 'Advanced options for Ubuntu>Ubuntu, with Linux <prev-kernel>-generic'
sudo reboot
```
Long-term fix: install `dkms` so `nvidia-driver-580` rebuilds on each kernel update.

### A.7 Verified Working Configuration (Blackwell)

| Component | Version |
|---|---|
| GPU | RTX 5090 (sm_120, 32 GB) |
| Driver | 580.126.09 |
| Kernel | 6.17.0-22-generic (Ubuntu 24.04.3) |
| `torch` | 2.11.0+cu128 |
| `xformers` | 0.0.35 |
| `pytorch3d` | 0.7.8 / 0.7.9 |
| `mmcv-full` | 1.7.2 (source built) |
| `detectron2` | 0.6 (source built) |
| `spconv` | spconv-cu126 2.3.8 |
| `kaolin` | 0.18.0 (CPU build) |
| `gsplat` | 1.5.3 (sm_120 source) |
| Phase 1A end-to-end | ✅ verified, 72 frames @ ~25 s total |
| Phase 1B end-to-end | ✅ verified, 5 episodes × 60 frames |
| SAM3D 217-object batch | ✅ 5.7 s / object, 0 failures |
