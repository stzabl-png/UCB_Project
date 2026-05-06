# Deployment Issues — Field Notes

> Compendium of real issues hit during fresh deployments and their fixes.
>
> **Audience:** industry users running large-scale experiments on **A100 (sm_80, Ampere)**.
> The main body covers issues that occur on any modern GPU. The Blackwell appendix at the
> end documents extra issues you'll only encounter on RTX 5090 / RTX 6000 Pro / B100 (sm_120).
>
> See also: `README.md` (canonical install) and `README.md` `T1`–`T16` (troubleshooting list).
> This doc is a *supplement* with nuance / gotchas not covered there.

---

## TL;DR for industry users (A100)

Pre-built object meshes are already published — **you don't need to run SAM3D yourself**:

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='UCBProject/ObjMesh', repo_type='dataset',
    local_dir='data_hub/ProcessedData/obj_meshes_ext')
"
# Move meshes into the path the pipeline reads from:
mv data_hub/ProcessedData/obj_meshes_ext/meshes/* data_hub/ProcessedData/obj_meshes/
```

The 217 mesh files (28 YCB + 89 EgoDex tasks + 100 OakInk) are SAM3D-reconstructed,
ready for Phase 1A Step 3 and Phase 1B E5.

If you want to reconstruct meshes yourself for new objects: `tools/batch_sam3d_recon.py`
(needs ~32 GB VRAM, runs ~5 s/object on A100).

---

## 1. HuggingFace + Credentials

### 1.1 setup_weights.py 401 — gated repo not authenticated
```
huggingface_hub.errors.RepositoryNotFoundError: 401 Client Error.
... UCBProject/Affordance2Grasp-Data ...
```
Most weight repos are gated. Login once, then `setup_weights.py` works:
```bash
huggingface-cli login    # paste a Read token
```
Make sure your account has been added to the `UCBProject` org or has dataset access.

### 1.2 setup_weights.py without `--tool` downloads 30 GB EgoDex
The `TOOLS` dict iterates **all** entries by default, including `download_egodex` (the
30 GB raw video dataset). Only run that if you need EgoDex for Phase 1B:
```bash
# Recommended: explicitly list tools, skip egodex unless needed
for t in fp hawor haptic megasam depthpro egomasks; do
  python setup_weights.py --tool $t
done
```

### 1.3 MANO is license-restricted, must download manually
No automation possible. Register at https://mano.is.tue.mpg.de, download
`mano_v1_2.zip`, place files in **all** of these locations:
- `third_party/haptic/assets/mano/MANO_{RIGHT,LEFT}.pkl`
- `third_party/haptic/_DATA/data/mano/MANO_{RIGHT,LEFT}.pkl`  ← the actual path HaPTIC reads
- `third_party/hawor/_DATA/data/mano/MANO_RIGHT.pkl`
- `third_party/hawor/_DATA/data_left/mano_left/MANO_LEFT.pkl`

### 1.4 HaPTIC `dl_model.sh` — gdown intermittent throttling
```
Cannot retrieve the public link of the file. ... many accesses.
```
Just retry with the raw file ID (no `--fuzzy`, that flag was removed):
```bash
gdown 1BX_gT__7hE47B_YUizUWfEfeqopLxloZ -O output/haptic_model.tar.gz
```

---

## 2. Submodule init / patches

### 2.1 ViTPose subdir not initialized in HaPTIC submodule
HaPTIC's `scripts/install.sh` does `pip install -e third-party/ViTPose` but the dir is
empty. Manually clone it:
```bash
cd third_party/haptic
git clone https://github.com/ViTAE-Transformer/ViTPose third-party/ViTPose
```

### 2.2 ViTPose installs as `mmpose` package + needs PEP 660 escape
PyPI's `kaolin` is a placeholder and ViTPose's `setup.py` doesn't support PEP 660
editable. Use legacy install:
```bash
cd third_party/haptic/third-party/ViTPose
python setup.py develop          # registers as `mmpose`
```

### 2.3 ViTPose's `mmpose/__init__.py` mmcv version cap
ViTPose's mmpose pins `mmcv >= 1.3.8, <= 1.5.0`, but mmcv 1.5.0 fails to build with
modern CUDA toolchains. mmcv-full 1.7.x works fine in practice — bump the cap:
```python
# third_party/haptic/third-party/ViTPose/mmpose/__init__.py
mmcv_maximum_version = '1.8.0'      # was '1.5.0'
```
Then `pip install --no-build-isolation 'mmcv-full>=1.7.0,<1.8.0'` (or via openmim).

### 2.4 HaPTIC requires `MANO_UV_*.obj` but doesn't ship it
`nnutils/hand_utils.py` tries `load_obj("_DATA/data/mano/MANO_UV_right_closed.obj")` —
the file is **not** in the public MANO release nor in HaPTIC's repo. Two options:
- (a) Download the UV mesh files from the SMPL-X / extras package (see HaPTIC README)
- (b) Skip UV loading — only needed for texture rendering, not for batch inference.
  Wrap `load_obj` in `if osp.exists(fname + '.obj')`, fall back to MANO `.pkl` faces.

### 2.5 `parse_det_seq` intrinsics patch (use the published patch)
Upstream HaPTIC's `parse_det_seq` doesn't accept an `intrinsics=` kwarg, but our
`data/batch_haptic_arctic.py` passes one. The fix is in `patches/haptic-intrinsics-fix.patch`:
```bash
git submodule update --init third_party/haptic
cd third_party/haptic
git apply ../../patches/haptic-intrinsics-fix.patch
```

---

## 3. Build / install gotchas

### 3.1 `detectron2` build sees no torch (build-isolation)
```
ModuleNotFoundError: No module named 'torch'  (during detectron2 wheel build)
```
pip's isolated build env doesn't see haptic env's torch. Add `--no-build-isolation`:
```bash
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2'
pip install --no-build-isolation 'git+https://github.com/mattloper/chumpy'
pip install --no-build-isolation 'git+https://github.com/hassony2/manopth.git'
```

### 3.2 `numpy` keeps getting upgraded to 2.x by transitive deps
Several common packages (`pyrender`, `scikit-image`, `xtcocotools`, etc.) pull `numpy>=2`,
but torch 2.1.1 + many older packages need `numpy<2`. Re-pin after every batch install:
```bash
pip install "numpy<2.0" --force-reinstall --no-deps
```

### 3.3 `fast-simplification` 0.1.13 needs Python 3.10
Uses `float | None` syntax (3.10+). If the env is 3.9:
```bash
pip install "fast-simplification>=0.1.6,<0.1.10"   # 0.1.9 is the last 3.9-compat
```
Alternative: bump the env to Python 3.10 (recommended; matches `bundlesdf` upstream now).

### 3.4 Depth Pro looks for `./checkpoints/depth_pro.pt` (relative)
`third_party/ml-depth-pro/src/depth_pro/depth_pro.py` hard-codes a relative path.
Symlink at the project root:
```bash
ln -s third_party/ml-depth-pro/checkpoints checkpoints
```

### 3.5 FoundationPose CMake — `boost_system` not found
Boost ≥ 1.69 made `boost_system` header-only — no separate cmake module. If you have
sudo, `apt install libboost-system-dev` works. Without sudo (conda Boost 1.91), patch
`mycpp/CMakeLists.txt`:
```cmake
find_package(Boost REQUIRED COMPONENTS program_options)   # remove "system"
```
Then:
```bash
conda install -c conda-forge libboost-devel libboost libboost-headers -y
pip install pybind11
export pybind11_DIR=$(python -m pybind11 --cmakedir)
bash build_all_conda.sh
```

### 3.6 FoundationPose pinned commit `25e225a` doesn't exist upstream
README says `git checkout 25e225a` but that hash isn't in `NVlabs/FoundationPose`.
Use `main` HEAD; the title "Local Conda install: fix mycpp import" is the right baseline.

### 3.7 HF cached file for `ProcessedData` (ycb meshes + masks) is required
```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='StZaBL/Affordance2Grasp-ProcessedData',
                  repo_type='dataset', local_dir='data_hub/ProcessedData')
"
```
Without this, FoundationPose can't find `obj_meshes/ycb/` or `obj_recon_input/ycb/` and
Step 3 yields zero processable sequences. Same applies to `UCBProject/EgoDataMask` for
Phase 1B masks.

---

## 4. Phase 1A pipeline gotchas

### 4.1 Step 4 contact alignment shows `human_prior max=0`
Two common causes:
1. **Sequence lacks grasp moments** — pick later sessions (frames 20–50 of a typical
   DexYCB clip have actual contact; first 10 are approach). Run on multiple sessions
   and aggregate per-object — that's how upstream produces meaningful priors.
2. **`fast-simplification` not installed** → mesh stays at 1.3M faces and alignment
   takes ~23 hours. Verify it's there:
   ```bash
   python -c "import fast_simplification; print(fast_simplification.__version__)"
   ```

### 4.2 Hardcoded paths from upstream author
`third_party/haptic/nnutils/hand_utils.py:18` defaults `mano_path` to
`/is/cluster/fast/yye/...`. Latest `data/batch_haptic_arctic.py` already overrides this
via `config.HAPTIC_MANO_DIR` — make sure your `config.py` defines it:
```python
HAPTIC_MANO_DIR = os.environ.get(
    "HAPTIC_MANO_DIR",
    os.path.join(HAPTIC_DIR, "assets", "mano"))
```

---

## 5. Phase 1B pipeline gotchas

### 5.1 `batch_megasam.py --seq-ids` requires all dataset metadata files
Even if you only want `egodex`, the script eagerly enumerates `ph2d_avp` / `taco` /
`hoi4d`, which crashes if those dataset metadata files don't exist. Workaround: use
`--start N --end M` instead of `--seq-ids`.

### 5.2 EgoDex registry uses absolute paths from another machine
`tools/egodex_sequence_registry.json` had original entries with `/home/lyh/...` paths
hardcoded. Many entries are also marked `skipped: true`. Add new entries for sequences
you want to process, with **your** absolute paths and `skipped` removed.

### 5.3 Tasks without object meshes
12 EgoDex tasks have no mask annotation (deformable / non-rigid actions: paper folding,
piano, keyboard, etc.). Their mesh entries don't exist. Skip them — single-image mesh
recon doesn't apply to deformable scenes.

### 5.4 Object scale estimation needs metric depth
SAM3D output is normalized (unitless). Run `data/estimate_obj_scale_ego.py --obj <name>`
to compute `scale_factor = d_real / d_mesh` from MegaSAM depth + mask. Result is saved
as `scale.json` next to `mesh.ply` and read automatically by Step 4 / E6.

### 5.5 Egocentric mask source vs. EgoDex name collision
The dataset `obj_recon_input/egocentric/` historically mixed EgoDex tasks **and**
TACO ego triplets. Filter to EgoDex tasks by intersecting with task dirs in
`RawData/EgoRawData/egodex/test/`. Tool: `tools/batch_sam3d_recon.py` does this filtering
when you pass `--datasets egodex`.

### 5.6 `weights_only=True` in PyTorch ≥ 2.6
PyTorch 2.6 reversed the `torch.load` default. Multiple checkpoints in HaWoR contain
`omegaconf.dictconfig.DictConfig` objects which are blocked. Latest main repo already
has these patched (`weights_only=False` in 7 files). On torch 2.1.1 it's a no-op, so
you don't need to revert anything for A100.

---

## Appendix A — Blackwell GPU notes (RTX 5090 / RTX 6000 Pro / B100)

> **Skip this if you're on A100 / V100 / RTX 4090.** The standard install just works.

Blackwell is **sm_120** (compute capability 12.0). The official install (torch 2.1.1 +
cu121 wheels) was compiled with `sm_50…sm_90` only. On Blackwell, model loading
succeeds but any GPU forward pass hits:

```
CUDA error: no kernel image is available for execution on the device
```

To run on Blackwell, the entire haptic / hawor / mega-sam env GPU stack must be
rebuilt against `cu128 + sm_120`. This is real work — 1.5–2 hours of source builds and
patching cascade — so if you have an A100 available, use it.

### A.1 Required upgrades (haptic env example)

```bash
# 1) CUDA 12.8 toolchain
conda install -n haptic -c nvidia cuda-nvcc=12.8 cuda-cudart-dev=12.8 cuda-libraries-dev=12.8 -y

# 2) Replace cu121 GPU stack with cu128
conda activate haptic
pip uninstall -y torch torchvision torchaudio xformers pytorch3d detectron2 mmcv-full
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install xformers --index-url https://download.pytorch.org/whl/cu128
# Re-pin torchvision after xformers possibly upgrades torch:
pip install torchvision --index-url https://download.pytorch.org/whl/cu128 --upgrade

# 3) Source-build the libs that have no cu128 wheels (with sm_120 arch flag)
export CUDA_HOME=$CONDA_PREFIX
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="12.0"
pip install --no-build-isolation 'mmcv-full>=1.7.0,<1.8.0'
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2'
pip install --no-build-isolation 'git+https://github.com/facebookresearch/pytorch3d.git'
```

### A.2 Same upgrade for `hawor` and `mega_sam` envs

Apply the same steps. Plus:
- **DROID-SLAM** (in `third_party/hawor/thirdparty/DROID-SLAM/setup.py`): add
  `'-gencode=arch=compute_120,code=sm_120'` to nvcc args, then
  `python setup.py install` from inside hawor env.
- **mega-sam droid_slam**: `mega-sam/base/setup.py` similarly needs the sm_120 line.
  After build, **delete the stale `mega-sam/base/droid_slam/droid_backends.cpython-310-x86_64-linux-gnu.so`**
  (committed at older ABI) so import resolves to the freshly built one in
  site-packages.
- **`torch_scatter`** (mega_sam env): the prebuilt cu128 wheel exists for newer torch:
  ```bash
  pip install --force-reinstall torch_scatter \
    -f https://data.pyg.org/whl/torch-2.11.0+cu128.html
  ```

### A.3 xformers `memory_efficient_attention` lacks sm_120 kernels

Even with cu128 torch + xformers latest, `memory_efficient_attention` errors out:
```
`fa3F@0.0.0` ... requires device with capability <= (9, 0) but your GPU has capability (12, 0) (too new)
```
Wrap calls in `try/except` and fall back to `torch.nn.functional.scaled_dot_product_attention`
(torch 2.11 SDPA natively supports sm_120):

Three call sites we patched on RTX 5090:
- `third_party/haptic/haptic/models/components/pose_transformer.py:145`
- `mega-sam/UniDepth/unidepth/models/backbones/metadinov2/attention.py:79`
- `~/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/attention.py:94` ← easy to miss; this is `torch.hub` cache

UniDepth's metadinov2 also calls `xformers.ops.SwiGLU` and `xformers.components.attention.NystromAttention`.
Easiest: force `XFORMERS_AVAILABLE = False` in those modules (UniDepth has clean
PyTorch fallbacks). For NystromAttention specifically, the `xformers.components`
namespace was removed in 0.0.28 — provide a small SDPA-based stub class.

### A.4 PyTorch 2.6+ `weights_only=True` rejects HaWoR / HaPTIC checkpoints

The `torch.load` default flipped in 2.6, blocking pickled `OmegaConf` configs. Either
patch every `torch.load(...)` to add `weights_only=False`, or monkey-patch globally at
the script entry:
```python
import torch as _t
_orig = _t.load
_t.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})
```

### A.5 SAM3D `kaolin` / `spconv` workarounds

For SAM3D 3D reconstruction:
- **kaolin**: NVIDIA S3 max is cu121 / torch 2.4, no cu128. Build CPU-only from source —
  SAM3D only uses `kaolin.render.camera` / `kaolin.visualize`, no CUDA kernels needed:
  ```bash
  git clone --recurse-submodules https://github.com/NVIDIAGameWorks/kaolin
  cd kaolin
  FORCE_CUDA=0 IGNORE_TORCH_VER=1 pip install --no-build-isolation -e .
  ```
- **spconv**: PyPI has no `spconv-cu128`, but `spconv-cu126` works on sm_120 thanks to
  PTX forward-compat:
  ```bash
  pip install spconv-cu126
  ```
- **gsplat**: source build from the SAM3D-pinned commit + `TORCH_CUDA_ARCH_LIST="12.0"`.
- **flash_attn**: skip entirely. SAM3D defaults to `ATTN_BACKEND=sdpa` (env var) and
  works without flash_attn on Blackwell.

### A.6 Kernel mismatch / missing nvidia driver after kernel upgrade

After kernel updates, the nvidia DKMS module sometimes isn't rebuilt for the new
kernel. `nvidia-smi` returns:
```
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
```
Check `lsmod | grep nvidia` (empty = not loaded). If a previous kernel still has the
modules built (`ls /lib/modules/*/`), boot into that kernel via GRUB:
```bash
sudo grub-reboot 'Advanced options for Ubuntu>Ubuntu, with Linux <prev-kernel>-generic'
sudo reboot
```
Long-term fix: install `dkms` so `nvidia-driver-580` rebuilds on each kernel update.

### A.7 Reported working configuration

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

Patches accumulated for Blackwell are kept locally in submodule working trees and not
pushed to the main repo — apply them on a per-deployment basis if you're on Blackwell.
