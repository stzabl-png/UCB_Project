# Affordance2Grasp

**Learning robot grasp poses from human hand-object interaction videos.**

Pipeline: `Video → Depth + Pose → Human Contact Prior → Sim Filtering → Train Policy → Robot Grasp`

---

## Pipeline Philosophy

This pipeline is **object-agnostic by design**.

The PointNet++ model does not learn "how to grasp object X". Instead, it learns a general principle: **where humans tend to make contact on an object's surface, and where the force center lies**. This geometric pattern transfers across object categories.

**Why use so many different datasets?**
Each dataset contributes a different type of diversity:

| Dataset | Object Category | What it contributes |
|---------|----------------|---------------------|
| DexYCB / HO3D | YCB household items | High-quality multi-view contact data; used as evaluation benchmark |
| OakInk | 100 diverse household objects | Broadens object category coverage for better generalization |
| TACO Allocentric | 100+ tool-object pairs | Tool grasping patterns; multi-camera 3rd-person view |
| EgoDex | Manipulation task objects | First-person viewpoint; diverse action-driven grasping |
| TACO Ego | Tool-object interactions | Egocentric complement to TACO Allocentric |

**The 7 YCB objects in DexYCB are an evaluation benchmark, not a deployment limit.** The model can be applied to any object given its mesh. Adding more diverse datasets is an ongoing process — more data leads to stronger generalization.

---

## Pipeline Overview

```
┌──────────────────────────────── PHASE 1A: Third-Person Data ────────────────────────────────┐
│                                                                                              │
│  DexYCB / HO3D v3 / OakInk / TACO Allocentric                                              │
│     │                                                                                        │
│     ├─ Depth Pro (two-pass) ─────→ K.txt + depths.npz  (metric depth, ~0.2–8% K error)    │
│     ├─ HaPTIC ──────────────────→ third_mano/{ds}/{seq}.npz  (MANO verts, camera space)   │
│     ├─ FoundationPose ──────────→ obj_poses/{ds}/{seq}/ob_in_cam/*.txt                     │
│     └─ batch_align_mano_fp.py ──→ training_fp/{ds}/{obj}.hdf5  (human_prior per object)   │
│                                                                                              │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
                       ▲ PARALLEL — runs independently of Phase 1B ▲
 ┌──────────────────────────────── PHASE 1B: First-Person Data (Egocentric) ────────────────────┐
│                                                                                              │
│  EgoDex / TACO Ego                                                                          │
│     │                                                                                        │
│     ├─ calibrate_dataset_fx.py ─→ CALIB_FX  (SLAM opt-intr ×1.10, no GT, ~1% error)      │
│     ├─ MegaSAM ─────────────────→ egocentric_depth/{ds}/{seq}/  K + depth + cam_c2w       │
│     ├─ HaWoR ──────────────────→ ego_mano/{ds}/{seq}.npz  (MANO verts, world frame)       │
│     ├─ FoundationPose ──────────→ obj_poses_ego/{ds}/{seq}/ob_in_cam/*.txt                 │
│     └─ batch_align_ego_mano_fp.py → training_fp_ego/{ds}/{obj}.hdf5  (human_prior)        │
│                                                                                              │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
                       ▲ PARALLEL — runs independently of Phase 1A ▲
                    ┌──────────────────────────────────────────────┐
                    │  Both feed into Phase 2 independently        │
                    └──────────────────────────────────────────────┘
                                            │
 ┌──────────────────────────────── PHASE 2: Aggregate HumanPrior ──────────────────────────────┐
│                                                                                              │
│  training_fp/ + training_fp_ego/  ──→  data_hub/human_prior/{obj}.hdf5                    │
│     point_cloud (4096, 3) · normals (4096, 3) · human_prior (4096,) · force_center (3,)   │
│                                                                                              │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
┌──────────────────────────────── PHASE 3: Robot Ground Truth ────────────────────────────────┐
│                                                                                              │
│  human_prior ──→ Grasp Candidates ──→ Isaac Sim ──→ robot_gt  (success / fail per pose)   │
│                                                                                              │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
┌──────────────────────────────── PHASE 4: Train Policy ──────────────────────────────────────┐
│                                                                                              │
│  human_prior + robot_gt ──→ Build HDF5 ──→ Train PointNet++ ──→ Checkpoint                │
│  New object mesh ─────────→ Predict Affordance ──→ Grasp Pose ──→ Sim Verify              │
│                                                                                              │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Supported Datasets

### Phase 1A — Third-Person

| Dataset | Objects | Sequences | K error | Role | Download |
|---|---|---|---|---|---|
| **DexYCB** | 20 YCB | ~3,600 (6 cams) | ~0.2% | Evaluation benchmark + training | [dex-ycb.github.io](https://dex-ycb.github.io) |
| **HO3D v3** | 8 YCB | 68 | ~1.5% | Additional YCB contact data | [tugraz.at HO3D](https://www.tugraz.at/institute/icg/research/team-lepetit/research-projects/hand-object-3d-pose-annotation) |
| **OakInk v1** | 100 | 778 | ~5.7% | Expands object diversity (100 categories) | [oakink.net](https://oakink.net) |
| **TACO Allocentric** | 100+ | 2,300+ (12 cams) | ~5–8% | Tool grasping; multi-camera diversity | [taco-group.github.io](https://taco-group.github.io) → `Allocentric_RGB_Videos` |

### Phase 1B — First-Person (Egocentric)

| Dataset | Camera | Sequences | K error | Role | Download |
|---|---|---|---|---|---|
| **EgoDex** | Apple Vision Pro | 3,051 | ~1% | First-person grasping patterns; action diversity | [egodex.cs.columbia.edu](https://egodex.cs.columbia.edu) |
| **TACO Ego** | GoPro 71° | 2,311 | ~1% | Egocentric tool grasping complement | [taco-group.github.io](https://taco-group.github.io) → `Egocentric_RGB_Videos` |

### Dataset Acquisition Checklist

> Use this checklist to confirm what you have before starting. Items marked **Required for gtfree** must be present to run Phase 4 training without Isaac Sim.

| Dataset | Size | Required for gtfree | Status | Notes |
|---------|------|-------------------|--------|-------|
| **DexYCB** | ~250 GB | ✅ Yes | `dex-ycb-20210415.tar.gz` | Extract to `RawData/ThirdPersonRawData/dexycb/` |
| **HO3D v3** | ~60 GB | ✅ Yes | ❌ Need to download | [tugraz.at](https://www.tugraz.at/institute/icg/research/team-lepetit/research-projects/hand-object-3d-pose-annotation) |
| **OakInk v1** | ~80 GB | ✅ Yes | ❌ Need to download | [oakink.net](https://oakink.net) |
| **TACO Allocentric** | ~120 GB | ⚠️ Optional | ❌ Need to download | Skip if storage is limited |
| **EgoDex** | ~30 GB | ⚠️ Optional | ❌ Need to download | [egodex.cs.columbia.edu](https://egodex.cs.columbia.edu) |
| **TACO Ego** | ~24 GB | ⚠️ Optional | ❌ Need to download | [taco-group.github.io](https://taco-group.github.io) |
| **MANO** | ~50 MB | ✅ Yes | `mano_v1_2.zip` | See MANO setup below |
| **SMPL-H** | ~300 MB | ✅ Yes | `smplh.tar.xz` | See MANO setup below |
| **Model weights** | ~12 GB | ✅ Yes | ❌ Run `setup_weights.py` | Automated download |
| **Ego masks** | ~70 MB | ✅ Yes (1B) | ❌ Run `setup_weights.py --tool egomasks` | Automated download |

> **Minimum viable dataset for gtfree training:** DexYCB + HO3D v3 + OakInk v1 (~390 GB).
> TACO and EgoDex extend coverage but are not required for an initial working model.

---

## Prerequisites

### Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| **GPU** | 1× RTX 3090 / A100 (24 GB VRAM) | 8× A100 80GB |
| **Storage** | 500 GB free (raw data + processed) | 2 TB (all datasets) |
| **RAM** | 64 GB | 128 GB |
| **OS** | Ubuntu 20.04+ (Linux only — case-sensitive paths) | Ubuntu 22.04 |

> **Storage breakdown:**
> EgoDex 30 GB · TACO Ego 24 GB · DexYCB ~250 GB · OakInk ~80 GB · HO3D ~60 GB
> Processed outputs (depth, poses, priors) add another ~100 GB.
> **Mount a dedicated `/data` volume before starting.** Do NOT install to `/home`.

---

## Multi-GPU Parallelization

All batch scripts support `--start N --end N` (or `--seqs`) for sharding across GPUs.
With 8 GPUs, Phase 1A + 1B can be completed in **~2–3 days** instead of ~10 days.

### Pattern: 8-GPU sharding

```bash
# Split N total sequences across 8 GPUs
# GPU 0: sequences 0..374,  GPU 1: 375..749,  GPU 2: 750..1124 ...
TOTAL=3000
CHUNK=$((TOTAL / 8))

for GPU in {0..7}; do
  START=$((GPU * CHUNK))
  END=$((START + CHUNK))
  CUDA_VISIBLE_DEVICES=$GPU python data/batch_megasam.py \
      --dataset egodex --start $START --end $END &
done
wait
```

### Per-script flags

| Script | Shard flag | Example |
|--------|-----------|---------|
| `batch_megasam.py` | `--start N --end N` | `--start 0 --end 375` |
| `batch_hawor.py` | `--seqs task1/0,task1/1,...` | comma-separated seq IDs |
| `batch_obj_pose_ego.py` | `--seq substring` | `--seq add_remove` |
| `batch_obj_pose.py` | `--seq substring` | `--seq dexycb` |
| `batch_depth_pro.py` | `--start N --end N` | same as MegaSAM |

### Time estimates (8× A100)

| Phase | Single GPU | 8× A100 |
|-------|-----------|---------|
| Phase 1A (4 datasets) | ~120 h | **~15 h** |
| Phase 1B EgoDex+TACO | ~150 h | **~19 h** |
| Phase 2 (aggregate) | ~2 h | ~2 h (single) |
| Phase 4 (training) | ~6 h | ~1 h |

---

## Setup

### 0. Pre-flight System Check

Run before installing anything to catch environment issues early:

```bash
nvidia-smi | grep "CUDA Version"   # check max supported CUDA (driver)
nvcc --version                      # check compiler CUDA version (must match torch build)
df -h /data /home                   # confirm > 500 GB free
free -h                             # confirm > 32 GB RAM
python3 --version                   # confirm Python ≥ 3.9
cmake --version                     # confirm cmake present (needed for FP build)
```

> **CUDA Version Alignment** — critical to avoid build failures:
> ```bash
> # Check your driver's max supported CUDA:
> nvidia-smi | grep "CUDA Version"   # e.g. "CUDA Version: 12.2"
> # Choose your torch build accordingly:
> #   Driver ≥ 12.0  →  use cu121
> #   Driver 11.x    →  use cu118
> # Install matching torch:
> pip install torch==2.1.1+cu121 torchvision --index-url https://download.pytorch.org/whl/cu121
> # or for 11.x:
> pip install torch==2.1.1+cu118 torchvision --index-url https://download.pytorch.org/whl/cu118
> ```
> Also install cmake and ninja (required for FoundationPose CUDA build):
> ```bash
> conda install -c conda-forge cmake ninja -y
> ```

> **GPU Architecture Compatibility** — check before choosing torch version:
> ```bash
> nvidia-smi --query-gpu=name,compute_cap --format=csv
> ```
>
> | Architecture | Examples | sm | torch 2.1.1 | Action |
> |---|---|---|---|---|
> | Ampere | A100, A30, RTX 3090 | sm_80/86 | ✅ | Use cu118 or cu121 |
> | Ada Lovelace | RTX 4080/4090, RTX 6000 Ada | sm_89 | ✅ | Use cu121 |
> | Blackwell | RTX 5090, RTX 6000 Pro (2025) | sm_120 | ❌ | See T12 below |
>
> If `compute_cap` ≥ `12.0` you have a **Blackwell GPU** — see **Troubleshooting T12**.

### 1. Clone (with all dependencies)

```bash
# --recursive pulls HaWoR, MegaSAM, and HaPTIC code automatically
git clone --recursive https://github.com/stzabl-png/UCB_Project.git Affordance2Grasp
cd Affordance2Grasp
```

> If you already cloned without `--recursive`, run:
> ```bash
> git submodule update --init --recursive
> ```

> **⚠️ Required: apply patches after submodule init**
> The `third_party/haptic` submodule is pinned to the upstream public commit.
> Our pipeline additions (metric-scale intrinsics injection) are maintained as patches.
> **You must apply them every time after `git submodule update`:**
> ```bash
> cd third_party/haptic
> git apply ../../patches/haptic-intrinsics-fix.patch
> cd ../..
> ```
> Without this, `batch_haptic.py` will crash with:
> `TypeError: parse_det_seq() got an unexpected keyword argument 'intrinsics'`

### 2. Download Model Weights

All weights are hosted on HuggingFace. One script downloads everything:

```bash
pip install huggingface_hub   # if not already installed
python setup_weights.py       # downloads FP + HaWoR + HaPTIC + MegaSAM + DepthPro (~12 GB total)
```

To download a specific tool only:
```bash
python setup_weights.py --tool fp       # FoundationPose only (248 MB)
python setup_weights.py --tool hawor    # HaWoR only (3.6 GB)
python setup_weights.py --tool haptic   # HaPTIC only (3.8 GB)
python setup_weights.py --tool megasam  # MegaSAM only (21 MB)
```

> **⚠️ MANO (license required — manual download)**
>
> MANO cannot be redistributed. Register and download from [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de/).
> You need **all of the following files** from the MANO package:
>
> | File | Source |
> |------|--------|
> | `MANO_RIGHT.pkl` | `mano_v1_2.zip` → `models/` |
> | `MANO_LEFT.pkl` | `mano_v1_2.zip` → `models/` |
> | `MANO_UV_right.obj` | MANO extras / SMPL-X data pack (see note below) |
> | `MANO_UV_left.obj` | MANO extras / SMPL-X data pack |
>
> > **Where to get `MANO_UV_*.obj`:** These UV mesh files are **not** in `mano_v1_2.zip`.
> > Download the **SMPL-X** or **SMPL+H** package from [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de/)
> > or look for "Additional body model files" on the MANO download page.
> > The files are typically under `body_models/mano/` or `extras/` in those archives.
>
> Place all files in **two locations**:
>
> **HaPTIC** (needs UV mesh + pkl):
> ```
> third_party/haptic/assets/mano/
>     MANO_RIGHT.pkl
>     MANO_LEFT.pkl
>     MANO_UV_right.obj        ← required for hand UV rendering
>     MANO_UV_left.obj
> ```
>
> **HaWoR** (pkl only):
> ```
> third_party/hawor/_DATA/data/mano/
>     MANO_RIGHT.pkl
> third_party/hawor/_DATA/data_left/mano_left/
>     MANO_LEFT.pkl
> ```
>
> Quick copy commands (after extracting mano_v1_2.zip and obtaining UV obj files):
> ```bash
> MANO=/path/to/mano_v1_2/models   # contains .pkl files
> UV=/path/to/mano_extras           # contains MANO_UV_*.obj files
>
> # HaPTIC (pkl + UV mesh) — default path: third_party/haptic/assets/mano/
> mkdir -p third_party/haptic/assets/mano
> cp $MANO/MANO_RIGHT.pkl $MANO/MANO_LEFT.pkl third_party/haptic/assets/mano/
> cp $UV/MANO_UV_right.obj $UV/MANO_UV_left.obj third_party/haptic/assets/mano/
>
> # HaWoR (pkl only)
> mkdir -p third_party/hawor/_DATA/data/mano
> mkdir -p third_party/hawor/_DATA/data_left/mano_left
> cp $MANO/MANO_RIGHT.pkl third_party/hawor/_DATA/data/mano/
> cp $MANO/MANO_LEFT.pkl  third_party/hawor/_DATA/data_left/mano_left/
> ```
>
> **⚠️ `MANO_UV_right.obj` is NOT optional.** HaPTIC's `ManopthWrapper` loads it at
> startup (before any inference). Missing this file causes an immediate crash even if
> you never use UV textures. There is no workaround except obtaining the file.
>
> The HaPTIC MANO directory defaults to `third_party/haptic/assets/mano/` and is
> controlled by `config.HAPTIC_MANO_DIR`. To use a custom path without copying files:
> ```bash
> export HAPTIC_MANO_DIR=/path/to/your/mano_v1_2/models
> # That directory must contain: MANO_RIGHT.pkl, MANO_LEFT.pkl,
> #   MANO_UV_right.obj, MANO_UV_left.obj
> ```


### 3. Conda Environments

> **First time only — install Miniconda if not present:**
> ```bash
> # Install Miniconda to your large disk (NOT /home if space is limited)
> wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
> bash Miniconda3-latest-Linux-x86_64.sh -b -p /data/miniconda3
> echo 'export PATH="/data/miniconda3/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
> # Accept the Terms of Service (required for new installs)
> conda tos accept
> ```

> **If your data is on a separate large disk (e.g. /data/5TB):**
> ```bash
> # Create symlink so all scripts find data_hub automatically
> mkdir -p /data/5TB/Affordance2Grasp
> ln -s /data/5TB/Affordance2Grasp data_hub
> ```

```bash
# Environment A — Depth Pro  (Phase 1A Step 1)
conda create -n depth-pro python=3.9 -y
conda activate depth-pro
# Install from submodule (avoids Apple CDN firewall issues)
pip install -e third_party/ml-depth-pro
pip install natsort tqdm pillow numpy h5py
# Download checkpoint via HuggingFace (Apple CDN may be blocked on remote servers)
python setup_weights.py --tool depthpro   # places depth_pro.pt in third_party/ml-depth-pro/checkpoints/

# Environment C — mega_sam  (Phase 1B Steps 1-2)
# Follow mega-sam/README for setup

# Environment D — hawor  (Phase 1B Step 3)
# ⚠️  DO NOT follow third_party/hawor/README blindly!
#     HaWoR's own README specifies torch 1.13+cu117 which is WRONG for our stack.
#     We use torch 2.1.0+cu121. See Section 3d below for the correct setup.
```

### 3b. Environment B — `bundlesdf` (main environment)

> **Used for:** HaPTIC (Step 2), FoundationPose (Steps 3, E5), contact alignment (Steps 4, E6), Sim (Phase 3), training (Phase 4).

**Step 1 — Create the environment**
```bash
# ⚠️  Must be Python 3.10 — NOT 3.9
# pytorch3d 0.7.5 pre-built wheel is py310 only; nvdiffrast ABI also requires 3.10
conda create -n bundlesdf python=3.10 -y
conda activate bundlesdf
conda install -c conda-forge cmake ninja -y  # ← required for FoundationPose CUDA build
pip install "numpy<2.0"                      # ← pin FIRST before other packages
pip install trimesh scipy h5py opencv-python natsort tqdm
pip install "fast-simplification>=0.1.6"     # ← CRITICAL: without this, contact alignment takes 23 h not 15 min
```

**Step 2 — Install HaPTIC**
```bash
conda create -n haptic python=3.10 -y
conda activate haptic
cd third_party/haptic
bash scripts/one_click.sh   # downloads HaPTIC weights automatically
cd ../..
```

**Step 3 — Clone and build FoundationPose**

> FoundationPose requires compiling CUDA C++ extensions. A simple `pip install` is not sufficient.

```bash
# System dependencies (requires sudo — only needed once per machine)
sudo apt-get install -y libboost-all-dev libomp-dev build-essential git

# cmake must be < 4.0: cmake ≥ 4.0 removes FindBoost, breaking FP's C++ build
pip install "cmake<4.0" ninja pybind11

# Clone and pin to a tested commit (main branch may have breaking changes)
git clone https://github.com/NVlabs/FoundationPose.git third_party/FoundationPose
cd third_party/FoundationPose
git checkout 25e225a   # last tested commit — update if you verify a newer one works

# Install Eigen3 3.4.0 (required for C++ build)
conda install conda-forge::eigen=3.4.0 -y
export CMAKE_PREFIX_PATH="$CONDA_PREFIX:$CMAKE_PREFIX_PATH"

# Install Python dependencies
pip install -r requirements.txt

# Install NVDiffRast — clone locally to avoid git metadata build issues
git clone --depth 1 https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast_src
pip install /tmp/nvdiffrast_src --no-build-isolation

# Build CUDA C++ extensions
# ⚠️  conda's gcc is a cross-compiler that conflicts with system glibc headers.
# Always force system gcc for this step:
bash build_all_conda.sh
# If build_all_conda.sh fails with 'bits/timesize.h not found', use system gcc:
# (see Troubleshooting T11 below)

# Register FP_ROOT so pipeline scripts can find it
echo 'export FP_ROOT="'"$(pwd)"'"' >> ~/.bashrc
export FP_ROOT="$(pwd)"
```

**Step 4 — Download model weights**

Weights are hosted on our HuggingFace repo. Run from inside the FoundationPose directory:

```bash
cd /path/to/FoundationPose

python3 -c "
from huggingface_hub import hf_hub_download, snapshot_download
import shutil, os

repo = 'UCBProject/Affordance2Grasp-Data'

for folder in ['2023-10-28-18-33-37', '2024-01-11-20-02-45']:
    snapshot_download(
        repo_id=repo,
        repo_type='dataset',
        allow_patterns=f'FoundationPose/weights/{folder}/*',
        local_dir='.',
    )
    # Move into weights/
    src = f'FoundationPose/weights/{folder}'
    dst = f'weights/{folder}'
    if os.path.exists(src):
        shutil.move(src, dst)
        print(f'✅ {dst}')
"
```

After download, your `weights/` folder should contain:
```
weights/
├── 2023-10-28-18-33-37/model_best.pth   ← refiner
└── 2024-01-11-20-02-45/model_best.pth   ← scorer
```

**Step 5 — Set path in `config.py`**
```python
FP_ROOT = "/path/to/FoundationPose"
```

### 3. Configure paths in `config.py`

```python
DATA_HUB   = "/path/to/data_hub"
HAPTIC_DIR = "/path/to/HaPTIC"
FP_ROOT    = "/path/to/FoundationPose"
```

### 3c. Environment D — `hawor` (Phase 1B Step E2)

> ⚠️ **Do NOT follow `third_party/hawor/README.md` for the conda setup.**
> HaWoR's own README specifies `torch==1.13+cu117`. That is its original dev environment.
> We run on **CUDA 12.1** and need `torch==2.1.0+cu121` to match the rest of the pipeline.

```bash
conda create -n hawor python=3.10 -y
conda activate hawor

# 1. PyTorch 2.1.0 + CUDA 12.1  (NOT torch 1.13 as HaWoR README says)
pip install torch==2.1.0 torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu121

# 2. torch-scatter — must use pre-built wheel matching torch+CUDA exactly
#    Building from source against the wrong CUDA (11.7 vs 12.1) will fail
pip install torch-scatter \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# 3. pytorch3d 0.7.5 pre-built for py310+cu121+torch2.1.0
pip install pytorch3d \
    -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt210/download.html

# 4. HaWoR dependencies (from submodule, skip their torch/scatter lines)
cd third_party/hawor
pip install -r requirements.txt   # install remaining deps (excludes torch/scatter above)
cd ../..

# 5. Pin pytorch-lightning + torchmetrics AFTER requirements.txt
#    requirements.txt may pull incompatible versions; these pins override them.
#    ⚠️  lightning 2.3+ has breaking changes; torchmetrics must match lightning 2.2.x
pip install pytorch-lightning==2.2.4 torchmetrics==1.4.0

# 6. Verify all key packages
python -c "
import torch; print('torch:', torch.__version__)
import torch_scatter; print('torch_scatter:', torch_scatter.__version__)
import pytorch3d; print('pytorch3d:', pytorch3d.__version__)
import pytorch_lightning as pl; print('pytorch_lightning:', pl.__version__)
import torchmetrics; print('torchmetrics:', torchmetrics.__version__)
"
# Expected:
# torch: 2.1.0+cu121
# torch_scatter: 2.1.2+pt21cu121
# pytorch3d: 0.7.5
# pytorch_lightning: 2.2.4
# torchmetrics: 1.4.0
```

### 4. Data layout

```
data_hub/
├── RawData/
│   ├── ThirdPersonRawData/
│   │   ├── dexycb/                      ← DexYCB raw
│   │   ├── ho3d_v3/                     ← HO3D v3 raw
│   │   ├── oakink_v1/                   ← OakInk v1 raw
│   │   └── taco/
│   │       └── Allocentric_RGB_Videos/  ← TACO Allocentric (download from taco-group.github.io)
│   └── EgoRawData/
│       ├── egodex/test/                 ← EgoDex raw
│       └── taco/
│           └── Egocentric_RGB_Videos/   ← TACO Ego (download from taco-group.github.io)
└── ProcessedData/                       ← generated by pipeline steps below
```

### 5. Download preprocessed assets (HuggingFace)

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='StZaBL/Affordance2Grasp-ProcessedData',
    repo_type='dataset',
    local_dir='data_hub/ProcessedData',
)
"
```

Includes: YCB meshes (`obj_meshes/ycb/`) + FoundationPose init masks (`obj_recon_input/ycb/`)

---

## Phase 1A — Third-Person Pipeline

### Step 1 · Depth Pro — Metric Depth + Camera Intrinsics

**Two-pass self-calibration (no GT required):** Pass 1 estimates fx per sequence → global median → Pass 2 fixes fx and estimates metric depth only.

```bash
conda activate depth-pro

# DexYCB — one camera at a time (6 cameras total)
for CAM in 841412060263 840412060917 932122060857 836212060125 932122061900 932122062010; do
    python data/batch_depth_pro.py --dataset dexycb --cam $CAM --two-pass --max-frames 150
done

# HO3D v3
python data/batch_depth_pro.py --dataset ho3d_v3 --two-pass --max-frames 150

# OakInk v1
python data/batch_depth_pro.py --dataset oakink --two-pass --max-frames 150

# TACO Allocentric (requires Allocentric_RGB_Videos downloaded)
python data/batch_depth_pro.py --dataset taco_allocentric --two-pass --max-frames 150
```

Output: `data_hub/ProcessedData/third_depth/{dataset}/{seq_id}/`
- `K.txt` — 3×3 intrinsics
- `depths.npz` — (N, H, W) float32 metric depth in metres

---

### Step 2 · HaPTIC — MANO Hand Estimation

```bash
conda activate haptic

python data/batch_haptic.py --dataset dexycb           --only-with-depth-k
python data/batch_haptic.py --dataset ho3d_v3          --only-with-depth-k
python data/batch_haptic.py --dataset oakink           --only-with-depth-k
python data/batch_haptic.py --dataset taco_allocentric --only-with-depth-k
```

Output: `data_hub/ProcessedData/third_mano/{dataset}/{seq_id}.npz`
- `verts_dict` — MANO vertices (778, 3) per frame, camera space

---

### Step 3 · FoundationPose — Object 6D Pose Tracking

```bash
conda activate bundlesdf

python tools/batch_obj_pose.py --dataset dexycb
python tools/batch_obj_pose.py --dataset ho3d_v3
python tools/batch_obj_pose.py --dataset oakink
python tools/batch_obj_pose.py --dataset taco_allocentric
```

Output: `data_hub/ProcessedData/obj_poses/{dataset}/{seq_id}/ob_in_cam/{frame}.txt`
- 4×4 object-in-camera transform per frame

---

### Step 4 · Contact Label Generation — Human Prior

> **Dependency:** install `fast-simplification` first — without it, mesh simplification (664k→5k verts) is skipped and DexYCB takes **23 h** instead of **15 min**.

**Alignment principle:** 3D distance in camera space, with depth-ratio correction (`scale_d = Z_obj/Z_hand`) to compensate HaPTIC weak-perspective depth overestimation (~1.3×). Guard: `scale_d ∈ [0.3, 3.0]`.

```bash
conda activate bundlesdf

python data/batch_align_mano_fp.py --dataset dexycb
python data/batch_align_mano_fp.py --dataset ho3d_v3
python data/batch_align_mano_fp.py --dataset oakink
python data/batch_align_mano_fp.py --dataset taco_allocentric
```

Output: `data_hub/ProcessedData/training_fp/{dataset}/{obj}.hdf5`
- `point_cloud` (4096, 3) · `normals` (4096, 3) · `human_prior` (4096,) · `force_center` (3,)

---

## Phase 1B — First-Person (Egocentric) Pipeline

### Step E0 · Calibrate Camera Intrinsics (once per dataset)

> **Fully automatic, no GT required.** Samples N sequences, runs DROID-SLAM opt-intr, filters 2σ outliers, applies ×1.10 systematic correction, and auto-patches `CALIB_FX` in `batch_megasam.py`.

```bash
cd mega-sam

# EgoDex (Apple Vision Pro 104°, ~15 min)
conda run --no-capture-output -n mega_sam \
  python -u ../data/calibrate_dataset_fx.py --dataset egodex --n 15 \
  2>&1 | tee ../output/calib_egodex.log

# TACO Ego (GoPro 71°, ~30 min)
conda run --no-capture-output -n mega_sam \
  python -u ../data/calibrate_dataset_fx.py --dataset taco --n 15 \
  2>&1 | tee ../output/calib_taco.log
```

**Pre-calibrated values** (already set — skip this step if using EgoDex / TACO Ego):
- `CALIB_FX["egodex"] = 249.4` px @640px  (~1% error)
- `CALIB_FX["taco"]   = 455.3` px @640px  (~1% error)

---

### Step E1 · MegaSAM — Metric Depth + Camera Poses

```bash
cd mega-sam

# EgoDex (3,051 sequences — use --resume to continue after interruption)
conda run --no-capture-output -n mega_sam \
  python -u data/batch_megasam.py --dataset egodex --resume \
  2>&1 | tee output/megasam_egodex.log

# TACO Ego (2,311 sequences)
conda run --no-capture-output -n mega_sam \
  python -u data/batch_megasam.py --dataset taco --resume \
  2>&1 | tee output/megasam_taco.log
```

Output per sequence: `data_hub/ProcessedData/egocentric_depth/{dataset}/{seq}/`
- `depth.npz` — (60, H, W) float32 metric depth
- `K.npy` — 3×3 camera intrinsic
- `cam_c2w.npy` — (60, 4, 4) camera-to-world transforms

---

### Step E2 · HaWoR — MANO Hand Estimation (egocentric)

> Each sequence runs in an isolated subprocess — no CUDA conflicts, failures are isolated.

```bash
conda activate bundlesdf   # dispatcher only; subprocess uses hawor env

python data/batch_hawor.py --dataset egodex
python data/batch_hawor.py --dataset taco_ego

# Options: --seq "task/ep" --force-rerun --max-frames 120
```

Output: `data_hub/ProcessedData/ego_mano/{dataset}/{seq_id}.npz`
- `right_verts` (T, 778, 3) · `left_verts` (T, 778, 3) — world frame
- `R_w2c` (T, 3, 3) · `t_w2c` (T, 3) — world→camera per frame

---

### Step E3 · Object Scale Estimation (once per object)

> Uses MegaSAM metric depth + SAM2 object mask to compute `scale_factor = d_real / d_mesh` to correct SAM3D reconstruction scale.

```bash
conda run -n hawor python data/estimate_obj_scale_ego.py
# or single object:
conda run -n hawor python data/estimate_obj_scale_ego.py --obj assemble_tile
```

Output: `data_hub/ProcessedData/obj_meshes/egocentric/{obj}/scale.json`

---

### Step E4 · SAM2 Mask (manual, once per object)

> **Skip this step** if the object already has a mask in `obj_recon_input/egocentric/`.
> All objects from existing datasets (EgoDex tasks, TACO Ego) already have masks.
> Only needed when adding a **new object category** not previously seen.

Annotate the object mask on one representative frame using the SAM2 GUI, then save to:
```
data_hub/ProcessedData/obj_recon_input/egocentric/{obj}/0.png
```

To check which objects already have masks:
```bash
ls data_hub/ProcessedData/obj_recon_input/egocentric/
```

---

### Step E5 · FoundationPose — Object 6D Pose (egocentric)

```bash
conda run -n bundlesdf python tools/batch_obj_pose_ego.py --dataset egodex
conda run -n bundlesdf python tools/batch_obj_pose_ego.py --dataset taco_ego
```

Output: `data_hub/ProcessedData/obj_poses_ego/{dataset}/{seq}/ob_in_cam/{frame}.txt`

---

### Step E6 · Contact Label Generation (egocentric)

> Sequences registered in `tools/egodex_sequence_registry.json` (EgoDex) or `tools/taco_ego_sequence_registry.json` (TACO Ego). See registry JSON for format.

```bash
conda activate bundlesdf

python data/batch_align_ego_mano_fp.py --dataset egodex
python data/batch_align_ego_mano_fp.py --dataset taco_ego
```

Output: `data_hub/ProcessedData/training_fp_ego/{dataset}/{obj}.hdf5`
- Same format as Phase 1A: `point_cloud` · `normals` · `human_prior` · `force_center`

---

## Phase 2 — Aggregate HumanPrior

Merge all Phase 1A + 1B outputs into a unified per-object prior:

```bash
conda activate bundlesdf
python data/aggregate_prior.py
```

Output: `data_hub/human_prior/{obj}.hdf5`
- `point_cloud` (4096, 3) · `normals` (4096, 3) · `human_prior` (4096,) · `force_center` (3,)

**Verify quality:**
```bash
python3 - <<'EOF'
import h5py, glob, numpy as np
base = "data_hub/human_prior"
print(f"{'Object':<28} {'max_hp':>8} {'cov>0.1':>9} {'cov>0.5':>9}")
for p in sorted(glob.glob(f"{base}/*.hdf5")):
    name = p.split("/")[-1].replace(".hdf5","")
    with h5py.File(p) as f: hp = f["human_prior"][()]
    print(f"  {name:<26} {hp.max():>8.3f} {(hp>0.1).mean()*100:>8.1f}% {(hp>0.5).mean()*100:>8.1f}%")
EOF
```

Expected: `max_hp ≥ 0.7`, `cov(>0.1) = 100%` for all objects.

---

## Phase 3 — Robot Ground Truth (Isaac Sim) — Optional

> **This phase is optional for training.** The current model (`checkpoints_gtfree_v2`, F1=0.642)
> was trained in **gtfree mode** (HumanPrior only, no robot simulation labels).
> Skip to Phase 4 if you want to train without Isaac Sim.
>
> Run Phase 3 only if you want to generate `robot_gt` labels to potentially improve accuracy.
>
> **Isaac Sim requirements:**
> - NVIDIA Omniverse account (free registration at [developer.nvidia.com](https://developer.nvidia.com/omniverse))
> - ~30 GB disk space
> - RTX-class GPU
> - Set: `export ISAAC_SIM_PATH=/path/to/isaac-sim`

### Step 5 · Sample Grasp Candidates

```bash
conda activate bundlesdf
python tools/random_grasp_sampler.py --all
```

Output: `data_hub/grasps/{obj}/candidates.hdf5`

---

### Step 6 · Run Grasp Simulation

```bash
# Requires Isaac Sim. Set path: export ISAAC_SIM_PATH=/path/to/isaac-sim

python sim/run_grasp_sim.py \
    --hdf5 data_hub/grasps/{obj}/candidates.hdf5 \
    --headless \
    --save-result
```

Output: `robot_gt` success/failure labels per grasp candidate

---

## Phase 4 — Train Policy

### Step 7 · Build Training Dataset

```bash
conda activate bundlesdf
python data/build_dataset.py --num_points 4096 --augment 3
```

Output: `dataset/train.hdf5`, `dataset/val.hdf5`
- `point_cloud` (N,3) · `normals` (N,3) · `human_prior` (N,) · `robot_gt` (N,) · `force_center` (3,)

---

### Step 8 · Train Affordance Policy

```bash
conda activate bundlesdf
python -m model.train \
    --epochs 200 \
    --batch_size 16 \
    --lr 0.001 \
    --fc_lambda 10.0
```

Checkpoint saved to `checkpoints/`

---

### Step 9 · Predict Affordance on New Object

```bash
conda activate bundlesdf
python -m inference.grasp_pose --mesh path/to/object.obj
```

Output: `grasps/{obj}/affordance_grasp.hdf5`

---

### Step 10 · Full Pipeline Verify

```bash
# Predict + sim verify
python run.py --mesh path/to/object.obj

# Prediction only (no sim):
python run.py --mesh path/to/object.obj --no-sim
```

---

## Estimated Runtime (RTX 4080 SUPER, single GPU)

| Step | DexYCB (6 cams) | HO3D v3 | OakInk | TACO Alloc | EgoDex | TACO Ego |
|---|---|---|---|---|---|---|
| 1/E0 · Depth Pro / SLAM calib | ~21 h | ~1.7 h | ~4 h | ~8 h | ~30 min | ~30 min |
| 2/E1 · HaPTIC / MegaSAM | ~48 h | ~1 h | ~4 h | ~10 h | ~50 h | ~35 h |
| 3/E2 · FP / HaWoR | ~8 h | ~15 min | ~3 h | ~6 h | ~40 h | ~30 h |
| 4/E6 · Contact Labels | **~15 min** † | **~1 min** | **~5 min** | **~10 min** | **~1 h** | **~45 min** |
| 5–6 · Sim (robot GT) | ~varies by object count | | | | | |
| 7–8 · Train Policy | ~2 h (all data combined) | | | | | |

† Requires `fast-simplification`; without it degrades to ~23 h.

> **Multi-GPU tip:** Steps 1–3 are embarrassingly parallel across cameras/sequences.
> With 8× A100, Phase 1A for all datasets completes in **< 4 h**.

---

## Verified Results

| Config | Result |
|---|---|
| DexYCB cam `841412060263` contact quality | coverage=100%, diverged=0 |
| Depth Pro fx — DexYCB (two-pass self-calib) | ~+0.2% residual |
| Depth Pro fx — HO3D v3 (two-pass) | ~−1.5% residual |
| Depth Pro fx — OakInk v1 (two-pass, 23 seqs) | ~−5.7% residual |
| SLAM opt-intr fx — EgoDex AVP (×1.10 corrected) | **~1%** |
| SLAM opt-intr fx — TACO Ego GoPro (×1.10 corrected) | **~1%** |
| FoundationPose tracking success rate | **100%** |
| DexYCB human_prior quality (`max_hp`) | **≥ 0.76** all 27 objects |
| HO3D v3 human_prior quality (`max_hp`) | **≥ 0.82** all 7 objects |

---

## Visualization

```bash
conda activate bundlesdf

# Contact heatmap (single object)
python tools/vis_contact_heatmap.py --dataset dexycb --obj 003_cracker_box

# Batch all objects
python tools/vis_contact_heatmap.py --dataset dexycb --batch --out output/vis_contact_heatmap

# Egocentric contact
python tools/vis_ego_contact.py --obj assemble_tile
```

---

## Key Design Decisions

| | Third-Person | First-Person |
|---|---|---|
| Depth source | Depth Pro (single image) | MegaSAM (SLAM, metric) |
| Hand estimation | HaPTIC → camera space | HaWoR → world frame |
| K estimation | Depth Pro two-pass self-calib | SLAM opt-intr ×1.10 |
| Scale correction | `estimate_obj_scale.py` (Depth Pro Z) | `estimate_obj_scale_ego.py` (MegaSAM Z) |
| Object pose | `batch_obj_pose.py` | `batch_obj_pose_ego.py` |
| Contact alignment | 3D distance + depth-ratio rescaling | 2D pixel distance (SLAM scale cancels) |
| Output HDF5 | `human_prior/{obj}.hdf5` | same format — directly mergeable |

---

## Troubleshooting

Common issues encountered during fresh deployment:

### T1 — `conda tos accept` error on `conda create`
```
CondaError: Please accept the Terms of Service
```
**Fix:** `conda tos accept` then retry.

### T2 — `fast-simplification` crashes with numpy 2.x
```
TypeError: ... numpy incompatible
```
**Fix:** Pin numpy before installing:
```bash
pip install "numpy<2.0"
pip install "fast-simplification>=0.1.6"
```

### T3 — `setup_weights.py` shows no output / seems stuck
The script is working — HuggingFace downloads large files silently. Check disk activity:
```bash
watch -n 2 'ls -lh third_party/*/weights/ third_party/*/checkpoints/ 2>/dev/null | grep -v total'
```
Expect ~12 GB total. Takes 5–30 min depending on connection speed.

### T4 — HaPTIC detectron2 build fails
```
ModuleNotFoundError: No module named 'pkg_resources'
# or: ImportError: cannot import name 'packaging' from 'pkg_resources'
```
**Root cause:** `setuptools ≥ 70` removes `pkg_resources`, which `torch` cpp_extension uses during detectron2 compilation.

**Fix:**
```bash
conda activate haptic
pip install --force-reinstall "setuptools<70"   # ← must do BEFORE detectron2
sudo apt-get install -y gcc g++ build-essential

# Clone locally (more reliable than pip install git+URL)
git clone --depth 1 https://github.com/facebookresearch/detectron2.git /tmp/detectron2_src
pip install /tmp/detectron2_src --no-build-isolation
```
> **Note:** Any subsequent `pip install` may upgrade setuptools back to ≥70.
> Always re-run `pip install --force-reinstall "setuptools<70"` after installing new packages in the `haptic` env.

### T13 — HaPTIC xformers: `No module named 'xformers'` or version conflict

```
ModuleNotFoundError: No module named 'xformers'
# or after installing latest xformers:
ERROR: xformers 0.0.35 requires torch >= 2.10, but you have torch 2.1.1
```

**Root cause:** `xformers` must be pinned to match the exact `torch` version.
The latest `xformers` (≥ 0.0.29) requires `torch ≥ 2.4`, which is incompatible with haptic's `torch 2.1.1`.

**Verified working combination** (confirmed on RTX 4080 SUPER):

| Package | Version |
|---------|---------|
| `torch` | `2.1.1+cu121` |
| `xformers` | **`0.0.23`** |
| `torchvision` | `0.16.1+cu121` |
| `numpy` | `1.26.4` |
| `detectron2` | `0.6` |
| `pytorch3d` | `0.7.5` |

**Fix:**
```bash
conda activate haptic

# Install exact matching versions
pip install torch==2.1.1+cu121 torchvision==0.16.1+cu121 torchaudio==2.1.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

pip install xformers==0.0.23 \
    --index-url https://download.pytorch.org/whl/cu121

# Verify
python -c "import xformers; import torch; print(torch.__version__, xformers.__version__)"
# Expected: 2.1.1+cu121  0.0.23
```

> **For Blackwell GPUs (sm_120):** Use the T12 upgrade path instead — torch 2.7+cu128 with
> matching xformers from `--index-url https://download.pytorch.org/whl/cu128`.

### T14 — HaPTIC: `No module named 'manopth'`

```
ModuleNotFoundError: No module named 'manopth'
```

**Root cause:** `manopth` is not on PyPI. It must be installed directly from GitHub.

**Fix:**
```bash
conda activate haptic
pip install git+https://github.com/hassony2/manopth.git

# Verify
python -c "import manopth; print(manopth.__file__)"
```

> Verified version: `0.0.1` from `hassony2/manopth` — no MANO model files needed for install,
> but MANO `.pkl` files must still be placed manually (see T7).

### T15 — HaPTIC: `No module named 'mmpose'` / mmengine version conflict

**Symptoms (may appear in sequence):**
```
ModuleNotFoundError: No module named 'mmpose'
# or after installing mmpose from PyPI:
ModuleNotFoundError: No module named 'mmengine'
# or after installing mmengine:
ERROR: mmengine requires mmcv >= 2.0.0rc4, but you have mmcv 1.3.9
```

**Root cause:** Two separate issues that compound each other:

1. `mmpose` is **not** the standalone PyPI package — it is `mmpose 0.24.0` bundled inside
   **ViTPose**, installed as an editable package. Installing from PyPI gives the wrong version.
2. `mmengine` belongs to the **mmcv 2.x** ecosystem. Our haptic env uses **mmcv 1.x**.
   The two are incompatible — installing `mmengine` into a 1.x env breaks everything.

**Verified working stack** (confirmed on RTX 4080 SUPER):

| Package | Version | Source |
|---------|---------|--------|
| `mmpose` | `0.24.0` | editable from `third-party/ViTPose/` |
| `mmcv` | `1.3.9` | pip (NOT mmcv-full, NOT 2.x) |
| `mmengine` | — | **not installed** |

**Fix (complete, in order):**
```bash
conda activate haptic

# Step 1: remove incorrectly installed packages
pip uninstall mmengine mmpose -y

# Step 2: clone ViTPose (if not already present)
cd third_party/haptic/third-party
git clone https://github.com/ViTAE-Transformer/ViTPose
cd ViTPose

# Step 3: editable install (this gives mmpose 0.24.0 from ViTPose)
pip install -e .

# Step 4: install mmcv 1.3.9 (1.x series, not 2.x, not mmcv-full)
pip uninstall mmcv-full mmcv -y 2>/dev/null
pip install mmcv==1.3.9

# Step 5: re-pin numpy (ViTPose install may upgrade it)
pip install "numpy<2.0"

# Verify
python -c "
import mmpose, mmcv
print('mmpose:', mmpose.__version__, mmpose.__file__)
print('mmcv:  ', mmcv.__version__)
try:
    import mmengine; print('mmengine: PRESENT (should not be!)')
except ImportError:
    print('mmengine: not installed (correct)')
"
# Expected output:
# mmpose: 0.24.0  .../third-party/ViTPose/mmpose/__init__.py
# mmcv:   1.3.9
# mmengine: not installed (correct)
```

### T16 — FoundationPose: `bundlesdf` Python 3.9 incompatibility (pytorch3d / nvdiffrast / mycpp)

**Symptoms:**
```
ModuleNotFoundError: No module named 'pytorch3d'
ImportError: version CXXABI_1.3.15 not found   # nvdiffrast binary wheel
ModuleNotFoundError: No module named 'mycpp'   # old .so is cpython-39
```

**Root cause:** `batch_obj_pose.py` requires `bundlesdf` to have `pytorch3d`, `nvdiffrast`,
and a compiled `mycpp.so` — all built for the **same Python ABI**. Our reference is
**Python 3.10**. A 3.9 env cannot use any of these.

**Verified working stack:**

| Package | Version | Notes |
|---------|---------|-------|
| Python | **3.10** | NOT 3.9 — ABI incompatible |
| torch | 2.1.1+cu121 | same as haptic env |
| pytorch3d | 0.7.5 | pre-built wheel py310/cu121/torch2.1.1 |
| nvdiffrast | 0.4.0 | compiled from source (NOT pip binary) |
| mycpp | — | compiled from source with cmake |

**Fix — full rebuild procedure:**
```bash
# 1. Remove old env
conda env remove -n bundlesdf -y
conda create -n bundlesdf python=3.10 -y
conda activate bundlesdf

# 2. PyTorch
pip install torch==2.1.1 torchvision==0.16.1 \
    --index-url https://download.pytorch.org/whl/cu121

# 3. pytorch3d 0.7.5 (pre-built for py310+cu121+torch2.1.1)
pip install pytorch3d \
    -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt211/download.html

# 4. nvdiffrast from source
#    ⚠️ setuptools>=70 breaks source builds — pin it first
pip install 'setuptools<70'
git clone https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast_src
export PATH=/usr/local/cuda/bin:$PATH   # nvcc must be on PATH
pip install /tmp/nvdiffrast_src --no-build-isolation

# 5. FoundationPose requirements
cd /path/to/FoundationPose
pip install -r requirements.txt
pip install fast-simplification   # not always in requirements.txt

# 6. Compile mycpp for Python 3.10
#    System deps first:
sudo apt-get install -y libeigen3-dev
pip install pybind11 'cmake<4.0'
#    Use pip-installed cmake (avoids system cmake version issues):
CMAKE=$(python -c "import cmake,os; print(os.path.join(os.path.dirname(cmake.__file__),'data','bin','cmake'))")
rm -rf /path/to/FoundationPose/mycpp/build
$CMAKE -B /path/to/FoundationPose/mycpp/build \
       /path/to/FoundationPose/mycpp/ \
       -DPYTHON_EXECUTABLE=$(which python) \
       -DCMAKE_BUILD_TYPE=Release \
       -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
$CMAKE --build /path/to/FoundationPose/mycpp/build -j$(nproc)
# → produces mycpp.cpython-310-x86_64-linux-gnu.so

# 7. Set FP_ROOT before running (config.py default is the empty submodule!)
export FP_ROOT=/path/to/FoundationPose

# 8. Verify all three
cd /path/to/FoundationPose
python -c "
import sys; sys.path.insert(0, '.')
import pytorch3d; print('pytorch3d:', pytorch3d.__version__)
import nvdiffrast.torch; print('nvdiffrast: ok')
from mycpp import *; print('mycpp: ok')
"
```

> **Why not just install pytorch3d into the existing 3.9 env?**
> The pre-built wheel is `py310` only. Source build on 3.9 + older CUDA often fails.
> Rebuilding as 3.10 takes ~15 min and matches the reference machine exactly.

### T5 — HaPTIC `gdown` / `one_click.sh`: `output/` not found + `_DATA/data/mano` symlink

**Symptom 1:** `FileNotFoundError: output/`
```bash
mkdir -p output
# then re-run one_click.sh
```

**Symptom 2 (after weights download):** `FileNotFoundError: _DATA/data/mano/mano_mean_params.npz`
HaPTIC looks for `_DATA/data/mano/` but model files are in `assets/mano/`:
```bash
# Run from inside third_party/haptic/
mkdir -p _DATA/data
ln -s ../../assets/mano _DATA/data/mano
```
> Note: `patches/haptic-intrinsics-fix.patch` (patch 1/2) adds the `mkdir -p output` fix
> automatically. The `_DATA/data/mano` symlink still needs to be created manually.

### T6 — DATA_HUB path mismatch (data on large disk, project in /home)
If your large dataset disk is mounted at e.g. `/data/5TB`:
```bash
# Inside the project directory:
ln -s /data/5TB/Affordance2Grasp/data_hub data_hub
# All scripts use config.DATA_HUB which auto-detects this symlink
```

### T7 — MANO download fails / not found
MANO cannot be redistributed. Download manually:
1. Register at [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de)
2. Download `mano_v1_2.zip` → extract `MANO_RIGHT.pkl` + `MANO_LEFT.pkl`
3. **Also get `MANO_UV_right.obj` + `MANO_UV_left.obj`** — these are **not** in `mano_v1_2.zip`:
   - Download the **SMPL-X** or **SMPL+H** package from [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de/)
   - Look for UV mesh files in `body_models/mano/` or `extras/` inside the archive
4. Place all files:
   - `third_party/haptic/assets/mano/MANO_RIGHT.pkl`
   - `third_party/haptic/assets/mano/MANO_LEFT.pkl`
   - `third_party/haptic/assets/mano/MANO_UV_right.obj`  ← **required for inference**
   - `third_party/haptic/assets/mano/MANO_UV_left.obj`
   - `third_party/hawor/_DATA/data/mano/MANO_RIGHT.pkl`
   - `third_party/hawor/_DATA/data_left/mano_left/MANO_LEFT.pkl`

### T7.3 — HaWoR: `pytorch-lightning` / `torchmetrics` version mismatch

**Symptom:** Error during HaWoR training/inference involving `torchmetrics` or `Callback` API:
```
AttributeError: ... has no attribute ...   # lightning 2.3+ broke API
TypeError: ... unexpected keyword argument  # torchmetrics incompatible with lightning 2.2
```

**Root cause:** `third_party/hawor/requirements.txt` does not pin lightning/torchmetrics
versions. `pip install -r requirements.txt` may pull the latest (e.g. lightning 2.3+),
which has breaking changes incompatible with HaWoR's training code.

**Verified compatible versions:**

| Package | Version |
|---------|---------|
| pytorch-lightning | **2.2.4** |
| torchmetrics | **1.4.0** |

**Fix — always run this AFTER `pip install -r requirements.txt`:**
```bash
conda activate hawor
pip install pytorch-lightning==2.2.4 torchmetrics==1.4.0
```

> The `requirements.txt` install may silently upgrade these. Always re-pin after.

### T7.5 — HaPTIC: `No module named 'webdataset'`

```
ModuleNotFoundError: No module named 'webdataset'
```
**Fix:**
```bash
conda activate haptic
pip install webdataset
```

### T7.7 — FoundationPose: `obj_recon_input/ycb/` missing (initial mask not found)

**Symptom:**
```
FileNotFoundError: .../obj_recon_input/ycb/ycb_dex_01/0.png
```
**Root cause:** FoundationPose requires a binary mask of the object in frame 0 to initialize
pose estimation. These masks were hand-annotated using `tools/annotate_obj_mask_recon.py`
and are stored in our HuggingFace repo. They are NOT generated automatically.

**Fix — download from HuggingFace:**
```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='UCBProject/Affordance2Grasp-Mesh',
    repo_type='dataset',
    local_dir='data_hub/ProcessedData',
    allow_patterns=['obj_recon_input/ycb/**', 'obj_meshes/ycb/**'],
    token='YOUR_HF_TOKEN',
)
"
# Downloads:
#   obj_recon_input/ycb/ycb_dex_01~20/0.png  (initial masks, 12 MB)
#   obj_meshes/ycb/ycb_dex_01~20/mesh.ply    (object meshes, 999 MB)
```

### T8 — DexYCB download (~250 GB)
DexYCB is not redistributable. Download from [dex-ycb.github.io](https://dex-ycb.github.io).
Requires registering and using their provided download script.
```bash
# Extract to correct location:
tar -xzf dex-ycb-*.tar.gz -C data_hub/RawData/ThirdPersonRawData/
```

### T9 — nvdiffrast / FoundationPose build: CUDA version mismatch
```
RuntimeError: The detected CUDA version (12.x) mismatches the version that was
used to compile PyTorch (11.8).
```
**Root cause:** `conda install cuda-toolkit` installs the *latest* CUDA (e.g. 12.9), not the version matching your torch wheel.

**Rule of thumb:** Match torch CUDA suffix to `nvidia-smi`'s "CUDA Version" (driver max), **not** `nvcc --version`.
```bash
# With CUDA 12.x driver → use cu121
pip install torch==2.1.1+cu121 torchvision==0.16.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

# With CUDA 11.x driver → use cu118
pip install torch==2.1.1+cu118 torchvision==0.16.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118

# Then install nvdiffrast via local clone (avoids git metadata build issues):
git clone --depth 1 https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast_src
pip install /tmp/nvdiffrast_src --no-build-isolation
```
> ⚠️ Never run `conda install -c nvidia cuda-toolkit` — it installs the latest CUDA version and will mismatch your torch wheel.

### T10 — FoundationPose cmake fails: FindBoost module removed
```
CMake Warning: Policy CMP0167 is not set: The FindBoost module is removed.
Could NOT find Boost (missing: system program_options)
```
**Root cause:** `cmake ≥ 4.0` dropped the legacy FindBoost module. You must use `cmake < 4.0`.
```bash
# Uninstall any cmake ≥ 4.0 first:
pip uninstall cmake -y
# Install pinned version:
pip install "cmake<4.0"
# Also ensure system Boost is installed:
sudo apt-get install -y libboost-all-dev
```

### T11 — FoundationPose mycpp compile fails: `bits/timesize.h: No such file or directory`
```
fatal error: bits/timesize.h: No such file or directory
```
**Root cause:** The bundlesdf conda environment ships a cross-compilation gcc (`x86_64-conda-linux-gnu-gcc`). When used with system headers from `/usr/include`, it cannot find glibc-internal headers because it expects its own isolated sysroot.

**Fix:** Bypass conda's gcc and use the system compiler directly:
```bash
cd third_party/FoundationPose
PYBIND11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")

mkdir -p mycpp/build && cd mycpp/build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX:/usr" \
  -DPython3_ROOT_DIR="$CONDA_PREFIX" \
  -DPYBIND11_PYTHON_EXECUTABLE="$CONDA_PREFIX/bin/python" \
  -Dpybind11_DIR="$PYBIND11_DIR" \
  -DBOOST_ROOT=/usr \
  -DBOOST_LIBRARYDIR=/usr/lib/x86_64-linux-gnu \
  -DBOOST_INCLUDEDIR=/usr/include \
  -DBoost_NO_SYSTEM_PATHS=OFF \
  -Wno-dev
cmake --build . -j$(nproc)
```
Expected result: `[100%] Built target mycpp` — `mycpp.cpython-3x-x86_64-linux-gnu.so` created. ✅

---

## Complete System Prerequisite Checklist

Run this **before** starting any conda environment setup:

```bash
# 1. Check GPU + driver
nvidia-smi | grep "CUDA Version"   # must be ≥ 11.8

# 2. System build dependencies (one-shot)
sudo apt-get install -y \
  libboost-all-dev \
  libomp-dev \
  build-essential \
  git

# 3. Python build tools (install in each conda env before other packages)
pip install "cmake<4.0" ninja pybind11 "setuptools<70" "numpy<2.0"
```

### T12 — Blackwell GPU (sm_120): `no kernel image is available for execution on the device`

**Affected GPUs:** RTX 5090, RTX 6000 Pro (2025 Blackwell generation)  
**Root cause:** `torch 2.1.1` was compiled only up to sm_90. Blackwell (sm_120) kernels are not included.

**Verify your architecture:**
```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
# compute_cap = 12.0 or higher → you have Blackwell → follow this fix
```

**Fix: upgrade the entire haptic env torch stack to 2.7+**
```bash
conda activate haptic

# Step 1: upgrade torch to Blackwell-compatible build
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128

# Step 2: upgrade xformers to matching version
pip install xformers --index-url https://download.pytorch.org/whl/cu128

# Step 3: pytorch3d has no cu128 prebuilt wheel — compile from source (~15 min)
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# Step 4: recompile detectron2 (torch ABI change requires rebuild)
pip install --force-reinstall "setuptools<70"
git clone --depth 1 https://github.com/facebookresearch/detectron2.git /tmp/det2
pip install /tmp/det2 --no-build-isolation

# Step 5: re-pin numpy (pytorch3d compile may upgrade it)
pip install "numpy<2.0"
```

> **Note on other envs:** `depth-pro` and `bundlesdf` environments can use `cu128` directly  
> (e.g. `pip install torch==2.7.0+cu128 ...`). FoundationPose CUDA extensions must also be  
> recompiled with the matching CUDA toolkit.

> **Verified working on:** RTX 4080 SUPER (sm_89) with torch 2.1.1+cu121  
> **Reported working on Blackwell after upgrade:** RTX 5090 (sm_120) with torch 2.7.0+cu128
