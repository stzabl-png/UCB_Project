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
                                           │
┌──────────────────────────────── PHASE 1B: First-Person Data ────────────────────────────────┐
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

---

## Setup

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
> You need: `MANO_RIGHT.pkl` and `MANO_LEFT.pkl` from the `mano_v1_2` package.
>
> Place the files in **two locations** (both tools need their own copy):
>
> **HaPTIC:**
> ```
> third_party/haptic/assets/mano/
>     MANO_RIGHT.pkl
>     MANO_LEFT.pkl
> ```
>
> **HaWoR:**
> ```
> third_party/hawor/_DATA/data/mano/
>     MANO_RIGHT.pkl
> third_party/hawor/_DATA/data_left/mano_left/
>     MANO_LEFT.pkl
> ```
>
> Quick copy commands (after extracting mano_v1_2.zip):
> ```bash
> MANO=/path/to/mano_v1_2/models
>
> # HaPTIC
> mkdir -p third_party/haptic/assets/mano
> cp $MANO/MANO_RIGHT.pkl $MANO/MANO_LEFT.pkl third_party/haptic/assets/mano/
>
> # HaWoR
> mkdir -p third_party/hawor/_DATA/data/mano
> mkdir -p third_party/hawor/_DATA/data_left/mano_left
> cp $MANO/MANO_RIGHT.pkl third_party/hawor/_DATA/data/mano/
> cp $MANO/MANO_LEFT.pkl  third_party/hawor/_DATA/data_left/mano_left/
> ```


### 3. Conda Environments

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
# Follow third_party/hawor/README for setup
```

### 3b. Environment B — `bundlesdf` (main environment)

> **Used for:** HaPTIC (Step 2), FoundationPose (Steps 3, E5), contact alignment (Steps 4, E6), Sim (Phase 3), training (Phase 4).

**Step 1 — Create the environment**
```bash
conda create -n bundlesdf python=3.9 -y
conda activate bundlesdf
pip install trimesh scipy h5py opencv-python natsort tqdm
pip install fast-simplification   # ← CRITICAL: without this, contact alignment takes 23 h not 15 min
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
# Clone
git clone https://github.com/NVlabs/FoundationPose.git /path/to/FoundationPose
cd /path/to/FoundationPose

# Install Eigen3 3.4.0 (required for C++ build)
conda install conda-forge::eigen=3.4.0 -y
export CMAKE_PREFIX_PATH="$CONDA_PREFIX:$CMAKE_PREFIX_PATH"

# Install Python dependencies
pip install -r requirements.txt

# Install NVDiffRast (rendering backend)
pip install git+https://github.com/NVlabs/nvdiffrast.git

# Build CUDA C++ extensions (takes ~3-5 min)
bash build_all_conda.sh
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

Annotate the object mask on one representative frame:
```
data_hub/ProcessedData/obj_recon_input/egocentric/{obj}/0.png
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

## Phase 3 — Robot Ground Truth (Isaac Sim)

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
