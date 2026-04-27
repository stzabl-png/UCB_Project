# Affordance2Grasp

**Learning robot grasp poses from human hand-object interaction videos.**

Pipeline: `Video → Depth + Pose Estimation → Human Contact Prior → Sim Filtering → Affordance Model → Robot Grasp`

---

## Pipeline Overview

```
┌─────────────────────────── PHASE 1: Data Generation ────────────────────────────┐
│                                                                                  │
│  Raw Video                                                                       │
│     │                                                                            │
│     ├─ Depth Pro  ──────→ Camera intrinsics (K) + Metric depth maps             │
│     │                                                                             │
│     ├─ HaPTIC     ──────→ MANO hand vertices  (778 pts, camera space)           │
│     │                                                                             │
│     └─ FoundationPose ─→ Object 6D pose  (4×4 transform, per frame)            │
│                │                                                                  │
│                └──── Align (3D distance) ──→ human_prior  (contact heatmap)     │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
                                     │
┌─────────────────────────── PHASE 2: Robot GT ────────────────────────────────────┐
│                                                                                  │
│  human_prior ──→ Grasp Candidates ──→ Isaac Sim ──→ robot_gt (success/fail)    │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
                                     │
┌─────────────────────────── PHASE 3: Model ───────────────────────────────────────┐
│                                                                                  │
│  human_prior + robot_gt ──→ Build HDF5 ──→ Train PointNet++ ──→ Checkpoint     │
│                                                                                  │
│  New object mesh ──→ Predict Affordance ──→ Generate Grasp Pose ──→ Sim Check   │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Supported Datasets

| Dataset | Objects | Sequences | Frames |
|---|---|---|---|
| **DexYCB** | 20 YCB objects | ~3,600 (6 cameras) | ~260K |
| **HO3D v3** | 8 YCB objects | 68 | ~90K |

Both datasets use **identical scripts** — just swap `--dataset dexycb` / `--dataset ho3d_v3`.

---

## Setup

### 1. Clone

```bash
git clone https://github.com/stzabl-png/UCB_Project.git Affordance2Grasp
cd Affordance2Grasp
```

### 2. Conda Environments

```bash
# Environment A — Depth Pro
conda create -n depth-pro python=3.9 -y
conda activate depth-pro
pip install git+https://github.com/apple/ml-depth-pro.git
pip install natsort tqdm pillow numpy h5py

# Environment B — Everything else (HaPTIC, FP, alignment, training)
conda create -n bundlesdf python=3.9 -y
conda activate bundlesdf
pip install trimesh scipy h5py opencv-python natsort tqdm fast_simplification torch
# Install HaPTIC per its official README
# Install FoundationPose: pip install -e /path/to/FoundationPose
```

### 3. Configure paths in `config.py`

```python
DATA_HUB   = "/path/to/data_hub"
HAPTIC_DIR = "/path/to/HaPTIC"
FP_ROOT    = "/path/to/FoundationPose"
```

### 4. Download raw data

| Dataset | Official link | Place at |
|---|---|---|
| DexYCB | https://dex-ycb.github.io | `data_hub/RawData/ThirdPersonRawData/dexycb/` |
| HO3D v3 | https://www.tugraz.at/...ho3d | `data_hub/RawData/ThirdPersonRawData/ho3d_v3/` |

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

Includes: YCB meshes (`obj_meshes/ycb/`) + FP init masks (`obj_recon_input/ycb/`)

---

## Phase 1 — Data Generation

### Step 1 · Depth Pro — Camera Intrinsics + Depth Maps

**Two-pass self-calibration** (no GT intrinsics needed):
Pass 1 estimates focal length per sequence → global median → Pass 2 uses fixed focal length.
Verified accuracy: DexYCB +0.2%, HO3D v3 -1.5%.

```bash
conda activate depth-pro

# DexYCB — run each camera separately (6 cameras, ~600 seqs each)
for CAM in 841412060263 840412060917 932122060857 836212060125 932122061900 932122062010; do
    python data/batch_depth_pro.py --dataset dexycb --cam $CAM --two-pass --max-frames 150
done

# HO3D v3 — all 68 sequences
python data/batch_depth_pro.py --dataset ho3d_v3 --two-pass --max-frames 150
```

Output: `data_hub/ProcessedData/third_depth/{dataset}/{seq_id}/`
- `K.txt` — 3×3 intrinsic matrix
- `depths.npz` — depth maps (N, H, W) float32, metres

---

### Step 2 · HaPTIC — MANO Hand Estimation

```bash
conda activate bundlesdf

python data/batch_haptic.py --dataset dexycb  --only-with-depth-k
python data/batch_haptic.py --dataset ho3d_v3 --only-with-depth-k
```

Output: `data_hub/ProcessedData/third_mano/{dataset}/{seq_id}.npz`
- `verts_dict` — MANO vertices (778, 3) per frame, camera space

---

### Step 3 · FoundationPose — Object 6D Pose Tracking

```bash
conda activate bundlesdf

python tools/batch_obj_pose.py --dataset dexycb
python tools/batch_obj_pose.py --dataset ho3d_v3
```

Output: `data_hub/ProcessedData/obj_poses/{dataset}/{seq_id}/ob_in_cam/{frame}.txt`
- 4×4 transformation matrix per frame (object-in-camera)

---

### Step 4 · Contact Label Generation — Human Prior

Uses **3D distance + depth normalization** to align MANO hand and FP mesh in metric camera space.

```bash
conda activate bundlesdf

python data/batch_align_mano_fp.py --dataset dexycb
python data/batch_align_mano_fp.py --dataset ho3d_v3
```

Output:
- `data_hub/ProcessedData/training_fp/{dataset}/{obj}.hdf5` — point cloud + human_prior
- `data_hub/ProcessedData/human_prior_fp/{dataset}/` — per-vertex contact probability

---

## Phase 2 — Robot Ground Truth (Isaac Sim)

### Step 5 · Sample Grasp Candidates

```bash
conda activate bundlesdf

# All objects (reads from human_prior)
python tools/random_grasp_sampler.py --all
```

Output: `data_hub/grasps/{obj}/candidates.hdf5`

### Step 6 · Run Grasp Simulation

```bash
# Requires Isaac Sim. Set path: export ISAAC_SIM_PATH=/path/to/isaac-sim

python sim/run_grasp_sim.py \
    --hdf5 data_hub/grasps/{obj}/candidates.hdf5 \
    --headless \
    --save-result
```

Output: `robot_gt` labels (grasp success/failure per candidate)

---

## Phase 3 — Model Training & Inference

### Step 7 · Build Training Dataset

```bash
conda activate bundlesdf

python data/build_dataset.py \
    --num_points 4096 \
    --augment 3
```

Output: `dataset/train.hdf5`, `dataset/val.hdf5`
- `point_cloud` (N, 3), `normals` (N, 3), `human_prior` (N,), `robot_gt` (N,), `force_center` (3,)

### Step 8 · Train Affordance Model

```bash
conda activate bundlesdf

python -m model.train \
    --epochs 200 \
    --batch_size 16 \
    --lr 0.001 \
    --fc_lambda 10.0
```

Checkpoint saved to `checkpoints/`

### Step 9 · Predict Affordance on New Object

```bash
conda activate bundlesdf

python -m inference.grasp_pose \
    --mesh path/to/object.obj
```

Output: `grasps/{obj}/affordance_grasp.hdf5`

### Step 10 · Verify Grasp in Simulation

```bash
# Full pipeline: predict + sim verify
python run.py --mesh path/to/object.obj

# Prediction only (no sim):
python run.py --mesh path/to/object.obj --no-sim
```

---

## Estimated Runtime (RTX 4080 SUPER, single GPU)

| Step | DexYCB (6 cams) | HO3D v3 |
|---|---|---|
| 1 · Depth Pro | ~21 h | ~1.7 h |
| 2 · HaPTIC | ~48 h | ~1 h |
| 3 · FoundationPose | ~8 h | ~15 min |
| 4 · Contact Labels | < 5 min | < 5 min |
| 5–6 · Sim (robot GT) | ~varies | ~varies |
| 7–8 · Train | ~2 h | — |

> **Multi-GPU tip**: Steps 1–3 are embarrassingly parallel across cameras/sequences.
> With 6× GPUs, DexYCB Phase 1 completes in ~13 h (one camera per GPU).
> With 8× A100, full Phase 1 for both datasets finishes in **< 2 h**.

> **Single camera baseline**: Using only `841412060263` (best camera, verified +0.2% bias),
> DexYCB Phase 1 completes in **~13 h** on a single GPU.

---

## Verified Results

| Config | Result |
|---|---|
| DexYCB cam `841412060263`, `ycb_dex_02` | coverage=100%, diverged=0 |
| Depth Pro fx bias — DexYCB | **+0.2%** vs GT |
| Depth Pro fx bias — HO3D v3 (two-pass) | **-1.5%** vs GT |
| FP tracking success rate | **100%** |
