# A100 Deployment Guide — Extending HumanPrior with EgoDex + OakInk

## 🤖 Context for Deployment Agent

**What this run does:**
This pipeline extends an existing HumanPrior dataset.
The company team already processed **DexYCB** and **HO3D** through the full pipeline
and has their HumanPrior HDF5 files ready. This deployment adds two new datasets:

| Dataset | View | Pipeline |
|---|---|---|
| **OakInk** (new) | Third-person | DepthPro → HaWoR → FoundationPose → Align |
| **EgoDex** (new) | Egocentric (first-person) | MegaSAM → HaWoR → FoundationPose → Align |

**End goal:** Merge all four datasets into a single unified HumanPrior:
```
DexYCB    (existing) ─┐
HO3D      (existing) ─┤  aggregate_prior.py  →  data_hub/human_prior/{obj}.hdf5
OakInk    (new)      ─┤
EgoDex    (new)      ─┘
```
This HumanPrior is then used to train the PointNet++ affordance model (`model/train.py`).

**Hardware target:** 8×A100, Ubuntu 22.04, CUDA 12.x  
**Estimated runtime:** ~16–17 hours total with all 8 GPUs running in parallel.

---

## Phase 1 Output — Unified ProcessedData Structure

> **⚠️ Critical:** `data_hub/ProcessedData/` is the most important Phase 1 output.
> All downstream pipeline steps (HaPTIC, FoundationPose alignment, model training) depend on it.
> **Upload to HuggingFace after each run completes.**

After running Steps 1 and 2 below, `ProcessedData/` will contain:

```
data_hub/ProcessedData/
├── egocentric/                    ← Phase 1B outputs (MegaSAM + HaWoR ego)
│   └── egodex/
│       └── {task}__{episode}/
│           ├── depth.npz          MegaSAM metric depth  (N, H, W)  metres
│           ├── cam_c2w.npy        camera-to-world poses (N, 4, 4)
│           ├── K.npy              camera intrinsics     (3, 3)
│           ├── mano.npz           HaWoR hand params     right/left verts + MANO
│           └── ob_in_cam/         FoundationPose object poses
│               ├── 000000.npy     (4, 4)  per frame
│               └── ...
└── third/                         ← Phase 1A outputs (DepthPro + HaWoR third)
    └── oakink/
        └── {seq_id}/
            ├── depths.npz         Depth Pro metric depth (N, H, W)  metres
            ├── K.npy              camera intrinsics      (3, 3)
            ├── mano.npz           HaWoR hand params
            └── ob_in_cam/
                ├── 000000.npy     (4, 4)
                └── ...
```

**Upload command (run after each step completes):**
```bash
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='data_hub/ProcessedData',
    repo_id='UCBProject/ProcessedData',
    repo_type='dataset',
    commit_message='Phase1 output: egocentric + third',
)
"
```

---

## Step 0 · Clone & Download Assets

```bash
git clone --recursive https://github.com/stzabl-png/UCB_Project.git
cd UCB_Project

pip install huggingface_hub

# Model weights + FP init masks + object meshes (~12 GB total)
python setup_weights.py

# OakInk raw data (~25 GB, third-person)
python setup_weights.py --tool oakink

# EgoDex raw data (~30 GB, egocentric)
python setup_weights.py --tool egodex
```

> **Already have DexYCB/HO3D results?**
> Copy `training_fp/dexycb/`, `training_fp/ho3d_v3/`, and `human_prior_fp/`
> into `data_hub/ProcessedData/` on the new machine.

---

## Step 1 · OakInk — Third-Person Pipeline (Phase 1A)

OakInk uses the same third-person pipeline as DexYCB:
DepthPro → HaWoR → FoundationPose → Align.

### 1a · Depth Pro (intrinsics + metric depth)

```bash
conda activate depth-pro

# Get total sequence count
TOTAL=$(python -c "
import sys; sys.path.insert(0,'.')
from data.batch_depth_pro import discover_oakink
import config
seqs = list(discover_oakink(config.DATA_HUB + '/RawData/ThirdPersonRawData/oakink_v1'))
print(len(seqs))
")
echo "OakInk total sequences: $TOTAL"

# Run all 8 GPUs in parallel (open in tmux or sbatch)
for GPU in 0 1 2 3 4 5 6 7; do
  START=$(( GPU * TOTAL / 8 ))
  END=$(( (GPU + 1) * TOTAL / 8 ))
  CUDA_VISIBLE_DEVICES=$GPU python data/batch_depth_pro.py \
    --dataset oakink --two-pass --start $START --end $END &
done
wait
```

Output: `data_hub/ProcessedData/third/oakink/{seq_id}/depths.npz + K.npy`

### 1b · HaWoR (hand pose estimation)

```bash
conda activate hawor

for GPU in 0 1 2 3 4 5 6 7; do
  START=$(( GPU * TOTAL / 8 ))
  END=$(( (GPU + 1) * TOTAL / 8 ))
  CUDA_VISIBLE_DEVICES=$GPU python data/batch_hawor.py \
    --dataset oakink --start $START --end $END &
done
wait
```

Output: `data_hub/ProcessedData/third/oakink/{seq_id}/mano.npz`

### 1c · FoundationPose (object pose)

> **Prerequisite**: `data_hub/ProcessedData/obj_recon_input/oakink/` must contain
> init masks — already downloaded by `python setup_weights.py` (via `thirdmasks`).

```bash
conda activate bundlesdf

for GPU in 0 1 2 3 4 5 6 7; do
  START=$(( GPU * TOTAL / 8 ))
  END=$(( (GPU + 1) * TOTAL / 8 ))
  CUDA_VISIBLE_DEVICES=$GPU python tools/batch_obj_pose.py \
    --dataset oakink --start $START --end $END &
done
wait
```

Output: `data_hub/ProcessedData/third/oakink/{seq_id}/ob_in_cam/`

### 1d · Align (generate HumanPrior)

```bash
conda activate bundlesdf

# OakInk has few objects — single GPU is fine
python data/batch_align_mano_fp.py --dataset oakink
```

Output:
- `data_hub/ProcessedData/training_fp/oakink/{obj}.hdf5`
- `data_hub/ProcessedData/human_prior_fp/{obj}.hdf5` (merged third-person prior)

---

## Step 2 · EgoDex — Egocentric Pipeline (Phase 1B)

EgoDex uses the egocentric pipeline:
MegaSAM → HaWoR → FoundationPose → Align.

### 2a · MegaSAM (depth + SLAM camera poses)

```bash
conda activate mega_sam
cd mega-sam

# EgoDex has 3051 sequences total
TOTAL=3051
for GPU in 0 1 2 3 4 5 6 7; do
  START=$(( GPU * TOTAL / 8 ))
  END=$(( (GPU + 1) * TOTAL / 8 ))
  CUDA_VISIBLE_DEVICES=$GPU python ../data/batch_megasam.py \
    --dataset egodex --start $START --end $END &
done
wait
```

Output: `data_hub/ProcessedData/egocentric/egodex/{task}__{episode}/depth.npz + cam_c2w.npy + K.npy`

### 2b · HaWoR (egocentric hand tracking)

```bash
conda activate hawor

for GPU in 0 1 2 3 4 5 6 7; do
  START=$(( GPU * TOTAL / 8 ))
  END=$(( (GPU + 1) * TOTAL / 8 ))
  CUDA_VISIBLE_DEVICES=$GPU python data/batch_hawor.py \
    --dataset egodex --start $START --end $END &
done
wait
```

Output: `data_hub/ProcessedData/egocentric/egodex/{task}__{episode}/mano.npz`

### 2c · FoundationPose (object pose, egocentric)

> **Prerequisite**: `data_hub/ProcessedData/obj_recon_input/egocentric/` must contain
> EgoDex init masks — already downloaded by `python setup_weights.py` (via `egomasks`).

```bash
conda activate bundlesdf

for GPU in 0 1 2 3 4 5 6 7; do
  START=$(( GPU * TOTAL / 8 ))
  END=$(( (GPU + 1) * TOTAL / 8 ))
  CUDA_VISIBLE_DEVICES=$GPU python tools/batch_obj_pose.py \
    --dataset egodex --start $START --end $END &
done
wait
```

Output: `data_hub/ProcessedData/egocentric/egodex/{task}__{episode}/ob_in_cam/`

### 2d · Align (generate HumanPrior)

```bash
conda activate bundlesdf

python data/batch_align_ego_mano_fp.py --dataset egodex
```

Output:
- `data_hub/ProcessedData/training_fp_ego/egodex/{obj}.hdf5`
- `data_hub/human_prior/{obj}.hdf5` (merged egocentric prior)

---

## Step 3 · Aggregate All Four Datasets

```bash
conda activate bundlesdf

# Verify outputs from each dataset
echo "=== HumanPrior counts ===" && \
ls data_hub/ProcessedData/training_fp/dexycb/     | wc -l && echo "DexYCB objects" && \
ls data_hub/ProcessedData/training_fp/ho3d_v3/    | wc -l && echo "HO3D objects" && \
ls data_hub/ProcessedData/training_fp/oakink/     | wc -l && echo "OakInk objects" && \
ls data_hub/ProcessedData/training_fp_ego/egodex/ | wc -l && echo "EgoDex objects"

# Merge training_fp + training_fp_ego → final human_prior
python data/aggregate_prior.py
```

Output: `data_hub/human_prior/{obj}.hdf5` (all four datasets merged)

**Quality check:**

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

Expected: `max_hp >= 0.7`, `cov(>0.1) = 100%`

---

## Step 4 · Build Training Set + Train Policy

```bash
conda activate bundlesdf

# Build HDF5 training set (GT-free mode, no Isaac Sim needed)
python data/build_dataset.py --num_points 4096 --augment 3

# Train on 8 GPUs (DDP)
python -m model.train \
    --epochs 200 \
    --batch_size 128 \
    --lr 0.001 \
    --fc_lambda 10.0
```

---

## HuggingFace Assets

| Tool | Repo | Size | Purpose |
|------|------|------|---------|
| Model weights | `UCBProject/Weights` | ~10 GB | FP/HaWoR/HaPTIC/MegaSAM/DepthPro |
| Third-person FP masks | `UCBProject/ThirdDataMask` | ~30 MB | Phase 1A init masks |
| Egocentric FP masks | `UCBProject/EgoDataMask` | ~70 MB | Phase 1B init masks |
| Object meshes | `UCBProject/ObjMesh` | ~1 GB | FP + scale estimation |
| OakInk raw data | `UCBProject/OakInk` | ~25 GB | Phase 1A input |
| EgoDex raw data | `UCBProject/EgoDex` | ~30 GB | Phase 1B input |
| **ProcessedData** | `UCBProject/ProcessedData` | ~TBD | **⚠️ Phase 1 unified output — upload after each run** |

One-command download (skip DexYCB/HO3D):

```bash
python setup_weights.py              # weights + masks + meshes
python setup_weights.py --tool oakink
python setup_weights.py --tool egodex
```

---

## Estimated Runtime (single A100)

| Step | OakInk | EgoDex |
|------|--------|--------|
| DepthPro / MegaSAM | ~4 h | ~50 h |
| HaWoR | ~3 h | ~40 h |
| FoundationPose | ~3 h | ~30 h |
| Align | ~5 min | ~1 h |
| **Total** | **~10 h** | **~121 h** |

**8-GPU parallel**: OakInk ~1.5 h, EgoDex ~15 h

---

## Troubleshooting

- **How to split `--start/--end`?** Divide total sequence count evenly by GPU index (see examples above).
- **FoundationPose cannot find mask?** Check that `obj_recon_input/{oakink,egocentric}/` is populated.
- **Sequences skipped?** Missing mask for that sequence — normal, does not affect others.
- **More issues** → see `README.md` Troubleshooting T1–T16 and `docs/DEPLOYMENT_ISSUES.md`.
