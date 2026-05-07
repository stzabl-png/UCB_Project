# Affordance2Grasp — Pipeline Instructions

> ⚠️ **This document is outdated (based on the early ARCTIC stage).**
> For the current pipeline, refer to **`README.md`** and **`HANDOVER.md`**.
> Kept as a historical reference only.

> **Goal:** Generate object contact heatmaps (Human Prior) from ARCTIC third-person videos,
> collect Robot GT data via Isaac Sim, and train the M5 PointNet++ model.

> **Paths:** `$PROJ` = project root (`/path/to/Affordance2Grasp`). `SAM3D_USER` = cloud username.

---

## Full Pipeline Flow

```
[ARCTIC third-person videos]
    ↓ Step 0a  annotate_trim.py        → data/trimmed/{seq}/
    ↓ Step 0b  annotate_obj_mask.py    → mask_bbox.png + bbox.json
    ↓ Step 0c  SAM3D (cloud)           → output/sam3d_obj_cache/{obj}/splat.ply
    ↓ Step 1   batch_depthpro.py       → output/depthpro_batch/{seq}/
    ↓ Step 2   batch_fp_register.py    → output/fp_register_batch/{seq}_T_obj_cam1.npy
    ↓ Step 3   batch_contact.py        → output/affordance_batch/{seq}/vert_contact_count.npy
    ↓ Step 4   export_arctic_prior.py  → data_hub/human_prior/{obj}.hdf5   ← key interface
    ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │  Step 5  random_grasp_sampler.py  (50% HP + 50% random)        │
    │          → output/grasps_random/{obj}_grasp.hdf5               │
    │  Step 6  Isaac Sim batch_random_sim.sh                          │
    │          → output/robot_gt_v4_physics/{obj}_robot_gt.hdf5      │
    │  Step 7  aggregate_robot_gt.py                                  │
    │          → data_hub/training/{obj}.hdf5                        │
    │  Step 8  model/train.py → output/checkpoints_m5/best_m5_model  │
    └─────────────────────────────────────────────────────────────────┘
    ↓ (M5 inference)
    inference/grasp_pose.py → candidate grasp poses HDF5 (13 candidates)
    sim/run_grasp.py        → Isaac Sim verification → posterior
```

---

## Environments

| Env | Purpose | Command |
|---|---|---|
| `base` | Video annotation, mask annotation (Qt GUI), random_grasp_sampler | `conda activate base` |
| `hawor` | Contact detection, prior export, visualization | `conda activate hawor` |
| `depth-pro` | Depth Pro depth estimation | `conda activate depth-pro` |
| `bundlesdf` | FoundationPose pose registration | `conda activate bundlesdf` |
| **IsaacSim** | Robot GT collection | `$ISAAC_SIM_PATH/python.sh` |
| Cloud server | SAM3D mesh generation | `ssh sam3d-gpu` |
| `sam3d-objects` | SAM3D inference | `conda activate sam3d-objects` |

---

## Key Paths

```bash
export PROJ=$HOME/Project/Affordance2Grasp     # project root
export ARCTIC_ROOT=$HOME/Project/arctic/unpack # ARCTIC data directory
export SAM3D_USER=lyh                          # cloud server username

# ARCTIC data layout
$ARCTIC_ROOT/arctic_data/data/cropped_images/s05/{seq}/1/   # image sequences
$ARCTIC_ROOT/meta/misc.json                                  # GT metadata
$ARCTIC_ROOT/meta/object_vtemplates/{obj}/mesh_tex.obj       # object mesh (mm)
$ARCTIC_ROOT/raw_seqs/s05/{seq}.[mano|object|smplx].npy

# Outputs
$PROJ/output/depthpro_batch/{seq}/
$PROJ/output/fp_register_batch/{seq}_T_obj_cam1.npy
$PROJ/output/affordance_batch/{seq}/vert_contact_count.npy
$PROJ/output/grasps_random/{obj}_grasp.hdf5
$PROJ/output/robot_gt_v4_physics/{obj}_robot_gt.hdf5
$PROJ/data_hub/human_prior/{obj}.hdf5                        ← key downstream interface
```

---

## Step 0a — Video Annotation + Trimming

```bash
conda activate base
cd $PROJ

# Launch annotation tool
python tools/annotate_trim.py \
    --arctic_root $ARCTIC_ROOT \
    --seq s05_ketchup_use_01
```

**Controls:**
- `← →` Navigate frames
- `[` Set start / `]` Set end
- `ENTER` Save trimmed segment
- `Q` Quit

Output: `data/trimmed/{seq}/` — extracted JPEG frames.

---

## Step 0b — Object Mask Annotation

```bash
conda activate base
python tools/annotate_obj_mask.py \
    --arctic_root $ARCTIC_ROOT \
    --seq s05_ketchup_use_01 \
    --cam 1
```

**Controls:** Click to define bounding box → SAM generates mask → `S` save.

Output: `data/trimmed/{seq}/mask_bbox.png` + `bbox.json`.

---

## Step 0c — SAM3D Cloud Mesh Generation

```bash
# Local: prepare + upload input
python tools/prep_sam3d_input.py --dataset arctic
rsync -avz /tmp/sam3d_input/ sam3d-gpu:~/input/

# Cloud: run SAM3D
ssh sam3d-gpu "
cd ~/lyh/sam-3d-objects
CUDA_VISIBLE_DEVICES=0 \
CONDA_PREFIX=~/miniconda3/envs/sam3d-objects \
~/miniconda3/envs/sam3d-objects/bin/python batch_infer.py \
    --input-dir ~/input --output-dir ~/output --dataset arctic
"

# Local: pull result
rsync -avz sam3d-gpu:~/output/arctic/ $PROJ/output/sam3d_obj_cache/
```

---

## Step 1 — Depth Pro

```bash
conda activate depth-pro
cd $PROJ

python data/batch_depthpro.py \
    --arctic_root $ARCTIC_ROOT \
    --seq_list data/trimmed/
```

Output: `output/depthpro_batch/{seq}/` — `K.txt` + `depths.npz`.

---

## Step 2 — FoundationPose Object Registration

```bash
conda activate bundlesdf
export FP_ROOT=/path/to/FoundationPose
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

python data/batch_fp_register.py \
    --arctic_root $ARCTIC_ROOT \
    --seq_list data/trimmed/
```

Output: `output/fp_register_batch/{seq}_T_obj_cam1.npy` — 4×4 object-in-camera pose.

---

## Step 3 — Contact Detection + Affordance Heatmap

```bash
conda activate hawor

python data/batch_contact.py \
    --arctic_root $ARCTIC_ROOT \
    --seq_list data/trimmed/
```

Output: `output/affordance_batch/{seq}/vert_contact_count.npy`.

---

## Step 4 — Export ARCTIC Prior

```bash
conda activate hawor

python data/export_arctic_prior.py
```

Output: `data_hub/human_prior/{obj}.hdf5` — per-vertex contact probability.

---

## Step 5 — Grasp Candidates (50% HP + 50% Random)

```bash
conda activate bundlesdf

python tools/random_grasp_sampler.py --all
```

Output: `output/grasps_random/{obj}_grasp.hdf5`.

---

## Step 6 — Isaac Sim Batch Verification

```bash
$ISAAC_SIM_PATH/python.sh sim/batch_random_sim.sh
```

Output: `output/robot_gt_v4_physics/{obj}_robot_gt.hdf5`.

---

## Step 7 — Aggregate Training Data

```bash
conda activate bundlesdf

python data/aggregate_robot_gt.py
```

Output: `data_hub/training/{obj}.hdf5` — merged `human_prior` + `robot_gt`.

---

## Step 8 — Train M5 Model

```bash
conda activate bundlesdf

python model/train.py \
    --data_dir data_hub/training/ \
    --save_dir output/checkpoints_m5/ \
    --epochs 200
```

---

## Common Debug Commands

```bash
# Check contact output for one sequence
python tools/vis_contact.py --seq s05_ketchup_use_01

# Verify human prior HDF5
python3 -c "
import h5py, numpy as np
with h5py.File('data_hub/human_prior/ketchup.hdf5') as f:
    hp = f['human_prior'][()]
    print('shape:', hp.shape, 'max:', hp.max(), 'cov>0.1:', (hp>0.1).mean())
"

# Check all objects
for f in data_hub/human_prior/*.hdf5; do
    echo -n "$f: "
    python3 -c "
import h5py, numpy as np
with h5py.File('$f') as f: hp = f['human_prior'][()]
print(f'max={hp.max():.3f}  cov>0.1={(hp>0.1).mean()*100:.1f}%')
"
done
```
