# Affordance2Grasp — GM Company Pipeline Guide
## OakInk + DexYCB Human Prior → Sim Grasp Validation → Model Training

> **Repo**: https://github.com/stzabl-png/UCB_Project  
> **Env**: conda `bundlesdf` for all Python steps; Isaac Sim for Steps 3–4  
> **Setup**: Follow **README.md Section 3b** (bundlesdf environment) before starting.

---

## Step 0 — Clone & Setup

```bash
git clone --recursive https://github.com/stzabl-png/UCB_Project.git Affordance2Grasp
cd Affordance2Grasp

# Apply required patch
cd third_party/haptic && git apply ../../patches/haptic-intrinsics-fix.patch && cd ../..

# HuggingFace login (required for private repos)
huggingface-cli login
```

---

## Step 1 — Download Data from HuggingFace

### 1A — Object Meshes
Source: [`UCBProject/ObjMesh`](https://huggingface.co/datasets/UCBProject/ObjMesh/tree/main/meshes)

```bash
huggingface-cli download UCBProject/ObjMesh \
    --repo-type dataset \
    --local-dir data_hub/meshes/SAM3DMesh
```

Verify:
```bash
conda run -n bundlesdf python3 -c "
import glob
print('OakInk:', len(glob.glob('data_hub/meshes/SAM3DMesh/oakink/**/*.ply', recursive=True)))
print('YCB:   ', len(glob.glob('data_hub/meshes/SAM3DMesh/ycb/**/*.ply', recursive=True)))
"
# Expected: OakInk >= 100, YCB >= 20
```

### 1B — Human Prior
Source: [`UCBProject/ProcessedData`](https://huggingface.co/datasets/UCBProject/ProcessedData/tree/main/training_fp)

```bash
huggingface-cli download UCBProject/ProcessedData \
    --repo-type dataset \
    --local-dir data_hub/ProcessedData \
    --include "training_fp/oakink/*" "training_fp/dexycb/*"
```

Verify:
```bash
conda run -n bundlesdf python3 -c "
import h5py, glob
for ds in ['oakink', 'dexycb']:
    files = glob.glob(f'data_hub/ProcessedData/training_fp/{ds}/*.hdf5')
    print(f'{ds}: {len(files)} objects')
    if files:
        with h5py.File(files[0]) as f:
            print('  keys:', list(f.keys()), '| shape:', f['point_cloud'].shape)
"
# Expected: oakink=100, dexycb=27; point_cloud shape = (4096, 3)
```

---

## Step 2 — Generate Grasp Candidates

**Script**: `tools/random_grasp_sampler.py` | **Env**: `bundlesdf`  
**Output**: `output/grasps_random/{obj_id}_grasp.hdf5`

```bash
conda activate bundlesdf

# OakInk (100 objects)
python3 tools/random_grasp_sampler.py --oakink --output-dir output/grasps_random

# DexYCB (20 objects)
python3 tools/random_grasp_sampler.py --dexycb --output-dir output/grasps_random
```

Verify:
```bash
ls output/grasps_random/ | wc -l   # >= 120
```

> **Note**: If the script reports `⚠️ mesh not found` for an object, the mesh in  
> `data_hub/meshes/SAM3DMesh/` has a different subfolder layout than expected.  
> Run `find data_hub/meshes/SAM3DMesh -name "*.ply" | head -5` to check the actual structure.

---

## Step 3 — Convert Meshes to Isaac Sim USD Format

**Script**: `sim/convert_batch_usd.py` | **Env**: Isaac Sim  
**Output**: `output/assets/{obj_id}.usd`

```bash
export ISAAC_SIM_PATH=/path/to/isaac-sim

$ISAAC_SIM_PATH/python.sh sim/convert_batch_usd.py --sam3d-only
```

> **Path check**: The script expects meshes at  
> `data_hub/meshes/SAM3DMesh/meshes/{dataset}/{obj}/mesh.ply`  
> If your downloaded structure differs, update `SAM3D_MESH_ROOT` at line 39 of the script.

Verify:
```bash
ls output/assets/*.usd | wc -l   # >= 120
```

---

## Step 4 — Isaac Sim Grasp Validation + Robot GT Collection

**Script**: `scripts/batch_auto_sim.sh` | **Env**: Isaac Sim  
**Output**: `output/robot_gt_auto/{obj_id}_robot_gt.hdf5`

Test one object first:
```bash
$ISAAC_SIM_PATH/python.sh sim/run_grasp_sim.py \
    --hdf5 output/grasps_random/A01001_grasp.hdf5 \
    --save-result --result-dir output/robot_gt_auto --headless
```

Run all objects:
```bash
export ISAAC_SIM_PATH=/path/to/isaac-sim
export GRASP_DIR=output/grasps_random
export GT_DIR=output/robot_gt_auto
bash scripts/batch_auto_sim.sh
```

Success criterion: `✅ GRASP SUCCESS!` — object lifted Δz > 3 cm.

---

## Step 5 — Build Per-Object Training Data

**Script**: `tools/gen_m5_training_data.py` | **Env**: `bundlesdf`  
**Output**: `data_hub/training_m5/{obj_id}.hdf5`

```bash
conda activate bundlesdf
ln -sfn $(pwd)/output/robot_gt_auto output/robot_gt
python3 tools/gen_m5_training_data.py
```

Expected output:
```
  ✅ A01001: 🧑 🤖(12g)
  ✅ ycb_dex_01: 🧑 🤖(8g)
  完成! 120 个训练样本
```

---

## Step 6 — Merge Training Data

Run this once to merge all per-object HDF5s into train/val split files:

```bash
conda run -n bundlesdf python3 - << 'EOF'
import h5py, numpy as np, glob, os

files = sorted(glob.glob('data_hub/training_m5/*.hdf5'))
print(f'Merging {len(files)} objects...')

all_pts, all_nrm, all_lbl, all_ids, all_fc = [], [], [], [], []
for fp in files:
    obj_id = os.path.splitext(os.path.basename(fp))[0]
    with h5py.File(fp) as f:
        pts, nrm, lbl, fc = f['point_cloud'][:], f['normals'][:], f['robot_gt'][:], f['force_center'][:]
    if pts.shape != (4096, 3): continue
    all_pts.append(pts[None]); all_nrm.append(nrm[None])
    all_lbl.append(lbl[None]); all_ids.append(obj_id); all_fc.append(fc[None])

pts_arr = np.concatenate(all_pts); nrm_arr = np.concatenate(all_nrm)
lbl_arr = np.concatenate(all_lbl); fc_arr = np.concatenate(all_fc)
ids_arr = np.array(all_ids, dtype='S64')

n = len(all_ids); np.random.seed(42); idx = np.random.permutation(n)
n_val = max(2, n // 5); val_idx, train_idx = idx[:n_val], idx[n_val:]

os.makedirs('output/dataset', exist_ok=True)
for sidx, name in [(train_idx, 'train'), (val_idx, 'val')]:
    with h5py.File(f'output/dataset/affordance_{name}.h5', 'w') as f:
        g = f.create_group('data')
        g.create_dataset('points',        data=pts_arr[sidx], compression='gzip')
        g.create_dataset('normals',       data=nrm_arr[sidx], compression='gzip')
        g.create_dataset('labels',        data=lbl_arr[sidx], compression='gzip')
        g.create_dataset('obj_ids',       data=ids_arr[sidx])
        g.create_dataset('force_centers', data=fc_arr[sidx],  compression='gzip')
    print(f'  {name}: {len(sidx)} objects')
print('Done!')
EOF
```

---

## Step 7 — Train PointNet++ Model

**Script**: `model/train.py` | **Env**: `bundlesdf` (GPU required)  
**Output**: `output/checkpoints_v5/best_model.pth`

```bash
conda activate bundlesdf

python3 -m model.train \
    --epochs 200 \
    --batch_size 16 \
    --lr 0.001 \
    --fc_lambda 10.0 \
    --dataset_dir output/dataset \
    --save_dir output/checkpoints_v5
```

Good training signs: **F1 > 70%**, **FC error < 15mm** by epoch 100+.

---

## Data Flow

```
UCBProject/ObjMesh          →  data_hub/meshes/SAM3DMesh/{oakink,ycb}/
UCBProject/ProcessedData    →  data_hub/ProcessedData/training_fp/{oakink,dexycb}/

Step 2: training_fp/ + meshes/     →  output/grasps_random/{obj}_grasp.hdf5
Step 3: meshes/                    →  output/assets/{obj}.usd
Step 4: grasps_random/ + assets/   →  output/robot_gt_auto/{obj}_robot_gt.hdf5
Step 5: training_fp/ + robot_gt/   →  data_hub/training_m5/{obj}.hdf5
Step 6: training_m5/               →  output/dataset/affordance_{train,val}.h5
Step 7: dataset/                   →  output/checkpoints_v5/best_model.pth
```

---

## Quick Troubleshooting

| Problem | Fix |
|---------|-----|
| `h5py numpy ABI error` | Use `bundlesdf` env, not `base` |
| `training_fp/oakink/ empty` | Re-run Step 1B |
| `mesh not found: A01001` | Re-run Step 1A; check SAM3DMesh path structure |
| `USD not found: A01001.usd` | Run Step 3 first |
| `CUDA out of memory` | Use `--batch_size 8` or `4` |
| FC error > 100mm after 200 epochs | Check that robot_gt is non-zero in training_m5 files |
