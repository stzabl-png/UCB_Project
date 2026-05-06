# Object Reconstruction Module

3D object mesh reconstruction from a single RGB image and object mask.

Based on publicly available model weights. Source code integrated to enable
standalone use without external dependencies.

## Quick Setup

```bash
# Install environment & download model weights (~12GB, one-time)
bash scripts/setup_obj_recon.sh
```

## Input Format

Each object to reconstruct needs a folder with:
```
input/
  {seq_id}/
    image.png   ← RGB scene image
    0.png       ← binary object mask (white = object)
```

## Usage

### Single object (test)
```bash
conda activate obj-recon
cd third_party/obj_recon
python demo.py
# Output: splat.ply (Gaussian Splat)
```

### Batch processing (all datasets)
```bash
# Step 1: Prepare inputs (select best frame + generate mask)
conda activate depth-pro
bash scripts/prepare_obj_recon_all.sh

# Step 2: Run reconstruction
conda activate obj-recon
python third_party/obj_recon/batch_infer.py \
  --input-dir  data_hub/ProcessedData/obj_recon_input \
  --output-dir data_hub/ProcessedData/obj_meshes

# Quick test (3 sequences per dataset)
python third_party/obj_recon/batch_infer.py \
  --input-dir  data_hub/ProcessedData/obj_recon_input \
  --output-dir data_hub/ProcessedData/obj_meshes \
  --limit 3
```

## Output

```
data_hub/ProcessedData/obj_meshes/{dataset}/{seq_id}/
  splat.ply   ← 3D Gaussian Splat
  mesh.ply    ← Triangle mesh (if available)
```

## Model Weights

Downloaded automatically by `setup_obj_recon.sh` from the official public release.
Weights are stored in `third_party/obj_recon/checkpoints/hf/` (~12GB).
This directory is excluded from git (see `.gitignore`).
