# SAM3D Cloud Reconstruction Guide

## Server Info

| Item | Value |
|---|---|
| SSH alias | `ssh sam3d-gpu` |
| GPU | 2× NVIDIA A800 80 GB (GPU 0 usually free) |
| SAM3D code | `/root/lyh/sam-3d-objects/` |
| Conda env | `sam3d-objects` |

---

## Directory Convention

```
~/input/{dataset}/{obj_name}/
    image.png    ← best RGB frame (object clearly visible)
    0.png        ← SAM2 binary mask (white = object, black = background)

~/output/{dataset}/{obj_name}/
    mesh.ply     ← SAM3D reconstructed mesh
    splat.ply    ← 3D Gaussian splat (optional, can be ignored)
```

---

## Step 1 — Local: Annotate Masks (once per new object)

Use the annotation tool to pick a representative frame and draw a mask:

```bash
conda activate base
cd $PROJ

# Annotate all egodex tasks (or use --start N to resume)
python tools/sam2_annotate_by_object.py --dataset egodex

# Re-annotate a specific task
python tools/sam2_annotate_by_object.py --dataset egodex \
    --task add_remove_lid --redo
```

**Annotation keys:**

| Key | Action |
|---|---|
| `← →` | Previous / next frame |
| `PgUp / PgDn` | Jump 10 frames |
| `Home / End` | Jump to first / last frame |
| **Left click** | Add foreground point → SAM2 generates mask |
| `B` | Toggle foreground / background mode |
| `D` | Toggle MegaSAM depth overlay |
| `C` | Clear all points and restart |
| `Enter` | **Save frame + mask**, move to next task |
| `S` | Skip this task |
| `Q` | Quit, keep progress |

**Output:**
```
data_hub/ProcessedData/obj_recon_input/egocentric/{task}/
    image.png    ← selected RGB frame
    0.png        ← SAM2 binary mask
```

**Check annotation status:**
```bash
for task in slot_batteries stack_unstack_cups fry_bread build_unstack_lego \
            flip_pages basic_pick_place basic_fold throw_collect_objects \
            assemble_disassemble_furniture_bench_desk; do
    if [ -f "data_hub/ProcessedData/obj_recon_input/egocentric/$task/0.png" ]; then
        echo "✅ $task"
    else
        echo "❌ $task — needs annotation"
    fi
done
```

---

## Step 2 — Local: Prepare Input

```bash
cd $PROJ
python tools/prep_sam3d_input.py --dataset egodex

# Transfer to cloud
rsync -avz /tmp/sam3d_input/ sam3d-gpu:~/input/
```

---

## Step 3 — Cloud: Run SAM3D Inference

> **Note**: Non-interactive SSH does not load `.bashrc`, so `conda` is unavailable.
> Use the full Python path instead.

```bash
# Option A: Run via SSH remote command (recommended)
ssh sam3d-gpu "
cd ~/lyh/sam-3d-objects
CUDA_VISIBLE_DEVICES=0 \
CONDA_PREFIX=~/miniconda3/envs/sam3d-objects \
~/miniconda3/envs/sam3d-objects/bin/python batch_infer.py \
    --input-dir ~/input \
    --output-dir ~/output \
    --dataset egodex 2>&1 | tee ~/output_log.txt
"

# Option B: SSH in and run interactively
ssh sam3d-gpu
cd ~/lyh/sam-3d-objects
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sam3d-objects
export CUDA_VISIBLE_DEVICES=0
python batch_infer.py --input-dir ~/input --output-dir ~/output --dataset egodex

# Debug: single object only
python batch_infer.py --input-dir ~/input --output-dir ~/output \
    --dataset egodex --seq battery
```

---

## Step 4 — Cloud → Local: Transfer Meshes

```bash
# On local machine
rsync -avz sam3d-gpu:~/output/egodex/ \
    $PROJ/data_hub/ProcessedData/obj_meshes/egocentric/

# Verify
ls $PROJ/data_hub/ProcessedData/obj_meshes/egocentric/
```

---

## Step 5 — Local: Scale Estimation

SAM3D outputs are in normalized coordinates, not metric meters.
Run scale estimation using MegaSAM depth:

```bash
cd $PROJ
conda activate hawor
python data/estimate_obj_scale_ego.py --obj battery
# Output: obj_meshes/egocentric/battery/scale.json
```

FoundationPose and alignment scripts read `scale.json` automatically.

---

## SAM3D Python API (single call)

```python
import sys
sys.path.append("/root/lyh/sam-3d-objects/notebook")
from inference import Inference, load_image, load_single_mask

# Load model once
inference = Inference("checkpoints/hf/pipeline.yaml", compile=False)

# Run inference
image  = load_image("path/to/image.png")
mask   = load_single_mask("path/to/mask_dir", index=0)   # 0 → 0.png
output = inference(image, mask, seed=42)

# Save mesh
output["mesh"].export("mesh.ply")
```

---

## EgoDex Object Registry (9 objects)

| obj_name | EgoDex sequence | Notes |
|---|---|---|
| `bench_desk` | `assemble_disassemble_furniture_bench_desk/15` | Table components |
| `cloth` | `basic_fold/38` | Deformable ⚠️ |
| `pick_object` | `basic_pick_place/259` | Generic pick-and-place |
| `lego` | `build_unstack_lego/9` | LEGO bricks |
| `book` | `flip_pages/14` | Book |
| `bread` | `fry_bread/0` | Bread |
| `battery` | `slot_batteries/1` | Battery |
| `cup` | `stack_unstack_cups/11` | Cup |
| `ball` | `throw_collect_objects/5` | Ball |

---

## Data Paths (Local)

| Type | Path |
|---|---|
| Raw RGB frames | `data_hub/RawData/EgoRawData/egodex/test/{task}/{ep}/extracted_images/` |
| MegaSAM depth + intrinsics | `data_hub/ProcessedData/egocentric_depth/egodex/{task}/{ep}/` |
| HaWoR MANO output | `data_hub/ProcessedData/ego_mano/egodex/{task}/{ep}.npz` |
| SAM2 init mask | `data_hub/ProcessedData/obj_recon_input/egocentric/{obj}/0.png` |
| SAM3D mesh | `data_hub/ProcessedData/obj_meshes/egocentric/{obj}/mesh.ply` |
| FP object poses | `data_hub/ProcessedData/obj_poses_ego/egodex/{task}/{ep}/ob_in_cam/` |
| Contact HDF5 | `data_hub/ProcessedData/training_fp_ego/egodex/{obj}.hdf5` |

---

## FAQ

**Q: GPU 1 is busy, how to use GPU 0?**
```bash
nvidia-smi
export CUDA_VISIBLE_DEVICES=0
```

**Q: Mask format requirements?**
- Single-channel or RGB PNG: white (255) = object, black (0) = background.
- Must match `image.png` dimensions exactly.
- File must be named `0.png` (required by `batch_infer.py`).

**Q: What unit does the output mesh use?**
SAM3D outputs normalized coordinates. Run `estimate_obj_scale_ego.py` to get metric scale via MegaSAM depth. The resulting `scale.json` is auto-read by FP and alignment scripts.
