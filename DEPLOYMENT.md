# Deployment Guide

> Quick reference for the company team. See **README.md** for the full pipeline.

---

## Commands by Step

### Environment: `depth-pro`

```bash
# Step 1 · Depth Pro — DexYCB (run each camera separately)
for CAM in 841412060263 840412060917 932122060857 836212060125 932122061900 932122062010; do
    python data/batch_depth_pro.py --dataset dexycb --cam $CAM --two-pass --max-frames 150
done

# Step 1 · Depth Pro — HO3D v3
python data/batch_depth_pro.py --dataset ho3d_v3 --two-pass --max-frames 150
```

### Environment: `haptic`

```bash
# Step 2 · HaPTIC — MANO hand estimation
python data/batch_haptic.py --dataset dexycb  --only-with-depth-k
python data/batch_haptic.py --dataset ho3d_v3 --only-with-depth-k
```

### Environment: `bundlesdf`

```bash
# Step 3 · FoundationPose — 6D object pose
python tools/batch_obj_pose.py --dataset dexycb
python tools/batch_obj_pose.py --dataset ho3d_v3

# Step 4 · Align — contact labels + human prior
python data/batch_align_mano_fp.py --dataset dexycb
python data/batch_align_mano_fp.py --dataset ho3d_v3

# Step 5 · Grasp candidates
python tools/random_grasp_sampler.py --all

# Step 7 · Build training dataset
python data/build_dataset.py --num_points 4096 --augment 3

# Step 8 · Train
python -m model.train --epochs 200 --batch_size 16

# Step 9 · Predict affordance
python -m inference.grasp_pose --mesh path/to/object.obj

# Step 10 · Verify in sim
python run.py --mesh path/to/object.obj
```

---

## DexYCB Camera Selection

| Camera | Depth Pro bias | Status |
|---|---|---|
| `841412060263` | +0.2% | ✅ Best (validated) |
| `840412060917` | −0.5% | ✅ |
| `932122060857` | −1.2% | ✅ |
| `836212060125` | −1.4% | ✅ |
| `932122061900` | +1.2% | ✅ |
| `932122062010` | +2.9% | 🟡 Acceptable |
| `839512060362` | −9.0% | ❌ Excluded |
| `932122060861` | −10.4% | ❌ Excluded |

Run each camera separately with `--two-pass`. Self-calibration corrects bias without GT intrinsics.

---

## Download Assets from HuggingFace

```bash
pip install huggingface_hub

# Model weights + FP masks + object meshes (~12 GB)
python setup_weights.py

# Or download specific datasets:
python setup_weights.py --tool oakink   # OakInk raw data (~25 GB)
python setup_weights.py --tool egodex   # EgoDex raw data (~30 GB)
```

---

## Raw Data Sources

| Dataset | Link |
|---|---|
| DexYCB | https://dex-ycb.github.io |
| HO3D v3 | https://www.tugraz.at/index.php?id=57823 |
