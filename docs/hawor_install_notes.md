# HaWoR Installation Notes

## Environment: `hawor` (Python 3.10, PyTorch 2.1.0+cu121)

> **Warning**: Do NOT follow the official HaWoR README for PyTorch version.
> The official doc says `torch==1.13` — this is wrong for our CUDA 12.1 stack.

---

## Installation

```bash
conda create -n hawor python=3.10 -y
conda activate hawor

# 1. PyTorch 2.1.0 + CUDA 12.1 (exact version required)
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# 2. Core dependencies
pip install pytorch-lightning==1.9.5
pip install smplx einops trimesh roma chumpy loguru yacs

# 3. torch-scatter (must match torch 2.1 + cu121 — pre-built wheel)
pip install torch-scatter==2.1.2+pt21cu121 \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# 4. pytorch3d (pre-built wheel for py310 + cu121 + torch2.1)
pip install --no-index --no-deps \
    https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt210/pytorch3d-0.7.5-cp310-cp310-linux_x86_64.whl

# 5. lietorch (must be compiled — no pip wheel)
# WARNING: compile only inside the hawor submodule, NOT from project root
cd third_party/hawor/thirdparty/lietorch
python setup.py install
cd -

# 6. Install HaWoR itself
cd third_party/hawor
pip install -e . --no-deps
cd -

# 7. Metric3D (optional, used by some HaWoR modes)
cd third_party/hawor/thirdparty/Metric3D
pip install -e . --no-deps
cd -

# 8. Final check
conda activate hawor
python -c "import hawor; print('HaWoR OK')"

# 9. Verify
python demo.py --video_path ./example/video_0.mp4 --vis_mode cam
```

---

## Model Weights

| File | Path | Source |
|------|------|--------|
| `detector.pt` | `weights/external/` | HuggingFace WiLoR |
| `droid.pth` | `weights/external/` | Google Drive (DROID-SLAM official) |
| `hawor.ckpt` | `weights/hawor/checkpoints/` | HuggingFace ThunderVVV/HaWoR |
| `infiller.pt` | `weights/hawor/checkpoints/` | HuggingFace ThunderVVV/HaWoR |
| `model_config.yaml` | `weights/hawor/` | HuggingFace ThunderVVV/HaWoR |
| `MANO_RIGHT.pkl` | `_DATA/data/mano/` | MANO website (registration required) |
| `MANO_LEFT.pkl` | `_DATA/data_left/mano_left/` | MANO website (registration required) |
| `metric_depth_vit_large_800k.pth` | `thirdparty/Metric3D/weights/` | Google Drive (Metric3D official) |

All weights are pre-downloaded via `python setup_weights.py --tool hawor`.

---

## HaWoR Output Format

After running `run_hawor_seq.py`, the output `.npz` file contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `right_verts` | (T, 778, 3) | Right hand MANO vertices (world coords) |
| `left_verts` | (T, 778, 3) | Left hand MANO vertices (world coords) |
| `R_w2c` | (T, 3, 3) | Camera rotation (world → camera) |
| `t_w2c` | (T, 3) | Camera translation (world → camera) |
| `R_c2w` | (T, 3, 3) | Camera rotation (camera → world) |
| `t_c2w` | (T, 3) | Camera translation (camera → world) |
| `img_focal` | scalar | Focal length (pixels) |
| `pred_trans` | (2, T, 3) | MANO root translation [0=left, 1=right] |
| `pred_rot` | (2, T, 3) | MANO root rotation (axis-angle) |
| `pred_hand_pose` | (2, T, 45) | MANO hand pose (15 joints × 3 axis-angle) |
| `pred_betas` | (2, T, 10) | MANO shape coefficients |
| `pred_valid` | (2, T) | Valid frame mask |
