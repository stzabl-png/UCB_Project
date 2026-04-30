# Affordance2Grasp

**Learning robot grasp poses from human hand-object interaction videos.**

Pipeline: `Video → Depth + Pose Estimation → Human Contact Prior → Sim Filtering → Affordance Model → Robot Grasp`

Supports both **third-person** (DexYCB, HO3D v3) and **egocentric / first-person** (EgoDex, PH2D-AVP) video sources.

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

### Third-Person Datasets

| Dataset | Objects | Sequences | Frames | Depth Pro bias |
|---|---|---|---|---|
| **DexYCB** | 20 YCB objects | ~3,600 (6 cameras) | ~260K | +0.2% |
| **HO3D v3** | 8 YCB objects | 68 | ~90K | -1.5% |
| **OakInk v1** | 100 objects (hand-object) | 778 | ~45K | -5.7% |

All three datasets use **identical scripts** — swap `--dataset dexycb` / `--dataset ho3d_v3` / `--dataset oakink`.

### Egocentric Datasets (First-Person)

| Dataset | Objects | Source |
|---|---|---|
| **EgoDex** | dexterous manipulation objects (e.g. `assemble_tile`) | Apple Vision Pro |
| **PH2D-AVP** | household objects (e.g. `orange`) | Apple Vision Pro |

Egocentric sequences use **HaWoR** (MANO) + **MegaSAM** (camera poses + metric depth) + **FoundationPose** (object pose).

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
pip install trimesh scipy h5py opencv-python natsort tqdm torch
pip install fast-simplification   # ← 关键：缺少此库会导致 mesh 不被简化，Step 4 慢 60×
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
| OakInk v1 | https://oakink.net | `data_hub/RawData/ThirdPersonRawData/oakink_v1/` |

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
conda activate haptic

python data/batch_haptic.py --dataset dexycb   --only-with-depth-k
python data/batch_haptic.py --dataset ho3d_v3  --only-with-depth-k
python data/batch_haptic.py --dataset oakink   --only-with-depth-k
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
```

Output: `data_hub/ProcessedData/obj_poses/{dataset}/{seq_id}/ob_in_cam/{frame}.txt`
- 4×4 transformation matrix per frame (object-in-camera)

---

### Step 4 · Contact Label Generation — Human Prior

> **依赖**: 必须先安装 `fast-simplification`（见 Setup），否则 mesh 不被简化（664k→5k 顶点），
> DexYCB 耗时从 **15 分钟变为 23 小时**。

**对齐原理**：
- 所有第三人称数据集统一使用 **3D + 深度比例修正**
  (`scale_d = Z_obj / Z_hand`，修正 HaPTIC weak-perspective 深度约 1.3× 高估)
- 守卫条件 `scale_d ∈ [0.3, 3.0]` 过滤深度异常帧
- 归一化方式：`count / max_count`（per-object），防止序列多的物体接触值被稀释
- Force center 使用 top 20% 高接触区域，防止接触弱时定位到零向量

```bash
conda activate bundlesdf

# 正式运行（直接写入 training_fp/，覆盖旧数据）
python data/batch_align_mano_fp.py --dataset dexycb
python data/batch_align_mano_fp.py --dataset ho3d_v3

# 对比运行（输出到独立目录，保留原数据）
python data/batch_align_mano_fp.py --dataset dexycb   --out_suffix v2
python data/batch_align_mano_fp.py --dataset ho3d_v3  --out_suffix v2
# 输出 → training_fp_v2/{dataset}/{obj}.hdf5

# 重新计算已存在的 HDF5
python data/batch_align_mano_fp.py --dataset dexycb --redo
```

Output:
- `data_hub/ProcessedData/training_fp/{dataset}/{obj}.hdf5`
  - `point_cloud` (4096, 3)、`normals` (4096, 3)、`human_prior` (4096,)、`force_center` (3,)
  - `attrs`: `n_seqs`、`threshold`、`dataset`
- `data_hub/ProcessedData/human_prior_fp/{obj}.hdf5` — 推理接口兼容格式

---

### Step 4b · 验证接触图数据质量

```bash
# 查看所有物体的 max_hp / coverage 统计
python3 - <<'EOF'
import h5py, glob, os, numpy as np
base = "data_hub/ProcessedData/training_fp/dexycb"
print(f"{'物体':<25} {'max_hp':>8} {'cov>0.1':>9} {'cov>0.5':>9}")
for p in sorted(glob.glob(f"{base}/*.hdf5")):
    name = os.path.splitext(os.path.basename(p))[0]
    with h5py.File(p) as f: hp = f["human_prior"][()]
    print(f"  {name:<25} {hp.max():>8.3f} {(hp>0.1).mean()*100:>8.1f}% {(hp>0.5).mean()*100:>8.1f}%")
EOF
```

**期望值**：v2 归一化后，所有物体 `max_hp ≥ 0.7`，`cov(>0.1) = 100%`。

---

## Visualization Tools

用一个统一工具查看任意数据集的接触热力图：

```bash
conda activate bundlesdf

# 单个物体 — Jet 热力图（默认）
python tools/vis_contact_heatmap.py --dataset dexycb   --obj 003_cracker_box
python tools/vis_contact_heatmap.py --dataset ho3d_v3  --obj 003_cracker_box

# 批量生成所有物体
python tools/vis_contact_heatmap.py --dataset dexycb   --batch --out output/vis_contact_heatmap
python tools/vis_contact_heatmap.py --dataset ho3d_v3  --batch --out output/vis_contact_heatmap
# 输出 → output/vis_contact_heatmap/{dataset}/{obj}.png

# 可选参数
#   --binary     同时显示二値面板（默认关闭）
#   --compare    对比 GT finger contacts（仅 oakink/grab）
#   --mode interactive  开启 Open3D 交互查看器
```

预期效果：**蓝→绿→红** Jet 色谱表示接触频率由低到高。
后缀语义：`max_hp ≈ 1.0`，`cov(>0.5) > 50%` 表示接触信号强。

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

| Step | DexYCB (6 cams) | HO3D v3 | OakInk v1 |
|---|---|---|---|
| 1 · Depth Pro | ~21 h | ~1.7 h | ~4 h |
| 2 · HaPTIC | ~48 h | ~1 h | ~4 h |
| 3 · FoundationPose | ~8 h | ~15 min | ~3 h |
| 4 · Contact Labels | **~15 min** † | **~1 min** | **~5 min** |
| 5–6 · Sim (robot GT) | ~varies | ~varies | ~varies |
| 7–8 · Train | ~2 h | — | — |

† 需安装 `fast-simplification`；未安装时退化为 ~23 h。

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
| Depth Pro fx bias — OakInk v1 (two-pass, 23 seqs) | **-5.7%** vs GT |
| FP tracking success rate | **100%** |
| EgoDex `assemble_tile` contact coverage | **8.0%** (2160 MANO frames) |
| DexYCB v2 contact quality (`max_hp`) | **≥ 0.76** all 27 objects |
| HO3D v3 v2 contact quality (`max_hp`) | **≥ 0.82** all 7 objects |

---

## Egocentric Extension (EgoDex / PH2D-AVP)

Egocentric sequences require a different toolchain from third-person:

```
Ego RGB Video
    │
    ├─ HaWoR (hawor env) ──────────→ world_space_res.pth  (MANO params, world frame)
    │
    ├─ MegaSAM ────────────────────→ cam_c2w.npy + depth.npz + K.npy  (metric depth)
    │
    ├─ SAM2 mask (manual, once) ───→ obj_recon_input/egocentric/{obj}/0.png
    │
    ├─ estimate_obj_scale_ego.py ──→ scale.json  (d_real / d_mesh via MegaSAM depth)
    │
    ├─ FoundationPose (bundlesdf) ─→ ob_in_cam/*.txt  (6D object pose per frame)
    │
    └─ gen_ego_contact_map.py ─────→ human_prior/{obj}.hdf5  (contact heatmap)
```

### Prerequisites

```bash
# Environments
conda activate hawor      # HaWoR, MANO, contact map generation
conda activate bundlesdf  # FoundationPose

# Data layout expected:
# data_hub/RawData/ThirdPersonRawData/egodex/test/{seq}/extracted_images/
# data_hub/ProcessedData/egocentric_depth/{dataset}/{seq}/  ← MegaSAM output
# data_hub/ProcessedData/obj_recon_input/egocentric/{obj}/0.png  ← SAM2 mask
# data_hub/ProcessedData/obj_meshes/egocentric/{obj}/mesh.ply    ← SAM3D mesh
```

### Step E1 · HaWoR — MANO Hand Estimation (egocentric)

> HaWoR runs on the extracted RGB frames and outputs world-space MANO parameters
> using MegaSAM SLAM poses as the world coordinate reference.

```bash
# Register sequence configuration in tools/gen_ego_contact_map.py (SEQUENCE_REGISTRY)
# then run HaWoR (see Video2MANO&Mesh/hawor for invocation details)
# Output: data_hub/RawData/ThirdPersonRawData/{dataset}/{seq}/world_space_res.pth
```

### Step E2 · MegaSAM — Camera Poses + Metric Depth

> MegaSAM provides cam_c2w (camera-to-world transforms) and metric depth maps
> that define the shared world coordinate frame used by both HaWoR and FoundationPose.

```bash
# Output layout (60-frame subsampled):
# data_hub/ProcessedData/egocentric_depth/{dataset}/{seq}/
#   depth.npz   # (60, H, W) float32 metric depth in metres
#   K.npy       # 3×3 camera intrinsic at depth resolution
#   cam_c2w.npy # (60, 4, 4) camera-to-world transforms
```

### Step E3 · Object Scale Estimation (egocentric)

> Uses MegaSAM metric depth + SAM2 object mask to estimate the real object diameter,
> then computes scale_factor = d_real / d_mesh to correct SAM3D reconstruction scale.
> This mirrors data/estimate_obj_scale.py (Depth Pro) for third-person data.

```bash
conda run -n hawor python data/estimate_obj_scale_ego.py
# or single object:
conda run -n hawor python data/estimate_obj_scale_ego.py --obj assemble_tile
# Output: data_hub/ProcessedData/obj_meshes/egocentric/{obj}/scale.json
```

### Step E4 · SAM2 Mask Annotation (once per object)

> Annotate the object mask on the first (or most representative) RGB frame.
> Save as `data_hub/ProcessedData/obj_recon_input/egocentric/{obj}/0.png`.

### Step E5 · FoundationPose — Object 6D Pose Tracking (egocentric)

> Uses the scale-corrected mesh (auto-read from scale.json) to estimate object
> pose in each depth/RGB frame. Output is compatible with gen_ego_contact_map.py.

```bash
# All egocentric sequences
conda run -n bundlesdf python tools/batch_obj_pose_ego.py

# Single dataset only
conda run -n bundlesdf python tools/batch_obj_pose_ego.py --dataset egodex

# Force redo (ignore cache)
conda run -n bundlesdf python tools/batch_obj_pose_ego.py --dataset egodex --redo

# Output:
# data_hub/ProcessedData/obj_poses_ego/{dataset}/{seq}/
#   ob_in_cam/000000.txt  ...  (4×4 pose per depth frame)
#   track_vis/000000.png  ...  (debug visualisation)
```

### Step E6 · Contact Map Generation (egocentric)

> Synchronises sparse HaWoR MANO frames with dense FoundationPose object trajectories
> via nearest-frame matching (not step-based) to maximise valid frame utilisation.
> Output format is identical to third-person `human_prior/*.hdf5`.

```bash
# All objects defined in SEQUENCE_REGISTRY
conda run -n hawor python tools/gen_ego_contact_map.py

# Single object
conda run -n hawor python tools/gen_ego_contact_map.py --obj assemble_tile

# Output:
# data_hub/human_prior/{obj}.hdf5
#   point_cloud   (4096, 3)  — sampled from mesh surface
#   normals       (4096, 3)
#   human_prior   (4096,)    — contact probability [0, 1]
```

### Step E7 · Visualise Contact Map

> Renders a 2-panel figure (binary + continuous jet) consistent with vis_3panel.py.
> Uses 20K-point dense mesh sampling + KNN interpolation, white background.

```bash
conda run -n hawor python tools/vis_ego_contact.py --obj assemble_tile
# Output: output/affordance_ego/{obj}_contact_3d.png
```

### Full EgoDex Runbook — 全量 3051 条序列

> **前提:** 帧已预提取完成 (EgoDex 官方数据自带 `extracted_images/`，无需手动提帧)。

```bash
cd /home/lyh/Project/Affordance2Grasp

# ── 实时进度监控 (另开 terminal，全程可用) ──────────────────────────
bash tools/egodex_progress.sh

# ── Step 1: MegaSAM — 相机位姿 + Metric Depth ──────────────────────
# 环境: mega_sam  |  必须在 mega-sam/ 子目录下运行
cd mega-sam
conda run -n mega_sam python ../data/batch_megasam.py --dataset egodex
# 断点续跑 (中断后重执行，已有 depth.npz 的自动跳过):
conda run -n mega_sam python ../data/batch_megasam.py --dataset egodex --resume
cd ..

# ── Step 2: HaWoR — MANO 手部估计 ─────────────────────────────────
# 环境: hawor  |  mp4 直接输入，输出 world_space_res.pth
HAWOR_DIR="/home/lyh/Project/Video2MANO&Mesh/hawor"
EGODEX_RAW="data_hub/RawData/ThirdPersonRawData/egodex/test"
conda run -n hawor bash -c "
  cd $HAWOR_DIR
  for mp4 in \$OLDPWD/$EGODEX_RAW/*/*.mp4; do
    task=\$(dirname \"\$mp4\" | xargs basename)
    subj=\$(basename \"\${mp4%.mp4}\")
    out=\"\$OLDPWD/$EGODEX_RAW/\$task/\$subj/world_space_res.pth\"
    [ -f \"\$out\" ] && echo \"  ⏭  \$task/\$subj\" && continue
    echo \"  ▶  \$task/\$subj\"
    python demo.py --video_path \"\$mp4\" --input_type file ||  \\
      echo \"  ❌ \$task/\$subj failed\"
  done
"

# ── Step 3: SAM2 标注 + SAM3D 重建 (每个新物体手动做一次) ──────────
# 见下方 "Registering a New Egocentric Sequence"

# ── Step 4: 注册新序列 (编辑 3 个文件，见下方说明) ─────────────────

# ── Step 5: Scale 估计 ─────────────────────────────────────────────
conda run -n hawor python data/estimate_obj_scale_ego.py

# ── Step 6: FoundationPose 物体位姿 ────────────────────────────────
conda run -n bundlesdf python tools/batch_obj_pose_ego.py --dataset egodex
# 强制重跑: 加 --redo

# ── Step 7: 接触图生成 ─────────────────────────────────────────────
conda run -n hawor python tools/gen_ego_contact_map.py

# ── Step 8: 可视化 ──────────────────────────────────────────────────
conda run -n hawor python tools/vis_ego_contact.py --obj assemble_tile
eog output/affordance_ego/assemble_tile_contact_3d.png
```

### 实时进度监控

另开 terminal，全程运行以查看各步骤完成数量（每 10 秒刷新）：

```bash
bash tools/egodex_progress.sh
```

或用 `watch` 一次性查看当前快照：

```bash
# 快速查看各步骤完成数
echo "Step 0 帧提取:" && find data_hub/RawData/ThirdPersonRawData/egodex/test -name "extracted_images" -type d | wc -l
echo "Step 1 MegaSAM:" && find data_hub/ProcessedData/egocentric_depth/egodex -name "depth.npz" | wc -l
echo "Step 2 HaWoR:"   && find data_hub/RawData/ThirdPersonRawData/egodex/test -name "world_space_res.pth" | wc -l
echo "Step 7 接触图:"  && ls data_hub/human_prior/*.hdf5 2>/dev/null | wc -l
```

### Registering a New Egocentric Sequence

1. Add entry to `SEQUENCE_REGISTRY` in `tools/batch_obj_pose_ego.py`
2. Add entry to `SEQUENCE_REGISTRY` in `tools/gen_ego_contact_map.py`
3. Add entry to `OBJ_DEPTH_MAP` in `data/estimate_obj_scale_ego.py`
4. Place SAM2 mask at `obj_recon_input/egocentric/{obj}/0.png`
5. Place SAM3D mesh at `obj_meshes/egocentric/{obj}/mesh.ply`
6. Run Steps E1–E4 above

### Estimated Runtime (single RTX 4080, full EgoDex 3051 sequences)

| Step | 耗时 (单 GPU) | 说明 |
|---|---|---|
| 0 · 帧提取 | ✅ 已完成 | EgoDex 官方数据自带 extracted_images |
| 1 · MegaSAM | ~15-30 min/条 × 3051 | 支持 `--resume` 断点续跑 |
| 2 · HaWoR | ~2-5 min/条 × 3051 | shell 循环，中断后重跑自动跳过 |
| 3 · SAM2+SAM3D | 手动，每物体一次 | — |
| 5 · Scale 估计 | < 1 min/物体 | — |
| 6 · FP 位姿 | ~15 min/序列 | 按注册序列数决定 |
| 7 · 接触图 | ~2 min/物体 | — |
| 8 · 可视化 | ~1 min/物体 | — |

### Key Differences vs Third-Person Pipeline

| | Third-Person (DexYCB / HO3D) | Egocentric (EgoDex / PH2D) |
|---|---|---|
| Hand estimation | HaPTIC → camera space | HaWoR → world space |
| Depth source | Depth Pro (single image) | MegaSAM (SLAM, metric) |
| Scale estimation | `estimate_obj_scale.py` (Depth Pro) | `estimate_obj_scale_ego.py` (MegaSAM) |
| Object pose | `batch_obj_pose.py` | `batch_obj_pose_ego.py` |
| Contact map | `gen_m5_training_data.py` | `gen_ego_contact_map.py` |
| Output HDF5 | `data_hub/human_prior/` (compatible) | `data_hub/human_prior/` (identical format) |
