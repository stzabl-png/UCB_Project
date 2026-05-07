# A100 快速部署指南 — 扩展 HumanPrior（加入 EgoDex + OakInk）

> **场景**：公司已有 DexYCB + HO3D 的 HumanPrior，现在需要加入 EgoDex（第一人称）和
> OakInk（第三人称），最终获得四个数据集的完整 HumanPrior 供 Policy 训练使用。
>
> **机器**：8×A100，Ubuntu 22.04，CUDA 12.x  
> **预计总时间**：EgoDex ~50 h + OakInk ~8 h（单 A100）；8 卡并行约 6-7 h 合计。

---

## Step 0 · 克隆仓库 & 下载必要资源

```bash
# 克隆（已有则跳过）
git clone --recursive https://github.com/stzabl-png/UCB_Project.git
cd UCB_Project

# 安装 HuggingFace CLI
pip install huggingface_hub

# 一键下载模型权重 + FP 初始化 mask + 物体 mesh（~12 GB）
python setup_weights.py

# 下载 OakInk 原始数据（~25 GB，第三人称）
python setup_weights.py --tool oakink

# 下载 EgoDex 原始数据（~30 GB，第一人称）
python setup_weights.py --tool egodex
```

> **已有 DexYCB/HO3D 的处理结果？** 把 `training_fp/dexycb/`、`training_fp/ho3d_v3/`
> 和 `human_prior_fp/` 直接复制到新机器的 `data_hub/ProcessedData/` 下即可。

---

## Step 1 · OakInk — 第三人称 Pipeline（Phase 1A）

OakInk 使用与 DexYCB 相同的第三人称流程（DepthPro → HaPTIC → FoundationPose → Align）。

### 1a · Depth Pro（内参 + 深度）

```bash
conda activate depth-pro

# 8 卡并行：每张卡处理 1/8 的序列（--start / --end 按总序列数分片）
TOTAL=$(python -c "
import sys; sys.path.insert(0,'.')
from data.batch_depth_pro import discover_oakink
import config
seqs = discover_oakink(config.DATA_HUB)
print(len(seqs))
")
echo "OakInk total sequences: $TOTAL"

# 示例：8 卡平均分（在 tmux / sbatch 里各开一个）
for GPU in 0 1 2 3 4 5 6 7; do
  START=$(( GPU * TOTAL / 8 ))
  END=$(( (GPU + 1) * TOTAL / 8 ))
  CUDA_VISIBLE_DEVICES=$GPU python data/batch_depth_pro.py \
    --dataset oakink --two-pass --start $START --end $END &
done
wait
```

输出：`data_hub/ProcessedData/third_depth/oakink/{seq_id}/`

### 1b · HaPTIC（手部姿态）

```bash
conda activate haptic

TOTAL=$(python -c "
import sys; sys.path.insert(0,'.')
from data.batch_haptic import discover_oakink
import config
seqs = discover_oakink(config.DATA_HUB)
print(len(seqs))
")

for GPU in 0 1 2 3 4 5 6 7; do
  START=$(( GPU * TOTAL / 8 ))
  END=$(( (GPU + 1) * TOTAL / 8 ))
  CUDA_VISIBLE_DEVICES=$GPU python data/batch_haptic.py \
    --dataset oakink --start $START --end $END &
done
wait
```

输出：`data_hub/ProcessedData/third_mano/oakink/{seq_id}.npz`

### 1c · FoundationPose（物体位姿）

> **前提**：`data_hub/ProcessedData/obj_recon_input/oakink/` 中已有初始化 mask  
> （`python setup_weights.py` 已通过 `thirdmasks` 下载）

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

输出：`data_hub/ProcessedData/obj_poses/oakink/{seq_id}/`

### 1d · Align（生成 HumanPrior）

```bash
conda activate bundlesdf

# OakInk 物体不多，单卡即可（或按 --obj 分片）
python data/batch_align_mano_fp.py --dataset oakink
```

输出：
- `data_hub/ProcessedData/training_fp/oakink/{obj}.hdf5`
- `data_hub/ProcessedData/human_prior_fp/{obj}.hdf5`（第三人称合并结果）

---

## Step 2 · EgoDex — 第一人称 Pipeline（Phase 1B）

EgoDex 使用第一人称流程（MegaSAM → HaWoR → FoundationPose → Align）。

### 2a · MegaSAM（深度 + SLAM 内参）

```bash
conda activate mega_sam

# EgoDex 共 3051 序列
TOTAL=3051
for GPU in 0 1 2 3 4 5 6 7; do
  START=$(( GPU * TOTAL / 8 ))
  END=$(( (GPU + 1) * TOTAL / 8 ))
  CUDA_VISIBLE_DEVICES=$GPU python data/batch_megasam.py \
    --dataset egodex --start $START --end $END &
done
wait
```

输出：`data_hub/ProcessedData/ego_depth/egodex/{seq_id}/`

### 2b · HaWoR（第一人称手部轨迹）

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

输出：`data_hub/ProcessedData/ego_mano/egodex/{seq_id}/`

### 2c · FoundationPose（物体位姿，第一人称视角）

> **前提**：`data_hub/ProcessedData/obj_recon_input/egocentric/` 中已有 EgoDex mask  
> （`python setup_weights.py` 已通过 `egomasks` 下载）

```bash
conda activate bundlesdf

for GPU in 0 1 2 3 4 5 6 7; do
  START=$(( GPU * TOTAL / 8 ))
  END=$(( (GPU + 1) * TOTAL / 8 ))
  CUDA_VISIBLE_DEVICES=$GPU python tools/batch_obj_pose_ego.py \
    --dataset egodex --start $START --end $END &
done
wait
```

输出：`data_hub/ProcessedData/obj_poses_ego/egodex/{seq_id}/`

### 2d · Align（生成 HumanPrior）

```bash
conda activate bundlesdf

python data/batch_align_ego_mano_fp.py --dataset egodex
```

输出：
- `data_hub/ProcessedData/training_fp_ego/egodex/{obj}.hdf5`
- `data_hub/human_prior/{obj}.hdf5`（第一人称合并结果）

---

## Step 3 · 汇总四个数据集的 HumanPrior

```bash
conda activate bundlesdf

# 验证各数据集输出
echo "=== 已有 HumanPrior ===" && \
ls data_hub/ProcessedData/training_fp/dexycb/    | wc -l && echo "DexYCB objects" && \
ls data_hub/ProcessedData/training_fp/ho3d_v3/   | wc -l && echo "HO3D objects" && \
ls data_hub/ProcessedData/training_fp/oakink/    | wc -l && echo "OakInk objects" && \
ls data_hub/ProcessedData/training_fp_ego/egodex/| wc -l && echo "EgoDex objects"

# Phase 2: aggregate — 合并 training_fp + training_fp_ego → 最终 human_prior
python data/aggregate_prior.py
```

输出：`data_hub/human_prior/{obj}.hdf5`（全部四个数据集合并）

质量验证：

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

期望：`max_hp ≥ 0.7`，`cov(>0.1) = 100%`

---

## Step 4 · 构建训练集 + 训练 Policy

```bash
conda activate bundlesdf

# 构建 HDF5 训练集（gtfree 模式，不需要 Isaac Sim）
python data/build_dataset.py --num_points 4096 --augment 3

# 训练（单机 8 卡 DDP）
python -m model.train \
    --epochs 200 \
    --batch_size 128 \
    --lr 0.001 \
    --fc_lambda 10.0
```

---

## HuggingFace 资源汇总

| 工具 | Repo | 大小 | 用途 |
|------|------|------|------|
| 模型权重 | `UCBProject/Affordance2Grasp-Weights` | ~10 GB | FP/HaWoR/HaPTIC/MegaSAM/DepthPro |
| 第三人称 FP mask | `UCBProject/ThirdDataMask` | ~30 MB | Phase 1A 初始化 mask (YCB/OakInk/TACO) |
| 第一人称 FP mask | `UCBProject/EgoDataMask` | ~70 MB | Phase 1B 初始化 mask (EgoDex/TACO) |
| 物体 mesh | `UCBProject/Affordance2Grasp-Mesh` | ~1 GB | FP + 尺度估计 |
| OakInk 数据 | `UCBProject/Affordance2Grasp-OakInk` | ~25 GB | Phase 1A 原始数据 |
| EgoDex 数据 | `UCBProject/Affordance2Grasp-EgoDex` | ~30 GB | Phase 1B 原始数据 |
| DexYCB | 官方下载 [dex-ycb.github.io](https://dex-ycb.github.io) | ~250 GB | 公司已有 |
| HO3D v3 | 官方下载 [tugraz.at](https://www.tugraz.at/index.php?id=57823) | ~6 GB | 公司已有 |

一键下载（跳过 DexYCB/HO3D）：

```bash
python setup_weights.py             # 权重 + mask + mesh
python setup_weights.py --tool oakink
python setup_weights.py --tool egodex
```

---

## 预计运行时间（单 A100）

| 步骤 | OakInk | EgoDex |
|------|--------|--------|
| DepthPro / MegaSAM | ~4 h | ~50 h |
| HaPTIC / HaWoR | ~3 h | ~40 h |
| FoundationPose | ~3 h | ~30 h |
| Align | ~5 min | ~1 h |
| **合计** | **~10 h** | **~121 h** |

**8 卡并行**：OakInk ~1.5 h，EgoDex ~15 h

---

## 常见问题

- **`--start/--end` 怎么分片？** 按 GPU 索引均分总序列数，见各步骤示例。
- **FoundationPose 找不到 mask？** 检查 `obj_recon_input/{oakink,egocentric}/` 是否已下载。
- **序列被跳过（skip）？** 该序列无对应 mask 文件，属正常，不影响其他序列。
- **更多问题** → 参考 `README.md` Troubleshooting T1-T16 和 `docs/DEPLOYMENT_ISSUES.md`
