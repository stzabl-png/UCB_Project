# Affordance2Grasp — DexYCB Pipeline Deployment Guide

> **目标**: 对 DexYCB top-5 cameras 全量 600 条序列跑完整 pipeline，生成接触标签训练数据。

---

## 1. 环境准备

### 1.1 克隆代码

```bash
git clone https://github.com/stzabl-png/UCB_Project.git Affordance2Grasp
cd Affordance2Grasp
```

### 1.2 创建三个 Conda 环境

```bash
# ① Depth Pro (内参估计 + 深度图)
conda create -n depth-pro python=3.9 -y
conda activate depth-pro
pip install git+https://github.com/apple/ml-depth-pro.git
pip install natsort tqdm h5py numpy

# ② HaPTIC (MANO 手部估计)
# 按照 HaPTIC 官方 README 安装:
# https://github.com/[haptic-repo]

# ③ BundleSDF / FoundationPose (物体位姿跟踪 + 对齐)
conda create -n bundlesdf python=3.9 -y
conda activate bundlesdf
# 按照项目 README 安装依赖 (trimesh, scipy, h5py, opencv, etc.)
pip install trimesh scipy h5py opencv-python natsort tqdm fast_simplification
# 安装 FoundationPose:
cd /path/to/FoundationPose && pip install -e .
```

### 1.3 修改 config.py

```python
# config.py 中设置路径
DATA_HUB   = "/your/path/to/data_hub"        # 数据根目录
HAPTIC_DIR = "/your/path/to/HaPTIC"          # HaPTIC 代码目录
FP_ROOT    = "/your/path/to/FoundationPose"  # FoundationPose 代码目录
```

---

## 2. 数据准备

### 2.1 下载 DexYCB 原始数据

```bash
# 官方: https://dex-ycb.github.io/
# 下载后放到:
$DATA_HUB/RawData/ThirdPersonRawData/dexycb/
# 目录结构:
# dexycb/
#   20200709-subject-01/
#     20200709_141754/
#       841412060263/  ← 相机序列号目录
#         color_000000.jpg
#         ...
```

### 2.2 从 Hugging Face 下载预处理数据

```bash
pip install huggingface_hub

python - << 'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="stzabl-png/Affordance2Grasp-ProcessedData",
    repo_type="dataset",
    local_dir="/your/path/to/data_hub/ProcessedData",
    ignore_patterns=["*.git*"]
)
EOF
```

Hugging Face 包含以下预处理数据（可跳过对应步骤）：

| 目录 | 大小 | 说明 |
|------|------|------|
| `obj_meshes/ycb/` | ~1GB | YCB 网格 + scale.json |
| `obj_recon_input/ycb/` | ~50MB | FP 初始化 mask |

---

## 3. Pipeline 执行（按顺序）

### 相机选择策略

根据 Depth Pro 精度分析，使用以下相机：

| 优先级 | 相机序列号 | fx 偏差 |
|--------|-----------|---------|
| **主选** | `841412060263` | +0.2% |
| 备选 | `840412060917` | -0.5% |
| 备选 | `932122060857` | -1.2% |
| 备选 | `836212060125` | -1.4% |
| 备选 | `932122061900` | +1.2% |
| 不用 | `932122062010` | +2.9% |
| 不用 | `839512060362` | -9.0% |
| 不用 | `932122060861` | -10.4% |

配置文件: `data/dexycb_camera_config.json`

---

### Step 1: Depth Pro 内参 + 深度估计

```bash
conda activate depth-pro
cd /path/to/Affordance2Grasp

# 主选相机全量 (600 条序列, ~3.3 小时)
python data/batch_depth_pro.py \
    --dataset dexycb \
    --cam 841412060263 \
    --max-frames 72

# 备选相机 (可选，如需全 5 台)
for CAM in 840412060917 932122060857 836212060125 932122061900; do
    python data/batch_depth_pro.py \
        --dataset dexycb \
        --cam $CAM \
        --max-frames 72
done
```

输出: `$DATA_HUB/ProcessedData/third_depth/dexycb/{seq_id}/K.txt + depths.npz`

---

### Step 2: HaPTIC MANO 手部估计

```bash
conda activate haptic
cd /path/to/Affordance2Grasp

# 仅处理有 Depth Pro K 的序列
python data/batch_haptic.py \
    --dataset dexycb \
    --only-with-depth-k
```

输出: `$DATA_HUB/ProcessedData/third_mano/dexycb/{seq_id}.npz`

---

### Step 3: FoundationPose 物体 6D 位姿跟踪

```bash
conda activate bundlesdf
cd /path/to/Affordance2Grasp

# 自动发现所有有深度数据的序列（无需指定物体/相机）
python tools/batch_obj_pose.py --dataset dexycb
```

输出: `$DATA_HUB/ProcessedData/obj_poses/dexycb/{seq_id}/ob_in_cam/{frame}.txt`

---

### Step 4: MANO × FP 对齐 → 接触标签生成

```bash
conda activate bundlesdf

python data/batch_align_mano_fp.py --dataset dexycb
```

输出:
- `$DATA_HUB/ProcessedData/training_fp/dexycb/{obj}.hdf5` — 训练数据
- `$DATA_HUB/ProcessedData/human_prior_fp/dexycb/` — 逐顶点接触先验

---

## 4. 预期时间 (RTX 4080 / 单卡)

| 步骤 | 600 条序列 | 备注 |
|------|-----------|------|
| Step 1 Depth Pro | ~3.3 h | 每帧 ~1.7s |
| Step 2 HaPTIC | ~8 h | 每帧 ~46s |
| Step 3 FP Pose | ~1.3 h | 每序列 ~8s |
| Step 4 Align | ~5 min | CPU |
| **合计** | **~13 h** | 流水线可并行 |

**流水线并行建议**: Step 1 和 Step 2 串行（同 GPU），Step 1/2 跑完后 Step 3/4 可以立刻对已完成序列开始处理。

---

## 5. 验证

```bash
conda activate bundlesdf

# 检查输出数量
ls $DATA_HUB/ProcessedData/third_depth/dexycb/ | wc -l   # 应为 600
ls $DATA_HUB/ProcessedData/third_mano/dexycb/  | wc -l   # 应为 600
ls $DATA_HUB/ProcessedData/obj_poses/dexycb/   | wc -l   # 应为 600

# 查看接触覆盖率
python - << 'EOF'
import h5py, os, glob
for f in glob.glob("data_hub/ProcessedData/training_fp/dexycb/*.hdf5"):
    with h5py.File(f) as h:
        hp = h['human_prior'][:]
        print(f"{os.path.basename(f)}: max={hp.max():.4f}  >0.05: {(hp>0.05).mean():.1%}")
EOF
```

---

## 6. 联系

如遇问题请联系 [lyh]。已验证配置：
- DexYCB + cam `841412060263` (003_cracker_box, 30 seqs): coverage=100%, diverged=0
- Depth Pro fx 偏差: +0.2%
- FP 跟踪成功率: 100%
