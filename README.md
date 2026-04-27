# Affordance2Grasp — 数据生成 Pipeline

从视频生成手-物接触标签，用于机器人抓取模型训练。

> **本文档面向公司协作人员**，说明如何跑完数据生成的全流程：
> `视频原始数据 → 内参+深度 → MANO手部估计 → 物体位姿 → 接触标签`

---

## 目标数据集

| 数据集 | 物体 | 序列数 | 总帧数 |
|---|---|---|---|
| **DexYCB** | 20 种 YCB 物体 | ~600 条 | ~43,200 帧 |
| **HO3D v3** | 8 种 YCB 物体 | 68 条 | ~90,000 帧 |

两个数据集用**完全相同的脚本**处理，流程一致。

---

## 一、 环境准备

### 1.1 克隆代码

```bash
git clone https://github.com/stzabl-png/UCB_Project.git Affordance2Grasp
cd Affordance2Grasp
```

### 1.2 创建 Conda 环境

需要两个独立环境：

```bash
# ── 环境 A: Depth Pro ──────────────────────────────────────────
conda create -n depth-pro python=3.9 -y
conda activate depth-pro
pip install git+https://github.com/apple/ml-depth-pro.git
pip install natsort tqdm h5py numpy pillow

# ── 环境 B: FoundationPose + HaPTIC + 对齐 ────────────────────
conda create -n bundlesdf python=3.9 -y
conda activate bundlesdf
pip install trimesh scipy h5py opencv-python natsort tqdm fast_simplification torch
# HaPTIC: 按 HaPTIC 官方 README 安装
# FoundationPose: pip install -e /path/to/FoundationPose
```

### 1.3 修改 `config.py`

```python
DATA_HUB   = "/your/path/to/data_hub"        # 数据根目录
HAPTIC_DIR = "/your/path/to/HaPTIC"
FP_ROOT    = "/your/path/to/FoundationPose"
```

---

## 二、 下载数据

### 2.1 下载原始视频

| 数据集 | 官网 | 放置路径 |
|---|---|---|
| DexYCB | https://dex-ycb.github.io | `data_hub/RawData/ThirdPersonRawData/dexycb/` |
| HO3D v3 | https://www.tugraz.at/institute/icg/research/team-lepetit/research-projects/hand-object-3d-pose-annotation | `data_hub/RawData/ThirdPersonRawData/ho3d_v3/` |

目录结构：
```
dexycb/
  20200709-subject-01/
    20200709_141754/
      841412060263/          ← 相机序列号
        color_000000.jpg
        ...
ho3d_v3/
  train/
    ABF10/
      rgb/
        0000.jpg
        ...
  evaluation/
    ...
```

### 2.2 下载预处理文件（HuggingFace）

```bash
pip install huggingface_hub

python - << 'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="StZaBL/Affordance2Grasp-ProcessedData",
    repo_type="dataset",
    local_dir="data_hub/ProcessedData",
)
EOF
```

包含内容：

| 文件夹 | 说明 |
|---|---|
| `obj_meshes/ycb/` | 所有 YCB 物体网格 + `scale.json` |
| `obj_recon_input/ycb/` | FoundationPose 初始化 mask |

---

## 三、 运行 Pipeline

> 每个步骤**先跑 DexYCB，再跑 HO3D v3**，命令完全对称。

---

### Step 1：Depth Pro — 估计内参 + 深度图

**环境：`depth-pro`**

```bash
conda activate depth-pro
cd /path/to/Affordance2Grasp

# ── DexYCB：每台相机单独跑 --two-pass（各自校准 fx）
# 共 6 台相机（排除最差的两台），每台约 600 条序列
for CAM in 841412060263 840412060917 932122060857 836212060125 932122061900 932122062010; do
    echo "=== 处理相机 $CAM ==="
    python data/batch_depth_pro.py \
        --dataset dexycb \
        --cam $CAM \
        --two-pass \
        --max-frames 150
done

# ── HO3D v3：全部 68 条序列，自动两阶段校准
python data/batch_depth_pro.py \
    --dataset ho3d_v3 \
    --two-pass \
    --max-frames 150
```

**`--two-pass` 说明**（两阶段自校准，无需 GT 内参）：
1. Pass 1：每条序列取 30 帧，快速估计各自 fx
2. 自动计算全局中位数 fx（收敛误差 < 2%）
3. Pass 2：以全局 fx 固定内参，重跑完整深度图

输出：`data_hub/ProcessedData/third_depth/{dataset}/{seq_id}/`
- `K.txt` — 3×3 内参矩阵
- `depths.npz` — 深度图 (N, H, W) float32，单位：米
- `frame_ids.txt` — 帧文件名

预计时间（RTX 4080 SUPER）：

| 数据集 | Pass 1 | Pass 2 | 合计 |
|---|---|---|---|
| DexYCB 6台相机 × 600条 | ~3 h | ~18 h | ~21 h |
| HO3D v3 68条 | ~10 min | ~1.5 h | ~1.7 h |

---

### Step 2：HaPTIC — MANO 手部估计（第三人称）

**环境：`bundlesdf`**（含 HaPTIC）

```bash
conda activate bundlesdf

# ── DexYCB
python data/batch_haptic.py \
    --dataset dexycb \
    --only-with-depth-k

# ── HO3D v3
python data/batch_haptic.py \
    --dataset ho3d_v3 \
    --only-with-depth-k
```

`--only-with-depth-k`：只处理 Step 1 已完成的序列（保证内参已就绪）。

输出：`data_hub/ProcessedData/third_mano/{dataset}/{seq_id}.npz`
- `verts_dict`：每帧的 MANO 手部顶点 (778, 3)，相机坐标系

预计时间：

| 数据集 | 时间 |
|---|---|
| DexYCB 600条 | ~8 h |
| HO3D v3 68条 | ~1 h |

---

### Step 3：FoundationPose — 物体 6D 位姿跟踪

**环境：`bundlesdf`**

```bash
conda activate bundlesdf

# ── DexYCB
python tools/batch_obj_pose.py --dataset dexycb

# ── HO3D v3
python tools/batch_obj_pose.py --dataset ho3d_v3
```

脚本自动：
- 根据序列名识别 YCB 物体
- 载入 `obj_recon_input/ycb/` 的初始 mask
- 用 FP 跑完整序列 6D 跟踪

输出：`data_hub/ProcessedData/obj_poses/{dataset}/{seq_id}/ob_in_cam/{frame}.txt`
- 每帧一个 4×4 变换矩阵（相机坐标系）

预计时间：

| 数据集 | 时间 |
|---|---|
| DexYCB 600条 | ~1.3 h |
| HO3D v3 68条 | ~15 min |

---

### Step 4：接触标签生成

**环境：`bundlesdf`**

```bash
conda activate bundlesdf

# ── DexYCB + HO3D v3（统一使用 3D 距离法 + 深度归一化）
python data/batch_align_mano_fp.py --dataset dexycb
python data/batch_align_mano_fp.py --dataset ho3d_v3
```

输出：
- `data_hub/ProcessedData/training_fp/{dataset}/{obj}.hdf5` — 训练数据
- `data_hub/ProcessedData/human_prior_fp/{dataset}/` — 逐顶点接触先验

**Step 4 < 5 分钟（CPU）**

---

## 四、 验证

```bash
conda activate bundlesdf

python - << 'EOF'
import h5py, os, glob
for ds in ['dexycb', 'ho3d_v3']:
    h5files = glob.glob(f"data_hub/ProcessedData/training_fp/{ds}/*.hdf5")
    for f in h5files:
        with h5py.File(f) as h:
            hp = h['human_prior'][:]
            print(f"{ds}/{os.path.basename(f)}: max={hp.max():.4f}  >0.05: {(hp>0.05).mean():.1%}")
EOF
```

正常输出示例：
```
dexycb/ycb_dex_02.hdf5: max=0.0942  >0.05: 3.2%
```

---

## 五、 预期总时间

| 步骤 | DexYCB | HO3D v3 | 合计 |
|---|---|---|---|
| Step 1 Depth Pro (6 cams) | ~21 h | ~1.7 h | ~23 h |
| Step 2 HaPTIC | ~48 h | ~1 h | ~49 h |
| Step 3 FP | ~8 h | ~15 min | ~8.5 h |
| Step 4 对齐 | <5 min | <5 min | <10 min |
| **合计** | **~77 h** | **~3 h** | **~80 h** |

> **并行建议**：6 台相机的 Depth Pro 可以分发到不同 GPU 同时跑。Step 2 HaPTIC 是最大瓶颈（~2天），建议多并行。
> > 若只用单台最优相机 `841412060263`：DexYCB 总计约 **13 h**。

---

## 六、 已验证配置

| 配置 | 结果 |
|---|---|
| DexYCB + cam `841412060263` + ycb_dex_02 | coverage=100%, diverged=0 |
| Depth Pro fx 偏差 (DexYCB) | **+0.2%** |
| Depth Pro fx 偏差 (HO3D v3, two-pass) | **-1.5%** |
| FP 跟踪成功率 | **100%** |

---

## 七、 常见问题

**Q：DexYCB 要跑哪台相机？**
A：默认只跑最优相机 `841412060263`（Depth Pro 误差 +0.2%）。已在 `dexycb_camera_config.json` 配置好，无需额外操作。

**Q：HO3D v3 为什么要 `--two-pass`？**
A：单条序列 Depth Pro fx 估计误差可达 ±20%，多序列全局中位数可收敛到 -1.5%。`--two-pass` 自动完成这个校准，无需 GT 内参。

**Q：HO3D 没有发散（diverged）怎么界定？**
A：Step 4 输出中 `diverged=0` 表示全部帧的 FP 位姿有效。

**Q：联系方式**
A：如遇问题请联系 lyh。
