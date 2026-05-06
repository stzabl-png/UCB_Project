# Affordance2Grasp — Project Pipeline Instructions

> ⚠️ **此文档已过时（基于早期 ARCTIC 阶段）。**  
> 最新部署指南请参阅 **`README.md`** 和 **`HANDOVER.md`**。  
> 本文档保留作为历史参考，路径变量已统一修正为可移植形式。

> **目的：** 从 ARCTIC 第三人称视频生成物体接触热力图（Human Prior），
> 接入 Robot GT 数据采集（Isaac Sim），最终训练 M5 PointNet++ 模型。
> 本文档记录所有环境、路径、命令，可直接复制执行。

> **路径说明：** 以下命令中 `$PROJ` = 你的项目根目录（`/path/to/Affordance2Grasp`）  
> `SAM3D_USER` = 你在云服务器上的用户名

---

## 完整 Pipeline 流程图

```
[ARCTIC 第三人称视频]
    ↓ Step 0a  annotate_trim.py       → data/trimmed/{seq}/
    ↓ Step 0b  annotate_obj_mask.py   → mask_bbox.png + bbox.json
    ↓ Step 0c  SAM3D (云端)           → output/sam3d_obj_cache/{obj}/splat.ply
    ↓ Step 1   batch_depthpro.py      → output/depthpro_batch/{seq}/
    ↓ Step 2   batch_fp_register.py   → output/fp_register_batch/{seq}_T_obj_cam1.npy
    ↓ Step 3   batch_contact.py       → output/affordance_batch/{seq}/vert_contact_count.npy
    ↓ Step 4   export_arctic_prior.py → data_hub/human_prior/{obj}.hdf5   ← ★ 上下游关键接口
    ↓
    ┌──────────────────────────────────────────────────────────────────┐
    │  Step 5  random_grasp_sampler.py  (50% HP + 50% 随机)           │
    │          → output/grasps_random/{obj}_grasp.hdf5                │
    │  Step 6  Isaac Sim  batch_random_sim.sh                          │
    │          → output/robot_gt_v4_physics/{obj}_robot_gt.hdf5       │
    │  Step 7  aggregate_robot_gt.py                                   │
    │          → data_hub/training/{obj}.hdf5 (human_prior + robot_gt)│
    │  Step 8  model/train.py  → output/checkpoints_m5/best_m5_model  │
    └──────────────────────────────────────────────────────────────────┘
    ↓ (M5 推理 inference/grasp_pose.py)
    inference/grasp_pose.py → 候选抓取位姿 HDF5 (13个候选，100% HP引导)
    sim/run_grasp.py        → Isaac Sim 验证 → posterior
```

---

## 目录

1. [环境总览](#1-环境总览)
2. [数据结构](#2-数据结构)
3. [Step 0a — 视频标注切割](#3-step-0a)
4. [Step 0b — 物体 Mask 标注](#4-step-0b)
5. [Step 0c — SAM3D 云端 Mesh 生成](#5-step-0c)
6. [Step 1 — Depth Pro 深度估计](#6-step-1)
7. [Step 2 — FoundationPose 物体位姿注册](#7-step-2)
8. [Step 3 — 接触检测 + Affordance 热力图](#8-step-3)
9. [Step 4 — ARCTIC Prior 导出（接入下游）](#9-step-4)
10. [Step 5 — Robot GT 数据采集（50/50 采样）](#10-step-5)
11. [Step 6 — Isaac Sim 批量验证](#11-step-6)
12. [Step 7 — 聚合训练数据](#12-step-7)
13. [Step 8 — M5 模型训练与推理](#13-step-8)
14. [常用调试命令](#14-常用调试命令)

---

## 1. 环境总览

| 环境 | 用途 | 激活命令 |
|------|------|----------|
| `base`         | 视频标注、Mask 标注（有 Qt GUI）、random_grasp_sampler | `conda activate base` |
| `hawor`        | 接触检测、Prior 导出、可视化   | `conda activate hawor` |
| `depth-pro`    | Depth Pro 深度估计              | `conda activate depth-pro` |
| `bundlesdf`    | FoundationPose 位姿注册         | `conda activate bundlesdf` |
| **IsaacSim**   | Robot GT 收集（无 conda）       | `$ISAAC_SIM_PATH/python.sh` |
| **云服务器**   | SAM3D Mesh 生成                 | `ssh sam3d-gpu` |
| `sam3d-objects`| SAM3D 推理（在云服务器上）      | `conda activate sam3d-objects` |

### 关键路径

```bash
# 使用前设置以下变量
export PROJ=$HOME/Project/Affordance2Grasp      # 项目根目录（按实际修改）
export ARCTIC_ROOT=$HOME/Project/arctic/unpack  # ARCTIC 数据目录
export SAM3D_USER=lyh                           # 云服务器用户名

# 本地
项目根:          $PROJ/
ARCTIC 数据:     $ARCTIC_ROOT/
  图像序列:        arctic_data/data/cropped_images/s05/<seq>/1/
  GT 元数据:       meta/misc.json
  物体 Mesh:       meta/object_vtemplates/<obj>/mesh_tex.obj  (单位: mm)
  GT 序列:         raw_seqs/s05/<seq>.[mano/object/smplx].npy
标注数据:        $PROJ/data/trimmed/

# 输出目录
output/
  haptic_arctic_cache/  s05__<seq>_cam1.npz       ← HaPTIC MANO verts
  sam3d_obj_cache/new/  <obj>/splat.ply            ← SAM3D Gaussian Splat
  depthpro_batch/       <seq>/depthpro_<fr>.npz
  fp_register_batch/    <seq>_T_obj_cam1.npy       ← 6DOF 位姿 4×4
  affordance_batch/     <seq>/vert_contact_count.npy + <obj>_affordance.png
  grasps_random/        <obj>_grasp.hdf5           ← 50/50 候选抓取
  robot_gt_v4_physics/  <obj>_robot_gt.hdf5        ← Sim 验证结果
  vis_prior/            <obj>_hp.png               ← Prior 可视化

# 数据中心
data_hub/
  human_prior/  <obj>.hdf5   ← ARCTIC HP ★ 关键接口
  training/     <obj>.hdf5   ← human_prior + robot_gt
  meshes/v1/    <A0xxxx>.obj ← OakInk 物体

# 云服务器 (ssh sam3d-gpu)
SSH: <ALIYUN_NLB_HOST>:<PORT>  user=$SAM3D_USER   # 联系 lyh 获取实际地址
SAM3D 代码: /root/$SAM3D_USER/sam-3d-objects/  (必须在此目录运行!)
输入:       /mnt/data/$SAM3D_USER/sam3d_input/
输出:       /mnt/data/$SAM3D_USER/sam3d_mesh_output/
```

---

## 2. 数据结构

```
data/trimmed/
└── s05_<obj>_grab_01/
    ├── meta.json                    ← 帧范围 (obj.start/end, mano.start/end)
    ├── s05_<obj>_grab_01_obj/       ← OBJ 帧 (无手，~20帧)
    │   ├── 00003.jpg ... 00026.jpg
    │   ├── bbox.json                ← 物体 bbox (Step 0b 生成)
    │   └── mask_bbox.png            ← SAM2 精细 mask
    └── s05_<obj>_grab_01_mano/      ← MANO 帧 (手入画，~20帧)

# 10 个物体序列 (subject s05):
# box, capsulemachine, espressomachine, ketchup, microwave,
# mixer, notebook, phone, scissors, waffleiron
```

---

## 3. Step 0a — 视频标注切割 ✅ 已完成

> **目的：** 从 ARCTIC 图像序列手动标注并切出 OBJ 段和 MANO 段

```bash
conda activate base
cd $PROJ

python tools/annotate_trim.py --seq s05/box_grab_01 --out_dir data/trimmed/s05_box_grab_01
# ... 10 个物体依次执行
```

**按键：** `← →` 逐帧 | `↑↓` 跳10帧 | `SPACE`×4 标记4个关键帧 | `ENTER` 导出

> ⚠️ OBJ 和 MANO 是独立区间，允许时间重叠。

---

## 4. Step 0b — 物体 Mask 标注 ✅ 已完成

> **目的：** 对 OBJ 帧画 bbox → SAM2 生成精细 mask（同时用于 SAM3D 和 FoundationPose）

```bash
conda activate base
python tools/annotate_obj_mask.py
```

**输出：**
- `data/trimmed/<seq>/*_obj/bbox.json`
- `data/trimmed/<seq>/*_obj/mask_bbox.png`
- `/tmp/sam3d_upload/<obj>/frame.jpg + mask.png`

---

## 5. Step 0c — SAM3D 云端 Mesh 生成 ✅ 已完成

### 上传到云服务器

```bash
# 整理文件
mkdir -p /tmp/sam3d_flat
for OBJ in box capsulemachine espressomachine ketchup microwave mixer notebook phone scissors waffleiron; do
    cp /tmp/sam3d_upload/${OBJ}/frame.jpg /tmp/sam3d_flat/${OBJ}.jpg
    cp /tmp/sam3d_upload/${OBJ}/mask.png  /tmp/sam3d_flat/${OBJ}_mask.png
done

# mask 用 tar 管道传 (防止云端频繁短连接报 Connection closed)
cd /tmp/sam3d_flat && tar czf - *_mask.png | ssh sam3d-gpu "tar xzf - -C /mnt/data/$SAM3D_USER/sam3d_input/" && cd -
```

### 云服务器批量推理

> ⚠️ **必须在 `/root/$SAM3D_USER/sam-3d-objects/` 目录运行！**（相对路径依赖）

```bash
ssh sam3d-gpu
conda activate sam3d-objects
cd /root/$SAM3D_USER/sam-3d-objects    # 必须！
python /tmp/batch_sam3d.py     # 约 15~20 分钟
```

### 下载结果回本地

```bash
# tar 管道一次性下载 (不要用 scp 循环)
ssh sam3d-gpu "cd /mnt/data/$SAM3D_USER/sam3d_mesh_output && tar czf - */splat.ply" | \
    tar xzf - -C $PROJ/output/sam3d_obj_cache/new/
```

---

## 6. Step 1 — Depth Pro 深度估计 ✅ 已完成

```bash
conda activate depth-pro
cd $PROJ/third_party/ml-depth-pro

python $PROJ/tools/batch_depthpro.py
python $PROJ/tools/batch_depthpro.py --seq s05_box_grab_01  # 单个
```

**输出：** `output/depthpro_batch/<seq>/depthpro_<frame>.npz`
- `depth_map`: (H,W) float32，单位：米
- `K_dp`: 估计内参矩阵（3×3）
- `focal_dp`: 估计焦距（像素）

---

## 7. Step 2 — FoundationPose 物体位姿注册 ✅ 已完成

> **目的：** OBJ 帧 + DepthPro 深度 + 物体 Mesh → T_obj_cam1（4×4 变换矩阵）

```bash
conda activate bundlesdf
cd $PROJ/third_party/FoundationPose   # 或你的 FoundationPose 安装目录

python $PROJ/tools/batch_fp_register.py --all
```

**输出：** `output/fp_register_batch/<seq>_T_obj_cam1.npy`

> **注意：** DepthPro 估计的焦距与 ARCTIC GT 焦距（4651px）有差异，
> 导致 FP 给出的 Z 值与旧校准基准不同。batch_contact.py 已用自动校准处理。
>
> **box** 的 FP 位姿不准（Z=3.35m 远，mesh 可能形状不够用于注册），
> 导致后续接触检测为 0，下游自动退化为 100% 随机采样。

### 当前 FP 位姿汇总

| 物体 | X(m) | Y(m) | Z(m) |
|------|------|------|------|
| box | -0.092 | -0.044 | 3.347 |
| capsulemachine | 0.021 | -0.080 | 2.020 |
| espressomachine | -0.010 | -0.036 | 2.210 |
| ketchup | 0.009 | -0.077 | 2.435 |
| microwave | 0.078 | -0.039 | 2.343 |
| mixer | 0.005 | -0.016 | 2.488 |
| notebook | -0.073 | 0.018 | 3.722 |
| phone | 0.072 | 0.015 | 2.396 |
| scissors | 0.004 | 0.007 | 2.246 |
| waffleiron | 0.103 | 0.024 | 2.501 |

---

## 8. Step 3 — 接触检测 + Affordance 热力图 ✅ 已完成

> **目的：** HaPTIC MANO 顶点 × 物体 Mesh（FP 变换后） → 接触检测 → 热力图

```bash
conda activate hawor
cd /home/lyh/Project/Affordance2Grasp

python tools/batch_contact.py                           # 全部序列
python tools/batch_contact.py --seq s05_scissors_grab_01  # 单个
python tools/batch_contact.py --contact_thresh 0.05    # 更严格（默认 0.07m=7cm）
```

### ⭐ 自动尺度校准（重要变更，勿改回固定值）

旧版固定参数 `SCALE_Z=3.315, SCALE_XY=0.713` 只对旧 scissors FP 准确（用 GT K 注册）。
新版 FP 用 DepthPro 估计 K，Z 值不同。**batch_contact.py 已改为 per-sequence 自动校准：**

```python
# 收集 MANO range 内所有帧的 HaPTIC Z 中位数
median_hz  = np.median([verts_dict[fi][:, 2].mean() for fi in mano_range])
scale_auto = Z_obj / median_hz        # 用 FP 给的物体 Z 深度动态对齐
scale_xy   = scale_auto * (SCALE_XY / SCALE_Z)   # 保持 XY/Z 原始比例
```

### 结果汇总（9/10 成功）

| 物体 | 接触帧 | 接触顶点 |
|------|--------|---------|
| box | 0 ❌ | 0 |
| capsulemachine | 22 ✅ | 2294 |
| espressomachine | 19 ✅ | 3023 |
| ketchup | 18 ✅ | 4139 |
| microwave | 21 ✅ | 3785 |
| mixer | 24 ✅ | 4148 |
| notebook | 15 ✅ | 1888 |
| phone | 20 ✅ | 4300 |
| scissors | 16 ✅ | 1138 |
| waffleiron | 17 ✅ | 3790 |

**输出：**
```
output/affordance_batch/s05_<seq>/
  <obj>_affordance.png      ← 4视角热力图 (Front/Side/Top/Iso)
  vert_contact_count.npy    ← 每顶点接触计数 (V,) raw 频率，供 Step 4 使用
```

---

## 9. Step 4 — ARCTIC Prior 导出（接入下游） ✅ 已完成

> **目的：** `vert_contact_count.npy` → `data_hub/human_prior/{obj}.hdf5`
> 与 OakInk 格式完全一致，供 random_grasp_sampler 和 M5 推理使用。

```bash
conda activate hawor
cd /home/lyh/Project/Affordance2Grasp

python tools/export_arctic_prior.py           # 全部 10 个物体
python tools/export_arctic_prior.py --obj scissors   # 单个
python tools/export_arctic_prior.py --force           # 覆盖重写
```

### Gaussian Diffusion 参数

```python
sigma  = 0.008   # 8mm  — 接触信号高斯扩散（标准差）
radius = 0.030   # 3cm  — 邻域搜索半径
```

Raw `vert_contact_count` 只在被接触顶点有值（稀疏）。Gaussian diffusion 将信号
扩散到 3cm 内的所有顶点，sigma 控制衰减速度。sigma=8mm 时 coverage 约 7~72%。

### 当前 Coverage

| 物体 | coverage(>0.1) | max_prior |
|------|---------------|-----------|
| box | 0.0% ❌ | 0.000 |
| capsulemachine | 12.3% | 0.935 |
| espressomachine | 10.4% | 0.717 |
| ketchup | 22.0% | 0.919 |
| microwave | 7.1% | 0.972 |
| mixer | 14.3% | 0.992 |
| notebook | 20.4% | 0.957 |
| phone | 55.0% | 0.982 |
| scissors | 71.7% | 0.953 |
| waffleiron | 25.1% | 0.962 |

> box 因 FP 失败→0接触→prior全零，下游 random_grasp_sampler 自动用 **100% 随机**（`has_hp=False`）。

### 可视化

```bash
python3 tools/vis_arctic_prior.py
# 输出: output/vis_prior/<obj>_hp.png + _overview_hp.png
```

### 输出 HDF5 格式（与 OakInk human_prior 完全一致）

```
data_hub/human_prior/{obj}.hdf5
  point_cloud: (1024, 3) float32  # mesh 采样点 (object local frame, m)
  normals:     (1024, 3) float32
  human_prior: (1024,)  float32   # [0,1] 归一化接触频率（Gaussian 扩散后）
```

---

## 10. Step 5 — Robot GT 数据采集（50/50 采样）

> **目的：** 在物体上采样抓取候选（50% HP引导 + 50%随机），
> 送入 Isaac Sim 验证，建立 Human Prior → Robot GT 的对应关系。

```bash
conda activate base
cd /home/lyh/Project/Affordance2Grasp

python3 tools/random_grasp_sampler.py --all            # OakInk 全部物体
python3 tools/random_grasp_sampler.py --obj scissors   # OakInk 单个
python3 tools/random_grasp_sampler.py --all --force    # 强制重新生成

# ARCTIC 物体 (输出加 arctic_ 前缀)
python3 tools/random_grasp_sampler.py --arctic \
    --output-dir output/grasps_arctic   # 全部 10 个
python3 tools/random_grasp_sampler.py --arctic --obj scissors \
    --output-dir output/grasps_arctic   # 单个
```

### ⚠️ 命名规则

```
输出目录:
  output/grasps_random/scissors_grasp.hdf5        obj_id="scissors"      ← OakInk
  output/grasps_arctic/arctic_scissors_grasp.hdf5 obj_id="arctic_scissors" ← ARCTIC

对应 USD:
  assets/usd/scissors.usd        ← OakInk 版本（已有）
  assets/usd/arctic_scissors.usd ← ARCTIC 版本（Step 6 生成）
```

HP lookup 时仍使用原始名（`scissors.hdf5`），`arctic_` 前缀仅用于区分 USD 和 HDF5 文件名。

### 采样策略（50/50，sample_points 已更新）

```python
# tools/random_grasp_sampler.py  sample_points() 逻辑：

n_hp   = n_total // 2   # 50% HP-guided
# 按 human_prior 概率加权采样高接触顶点 → 加 ±5mm jitter → 保留 mesh 内部点

n_rand = n_total // 2   # 50% 纯随机
# bbox 内均匀随机 → 保留 mesh 内部点

# 若 has_hp=False（box 等无先验物体）→ 自动 100% 随机
```

### 候选评分（score_candidate）

| 维度 | 权重 | 说明 |
|------|------|------|
| 反力（Antipodal） | 30% | 两侧接触法线近似反平行 → 力闭合 |
| 平整度（Flatness）| 25% | 接触区域法线一致性 |
| 重心对齐（CoM）  | 25% | 抓取中心靠近物体重心 |
| 宽度（Width）    | 20% | 2~5cm 最优 |

目标：每个物体输出 **20 个 score ≥ 60 的候选**，6个固定接近方向（±X/Y/Z），迭代至满足数量。

**输出：** `output/grasps_random/<obj>_grasp.hdf5`

---

## 11. Step 6 — Isaac Sim 批量验证

### 6.0 — 先转换 ARCTIC mesh → USD

> Isaac Sim 需要 USD 格式。ARCTIC USD 文件命名为 `arctic_{obj}.usd`，
> 与 OakInk `scissors.usd` 等明确区分。

```bash
conda activate base

# 安装 pxr (Pixar 官方独立包，只需安装一次)
pip install usd-core

# 转换全部 10 个物体 (约 5~10 分钟)
python3 tools/convert_arctic_to_usd.py

# 单个物体（force 覆盖）
python3 tools/convert_arctic_to_usd.py --obj scissors --force

# 确认 10 个 USD 生成
ls assets/usd/arctic_*.usd | wc -l   # 应为 10
```

> ⚠️ **不要用 `sim45` 运行此脚本** — sim45 的 Python 没有 usd-core。
> 直接在 `conda activate base` 环境里跑即可。

### 6.1 — 批量跑 Sim 验证

```bash
# sim45 会自动激活 Isaac Sim 环境，无需 conda deactivate
bash scripts/batch_arctic_sim.sh
```

> ⚠️ `batch_arctic_sim.sh` 内部调用:
> ```
> sim45 sim/run_grasp_sim.py --hdf5 output/grasps_arctic/arctic_{obj}_grasp.hdf5 ...
> ```
> 它读取 HDF5 里的 `metadata.obj_id = "arctic_scissors"`，
> 然后加载 `assets/usd/arctic_scissors.usd`

**输出：** `output/robot_gt_arctic/{obj}_robot_gt.hdf5`

关键字段：
- `attrs['success']`: bool — 至少有一个候选成功
- `attrs['n_successful']`: int — 成功候选数
- `winning_candidate/grasp_point`: (3,) — 成功抓取点（object local frame）
- `winning_candidate/approach_dir`: (3,) — 接近方向
- `winning_candidate/finger_dir`: (3,) — 手指方向

---

## 12. Step 7 — 聚合训练数据

```bash
conda activate base
cd /home/lyh/Project/Affordance2Grasp

python3 data/aggregate_robot_gt.py
```

**流程：**
1. 读 `output/robot_gt_v4_physics/<obj>_robot_gt.hdf5`（仅处理 `success=True` 的）
2. 找对应 `data_hub/human_prior/<obj>.hdf5`
3. 用成功的夹爪位姿在物体点云上计算 `robot_gt` affordance（Gaussian σ=10mm）
4. 合并保存

**输出：**
```
data_hub/training/{obj}.hdf5
  point_cloud: (N, 3) float32   # 同 human_prior
  normals:     (N, 3) float32
  human_prior: (N,)  float32    # 来自 ARCTIC 接触检测（Step 4）
  robot_gt:    (N,)  float32    # 来自 Isaac Sim 成功抓取（Step 6）
  force_center: (3,) float32    # 实际受力中心
```

---

## 13. Step 8 — M5 模型训练与推理

### 训练

```bash
conda activate base
cd /home/lyh/Project/Affordance2Grasp

python3 model/train.py            # 使用 data_hub/training/ 下所有 HDF5
python3 model/train.py --epochs 50  # 快速 finetune
```

**已有模型：** `output/checkpoints_m5/best_m5_model.pth`
（OakInk 数据训练，可直接对 ARCTIC 物体做**零样本推理**）

### M5 推理（生成候选抓取）

```bash
conda activate base
cd /home/lyh/Project/Affordance2Grasp

# 自动读取 data_hub/human_prior/{obj}.hdf5 作为第7通道输入
python -m inference.grasp_pose \
    --mesh $ARCTIC_ROOT/meta/object_vtemplates/scissors/mesh_tex.obj
```

> ⚠️ ARCTIC mesh 单位为 **mm**，inference 时 predictor.py 内部不做单位转换，
> 请先将 mesh 缩放 /1000 或使用 `output/sam3d_obj_cache/` 中已经是米单位的 mesh。

### 候选生成设计（inference/grasp_pose.py）

- 输入：M5 预测的接触点（`contact_prob > threshold`）
- **100% Human Prior 引导**，无随机成分
- PCA(contact_pts) → 手指方向 → 叉乘主轴 → 接近方向
- ±30°/±15°/0° jitter × 2 反向 + top-down = **13 个候选**
- 几何打分排序 → cuRobo 可达性预检 → 依次执行

---

## 14. 常用调试命令

```bash
cd /home/lyh/Project/Affordance2Grasp

# 检查各步骤输出完整性
ls output/fp_register_batch/*_T_obj_cam1.npy | wc -l    # 应为 10
ls output/affordance_batch/*/*.png | wc -l               # 应为 10
ls data_hub/human_prior/*.hdf5                           # 9~10 个

# 查看所有 FP 位姿的 XYZ 平移
python3 tools/_diag_fp.py

# 重新导出某个物体的 Prior（覆盖）
python3 tools/export_arctic_prior.py --obj box --force

# 可视化所有 HP 热力图
python3 tools/vis_arctic_prior.py
# 输出: output/vis_prior/<obj>_hp.png

# 查看 Affordance 热力图
eog output/affordance_batch/s05_scissors_grab_01/scissors_affordance.png &

# 检查 HaPTIC cache
ls output/haptic_arctic_cache/s05__*grab*_cam1.npz | wc -l  # 应为 10

# 云服务器状态
ssh sam3d-gpu "ls /mnt/data/$SAM3D_USER/sam3d_mesh_output/ && df -h /mnt/data"

# 检查某个 human_prior 文件内容
python3 -c "
import h5py, numpy as np
with h5py.File('data_hub/human_prior/scissors.hdf5','r') as f:
    hp = f['human_prior'][()]
    print(f'shape={hp.shape}  max={hp.max():.3f}  coverage={np.mean(hp>0.1):.1%}')
"
```

---

*最后更新: 2026-04-20 | Affordance2Grasp Pipeline*
*本次新增: Step 3 自动尺度校准 | Step 4 ARCTIC Prior 导出 | Step 5 50/50 Robot GT 采样 | 完整 Step 6-8 文档*
