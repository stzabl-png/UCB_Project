# Affordance 模型训练指南

PointNet++ affordance 训练，数据来自 **no_rot executed**（[`UCBProject/train_affordance`](https://huggingface.co/datasets/UCBProject/train_affordance/tree/main)）。

**两条训练流水线：**

| 流水线 | 入口 | 监督 GT | 模型 |
|--------|------|---------|------|
| **Robot 接触 / soft map**（主） | `python -m model.affordance.train` | `labels` → 在线 Gaussian **soft**；指标可用 binary | `PointNet2Seg`，`seg_head=mlp_sigmoid` → 每点标量 score |
| **Human prior**（对照） | `python model/train_affordance_hp.py` | HDF5 `human_priors`（**L1**） | `PointNet2SegOnly`（无 FC 头） |

**代码目录：** `model/affordance/`（`train.py`、`train_hp.py`、`dataset.py`、`dataset_hp.py` 等）

---

## 目录

1. [整体流程](#整体流程)
2. [数据准备](#数据准备)（含 [HF：`UCBProject/train_affordance`](#从-hugging-face-下载collaborator)）
3. [环境与快速开始](#环境与快速开始)
4. [正式训练](#正式训练)
5. [输出目录与 checkpoint](#输出目录与-checkpoint)
6. [损失函数与模型](#损失函数与模型)
7. [多卡 DDP](#多卡-ddp)
8. [续训 `--resume`](#续训-resume)
9. [Debug 过拟合模式](#debug-过拟合模式)
10. [超参 Sweep](#超参-sweep)
11. [Human prior 监督训练](#human-prior-监督训练)
12. [命令行参数速查](#命令行参数速查)
13. [常见问题](#常见问题)
14. [相关文档](#相关文档)

---

## 整体流程

```text
抓取采集 (Isaac Sim)
  output/grasp_collect_no_rot/merged/{obj}_robot_gt_merged.hdf5
        +
  data_hub/meshes/SAM3DMesh/rotated_mesh/...
        +
  data_hub/ProcessedData/train_fp_rotated/...  (prepare 默认用，训练不读)
        │
        ▼
  tools/prepare_affordance_executed.py
        │
        ▼
  output/affordance_no_rot_executed/
    affordance_train.h5 / affordance_val.h5
        │
        ▼
  python -m model.affordance.train
        │
        ▼
  output/affordance_no_rot_executed/training_runs/<run>/  # 时间戳 | run-name | group/name
    ckpt/  vis/  log/
```

上游数据与 prepare 细节见 [`docs/prepare_affordance_executed.md`](prepare_affordance_executed.md)、[`docs/grasp_collect_pipeline.md`](grasp_collect_pipeline.md)。

---

## 数据准备

### 训练直接读取的文件

默认 `--dataset_dir`：

```text
output/affordance_no_rot_executed/
├── affordance_train.h5          # 必需：二值标签 + 点云
├── affordance_val.h5            # 必需
├── affordance_train_soft.h5     # 推荐：预计算 soft GT（HF 已含）
├── affordance_val_soft.h5       # 推荐
├── soft_gt_export_meta.json     # soft 导出参数记录（σ ratio 等）
├── dataset_info.json
├── objects_trainable.txt
└── objects_train_val_split.json
```

**训练脚本只打开** `affordance_{train,val}.h5`。  
`*_soft.h5` 用于对照、可视化与团队共享；**loss 里的 soft target 仍在每个 step 按增广后的 xyz 在线重算**（与 `export_soft_affordance_gt.py` 公式一致，见下）。

### Soft GT：在线计算 vs 磁盘文件

| 来源 | 何时用 | σ 由谁定 |
|------|--------|----------|
| **在线**（`SoftAffordanceDataset.__getitem__`） | **simple** 的 `L1(score, soft_gt)`、**full** 的 `L_aff`、vis 里的「GT soft」 | `--heatmap-sigma-ratio`（训练默认 **0.05**） |
| **磁盘**（`*_soft.h5`） | HF 分发、QC、与 prepare 点云对齐的参考图 | 导出时 `--heatmap-sigma-ratio`（脚本默认 **0.03**，见 `soft_gt_export_meta.json`） |

公式（`model/affordance/heatmap.py`）：对二值接触点做 Gaussian，`σ = ratio × 物体 bbox 对角线`，接触点处为 1.0。

本地重新生成 soft 文件（需已有 base H5）：

```bash
python3 tools/export_soft_affordance_gt.py \
  --dataset-dir output/affordance_no_rot_executed \
  --heatmap-sigma-ratio 0.03 \
  --overwrite
```

若希望磁盘 soft 与训练 σ 一致，导出时把 `--heatmap-sigma-ratio` 设为与训练相同（例如 `0.05`）。

### HDF5 字段（`affordance_{train,val}.h5` 的 `data/`）

| 字段 | 形状 | 训练是否使用 |
|------|------|-------------|
| `points` | (N, 4096, 3) | ✅ 网络输入（**仅 xyz，3 通道**） |
| `normals` | (N, 4096, 3) | ❌ 不送入 PointNet++ |
| `labels` | (N, 4096) | ✅ 二值接触 GT；并用于**在线**生成 soft heatmap |
| `force_centers` | (N, 3) | ✅ center 相关 loss（默认 λ=0） |
| `obj_ids` | (N,) | ✅ 物体级 train/val 划分 |
| `human_priors` | (N, 4096) | ❌ 读出仅用于 **val 可视化** 第一行，不进网络/loss |

### `affordance_{train,val}_soft.h5` 额外字段

| 字段 | 形状 | 说明 |
|------|------|------|
| `soft_labels` | (N, 4096) float32 | 在**未增广** canonical 点云上预计算的 soft map |
| `soft_sigma` | (N,) | 每样本使用的 σ（米） |
| 其余 | 与 base H5 相同 | `points`, `labels`, `human_priors`, … |

每样本对应 **一个物体** 的一次 mesh 采样（4096 点）。train/val 在脚本内按 **物体 ID** 再划分（`--val_ratio`，默认 0.15），并合并 train.h5 与 val.h5 中属于 val 物体的样本做验证。

### 生成训练 HDF5

```bash
conda activate bundlesdf
cd ~/Project/Affordance2Grasp

python3 tools/prepare_affordance_executed.py
# 或
bash scripts/run_prepare_affordance_executed.sh
```

单物体 + QC 图：

```bash
python3 tools/prepare_affordance_executed.py --obj A01001 --qc-vis
```

生成 soft GT 磁盘文件（与 HF 上 `*_soft.h5` 同格式）：

```bash
python3 tools/export_soft_affordance_gt.py \
  --dataset-dir output/affordance_no_rot_executed \
  --heatmap-sigma-ratio 0.03
```

### 从 Hugging Face 下载（Collaborator）

训练数据**不进 Git**。官方数据集仓库：

**[`UCBProject/train_affordance`](https://huggingface.co/datasets/UCBProject/train_affordance/tree/main)** — object-wise executed grasp + **预计算 soft GT map**（base + `*_soft.h5`）

| HF 仓库根目录文件 | 本地路径（`--dataset_dir` 默认） |
|-------------------|----------------------------------|
| `affordance_train.h5` | `.../affordance_train.h5` |
| `affordance_val.h5` | `.../affordance_val.h5` |
| `affordance_train_soft.h5` | `.../affordance_train_soft.h5` |
| `affordance_val_soft.h5` | `.../affordance_val_soft.h5` |
| `soft_gt_export_meta.json` | `.../soft_gt_export_meta.json` |
| `dataset_info.json` | 同目录 |
| `objects_trainable.txt` | 同目录 |
| `objects_train_val_split.json` | 同目录 |

```bash
conda activate bundlesdf   # pip install huggingface_hub
cd ~/Project/Affordance2Grasp

# 整库下载到默认数据目录
huggingface-cli download UCBProject/train_affordance \
  --repo-type dataset \
  --local-dir output/affordance_no_rot_executed

# 校验（训练最少需要 base；建议 soft 一并保留）
ls -lh output/affordance_no_rot_executed/affordance_{train,val}.h5
ls -lh output/affordance_no_rot_executed/affordance_{train,val}_soft.h5
python3 -c "import h5py; f=h5py.File('output/affordance_no_rot_executed/affordance_train_soft.h5'); print(list(f['data'].keys()))"
# 应含 soft_labels, soft_sigma
```

需有该 **private dataset** 的访问权限（请仓库管理员在 HF 上添加 collaborator）。

本地若已用 `tools/prepare_affordance_executed.py` 生成过同名文件，下载会覆盖；以 HF 版本为准便于与团队对齐。

---

## 环境与快速开始

```bash
conda activate bundlesdf
cd ~/Project/Affordance2Grasp

# 单卡：默认 simple = L1(score, soft_gt)，robot executed 接触
python -m model.affordance.train --gpus 0

# 命名 run（示例：full_train/robot_gt_l1_full）
python -m model.affordance.train --gpus 0 \
  --run-group full_train \
  --run-name robot_gt_l1_full \
  --loss-mode simple

# legacy 多 term loss（focal/Tversky/λ；mlp_sigmoid 下 binary 用伪 logits）
python -m model.affordance.train --gpus 0 --loss-mode full \
  --lambda-aff 0.3 --lambda-binary 1.0

# Human prior 监督（L1 on human_priors）
python model/train_affordance_hp.py --gpus 0
# 或: bash model/affordance/run_train_hp.sh
# （脚本默认 --run-name hp_mse_full 为历史命名；loss 已是 L1）
```

需要 **CUDA**；debug 模式仅支持单卡。

---

## 正式训练

### 网络与输入（`train.py` / robot GT）

- 模型：`PointNet2Seg`，`in_channel=3`（**仅 xyz**）
- 分割头：`seg_head=mlp_sigmoid` → 输出 **`(B, N)` score ∈ [0,1]**（非 2-class softmax logits）
- FC 分支：默认 **关闭**；`--predict-force-center` 构建 FC 头（仅 `--loss-mode full` 时常用）
- GT binary：HDF5 `labels`（robot executed 接触）
- GT soft：**在线** Gaussian（`σ = --heatmap-sigma-ratio × bbox 对角线`，默认 **0.05**）；与 HF `*_soft.h5` 的 canonical `soft_labels`（常 σ=0.03）对照用
- 可视化：`human_priors` 显示在 val/train 图第一行（不进 loss）

### 学习率与早停

| 项 | 默认 |
|----|------|
| 优化器 | AdamW，`lr=1e-3`，`weight_decay=1e-4` |
| 调度 | Cosine，`lr_min=1e-5`；`warmup_epochs=0`（默认关闭 linear warmup） |
| 早停 | `patience=10`；`warmup_epochs>0` 时前 N 个 epoch 不计入 patience |
| Best checkpoint | `best_score = ckpt_f1_weight·F1 + (1-w)·AP`（默认各 50%） |
| 塌缩检测 | 预测全正/概率 span 过小会标 `COLLAPSE`，默认不写入 best |

### 数据增广（`--augment-mode`）

| 模式 | rotation | scale | shift | jitter | dropout |
|------|----------|-------|-------|--------|---------|
| `full`（默认） | ✅ | ✅ | ✅ | ✅ | ✅ |
| `weak` | ✅ | ✅ | ✅ | ✅ | ❌ |
| `none` | ❌ | ❌ | ❌ | ❌ | ❌ |

`--no-augment` 等价于关闭全部增广。

### 验证 / 训练可视化

| 文件 | 说明 |
|------|------|
| `vis/val_epoch_XXXX.png` | 验证集物体（`--val-vis-max-objects`，默认 10） |
| `vis/train_epoch_XXXX.png` | 随机抽训练物体（`--train-vis-max-objects`，默认 10；`0` 关闭） |
| `vis/train_vis_objects.json` | 记录 train vis 抽到的 object id 与 seed |

面板行（robot 流水线）：Human prior → GT bin → GT soft → Pred bin → Pred score。

---

## 输出目录与 checkpoint

每次**新 run**（无 `--resume`）在 `training_runs/` 下创建目录，**优先级**：

1. `--resume` → 沿用 ckpt 所在 run  
2. `--save_dir` → 指定 run 根或 `ckpt/`  
3. `--run-name` [+ `--run-group`] → 命名目录（已存在则自动加 `__YYYYMMDD_HHMMSS` 后缀）  
4. 否则 → `training_runs/<YYYYMMDD_HHMMSS>/`

```text
# 默认时间戳
training_runs/20260523_120000/

# 命名实验
training_runs/my_exp_A01/

# 分组实验
training_runs/full_train/robot_gt_l1_full/
training_runs/overnight_affordance_sweep_20260523/A01_noFC_normNone_base/
├── ckpt/   best_ckpt.pth, last_ckpt.pth, training_history.json, split_info.json
├── vis/    val_epoch_*.png
├── log/    train.log
└── run_manifest.json
```

`vis/`、`log/` 与 `ckpt/` **同级**，不在 `ckpt/` 内。

命名训练示例：

```bash
python -m model.affordance.train --gpus 0 \
  --run-group overnight_sweep \
  --run-name A01_noFC_normNone_base \
  --head-norm none
```

---

## 损失函数与模型（robot GT 流水线）

由 **`--loss-mode`** 选择（`build_affordance_criterion`）：

### `simple`（默认，推荐）

```text
L = mean( |score − soft_gt| )     # PyTorch L1Loss，即 MAE on soft map
```

| 项 | 说明 |
|----|------|
| `score` | `(B, N)`，`mlp_sigmoid` 头 sigmoid 输出 ∈ [0,1] |
| `soft_gt` | 增广后 xyz 上由 `labels` 在线生成的 Gaussian heatmap |
| `--lambda-l1-reg` | **已废弃**（CLI 保留兼容，simple 模式不读） |
| 验证指标 | 仍用 `labels` 二值算 F1/AP（与 loss 无关） |

```bash
python -m model.affordance.train --gpus 0

python -m model.affordance.train --gpus 0 \
  --run-group full_train \
  --run-name robot_gt_l1_full
```

`simple` 模式下 **不**使用 balanced binary 采样（`train_loop` 会跳过）。

### `full`（legacy 多 term）

```text
L_total = λ_aff·L_aff + λ_bin·L_binary + λ_peak·L_peak + …
```

| 项 | 定义 |
|----|------|
| **L_aff** | `0.7·balanced_soft_focal + 0.3·soft_Dice` |
| **L_binary** | `0.6·Focal + 0.4·Tversky`（`mlp_sigmoid` 时由 score 构造 2-class 伪 logits） |
| **L_peak** | 接触点 BCE |
| **L_center_*** | FC / heatmap 质心（需 `--predict-force-center` 与对应 λ） |

默认 λ：`λ_aff=0.3`，`λ_binary=1.0`，其余多为 0。

```bash
python -m model.affordance.train --gpus 0 --loss-mode full \
  --lambda-aff 0.3 --lambda-binary 1.0
```

Sweep 脚本 [`run_affordance_sweep.sh`](../model/affordance/run_affordance_sweep.sh) 面向 **full**；若未传 `--loss-mode`，当前默认会变成 **simple**，请在 `COMMON_ARGS` 中加 `--loss-mode full`。

`--disable-center-loss` 在 full 模式下将 center 相关 λ 置 0。

Debug 与正式训练共用 `build_affordance_criterion`（建议 debug 用 `--loss-mode simple`）。

### Tversky / 采样（正式与 debug 共用）

| 参数 | 含义 |
|------|------|
| `--binary-tversky-alpha` | Tversky FP 权重（↑ 更惩罚误检） |
| `--binary-tversky-beta` | Tversky FN 权重 |
| `--binary-neg-ratio` | debug 训练步中 balanced binary 的负:正采样比 |
| `--soft-background-weight` | soft focal 背景项权重（默认 0.25） |

### `--head-norm`

分割头前特征归一化：`none` | `layernorm` | `groupnorm`。

---

## 多卡 DDP

```bash
python -m model.affordance.train --gpus 0,1,2,3 --batch_size 32
```

- **1 张卡**：单进程
- **≥2 张卡**：`torch.multiprocessing.spawn` + DDP
- `--batch_size` 为 **每张 GPU** 的 batch；有效 batch ≈ `batch_size × GPU 数`
- 样本过少时会自动 **cap** per-GPU batch（避免 DDP `drop_last` 无 batch）
- `--master_port` 默认 `29500`（端口冲突时修改）
- `--num_workers` 为每个进程的 DataLoader worker 数

**Debug 模式不支持多卡**（会自动只用第一张 GPU）。

---

## 续训 `--resume`

```bash
python -m model.affordance.train \
  --resume output/affordance_no_rot_executed/training_runs/20260522_213801/ckpt/last_ckpt.pth \
  --gpus 0
```

行为：

- 从 checkpoint 恢复 **model / optimizer / scheduler / epoch / best 指标 / history**
- **沿用原 run 目录**（根据 ckpt 路径解析 `run_dir`、`vis_dir`、`log_dir`）
- **不会** 覆盖 `run_manifest.json`（仅新 run 写入）
- `train.log` 以 append 方式继续写

也可从 `best_ckpt.pth` 恢复，但通常续训用 `last_ckpt.pth` 以保持 epoch 连续。

---

## Debug 过拟合模式

用于在 **少量物体 / 少量样本** 上快速检查：数据管线、标签、loss、梯度、可视化是否正常。  
**不是**正式训练流程；与正式训练的主要区别见下表。

| 项 | 正式训练 | Debug (`--debug-overfit-one-object`) |
|----|----------|--------------------------------------|
| 优化单位 | epoch | **step**（默认 1000 步） |
| 数据子集 | 全量 + 物体划分 | 手动选 K 个物体或指定 object id |
| train / val | 分开 | **相同子集**（过拟合自检） |
| 增广 | 默认 full | **强制关闭**（除非你先设了 `--no-augment` 逻辑仍会被 debug 打开 no_augment） |
| FC 头 | `--predict-force-center`（默认关） | 同左，由 CLI 控制 |
| 损失权重 | `--lambda-*` CLI | **同一套 CLI**（与正式训练相同类） |
| early stop | 默认开 | **关闭** |
| weight decay | 1e-4 | **0** |
| checkpoint | best/last 按 epoch | 结束写 `debug_last_ckpt.pth` |
| 可视化 | `val_epoch_*.png` | **`debug_step_*.png`**（按 step） |
| DDP | 支持 | **仅单卡** |

启用方式：加上 **`--debug-overfit-one-object`**。

### Debug 自动设置的默认值

`model/affordance/debug.py` 中 `apply_debug_config()` 会强制：

- `no_augment=True`，`augment_mode=none`
- `disable_early_stop=True`
- `weight_decay=0`，`warmup_epochs=0`
- `ckpt_reject_collapse=False`（塌缩也允许记 best，便于观察）

**不会**改写 `--lambda-*`（除非你传了 `--disable-center-loss`，则 center 相关 λ 置 0）。  
启动时 `log_loss_weights` 会打印当前 λ，并写入 `debug_manifest.json` 的 `loss_weights` 字段。

### 如何选择 debug 子集

优先级（互斥逻辑）：

1. **`--debug-object-id OBJ`**  
   使用该物体的全部样本（优先含接触点的样本；可用 `--debug-samples-per-object M` 限制每物体最多 M 条）。

2. **`--debug-use-sample-mode`**（legacy）  
   取前 `--debug-num-samples` 个 **含接触点** 的样本（默认 1）。

3. **默认：物体级**  
   - `--debug-num-objects K`：选 K 个 **至少有一个接触点** 的物体（默认 K=1）  
   - `--debug-object-mode first|random`：按 obj_id 排序取前 K 个，或随机（`--debug-seed`）  
   - `--debug-samples-per-object M`：每物体最多 M 条样本（0=不限制）

Debug 会从 `affordance_train.h5` + `affordance_val.h5` **合并池** 里选样本（不受正式 train/val 物体划分限制）。

### 损失权重（用 `--lambda-*`，无 `--debug-loss-mode`）

建议排查顺序（通过 CLI 调 λ，不再使用 preset loss mode）：

| 阶段 | 命令要点 |
|------|----------|
| 1. 查网络 | `--lambda-aff 0 --lambda-binary 1 --lambda-peak 0` + `--debug-synthetic-label x_positive` |
| 2. 查 GT | 同上合成标签能过拟合后，去掉 `--debug-synthetic-label` |
| 3. 加 soft | `--lambda-aff 0.3 --lambda-binary 1.0` |
| 4. 加 peak | `--lambda-peak 0.3`（可选） |

### 合成标签（排除 GT 问题）

```bash
python -m model.affordance.train \
  --gpus 0 \
  --debug-overfit-one-object \
  --debug-synthetic-label x_positive \
  --lambda-aff 0 --lambda-binary 1 --lambda-peak 0 \
  --debug-max-steps 500
```

- `x_positive`：label = 点 x 坐标大于中位数  
- `z_positive`：label = 点 z 坐标大于中位数  

若合成标签都无法过拟合，优先查模型/优化器/数据加载。

### Debug 推荐命令示例

**单物体，只看 binary（最常见起步）：**

```bash
python -m model.affordance.train \
  --gpus 0 \
  --debug-overfit-one-object \
  --debug-object-id A01001 \
  --lambda-aff 0 --lambda-binary 1 --lambda-peak 0 \
  --debug-max-steps 1000 \
  --lr 1e-3 \
  --debug-log-interval 20 \
  --debug-vis-interval 50
```

**随机 5 个物体，每物体最多 3 条样本，binary + soft：**

```bash
python -m model.affordance.train \
  --gpus 0 \
  --debug-overfit-one-object \
  --debug-num-objects 5 \
  --debug-object-mode random \
  --debug-seed 42 \
  --debug-samples-per-object 3 \
  --lambda-aff 0.3 --lambda-binary 1.0 --lambda-peak 0 \
  --debug-max-steps 2000
```

**加大 Tversky FP 惩罚（减少全图高亮）：**

```bash
python -m model.affordance.train \
  --gpus 0 \
  --debug-overfit-one-object \
  --debug-object-id A01001 \
  --lambda-aff 0 --lambda-binary 1 --lambda-peak 0 \
  --binary-tversky-alpha 0.7 \
  --binary-tversky-beta 0.3
```

### Debug 输出解读

**终端日志列（每 `--debug-log-interval` 步）：**

```text
 step | loss    bin     soft    peak | prob stats | μ+/μ- | F1 AP top1% | grad
```

| 字段 | 含义 |
|------|------|
| `pμ`, `p[min,max]`, `span` | 全图预测概率统计；span 过小可能塌缩 |
| `μ+` / `μ-` | GT 接触点 / 非接触点上预测均值；理想情况 μ+ ≫ μ- |
| `F1` / `AP` | 在 **当前 debug 子集** 上的指标（train=val） |
| `top1%` | 概率最高 1% 点的 precision |
| `grad` | 分割头梯度范数；长期 ≈0 可能 dead head |
| `objF1 μ/min` | 按物体平均 F1（需 `debug-log-interval` 与 object metrics 对齐） |
| `gapμ`, `cμ`, `ncμ` | 接触/非接触 logit 间隔（诊断分类头是否拉开） |

**`vis/debug_step_XXXX.png`：**

- 每 `--debug-vis-interval` 步保存（含 step 0）  
- 最多 `--debug-vis-max-objects` 列（默认 10）；物体多时会选前若干 + F1 最差的一个  
- 用于肉眼对比 GT 接触区与 pred 高亮是否对齐  

**其它 debug 产物：**

```text
training_runs/<timestamp>/
├── debug_manifest.json      # 子集 indices、object ids、args
├── ckpt/debug_last_ckpt.pth
├── ckpt/debug_history.json  # 逐步 loss/metrics
└── vis/debug_step_*.png
```

### Debug 通过标准（经验）

在 **单物体、`--loss-mode simple` 或 full 下 `λ_bin=1`、无增广** 下，通常期望数百步内：

- `μ+` 明显高于 `μ-`，`span` 明显大于 0.1  
- `F1` / `AP` 持续上升（小物体上可到很高）  
- `debug_step_*.png` 中 pred 热点与 GT 接触带重合  

若 loss 下降但 F1 不升：查标签是否极稀疏、Tversky α/β、或提高 `--lambda-aff`。  
若 `grad` 长期为 0：查 CUDA、学习率、是否 batch 为空。

---

## 超参 Sweep

批量实验脚本：[`model/affordance/run_affordance_sweep.sh`](../model/affordance/run_affordance_sweep.sh)

从仓库根目录运行（脚本会自动 `cd` 到项目根）：

```bash
conda activate bundlesdf
cd ~/Project/Affordance2Grasp

# 默认：GPU=0, EPOCHS=60, AUG=weak, GROUP=overnight_affordance_sweep_<时间戳>
bash model/affordance/run_affordance_sweep.sh

# 自定义
GPU=0 EPOCHS=80 AUG=weak GROUP=my_sweep_001 bash model/affordance/run_affordance_sweep.sh
```

脚本内 **COMMON_ARGS** 默认（与单次训练不同，偏 sweep 用）：

| 项 | Sweep 默认 |
|----|------------|
| `lr` | `3e-4` |
| `warmup-epochs` | `0` |
| `patience` | 等于 `EPOCHS`（跑满，不 early stop） |
| `augment-mode` | `weak`（环境变量 `AUG`） |
| `heatmap-sigma-ratio` | `0.03` |
| `binary-tversky` | α=0.7, β=0.3 |
| `binary-neg-ratio` | `2` |
| `soft-background-weight` | `0.5` |
| `lambda-aff` / `lambda-binary` | `0.3` / `1.0` |

阶段概览：

| Stage | 内容 |
|-------|------|
| A | FC 分支 off/on × head norm none/GN |
| B | `lambda-aff` ∈ {0.1, 0.5, 0.8} |
| C | `binary-neg-ratio` ∈ {1, 3} |
| D | Tversky α/β 更严 (0.8/0.2) |
| E | `soft-background-weight` ∈ {0.25, 1.0} |
| F | 小 `lambda-peak` ∈ {0.02, 0.05} |
| G | `--predict-force-center` + `lambda-center-head` |
| H | `lambda-center-heatmap` 探索 |

结果目录：`output/affordance_no_rot_executed/training_runs/${GROUP}/<run-name>/`。  
失败 run 会列在脚本末尾；`CONTINUE_ON_ERROR=1`（默认）时继续后续实验。

> **注意：** 脚本写于 `full` 多 term loss 时期。若未传 `--loss-mode`，当前训练默认是 **`simple`**。跑 sweep 时请在 `COMMON_ARGS` 中加 `--loss-mode full`，或改 sweep 脚本。

---

## Human prior 监督训练

用 HDF5 里的 **`human_priors`**（clip 到 [0,1]）作软标签，**仅 L1**，与 robot 接触流水线分开。

| 项 | 说明 |
|----|------|
| 入口 | `python model/train_affordance_hp.py` |
| 快捷脚本 | `bash model/affordance/run_train_hp.sh` |
| 模型 | `PointNet2SegOnly`（无 FC 头） |
| Loss | `L1(pred_score, human_prior)`（`L1HumanPriorLoss`） |
| 默认 run 组 | `training_runs/hp_supervision/<run-name>/` |
| 指标阈值 | `--hp-threshold`（默认 0.5，用于 F1/AP） |

```bash
python model/train_affordance_hp.py \
  --run-group hp_supervision \
  --run-name hp_l1_full \
  --gpus 0 \
  --epochs 200 \
  --patience 30 \
  --lr 3e-4 \
  --augment-mode full \
  --hp-threshold 0.5 \
  --train-vis-max-objects 10 \
  --val-vis-max-objects 10
```

可视化：`val_epoch_*.png` / `train_epoch_*.png`。  
- 监督行用 **human_prior** 作为 GT soft/bin  
- 额外一行 **robot_gt**（来自 `labels`）仅作对比，**不参与 loss**

数据仍用同一 `affordance_{train,val}.h5`（需含 `human_priors` 字段；与 HF 包一致）。

---

## 命令行参数速查

### 训练通用

| 参数 | 默认 | 说明 |
|------|------|------|
| `--gpus` | `0` | 逗号分隔 GPU id |
| `--epochs` | `200` | 总 epoch |
| `--batch_size` | `64` | 每 GPU batch |
| `--lr` | `1e-3` | 初始学习率 |
| `--lr-min` | `1e-5` | cosine 下限 |
| `--weight_decay` | `1e-4` | |
| `--warmup-epochs` | `0` | >0 时 linear warmup + cosine；前 N ep 不计 early stop |
| `--warmup-start-factor` | `0.1` | warmup 起始 LR = factor × lr |
| `--dataset_dir` | `output/affordance_no_rot_executed` | |
| `--save_dir` | — | 最高优先级；run 根或 `ckpt/` |
| `--run-name` | — | `training_runs/[group/]name` |
| `--run-group` | — | 可选分组目录（sweep） |
| `--predict-force-center` | off | 是否构建 FC 头 |
| `--val-vis-max-objects` | `10` | val 可视化列数；`0`=全部 |
| `--train-vis-max-objects` | `10` | train 可视化；`0`=关闭 |
| `--train-vis-seed` | `split_seed+7919` | train vis 抽样种子 |
| `--resume` | — | checkpoint 路径 |
| `--patience` | `10` | early stop（warmup 期间不计） |
| `--disable-early-stop` | off | |
| `--num_workers` | `4` | DataLoader |
| `--master_port` | `29500` | DDP 端口 |

### Human prior 流水线（`train_hp.py`）

| 参数 | 默认 | 说明 |
|------|------|------|
| `--run-group` | `hp_supervision` | |
| `--hp-threshold` | `0.5` | F1/AP 二值化阈值 |
| `--patience` | `30` | |
| `--lr` | `3e-4` | |
| `--master_port` | `29501` | 避免与主训练 29500 冲突 |

### 损失 / 热力图（robot 流水线，`train.py`）

| 参数 | 默认 | 说明 |
|------|------|------|
| `--loss-mode` | `simple` | `full` = legacy 多 term |
| `--lambda-l1-reg` | `1e-4` | **simple 下无效**，仅兼容保留 |
| `--heatmap-sigma-ratio` | `0.05` |
| `--lambda-aff` | `0.3`（仅 full） |
| `--lambda-binary` | `1.0` |
| `--lambda-peak` | `0.0` |
| `--lambda-center-heatmap` | `0.0` |
| `--lambda-center-head` | `0.0` |
| `--lambda-consistency` | `0.0` |
| `--lambda-smooth` | `0.0` |
| `--ckpt-f1-weight` | `0.5` |
| `--disable-center-loss` | off |
| `--binary-tversky-alpha/beta` | `0.5` / `0.5` |
| `--binary-neg-ratio` | `1.0` |
| `--soft-background-weight` | `0.25` |
| `--head-norm` | `none` |

### 增广

| 参数 | 默认 |
|------|------|
| `--augment-mode` | `full` |
| `--no-augment` | off |

### Debug 专用

| 参数 | 默认 | 说明 |
|------|------|------|
| `--debug-overfit-one-object` | off | **启用 debug 模式** |
| `--debug-object-id` | — | 指定物体 |
| `--debug-num-objects` | `1` | 选 K 个物体 |
| `--debug-object-mode` | `first` | `first` / `random` |
| `--debug-samples-per-object` | `0` | 每物体样本上限，0=不限 |
| `--debug-use-sample-mode` | off | 按样本而非物体选 |
| `--debug-num-samples` | `1` | sample 模式下条数 |
| `--debug-seed` | `42` | 随机选物体/样本 |
| `--debug-max-steps` | `1000` | 优化步数 |
| `--debug-log-interval` | `20` | 日志间隔 |
| `--debug-vis-interval` | `50` | 存图间隔 |
| `--debug-vis-max-objects` | `10` | 可视化列数上限 |
| `--debug-synthetic-label` | — | `x_positive` / `z_positive` |

完整列表以 `python -m model.affordance.train --help` 为准。

---

## 常见问题

**Q: `affordance_train.h5` 不存在？**  
从 [UCBProject/train_affordance](https://huggingface.co/datasets/UCBProject/train_affordance/tree/main) 下载到 `output/affordance_no_rot_executed/`，或本地 `prepare` + 可选 `export_soft_affordance_gt.py`（见 [数据准备](#数据准备)）。

**Q: 有 `*_soft.h5` 但训练还用在线 soft？**  
`simple` / `full` 的 soft 监督均在 `Dataset` 里按增广后 xyz **在线**重算。磁盘 `soft_labels` 用于 HF/QC。

**Q: `train.py` 和 `train_hp.py` 选哪个？**  
学 **executed 接触** → `model.affordance.train`（默认 `simple` = L1 on 在线 soft）。学 **human prior** → `train_affordance_hp.py`（L1 on `human_priors`）。

**Q: `--lambda-l1-reg` 还有用吗？**  
**simple 模式不用**；loss 仅为 `L1(score, soft_gt)`。full 模式也不读该参数。

**Q: Sweep 结果和默认训练不一致？**  
默认已是 `simple`；sweep 需加 `--loss-mode full` 才与脚本内 λ/Tversky 设计一致。

**Q: 正式训练与 `model/train.py` 的关系？**  
`model/train.py` 是旧版多任务训练（`build_dataset.py` 数据）。**当前 no_rot executed 流水线请用 `model.affordance.train`**。

**Q: `human_priors` / `normals` 为何不用？**  
当前 PointNet++ 只吃 **xyz（3 通道）**。H5 里的 `normals` / `human_priors` 会加载但**不进网络、不进 loss**；接入需改 `in_channel` 与 forward。

**Q: DDP 报 batch 相关错误？**  
减小 `--batch_size` 或增加数据；脚本会自动 cap，但若每卡不足 1 个 batch 仍会失败。

**Q: Debug 正常但全量训练不收敛？**  
逐步放开：增广 `weak`→`full`、提高 `--lambda-aff`、按需加 `--lambda-peak`、调 Tversky α/β、检查 `heatmap_sigma_ratio`。

**Q: `COLLAPSE` 是什么？**  
验证时预测几乎全为正或概率动态范围极小；best_ckpt 默认会拒绝此类 epoch。

---

## 相关文档

| 文档 | 内容 |
|------|------|
| [`prepare_affordance_executed.md`](prepare_affordance_executed.md) | 训练 HDF5 生成、接触点 C/B/A、HP 坐标系 |
| [`grasp_collect_pipeline.md`](grasp_collect_pipeline.md) | merged hdf5 上游采集 |
| 根目录 [`README.md`](../README.md) Step 7–8 | 旧版 `build_dataset` + `model/train.py`（勿与本文混淆） |

