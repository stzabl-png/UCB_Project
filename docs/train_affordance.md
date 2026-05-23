# Affordance 模型训练指南

PointNet++ affordance 训练流水线，基于 **no_rot executed** 数据集：从 Isaac Sim 验证过的 executed grasp 生成 per-point 接触标签，用 Gaussian soft heatmap 作为主监督。

**入口脚本：**

- `python -m model.affordance.train`（推荐）
- `python model/train_affordance.py`（薄封装，等价）

**代码目录：** `model/affordance/`（`train.py`、`dataset.py`、`losses.py`、`debug.py` 等）

---

## 目录

1. [整体流程](#整体流程)
2. [数据准备](#数据准备)
3. [环境与快速开始](#环境与快速开始)
4. [正式训练](#正式训练)
5. [输出目录与 checkpoint](#输出目录与-checkpoint)
6. [损失函数与模型](#损失函数与模型)
7. [多卡 DDP](#多卡-ddp)
8. [续训 `--resume`](#续训-resume)
9. [Debug 过拟合模式](#debug-过拟合模式)
10. [命令行参数速查](#命令行参数速查)
11. [常见问题](#常见问题)
12. [相关文档](#相关文档)

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
  output/affordance_no_rot_executed/training_runs/<时间戳>/
    ckpt/  vis/  log/
```

上游数据与 prepare 细节见 [`docs/prepare_affordance_executed.md`](prepare_affordance_executed.md)、[`docs/grasp_collect_pipeline.md`](grasp_collect_pipeline.md)。

---

## 数据准备

### 训练直接读取的文件

默认 `--dataset_dir`：

```text
output/affordance_no_rot_executed/
├── affordance_train.h5    # 必需
├── affordance_val.h5      # 必需
├── dataset_info.json      # 统计信息（可选参考）
├── objects_trainable.txt
└── objects_train_val_split.json
```

### HDF5 字段（`data/` 组）

| 字段 | 形状 | 训练是否使用 |
|------|------|-------------|
| `points` | (N, 4096, 3) | ✅ 点云 xyz |
| `normals` | (N, 4096, 3) | ✅ 法线 |
| `labels` | (N, 4096) | ✅ 二值接触 GT → soft heatmap |
| `force_centers` | (N, 3) | ✅ 力中心（center heatmap loss，debug 时关闭） |
| `obj_ids` | (N,) | ✅ 物体级 train/val 划分 |
| `human_priors` | (N, 4096) | ❌ 仅 prepare 写入，**当前网络不读** |

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

---

## 环境与快速开始

```bash
conda activate bundlesdf
cd ~/Project/Affordance2Grasp

# 单卡正式训练
python -m model.affordance.train --gpus 0

# 指定数据目录
python -m model.affordance.train \
  --dataset_dir output/affordance_no_rot_executed \
  --gpus 0 \
  --batch_size 64
```

需要 **CUDA**；debug 模式仅支持单卡。

---

## 正式训练

### 网络与输入

- 模型：`PointNet2Seg`，`in_channel=6`（xyz + normals）
- 输出：per-point 接触 logits → softmax 概率；`predict_force_center=True`（正式训练）
- GT：由 `labels` 生成 Gaussian soft heatmap（`σ = heatmap_sigma_ratio × 物体 bbox 对角线`，默认 ratio=0.05）

### 学习率与早停

| 项 | 默认 |
|----|------|
| 优化器 | AdamW，`lr=1e-3`，`weight_decay=1e-4` |
| 调度 | Cosine，`lr_min=1e-5`，`warmup_epochs=5` |
| 早停 | `patience=10`（warmup 期内不计入）；指标无提升则停 |
| Best checkpoint | `best_score = ckpt_f1_weight·F1 + (1-w)·AP`（默认各 50%） |
| 塌缩检测 | 预测全正/概率 span 过小会标 `COLLAPSE`，默认不写入 best |

### 数据增广（`--augment-mode`）

| 模式 | rotation | scale | shift | jitter | dropout |
|------|----------|-------|-------|--------|---------|
| `full`（默认） | ✅ | ✅ | ✅ | ✅ | ✅ |
| `weak` | ✅ | ✅ | ✅ | ✅ | ❌ |
| `none` | ❌ | ❌ | ❌ | ❌ | ❌ |

`--no-augment` 等价于关闭全部增广。

### 验证可视化

每个 epoch 在 `vis/val_epoch_XXXX.png` 保存验证物体网格（GT / pred / heatmap 等）。

---

## 输出目录与 checkpoint

每次**新 run**（无 `--resume`）创建：

```text
output/affordance_no_rot_executed/training_runs/<YYYYMMDD_HHMMSS>/
├── ckpt/
│   ├── best_ckpt.pth          # val score 最优
│   ├── last_ckpt.pth          # 最新 epoch（含 optimizer/scheduler/history）
│   ├── training_history.json
│   ├── split_info.json
│   └── checkpoint_epoch{N}.pth  # 每 10 epoch 仅存 model（轻量）
├── vis/
│   └── val_epoch_*.png
├── log/
│   └── train.log              # 与终端同步的配置 + epoch 表
└── run_manifest.json          # 本次 run 的 args 快照
```

`--save_dir` 可指定 run 根目录或 `.../ckpt/` 路径；`vis/`、`log/` 与 `ckpt/` 同级，**不会**放在 `ckpt/` 里面。

---

## 损失函数与模型

### `--loss-mode`（正式训练）

| 模式 | 公式概要 |
|------|----------|
| `current_soft`（默认） | `L_aff + λ_bin·L_binary + λ_ch·L_center_hm` |
| `peak_soft_no_center` | `L_peak + 0.5·L_binary + 0.5·L_aff`（关闭 center） |
| `full_with_center` | `peak_soft` + `λ_ch·L_center_hm`（默认 λ_ch=2） |

- **L_aff**：`0.7·balanced_soft_focal + 0.3·soft_Dice`（对 soft heatmap）
- **L_binary**：`0.6·Focal + 0.4·Tversky`（硬二值标签）
- **L_center_hm**：预测 heatmap 质心与 `force_centers` 的 L1（按物体尺度归一化）

默认权重：`λ_binary=0.5`，`λ_center_heatmap=10.0`，`λ_center_head=0`，`λ_consistency=0`，`λ_smooth=0`。

`--disable-center-loss` 会将 center 相关 λ 置 0。

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
| force center 头 | 开启 | **关闭**（`predict_force_center=False`） |
| center loss | 可开 | **强制为 0** |
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
- 所有 center / consistency / smooth λ = 0
- `weight_decay=0`，`warmup_epochs=0`
- `ckpt_reject_collapse=False`（塌缩也允许记 best，便于观察）

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

### `--debug-loss-mode`

| 模式 | 损失 |
|------|------|
| `binary_only`（默认） | 仅 `L_binary` |
| `peak_soft` | `L_peak + 0.5·L_aff` |
| `soft_only` | 仅 `L_aff` |
| `peak_binary_soft` | `0.3·L_peak + 0.7·L_binary + 0.5·L_aff` |
| `binary_soft_strict` | `L_binary + 0.3·L_aff` |

建议排查顺序：

1. `binary_only` + 合成标签 → 验证网络能学简单模式  
2. `binary_only` + 真实标签 → 验证 GT 与可视化  
3. `peak_soft` / `binary_soft_strict` → 逐步加入 soft heatmap  

### 合成标签（排除 GT 问题）

```bash
python -m model.affordance.train \
  --gpus 0 \
  --debug-overfit-one-object \
  --debug-synthetic-label x_positive \
  --debug-loss-mode binary_only \
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
  --debug-loss-mode binary_only \
  --debug-max-steps 1000 \
  --lr 1e-3 \
  --debug-log-interval 20 \
  --debug-vis-interval 50
```

**随机 5 个物体，每物体最多 3 条样本，带 soft loss：**

```bash
python -m model.affordance.train \
  --gpus 0 \
  --debug-overfit-one-object \
  --debug-num-objects 5 \
  --debug-object-mode random \
  --debug-seed 42 \
  --debug-samples-per-object 3 \
  --debug-loss-mode binary_soft_strict \
  --debug-max-steps 2000
```

**不增广、加大 Tversky FP 惩罚（减少全图高亮）：**

```bash
python -m model.affordance.train \
  --gpus 0 \
  --debug-overfit-one-object \
  --debug-object-id A01001 \
  --debug-loss-mode binary_only \
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

在 **单物体、binary_only、无增广** 下，通常期望数百步内：

- `μ+` 明显高于 `μ-`，`span` 明显大于 0.1  
- `F1` / `AP` 持续上升（小物体上可到很高）  
- `debug_step_*.png` 中 pred 热点与 GT 接触带重合  

若 loss 下降但 F1 不升：查标签是否极稀疏、Tversky α/β、或改用 `peak_soft`。  
若 `grad` 长期为 0：查 CUDA、学习率、是否 batch 为空。

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
| `--warmup-epochs` | `5` | warmup 不计入 early stop |
| `--dataset_dir` | `output/affordance_no_rot_executed` | |
| `--save_dir` | 自动时间戳 | run 或 ckpt 路径 |
| `--resume` | — | checkpoint 路径 |
| `--val_ratio` | `0.15` | 物体级 val 比例 |
| `--split_seed` | `42` | 划分随机种子 |
| `--patience` | `10` | early stop |
| `--disable-early-stop` | off | |
| `--num_workers` | `4` | DataLoader workers |
| `--master_port` | `29500` | DDP 端口 |

### 损失 / 热力图

| 参数 | 默认 |
|------|------|
| `--loss-mode` | `current_soft` |
| `--heatmap-sigma-ratio` | `0.05` |
| `--lambda-binary` | `0.5` |
| `--lambda-center-heatmap` | `10.0` |
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
| `--debug-loss-mode` | `binary_only` | 见上文 |
| `--debug-max-steps` | `1000` | 优化步数 |
| `--debug-log-interval` | `20` | 日志间隔 |
| `--debug-vis-interval` | `50` | 存图间隔 |
| `--debug-vis-max-objects` | `10` | 可视化列数上限 |
| `--debug-synthetic-label` | — | `x_positive` / `z_positive` |

完整列表以 `python -m model.affordance.train --help` 为准。

---

## 常见问题

**Q: `affordance_train.h5` 不存在？**  
先运行 `tools/prepare_affordance_executed.py`（见 [数据准备](#数据准备)）。

**Q: 正式训练与 `model/train.py` 的关系？**  
`model/train.py` 是旧版多任务训练（`build_dataset.py` 数据）。**当前 no_rot executed 流水线请用 `model.affordance.train`**。

**Q: `human_priors` 何时进网络？**  
尚未接入；H5 里仅有字段。若要使用需改 `SoftAffordanceDataset` 与 `in_channel`。

**Q: DDP 报 batch 相关错误？**  
减小 `--batch_size` 或增加数据；脚本会自动 cap，但若每卡不足 1 个 batch 仍会失败。

**Q: Debug 正常但全量训练不收敛？**  
逐步放开：增广 `weak`→`full`、加入 `soft` loss、调 `lambda_binary` / Tversky、检查类别不平衡与 `heatmap_sigma_ratio`。

**Q: `COLLAPSE` 是什么？**  
验证时预测几乎全为正或概率动态范围极小；best_ckpt 默认会拒绝此类 epoch。

---

## 相关文档

| 文档 | 内容 |
|------|------|
| [`prepare_affordance_executed.md`](prepare_affordance_executed.md) | 训练 HDF5 生成、接触点 C/B/A、HP 坐标系 |
| [`grasp_collect_pipeline.md`](grasp_collect_pipeline.md) | merged hdf5 上游采集 |
| 根目录 [`README.md`](../README.md) Step 7–8 | 旧版 `build_dataset` + `model/train.py`（勿与本文混淆） |

