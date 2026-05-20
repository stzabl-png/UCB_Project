# OakInk 抓取采集流水线使用说明

从 **USD 转换** → **候选生成** → **Isaac Sim 验证** → **可视化 / 全量 batch**。

默认策略：**不旋转 mesh**（`--no-rotation`，与 SAM3D 原始朝向 + `scale.json` 一致）。

---

## 0. 环境准备（每次新开终端）

**路径因机器而异**，先把下面两个变量改成你自己的：

| 变量 | 含义 | 示例 |
|------|------|------|
| `PROJ` | 本仓库根目录 | `~/Project/Affordance2Grasp` |
| `ISAAC_SIM_PATH` | Isaac Sim 安装根目录（含 `python.sh`） | 见下方说明 |

**如何找到 Isaac Sim 路径：** 安装目录下应有 `python.sh`，例如：

- `~/isaacsim`
- `~/.local/share/ov/pkg/isaac-sim-4.x.x`
- NVIDIA 默认安装路径（按你本机文档为准）

```bash
# 任选一种方式确认
ls "$ISAAC_SIM_PATH/python.sh"
# 或: find ~ -name 'python.sh' -path '*/isaac*' 2>/dev/null | head
```

```bash
conda activate bundlesdf   # 或你用于 sampler/convert 的环境名

export PROJ=/path/to/Affordance2Grasp
cd "$PROJ"

export ISAAC_SIM_PATH=/path/to/your/isaac-sim   # ← 改成你的 Isaac 根目录
```

| 阶段 | Python | 说明 |
|------|--------|------|
| USD 转换 / sampler / batch 主进程 | `bundlesdf` 里的 `python3` | 需要 `trimesh`, `h5py`, `scipy`, `rtree`；USD 需要 `usd-core`（`pxr`） |
| Isaac Sim | `$ISAAC_SIM_PATH/python.sh` | batch 会自动调用；也可传 `--isaac-python /path/to/python.sh` |

安装缺失依赖（若报错）：

```bash
pip install rtree usd-core
```

---

## 1. USD 转换（全量或单个）

输出目录：`output/obj_usd/oakink/{OBJ}.usd` 与 `{OBJ}_meta.json`。

### 1.1 全量 OakInk（推荐先跑一遍）

```bash
conda activate bundlesdf
cd "$PROJ"

python3 tools/convert_obj_usd.py \
  --dataset oakink \
  --no-rotation \
  --force
```

### 1.2 单个物体 smoke test

```bash
python3 tools/convert_obj_usd.py \
  --obj A01001 \
  --dataset oakink \
  --no-rotation \
  --force
```

### 1.3 检查是否生成成功

```bash
ls -lh output/obj_usd/oakink/A01001.usd output/obj_usd/oakink/A01001_meta.json
```

---

## 2. 单物体：候选生成 + Sim + 可视化

### 2.1 生成 20 个候选（sampler）

```bash
conda activate bundlesdf
cd "$PROJ"

python3 tools/random_grasp_sampler.py \
  --obj A01001 \
  --dataset oakink \
  --no-rotation \
  --force \
  --target 20 \
  --output-dir output/grasp_collect/candidates/round_0000
```

输出：`output/grasp_collect/candidates/round_0000/A01001_grasp.hdf5`

### 2.2 Isaac Sim 验证（单物体）

```bash
# 确保已 export ISAAC_SIM_PATH（见第 0 节）

$ISAAC_SIM_PATH/python.sh sim/run_grasp_sim.py \
  --hdf5 output/grasp_collect/candidates/round_0000/A01001_grasp.hdf5 \
  --result-dir output/grasp_collect/robot_gt/round_0000 \
  --save-result \
  --headless \
  --max-candidates 20
```

输出：`output/grasp_collect/robot_gt/round_0000/A01001_robot_gt.hdf5`

### 2.3 可视化（matplotlib PNG）

全部候选 + 成功标绿：

```bash
conda activate bundlesdf
cd "$PROJ"

python3 tools/vis_grasp_candidates.py \
  --hdf5 output/grasp_collect/candidates/round_0000/A01001_grasp.hdf5 \
  --robot-gt output/grasp_collect/robot_gt/round_0000/A01001_robot_gt.hdf5
```

只画成功的：

```bash
python3 tools/vis_grasp_candidates.py \
  --hdf5 output/grasp_collect/candidates/round_0000/A01001_grasp.hdf5 \
  --robot-gt output/grasp_collect/robot_gt/round_0000/A01001_robot_gt.hdf5 \
  --only-success
```

PNG 目录：`output/grasp_vis/A01001/`（成功-only 在 `success/` 子目录）。

### 2.4 查看成功数量与名称

```bash
python3 -c "
import h5py
p='output/grasp_collect/robot_gt/round_0000/A01001_robot_gt.hdf5'
with h5py.File(p) as f:
    print('n_successful:', f.attrs.get('n_successful'))
    for k in sorted(f['successful_grasps'].keys()):
        print(' ', f['successful_grasps/'+k].attrs.get('name'))
"
```

---

## 3. 显存压测：每张 GPU 最多几个并行 sim

在正式 batch 前，用 A01001 测 `--sim-per-gpu`。压测会**真实启动多个 Isaac sim**（默认每个只验 1 个候选），因此需要一份**候选 HDF5** 作为输入。

### 3.0 前置：候选 HDF5 从哪来？

默认路径：

```text
output/grasp_collect/candidates/round_0000/A01001_grasp.hdf5
```

**检查是否已有：**

```bash
ls -lh output/grasp_collect/candidates/round_0000/A01001_grasp.hdf5
```

**若没有**（压测脚本报 `HDF5 不存在`），先跑 sampler 生成（只需几分钟，与第 2.1 节相同）：

```bash
conda activate bundlesdf
cd "$PROJ"

mkdir -p output/grasp_collect/candidates/round_0000

python3 tools/random_grasp_sampler.py \
  --obj A01001 \
  --dataset oakink \
  --no-rotation \
  --force \
  --target 20 \
  --output-dir output/grasp_collect/candidates/round_0000
```

生成后再确认：

```bash
ls -lh output/grasp_collect/candidates/round_0000/A01001_grasp.hdf5
```

也可用其它物体 / 路径，压测时显式指定：

```bash
python3 scripts/benchmark_sim_parallel.py \
  --obj A01010 \
  --hdf5 output/grasp_collect/candidates/round_0000/A01010_grasp.hdf5 \
  ...
```

> 压测**不需要** `robot_gt`（sim 成功结果），只要 **`*_grasp.hdf5`（候选）**。  
> 若还没有 USD，可先跑第 1 节，或确保 `output/obj_usd/oakink/A01001.usd` 存在，否则 sim 子进程会失败、显存读数不准。

### 3.1 运行压测

```bash
conda activate bundlesdf
cd "$PROJ"
# 确保已 export ISAAC_SIM_PATH（见第 0 节）

python3 scripts/benchmark_sim_parallel.py \
  --gpu 0 \
  --start-n 6 \
  --min-free-gb 3
```

### 3.2 双卡各测一次

```bash
python3 scripts/benchmark_sim_parallel.py --gpu 0 --start-n 6 --min-free-gb 3
python3 scripts/benchmark_sim_parallel.py --gpu 1 --start-n 6 --min-free-gb 3
```

### 3.3 看结果

```bash
cat output/benchmark_sim_parallel/gpu0_A01001/result_min_free_3.0gb.json | grep max_sim_per_gpu
```

记下推荐值，例如 `max_sim_per_gpu: 1` → batch 用 `--sim-per-gpu 1`。

---

## 4. 正式全量 batch（推荐）

脚本：`scripts/batch_grasp_collect.py`

**每轮两阶段：**

1. 并行生成**全部**物体的 candidate（`--sampler-workers`）
2. 并行 sim（`--sim-per-gpu` × `--sim-gpu-ids`）

输出根目录：`output/grasp_collect/`

| 路径 | 内容 |
|------|------|
| `candidates/round_XXXX/{OBJ}_grasp.hdf5` | 候选抓取 |
| `robot_gt/round_XXXX/{OBJ}_robot_gt.hdf5` | Sim 成功结果 |
| `merged/{OBJ}_robot_gt_merged.hdf5` | 多轮成功合并 |
| `summary.csv` | 每物体每轮状态（追加，不删历史） |
| `state.json` | 下次 `--resume` 从哪一轮开始 |
| `sim_logs/round_XXXX/` | convert / gen / sim 日志 |

**轮次与覆盖：** 不同 `round_XXXX/` **互不覆盖**。危险操作：不带 `--resume` 又从 `round_0000` 跑——sampler `--force` 会**重写**该轮已有 HDF5。

### 4.1 首次全量（默认 10 轮，已转 USD，双卡示例）

```bash
conda activate bundlesdf
cd "$PROJ"
# 确保已 export ISAAC_SIM_PATH

python3 scripts/batch_grasp_collect.py \
  --dataset oakink \
  --sampler-workers 8 \
  --sim-gpu-ids 0,1 \
  --sim-per-gpu 1 \
  --target 20 \
  --headless \
  --no-convert
```

- 默认 `--max-rounds 10` → `round_0000` … `round_0009`（可不写该参数）。
- 不加 `--rotation` = **no-rotation**。
- `--no-convert`：跳过 USD（需已完成第 1 节）。

### 4.2 跑完预设轮数后继续加轮（不覆盖旧 round）

例如已跑满默认 10 轮（`state.json` 里 `"round": 10`），再加 **5** 轮 → 只写 `round_0010`…`round_0014`：

```bash
python3 scripts/batch_grasp_collect.py \
  --dataset oakink \
  --max-rounds 5 \
  --resume \
  --sampler-workers 8 \
  --sim-gpu-ids 0,1 \
  --sim-per-gpu 1 \
  --headless \
  --no-convert
```

`--resume` 作用：

1. 从 `state.json` 的 `round` 继续编号（不回到 0000）。
2. 某一 round 目录里已有 `*_grasp.hdf5` / `*_robot_gt.hdf5` 的物体会跳过 gen/sim。

**不要**在已有 10 轮数据后去掉 `--resume` 再跑默认 10 轮——会从 `round_0000` 重来并覆盖该轮文件。

另开一套输出（与旧实验完全隔离）：

```bash
python3 scripts/batch_grasp_collect.py \
  --dataset oakink \
  --outdir output/grasp_collect_exp2 \
  ...
```

### 4.3 断点续跑（同一轮未完成）

batch 中途断了，同一轮里补跑未完成物体：

```bash
python3 scripts/batch_grasp_collect.py \
  --dataset oakink \
  --max-rounds 1 \
  --resume \
  --sampler-workers 8 \
  --sim-gpu-ids 0,1 \
  --sim-per-gpu 1 \
  --headless \
  --no-convert
```

若 `state.json` 已进位到下一轮，但你想**只补** `round_0003`，需手动把 `state.json` 里 `"round"` 改回 `3`，或只删该轮缺失物体的 HDF5 后 `--resume`。

### 4.4 单卡

```bash
python3 scripts/batch_grasp_collect.py \
  --dataset oakink \
  --sampler-workers 8 \
  --sim-gpu-ids 0 \
  --sim-per-gpu 1 \
  --headless \
  --no-convert
```

### 4.5 监控进度

```bash
tail -f output/grasp_collect/summary.csv
```

```bash
# 统计已有 robot_gt 数量
ls output/grasp_collect/robot_gt/round_0000/*_robot_gt.hdf5 2>/dev/null | wc -l
```

建议用 `tmux` / `screen` 长时间挂后台。

---

## 5. 参数速查

| 参数 | 默认 | 含义 |
|------|------|------|
| `--sampler-workers` | 4 | Phase 1 CPU 并行数 |
| `--sim-gpu-ids` | `0` | 物理 GPU 列表，如 `0,1` |
| `--sim-per-gpu` | 1 | **每张 GPU** 同时跑的 Isaac 数 |
| `--target` | 20 | 每物体每轮候选数 |
| `--max-rounds` | 10 | 轮数（默认 `round_0000`…`0009`） |
| `--no-convert` | 关 | 跳过 USD 转换 |
| `--resume` | 关 | 跳过已有 HDF5 |
| `--rotation` | 关 | 加上则使用 `rotation.json`（一般不要） |

**总 sim 并行数** = `len(sim-gpu-ids) × sim-per-gpu`  
例：`0,1` + `sim-per-gpu 1` → 最多 2 个 Isaac 同时跑。

---

## 6. 常见问题

### `No module named 'pxr'`

在 bundlesdf 中：`pip install usd-core`，或用已装好 pxr 的环境跑 `convert_obj_usd.py`。

### Sim 找不到 USD

确认存在：`output/obj_usd/oakink/{OBJ}.usd`，或去掉 `--no-convert` 让 batch 自动转换。

### 可视化 mesh 躺着、抓取不对齐

候选需 **no-rotation**；`vis_grasp_candidates` 会从 HDF5 `metadata/no_rotation` 自动对齐。旧图请重新生成。

### 可视化全是红色 FAILED

需加 `--robot-gt` 或 `--success raycast_1 ...`；仅候选 HDF5 不会标成功。

### 多轮候选是否相同？

**不同**。无固定 seed，每轮 `--force` 重新随机采样。

---

## 7. 最小命令清单（复制即用）

```bash
# 0. 环境（路径改成你的）
conda activate bundlesdf
export PROJ=/path/to/Affordance2Grasp
export ISAAC_SIM_PATH=/path/to/your/isaac-sim
cd "$PROJ"

# 1. USD（全量，只需一次）
python3 tools/convert_obj_usd.py --dataset oakink --no-rotation --force

# 2. 显存压测（可选；若无 A01001_grasp.hdf5 先做第 3.0 节 sampler）
python3 scripts/benchmark_sim_parallel.py --gpu 0 --start-n 6 --min-free-gb 3

# 3. 正式 batch（双卡，默认 10 轮 round_0000..0009）
python3 scripts/batch_grasp_collect.py \
  --dataset oakink \
  --sampler-workers 8 \
  --sim-gpu-ids 0,1 \
  --sim-per-gpu 1 \
  --headless \
  --no-convert

# 3b. 跑满 10 轮后再加 5 轮（不覆盖旧 round）:
# python3 scripts/batch_grasp_collect.py --dataset oakink --max-rounds 5 --resume ...

# 4. 抽查可视化
python3 tools/vis_grasp_candidates.py \
  --hdf5 output/grasp_collect/candidates/round_0000/A01001_grasp.hdf5 \
  --robot-gt output/grasp_collect/robot_gt/round_0000/A01001_robot_gt.hdf5 \
  --only-success
```

---

## 8. 相关文件

| 脚本 | 作用 |
|------|------|
| `tools/convert_obj_usd.py` | PLY → USD |
| `tools/random_grasp_sampler.py` | 候选抓取 |
| `sim/run_grasp_sim.py` | Isaac 验证 |
| `scripts/batch_grasp_collect.py` | 全量两阶段 batch |
| `scripts/benchmark_sim_parallel.py` | 显存 / 并行数压测 |
| `tools/vis_grasp_candidates.py` | 候选 + 成功 PNG |
| `tools/merge_robot_gt.py` | 多轮 GT 合并（batch 内自动调用） |
