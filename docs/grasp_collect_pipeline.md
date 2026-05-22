# OakInk 抓取采集流水线使用说明

从 **USD 转换** → **候选生成** → **Isaac Sim 验证** → **可视化 / 全量 batch**。

## 输出目录约定

| 目录 | 用途 |
|------|------|
| `output/grasp_collect_no_rot/` | **当前** 修改版 legacy batch 默认（rotated SAM3D + `train_fp_rotated` sampler） |
| `output/grasp_collect_legacy/` | 旧一轮实验（原 `grasp_collect` 改名保留） |
| `output/grasp_collect/` | placement batch 等仍可能共用（见 `batch_grasp_collect_placement.py`） |

下文示例路径以 **`grasp_collect_no_rot`** 为准；续跑时 `--outdir` 必须与已有 `state.json` 一致。

默认策略：**不旋转 mesh**（`--no-rotation` USD；采样 mesh 用 `rotated_mesh` + `train_fp_rotated` HP）。

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

输出目录：`output/obj_usd/{dataset}/{OBJ}.usd` 与 `{OBJ}_meta.json`。

**默认 mesh 源**（与 legacy sampler 一致）:

- `data_hub/meshes/SAM3DMesh/rotated_mesh/{oakink|ycb}/{obj}/mesh.ply`（已含 +90°X）
- `scale.json` 仍从 `data_hub/ProcessedData/obj_meshes/` 读取

请始终加 **`--no-rotation`**（脚本在 rotated_mesh 模式下会自动启用）。**不要**再叠 `rotation.json`。

### 1.1 OakInk + YCB 全量重转（换 rotated mesh 后必做）

```bash
conda activate bundlesdf
cd "$PROJ"

python3 tools/convert_obj_usd.py --dataset oakink --no-rotation --force
python3 tools/convert_obj_usd.py --dataset ycb --no-rotation --force
```

### 1.2 单个物体 smoke test

```bash
python3 tools/convert_obj_usd.py --obj A01001 --dataset oakink --no-rotation --force
python3 tools/convert_obj_usd.py --obj ycb_dex_01 --dataset ycb --no-rotation --force
```

### 1.3 回退 obj_meshes（旧资产）

```bash
python3 tools/convert_obj_usd.py --obj A01001 --legacy-assets --no-rotation --force
```

### 1.4 检查是否生成成功

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
  --output-dir output/grasp_collect_no_rot/candidates/round_0000
```

输出：`output/grasp_collect_no_rot/candidates/round_0000/A01001_grasp.hdf5`

### 2.2 Isaac Sim 验证（单物体）

```bash
# 确保已 export ISAAC_SIM_PATH（见第 0 节）

$ISAAC_SIM_PATH/python.sh sim/run_grasp_sim.py \
  --hdf5 output/grasp_collect_no_rot/candidates/round_0000/A01001_grasp.hdf5 \
  --result-dir output/grasp_collect_no_rot/robot_gt/round_0000 \
  --save-result \
  --headless \
  --max-candidates 20
```

输出：`output/grasp_collect_no_rot/robot_gt/round_0000/A01001_robot_gt.hdf5`

### 2.2.1 `robot_gt` HDF5 里有什么（schema v2）

整物体一个 `{OBJ}_robot_gt.hdf5`（不是每个 try 一个文件）。

| 位置 | 内容 |
|------|------|
| **根 attrs** | `obj_id`, `n_successful`, `robot_gt_schema_version=2`, `executed_pose_frame=object_mesh`, `executed_ee_frame=panda_hand` |
| **`candidate_results/candidate_i`** | **每个 try 都有**：候选 `grasp_point`/`rotation`、`success`、（跑完闭合+提起后）`executed_panda_hand_at_close` / `executed_panda_hand_post_lift` |
| **`successful_grasps/grasp_j`** | 仅 **Sim 成功** 的 try：同上 + `approach_dir`/`finger_dir`（来自候选旋转） |
| **`winning_candidate`** | 第一个成功者（候选 + executed，无 `contact_points_local`） |

**语义（必读）：**

- **`grasp_point` / `rotation`**：Stage A **规划候选**（grasp 帧），不是 Sim 执行真值。
- **`executed_panda_hand_at_close`**：闭合结束、**lift 前**，真实 **`panda_hand` 手腕** 在物体系下的位姿（`position`, `rotation`, `approach_dir`, `finger_dir`）。
- **`executed_panda_hand_post_lift`**：提起稳定后、与 **Δz>3cm 成功判定** 同刻的 `panda_hand` 位姿。
- **`gripper_tips_loc`** `(2,3)`：闭合后、lift 前（与 `executed_panda_hand_at_close` 同时刻）左右指尖在 **物体 mesh 局部系**；`finger_width_actual` 为两点距离。凡执行到 close 的候选都会写入 `candidate_results`（不限 success）。

**录屏：** batch **不**录屏。调试时用 `sim/run_grasp_sim_rec.py`（每 try 一个 mp4，与 headless sim 结果可能不一致）。

### 2.3 可视化（matplotlib PNG）

全部候选 + 成功标绿：

```bash
conda activate bundlesdf
cd "$PROJ"

python3 tools/vis_grasp_candidates.py \
  --hdf5 output/grasp_collect_no_rot/candidates/round_0000/A01001_grasp.hdf5 \
  --robot-gt output/grasp_collect_no_rot/robot_gt/round_0000/A01001_robot_gt.hdf5
```

只画成功的：

```bash
python3 tools/vis_grasp_candidates.py \
  --hdf5 output/grasp_collect_no_rot/candidates/round_0000/A01001_grasp.hdf5 \
  --robot-gt output/grasp_collect_no_rot/robot_gt/round_0000/A01001_robot_gt.hdf5 \
  --only-success
```

PNG 目录：`output/grasp_vis/A01001/`（成功-only 在 `success/` 子目录）。

### 2.4 查看成功数量与名称

```bash
python3 -c "
import h5py
p='output/grasp_collect_no_rot/robot_gt/round_0000/A01001_robot_gt.hdf5'
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
output/grasp_collect_no_rot/candidates/round_0000/A01001_grasp.hdf5
```

**检查是否已有：**

```bash
ls -lh output/grasp_collect_no_rot/candidates/round_0000/A01001_grasp.hdf5
```

**若没有**（压测脚本报 `HDF5 不存在`），先跑 sampler 生成（只需几分钟，与第 2.1 节相同）：

```bash
conda activate bundlesdf
cd "$PROJ"

mkdir -p output/grasp_collect_no_rot/candidates/round_0000

python3 tools/random_grasp_sampler.py \
  --obj A01001 \
  --dataset oakink \
  --no-rotation \
  --force \
  --target 20 \
  --output-dir output/grasp_collect_no_rot/candidates/round_0000
```

生成后再确认：

```bash
ls -lh output/grasp_collect_no_rot/candidates/round_0000/A01001_grasp.hdf5
```

也可用其它物体 / 路径，压测时显式指定：

```bash
python3 scripts/benchmark_sim_parallel.py \
  --obj A01010 \
  --hdf5 output/grasp_collect_no_rot/candidates/round_0000/A01010_grasp.hdf5 \
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

1. 并行生成**全部**物体的 candidate（`--sampler-workers`，**仅 raycast**，不传 `--structured-contacts`）
2. 并行 sim（`--sim-per-gpu` × `--sim-gpu-ids`，**`sim/run_grasp_sim.py --headless`**，不录屏）

**物体列表：** 与 `random_grasp_sampler.list_dataset_objs` 相同（**不是** `obj_meshes` 目录扫描）：

| `--dataset` | 条件 |
|-------------|------|
| `oakink` | `rotated_mesh/oakink/{id}/mesh.ply` + `train_fp_rotated/oakink/{id}.hdf5` + `scale.json` |
| `ycb` | `rotated_mesh/ycb/ycb_dex_*/mesh.ply` + `train_fp_rotated/dexycb/ycb_dex_*.hdf5` + scale |

**每轮可同时跑多个 dataset**（默认 `oakink+ycb` 共 120 物）：`--dataset oakink,ycb` 或 `--dataset all`。Sampler 按物传各自 `--dataset`；HDF5 文件名仅含 `obj_id`（OakInk / YCB id 不冲突）。全量前对两个 dataset 都做第 1 节 USD 转换。

输出根目录：`output/grasp_collect_no_rot/`（`batch_grasp_collect.py` 默认 `--outdir`）

| 路径 | 内容 |
|------|------|
| `candidates/round_XXXX/{OBJ}_grasp.hdf5` | 候选抓取（raycast） |
| `robot_gt/round_XXXX/{OBJ}_robot_gt.hdf5` | Sim 结果（schema v2，含 `executed_*`） |
| `merged/{OBJ}_robot_gt_merged.hdf5` | 多轮 **successful_grasps** 合并（**默认不去重**；含 `executed_*`） |
| `summary.csv` | 每物体每轮状态（追加，不删历史） |
| `state.json` | 下次 `--resume` 从哪一轮开始 |
| `sim_logs/round_XXXX/` | convert / gen / sim 日志 |

**轮次与覆盖：** 不同 `round_XXXX/` **互不覆盖**。危险操作：不带 `--resume` 又从 `round_0000` 跑——sampler `--force` 会**重写**该轮已有 HDF5。

### 4.1 首次全量（默认 10 轮，已转 USD，双卡示例）

```bash
conda activate bundlesdf
cd "$PROJ"
# 确保已 export ISAAC_SIM_PATH

# OakInk + YCB 同一 batch（默认即两者；每轮 round_XXXX 下 100+20 个物体）
python3 scripts/batch_grasp_collect.py \
  --dataset all \
  --sampler-workers 8 \
  --sim-gpu-ids 0,1 \
  --sim-per-gpu 1 \
  --target 20 \
  --headless \
  --no-convert

# 仅 OakInk: --dataset oakink
# 仅 YCB:     --dataset ycb
```

- 默认 `--max-rounds 10` → `round_0000` … `round_0009`（可不写该参数）。
- 不加 `--rotation` = **no-rotation**。
- `--no-convert`：跳过 USD（需已完成第 1 节）。
- **成功判据**由 sim 固定为物体抬高 Δz>3cm，batch 不修改。

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
tail -f output/grasp_collect_no_rot/summary.csv
```

```bash
# 统计已有 robot_gt 数量
ls output/grasp_collect_no_rot/robot_gt/round_0000/*_robot_gt.hdf5 2>/dev/null | wc -l
```

建议用 `tmux` / `screen` 长时间挂后台。

### 4.6 单独重跑 merge（不跑 batch）

batch 每个物体 sim 结束后会自动 merge；也可手动合并已有各轮 `robot_gt`：

```bash
conda activate bundlesdf
cd "$PROJ"

python3 tools/merge_robot_gt.py --obj A01001 \
  --inputs \
    output/grasp_collect_no_rot/robot_gt/round_0000/A01001_robot_gt.hdf5 \
    output/grasp_collect_no_rot/robot_gt/round_0001/A01001_robot_gt.hdf5 \
  --output output/grasp_collect_no_rot/merged/A01001_robot_gt_merged.hdf5
```

默认 **保留全部** 成功条目（不去重），并拷贝每条成功的 **`executed_panda_hand_at_close` / `post_lift`**。需要 dedup 时加 `--deduplicate`（按候选 `grasp_point` 判近，不是 executed）。

**指尖 / 接触点来源（merged schema v3，必读）：**

| 字段 | 含义 | 训练可用？ |
|------|------|------------|
| `gripper_tips_loc` | 新 Sim：`at_close` 真指尖（物体系） | ✅ `gripper_tips_trusted=True` |
| `contact_points_local` | 旧轮：lift 后伪接触点 | ❌ `contact_points_trusted=False` |

每条 `successful_grasps/grasp_*` 有 **`gripper_tips_source`**：`at_close` / `legacy_post_lift` / `none`。**不会**再把旧 `contact_points_local` 改名成 `gripper_tips_loc`。训练读 merged 时请只看 `gripper_tips_trusted`（`build_dataset.py` 已按此过滤）。若 merged 里不要任何 legacy 指尖，merge 时加 **`--exclude-legacy-contact`**（只丢接触点，抓取 pose 仍保留）。

整库重 merge 示例（bash）：

```bash
for f in output/grasp_collect_no_rot/robot_gt/round_0000/*_robot_gt.hdf5; do
  obj=$(basename "$f" _robot_gt.hdf5)
  inputs=$(ls output/grasp_collect_no_rot/robot_gt/round_*/${obj}_robot_gt.hdf5 2>/dev/null)
  [ -n "$inputs" ] || continue
  python3 tools/merge_robot_gt.py --obj "$obj" --output \
    "output/grasp_collect_no_rot/merged/${obj}_robot_gt_merged.hdf5" \
    --inputs $inputs
done
```

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
| `--merge-deduplicate` | 关 | 写 `merged/` 时去掉相近 pose |
| `--rotation` | 关 | 加上则使用 `rotation.json`（一般不要） |

### HDF5 中的 `mesh_prerotation/`（pose 级）

**每个 pose**（`candidates/candidate_i`、`successful_grasps/grasp_i` 等）下各有自己的 `mesh_prerotation/`，记录该 pose 生成/测试时 mesh 实际用的预旋转。文件根级不再写。

| 情况 | `euler_xyz_deg` / `matrix` |
|------|----------------------------|
| 默认 `--no-rotation` | `[0,0,0]` + 单位阵，`method=identity` |
| `--rotation` | 与 `rotation.json` 一致（经 canonical 阈值） |

| 字段 | 含义 |
|--------|------|
| `euler_xyz_deg` | 该 pose 对应的欧拉角 (度, xyz) |
| `matrix` | 对应 3×3 旋转矩阵 |
| `@method` | `identity` 或 `rotation.json` 的 method |

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
  --hdf5 output/grasp_collect_no_rot/candidates/round_0000/A01001_grasp.hdf5 \
  --robot-gt output/grasp_collect_no_rot/robot_gt/round_0000/A01001_robot_gt.hdf5 \
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
