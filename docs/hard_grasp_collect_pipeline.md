# 低成功物体抓取补采：候选池 Sim 流水线

针对 **merged 成功抓取数偏少** 的物体，用候选池 + Isaac Sim 批量验证，持续补充 `robot_gt` 与 `merged` 中的成功 pose。

流水线拆成两步：

| 组件 | 脚本 | 作用 |
|------|------|------|
| **候选池 sim batch** | `batch_sim_candidates_pool.py` | 从 pool 按 merged 成功数 **加权**抽 candidate，固定 4 个 Z-yaw sim，写 `robot_gt/round_R/` 并 merge |
| **候选池生成** | `batch_gen_candidates_pool.py` | CPU 上跑 `random_grasp_sampler`，维护 `candidates/pool/`；pool 空时 sim batch **自动 refill**（可关） |

典型工作目录：`output/grasp_collect_no_rot/`（`state.json`、`merged/`、`candidates/pool/`、`robot_gt/round_*`）。

HDF5 schema、crash/resume、registry 语义见 [`grasp_collect_pipeline.md`](grasp_collect_pipeline.md) 第 5 节。

---

## 1. 获取代码

**整仓 clone**（GitHub 含脚本与 `sim/env_config/`，不含大数据与二进制 sim 资产）：

```bash
git clone <repo-url> Affordance2Grasp
cd Affordance2Grasp
```

流水线相关入口：

| 脚本 | 作用 |
|------|------|
| `scripts/batch_sim_candidates_pool.py` | 候选池 sim batch（主入口） |
| `scripts/batch_gen_candidates_pool.py` | 候选池生成（auto-refill） |
| `sim/run_grasp_sim_pool.py` | Isaac 长驻 worker |

---

## 2. HuggingFace 数据集：`UCBProject/hard_obj_grasp_collect_pipeline`

代码在 GitHub；二进制 / HDF5 / USD 在 [UCBProject/hard_obj_grasp_collect_pipeline](https://huggingface.co/datasets/UCBProject/hard_obj_grasp_collect_pipeline)（[UCBProject](https://huggingface.co/UCBProject) org）。

Dataset 内 **直接 mirror 仓库相对路径**（顶层即 `sim/`、`output/`、`data_hub/`），下载时 `--local-dir $PROJ` 即可对齐。

维护者上传步骤见 [`hf_upload_hard_obj_grasp_collect.md`](hf_upload_hard_obj_grasp_collect.md)。

### 2.1 Dataset 内容与放置位置

| Dataset 内路径 | 必须 | 下载后位于 `$PROJ/` |
|----------------|------|---------------------|
| `sim/assets_franka/` | ✅ | `sim/assets_franka/` |
| `sim/assets_scene/` | ✅ | `sim/assets_scene/` |
| `output/obj_usd/` | ✅ | `output/obj_usd/` |
| `output/grasp_collect_no_rot/candidates/pool/` | ✅ | 同上（**见 §2.3**；HF 当前为旧 ~200 cand/物体，新 pool 上传后 **同路径覆盖**） |
| `output/grasp_collect_no_rot/merged/` | ✅ | 同上 |
| `output/grasp_collect_no_rot/state.json` | ✅ | 同上 |
| `output/grasp_collect_no_rot/robot_gt/` | 推荐 | 同上 |
| `data_hub/meshes/SAM3DMesh/rotated_mesh/` | ⚠️ auto-refill | `data_hub/meshes/SAM3DMesh/rotated_mesh/` |
| `data_hub/ProcessedData/train_fp_rotated/` | ⚠️ auto-refill | `data_hub/ProcessedData/train_fp_rotated/` |
| `data_hub/ProcessedData/obj_meshes/**/scale.json` | ⚠️ auto-refill | `data_hub/ProcessedData/obj_meshes/`（**仅 scale.json**） |

**Isaac Sim / cuRobo** 不在 HF，见 §4。

### 2.2 下载

```bash
export PROJ=/path/to/Affordance2Grasp
cd "$PROJ"

pip install -U huggingface_hub
hf auth login   # 或 export HF_TOKEN=...

# 整包（推荐）
hf download UCBProject/hard_obj_grasp_collect_pipeline \
  --repo-type dataset \
  --local-dir "$PROJ"
```

按需只下子目录：

```bash
hf download UCBProject/hard_obj_grasp_collect_pipeline output/grasp_collect_no_rot \
  --repo-type dataset --local-dir "$PROJ"
```

**整包下载后**，`$PROJ` 下应出现（路径与 git clone 后的仓库根 **合并**，不是另建子目录）：

```text
$PROJ/                                          ← git clone 的 Affordance2Grasp 根
├── scripts/ ...                                ← 来自 GitHub
├── sim/
│   ├── assets_franka/franka.usd               ← 来自 HF
│   ├── assets_scene/...                        ← 来自 HF
│   └── run_grasp_sim_pool.py                   ← 来自 GitHub
├── output/
│   ├── obj_usd/{oakink,ycb}/{OBJ}.usd          ← 来自 HF
│   └── grasp_collect_no_rot/
│       ├── candidates/pool/{OBJ}_grasp.hdf5    ← 来自 HF
│       ├── merged/{OBJ}_robot_gt_merged.hdf5   ← 来自 HF
│       ├── state.json                          ← 来自 HF
│       └── robot_gt/round_*/                   ← 来自 HF（推荐）
└── data_hub/                                   ← 来自 HF（auto-refill）
    ├── meshes/SAM3DMesh/rotated_mesh/...
    └── ProcessedData/
        ├── train_fp_rotated/...
        └── obj_meshes/{oakink,ycb}/**/scale.json
```

`hf download ... --local-dir "$PROJ"` 会把 HF 里与上表同名的路径 **写进现有 clone**；无需手动移动文件。

### 2.3 Candidate pool 版本（HF 旧 pool → 新 500 pool）

> **部署前必读：** HuggingFace dataset 里当前的 `output/grasp_collect_no_rot/candidates/pool/` 是 **旧版 pool**（每物体约 **200** candidate）。Titan 上正在生成的新 pool（500 cand/物体，merged 成功 &lt; 30）完成后，**上传时直接替换该目录下的文件**，**不要**在 HF / 部署机上改用别的 pool 路径。

| 阶段 | 位置 | 说明 |
|------|------|------|
| **HF / 部署默认 pool** | `output/grasp_collect_no_rot/candidates/pool/` | 标准路径；sim batch 默认 `--pool-dir` 指向此处 |
| **HF 上（当前）** | 同上 | **旧版** ~200 cand/物体 |
| **Titan 生成 staging（进行中）** | `/home/vision/Project/Affordance2Grasp/output/pool_500_threshold30` | 与正在 sim 的 `candidates/pool/` **隔离**，避免覆盖 Titan 在跑实验 |

新 pool 生成参数（Titan staging）：

- 物体：`merged` 成功数 **&lt; 30**
- 每物体 **500** candidate

Titan 上生成（写入 staging，**不碰**正在 sim 的 pool）：

```bash
python3 scripts/batch_gen_candidates_pool.py \
  --merged-dir output/grasp_collect_no_rot/merged \
  --output-dir output/pool_500_threshold30 \
  --success-threshold 30 \
  --target 500 \
  --sampler-workers 16
```

#### 上传 HF：同路径替换旧 pool（不要换 directory）

新 pool 生成完成后，**整包 rsync 进标准 pool 目录**，覆盖旧 `{OBJ}_grasp.hdf5`：

```bash
# Titan 上（示例）；目标必须是 candidates/pool/，不要新建 pool_500_* 给下游用
rsync -a --delete output/pool_500_threshold30/ \
  output/grasp_collect_no_rot/candidates/pool/

# 再按 hf_upload_hard_obj_grasp_collect.md staging → hf upload
rsync -a output/grasp_collect_no_rot/candidates/pool/ \
  "$STAGING/output/grasp_collect_no_rot/candidates/pool/"
```

上传后 HF dataset 内路径仍为 **`output/grasp_collect_no_rot/candidates/pool/`**；部署机 `hf download` 后 **无需** `--pool-dir`，与文档其它章节一致。

#### Registry：替换 pool 后可能需要手动清理

`sim_pool_registry.json` 按 `(obj_id, candidate_name#pool_idx)` 记录哪些 slot 已 sim。  
**整包替换 pool HDF5 后**，旧 registry 里「已 simulated」的下标可能对应 **新 pool 里不同的 pose**。

| 场景 | 建议 |
|------|------|
| **新机器 / 新实验**，从 HF 拉新 pool | **不要**从 Titan 拷贝 `sim_pool_registry.json`；无文件则 batch 从空 registry 开始 |
| **同一 outdir** 在 Titan 上曾用旧 pool 跑过 sim | 对 **被新 pool 覆盖的物体** 清 registry（见下）；或删除整份 `sim_pool_registry.json` 重来 |
| **auto-refill**（`--force`） | batch 会自动清 **被 refill 物体** 的 registry（§7）；**整包换 pool 不会自动清** |

手动清理（示例：新 pool 覆盖的所有低成功物体；或干脆整文件删除）：

```bash
# 整文件删除（最简单；会丢失所有物体的 sim 进度记录）
rm -f output/grasp_collect_no_rot/sim_pool_registry.json

# 或只删部分物体（需与 pool 中实际 obj_id 一致）
python3 - <<'PY'
import json
path = "output/grasp_collect_no_rot/sim_pool_registry.json"
objs_to_clear = {"ycb_dex_01", "A01002"}  # 换成被新 pool 覆盖的 obj_id
with open(path) as f:
    reg = json.load(f)
for oid in objs_to_clear:
    reg.get("candidates", {}).pop(oid, None)
with open(path, "w") as f:
    json.dump(reg, f, indent=2)
print("cleared", len(objs_to_clear), "objects")
PY
```

**与 Titan 在跑 round 的关系：** staging 在 `pool_500_threshold30/`，不会动 Titan 当前 `candidates/pool/` + `sim_pool_registry.json`；**上传 HF 并替换标准 pool 是给其它部署机用的**，不影响 Titan 上已开始的 round，除非你在 Titan 本机也对 `candidates/pool/` 做 `--delete` 替换（**不要在 sim 进行中做**）。

#### 部署机跑 sim（用 HF 新 pool 后）

```bash
python3 scripts/batch_sim_candidates_pool.py \
  --outdir output/grasp_collect_no_rot \
  --resume \
  ...
```

**不要**加 `--pool-dir output/pool_500_threshold30`；默认 `candidates/pool/` 即为新 pool。

### 2.4 本地从 mesh 生成 USD（备选）

若 dataset 未含 `output/obj_usd/`，但已有 `data_hub/.../rotated_mesh`：

```bash
conda activate bundlesdf
cd "$PROJ"
pip install usd-core
python3 tools/convert_obj_usd.py --dataset oakink --no-rotation --force
python3 tools/convert_obj_usd.py --dataset ycb --no-rotation --force
```

---

## 3. 运行前数据检查

| 检查项 | 命令 / 路径 |
|--------|-------------|
| Franka USD | `test -f sim/assets_franka/franka.usd` |
| 场景 USD | `ls sim/assets_scene/Collected_default_environment/default_environment.usd` |
| 物体 USD | `ls output/obj_usd/oakink/*.usd \| head -1` |
| Candidate pool | `ls output/grasp_collect_no_rot/candidates/pool/*_grasp.hdf5 \| head -1` |
| Merged GT | `ls output/grasp_collect_no_rot/merged/*_merged.hdf5 \| head -1` |
| 续跑 round | `cat output/grasp_collect_no_rot/state.json` |
| auto-refill | `ls data_hub/meshes/SAM3DMesh/rotated_mesh/oakink/*/mesh.ply \| head -1` |

若无 `data_hub` 且需要 auto-refill，运行 sim batch 时加 **`--no-auto-refill`**。

---

## 4. 软件环境

### 4.1 依赖

| 组件 | 说明 |
|------|------|
| **Isaac Sim** | 设置 `ISAAC_SIM_PATH`（目录含 `python.sh`） |
| **cuRobo** | 源码安装；`run_grasp_sim_pool.py` 会搜 `~/Project/curobo/src`、`~/curobo/src` 等，否则改该文件内 `sys.path` |
| **Conda 环境** | 如 `bundlesdf`：跑 batch 主进程 + 候选池生成 |

```bash
pip install -r requirements.txt
pip install rtree          # sampler
pip install usd-core       # 仅 convert_obj_usd 需要
```

### 4.2 每次开终端

```bash
conda activate bundlesdf
export PROJ=/path/to/Affordance2Grasp
export ISAAC_SIM_PATH=/path/to/isaac-sim
cd "$PROJ"
```

### 4.3 启动前检查

见 **§3 运行前数据检查**，并确认：

```bash
test -f "$ISAAC_SIM_PATH/python.sh" && echo "Isaac OK"
test -f scripts/batch_sim_candidates_pool.py && test -f sim/run_grasp_sim_pool.py && echo "Scripts OK"
```

---

## 5. 运行

### 5.1 并行度

**Isaac 进程总数** = `len(--sim-gpu-ids) × --sim-per-gpu`

同 GPU 多 worker **错开 10s** 启动（`SAME_GPU_STAGGER_S=10`）。`--sim-per-gpu` 按机器显存调整；多卡示例：

```bash
--sim-gpu-ids 0,1,2,3 --sim-per-gpu 2    # 8 个 worker
--sim-gpu-ids 0,1 --sim-per-gpu 4        # 8 个 worker（双卡）
--sim-gpu-ids 0 --sim-per-gpu 1          # 单卡 smoke
```

### 5.2 生产命令

```bash
conda activate bundlesdf
export ISAAC_SIM_PATH=/path/to/isaac-sim
cd "$PROJ"

python3 scripts/batch_sim_candidates_pool.py \
  --outdir output/grasp_collect_no_rot \
  --resume \
  --max-rounds 10 \
  --sim-gpu-ids 0,1,2,3 \
  --sim-per-gpu 2 \
  --headless
```

> 默认 `--pool-dir` 为 `{outdir}/candidates/pool`。从 HF 拉下 **§2.3 新 pool** 后直接用即可，**无需**改 pool 路径。

每轮默认 **500 candidate 槽位 × 4 yaw = 2000** sim task；低成功物体因加权会被更频繁抽到。

**Early-stop（默认开启）：** 同一 `(obj, candidate)` **任一 yaw 抓取成功** 后，其余未试 yaw **不再进 Isaac sim**，在 chunk 中写 `skipped: true` 行；registry 将该 candidate 标为 resolved，下轮规划不再抽取。禁用：`--no-early-stop-yaw-on-success`。

| 参数 | 默认 | 含义 |
|------|------|------|
| `--outdir` | `output/grasp_collect_no_rot` | 实验根目录 |
| `--pool-dir` | `{outdir}/candidates/pool` | candidate 池 HDF5（HF 标准路径；见 §2.3 上传替换） |
| `--slots-per-round` | 500 | 每轮 candidate 槽位数 |
| `--pool-target` | 50 | auto-refill 时每个物体的 pool `--target` |
| `--score-threshold` | 70 | refill sampler 分数门槛 |
| `--no-auto-refill` | 关 | pool 空时不调候选池生成 |
| `--resume` | 关 | 读 `state.json`；续跑未完成 `task_queue` |
| `--sim-timeout` | 7200 | 单 worker chunk 超时（秒） |

建议用 `tmux` / `screen` 挂后台。

### 5.3 监控

```bash
tail -f output/grasp_collect_no_rot/summary.csv
cat output/grasp_collect_no_rot/state.json
ls output/grasp_collect_no_rot/robot_gt/round_0015/ | wc -l
tail -f output/grasp_collect_no_rot/sim_logs/round_0015/chunk_*_gpu*.log
```

---

## 6. Smoke test（正式跑之前）

```bash
python3 scripts/batch_sim_candidates_pool.py \
  --outdir output/grasp_collect_smoke \
  --pool-dir output/grasp_collect_no_rot/candidates/pool \
  --merged-dir output/grasp_collect_no_rot/merged \
  --max-rounds 1 \
  --slots-per-round 4 \
  --sim-gpu-ids 0 \
  --sim-per-gpu 1 \
  --headless \
  --no-auto-refill
```

期望：

- `output/grasp_collect_smoke/robot_gt/round_0000/*_robot_gt.hdf5` 有输出
- worker log 中同 chunk 内出现 **物体 swap**（需每 chunk ≥2 个不同物体：`slots-per-round` 足够大且多 worker 分 chunk）

---

## 7. 候选池生成

日常由 sim batch **自动**触发。手动补货：

```bash
python3 scripts/batch_gen_candidates_pool.py \
  --merged-dir output/grasp_collect_no_rot/merged \
  --output-dir output/grasp_collect_no_rot/candidates/pool \
  --success-threshold 20 \
  --target 50 \
  --sampler-workers 16 \
  --force
```

auto-refill 时 sim batch 用 **merged 成功数中位数（round≥3）** 作 `--success-threshold`，并固定 `--force`。

**Auto-refill 与 registry：** refill 会用 `--force` **覆盖** 低成功物体的 pool HDF5。成功后 batch 会：

1. **清除被 refill 物体**在 `sim_pool_registry.json` 里的记录（不是整文件删除；未 refill 的高成功物体保留 registry）
2. 若当轮 queue **未完成**，删除 `round_R_task_queue.json`，避免 resume 仍引用旧 pool 里的 candidate

若手动 refill（`batch_gen_candidates_pool.py --force`），需同样清理对应物体的 registry 条目。

---

## 8. 续跑说明

| 场景 | 做法 |
|------|------|
| 正常续跑 | 保留 `state.json`、`merged/`、`pool/`、`robot_gt/`、`sim_pool_registry.json`；加 `--resume` |
| 某轮 worker crash / Ctrl+C | `--resume` 从 **chunk results** 重建 `completed_task_ids`；只 sim **chunk 无记录** 的 task |
| mid-round 中断 | worker 结束 / 每 ~5s / Ctrl+C / 轮末：从 **chunk 扫盘** 同步 queue+registry（chunk 为唯一真相） |
| 进下一轮 | `state.round` 仅在 **chunk↔queue sync_ok** 后 +1；**不要求** 2000/2000 sim 跑满（pending=无 chunk 记录可保留） |
| 新 outdir、round 从 0 | 新 `--outdir` 或 `state.json` → `{"round": 0}`；仍需 `merged/` 做加权 |
| 换 pool / 上传新 pool 后 | **手动**清 `sim_pool_registry.json` 或删被覆盖物体的 registry 条目（§2.3）；auto-refill 仅自动清被 refill 物体（§7） |
| pool 已 sim 过 | 保留 `sim_pool_registry.json`，否则同一 candidate 可能被重抽 |

---

## 9. 常见问题

| 现象 | 处理 |
|------|------|
| `USD not found` | 检查 `output/obj_usd/` 或跑 `convert_obj_usd.py` |
| `merged dir not found` | 准备 `merged/{obj}_robot_gt_merged.hdf5` |
| `cuRobo` import 失败 | 安装 cuRobo 或改 `run_grasp_sim_pool.py` 的 `sys.path` |
| pool 空 + refill 失败 | 检查 `data_hub` 或 `--no-auto-refill` |
| 同 GPU 多 worker OOM / crash | 降低 `--sim-per-gpu` |
| worker 0 results | 该 task 不在 `chunk_*_results.json` 中 → `--resume` 会补跑；不会标 `simulated` |
| 换 pool 后 registry 与 pose 对不上 | 删 `sim_pool_registry.json` 或按物体清 registry（§2.3） |
| mid-round Ctrl+C 后重跑 | `completed_task_ids` 由 chunk 重建；有 chunk 行的 task **不会**重 sim |
| chunk 有、queue 无（旧 ingest 缺口） | 轮末 sync 会补齐；或 `sync_queue_and_registry_from_chunks()` 手动跑一次 |

---

## 10. 部署核对清单

**代码**

- [ ] `git clone` 完整仓库，`cd Affordance2Grasp`

**HF / 数据（§2）**

- [ ] `sim/assets_franka/`、`sim/assets_scene/` → `$PROJ/sim/`
- [ ] `output/obj_usd/` → `$PROJ/output/obj_usd/`
- [ ] `output/grasp_collect_no_rot/candidates/pool/`（§2.3 **新 500 pool**，HF 上应已替换旧 ~200 pool；默认路径，无需 `--pool-dir`）
- [ ] `output/grasp_collect_no_rot/`（merged、state.json；推荐 robot_gt）
- [ ] 新部署 **勿** 从 Titan 拷贝 `sim_pool_registry.json`；换 pool 后按需手动清 registry（§2.3）
- [ ] （auto-refill）`data_hub/` 三包或等价 ProcessedData 子集

**环境（§4）**

- [ ] Isaac Sim + `ISAAC_SIM_PATH`
- [ ] cuRobo
- [ ] conda + `requirements.txt` + `rtree`

**验证**

- [ ] §3 数据检查通过
- [ ] §6 smoke 通过
- [ ] §5.2 生产命令已启动

---

## 11. 相关文档

| 文档 | 内容 |
|------|------|
| [`grasp_collect_pipeline.md`](grasp_collect_pipeline.md) | Pool 全流程、HDF5 schema、Legacy 对比 |
| `scripts/batch_sim_candidates_pool.py` | 候选池 sim batch CLI |
| `scripts/batch_gen_candidates_pool.py` | 候选池生成 CLI |
