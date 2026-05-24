# 上传 `UCBProject/hard_obj_grasp_collect_pipeline`

维护者：把低成功物体抓取补采流水线所需 **最小** 资产打进一个 HF Dataset。  
使用者下载见 [`hard_grasp_collect_pipeline.md`](hard_grasp_collect_pipeline.md) §2。

Dataset 顶层 **mirror 仓库路径**（`sim/`、`output/`、`data_hub/`），便于：

```bash
hf download UCBProject/hard_obj_grasp_collect_pipeline \
  --repo-type dataset --local-dir "$PROJ"
```

---

## 0. 前置

```bash
pip install -U huggingface_hub
hf auth login
# 或: export HF_TOKEN=hf_...

export PROJ=/home/vision/Project/Affordance2Grasp
export REPO=UCBProject/hard_obj_grasp_collect_pipeline
export STAGING=/tmp/hf_hard_obj_grasp_collect_staging
```

需对 [UCBProject](https://huggingface.co/UCBProject) org 有 **write** 权限。

---

## 1. 创建空 Dataset

**网页：** [UCBProject](https://huggingface.co/UCBProject) → New dataset → 名称 `hard_obj_grasp_collect_pipeline` → Private（按需）。

**CLI：**

```bash
hf repos create UCBProject/hard_obj_grasp_collect_pipeline \
  --repo-type dataset \
  --private
```

---

## 2. 准备 staging（只拷必要文件）

```bash
rm -rf "$STAGING"
mkdir -p "$STAGING"
cd "$PROJ"
```

### 2.1 Sim 资产

```bash
mkdir -p "$STAGING/sim"
rsync -a sim/assets_franka/ "$STAGING/sim/assets_franka/"
rsync -a sim/assets_scene/  "$STAGING/sim/assets_scene/"
```

> 若本地无 `assets_scene`，可从已有 [`UCBProject/assets_scene`](https://huggingface.co/datasets/UCBProject/assets_scene) 下载后放进 staging，再改路径为 `sim/assets_scene/`（代码硬编码此路径）。

### 2.2 物体 USD

```bash
mkdir -p "$STAGING/output"
rsync -a output/obj_usd/ "$STAGING/output/obj_usd/"
```

### 2.3 实验起点 `grasp_collect_no_rot`

**Candidate pool 始终在标准路径** `output/grasp_collect_no_rot/candidates/pool/`（不要在 dataset 里另建 `pool_500_*` 给下游用）。详见 [`hard_grasp_collect_pipeline.md`](hard_grasp_collect_pipeline.md) §2.3。

**用新 500 pool 替换 HF 上旧 ~200 pool 时**（Titan 生成在 `output/pool_500_threshold30/`，与在跑 sim 的 pool 隔离）：

```bash
# Titan：staging → 标准 pool（覆盖旧文件）
rsync -a --delete output/pool_500_threshold30/ \
  output/grasp_collect_no_rot/candidates/pool/
```

再 staging 其余 grasp_collect 资产：

```bash
mkdir -p "$STAGING/output/grasp_collect_no_rot/candidates/pool"

rsync -a output/grasp_collect_no_rot/candidates/pool/ \
  "$STAGING/output/grasp_collect_no_rot/candidates/pool/"

rsync -a output/grasp_collect_no_rot/merged/ \
  "$STAGING/output/grasp_collect_no_rot/merged/"

cp output/grasp_collect_no_rot/state.json \
  "$STAGING/output/grasp_collect_no_rot/state.json"

# 推荐：历史 robot_gt（merge + 加权）
rsync -a output/grasp_collect_no_rot/robot_gt/ \
  "$STAGING/output/grasp_collect_no_rot/robot_gt/"

# 续跑 registry（可选）；整包换 pool 上传时通常 **不要** 带 Titan 的 registry
[ -f output/grasp_collect_no_rot/sim_pool_registry.json ] && \
  cp output/grasp_collect_no_rot/sim_pool_registry.json \
     "$STAGING/output/grasp_collect_no_rot/"
```

**不要**上传：`sim_logs/`、`summary.csv`、`*_task_queue.json`（运行时会再生；task_queue 与具体机器续跑绑定）。

**换 pool 后 registry：** 部署机用新 pool 时应空 registry 或手动清理（[`hard_grasp_collect_pipeline.md`](hard_grasp_collect_pipeline.md) §2.3）。

### 2.4 auto-refill：`rotated_mesh` + `train_fp_rotated`

```bash
mkdir -p "$STAGING/data_hub/meshes/SAM3DMesh" "$STAGING/data_hub/ProcessedData"
rsync -a data_hub/meshes/SAM3DMesh/rotated_mesh/ \
  "$STAGING/data_hub/meshes/SAM3DMesh/rotated_mesh/"

rsync -a data_hub/ProcessedData/train_fp_rotated/ \
  "$STAGING/data_hub/ProcessedData/train_fp_rotated/"
```

### 2.5 auto-refill：仅 `scale.json`（不要 mesh.ply）

只拷贝 `obj_meshes` 下 **oakink / ycb** 的 `scale.json`，保留目录结构：

```bash
BASE=data_hub/ProcessedData/obj_meshes
DEST="$STAGING/data_hub/ProcessedData/obj_meshes"

for ds in oakink ycb; do
  find "$BASE/$ds" -name scale.json | while read -r f; do
    rel="${f#$BASE/}"
    mkdir -p "$DEST/$(dirname "$rel")"
    cp "$f" "$DEST/$rel"
  done
done

# 核对：应约 120 个 scale.json，无 mesh.ply
find "$DEST" -name scale.json | wc -l
find "$DEST" -name '*.ply' | wc -l   # 期望 0
```

### 2.6 Dataset README（可选）

```bash
cat > "$STAGING/README.md" <<'EOF'
# hard_obj_grasp_collect_pipeline

Assets for low-success grasp collection (pool sim batch).

Download into repo root:
```bash
hf download UCBProject/hard_obj_grasp_collect_pipeline \
  --repo-type dataset --local-dir /path/to/Affordance2Grasp
```

See Affordance2Grasp `docs/hard_grasp_collect_pipeline.md`.
EOF
```

### 2.7 检查 staging 体积

```bash
du -sh "$STAGING"/*
du -sh "$STAGING"
```

---

## 3. 上传到 HuggingFace

**整包上传（简单）：**

```bash
hf upload "$REPO" "$STAGING/." . \
  --repo-type dataset \
  --commit-message "Initial hard_obj_grasp_collect pipeline assets"
```

> 体积较大、中断需续传时，可改用 `hf upload-large-folder "$REPO" "$STAGING" --repo-type dataset`。

**分目录上传（大文件、可断点重传）：**

```bash
hf upload "$REPO" "$STAGING/sim" sim \
  --repo-type dataset --commit-message "sim assets"

hf upload "$REPO" "$STAGING/output/obj_usd" output/obj_usd \
  --repo-type dataset --commit-message "obj usd"

hf upload "$REPO" "$STAGING/output/grasp_collect_no_rot" output/grasp_collect_no_rot \
  --repo-type dataset --commit-message "grasp collect bootstrap"

hf upload "$REPO" "$STAGING/data_hub" data_hub \
  --repo-type dataset --commit-message "data_hub auto-refill"
```

---

## 4. 上传后验证

```bash
# 试下载到临时目录
TMP=$(mktemp -d)
hf download "$REPO" sim/assets_franka/franka.usd \
  --repo-type dataset --local-dir "$TMP"
ls -la "$TMP/sim/assets_franka/franka.usd"
rm -rf "$TMP"
```

上传成功并确认 HF 网页上文件齐全后，**可删除本机 staging** 释放磁盘（仅上传用的临时副本，删了不影响 HF 与 `$PROJ` 原数据）：

```bash
rm -rf "$STAGING"
```

---

## 5. staging 目录树（参考）

```text
staging/
├── README.md
├── sim/
│   ├── assets_franka/
│   └── assets_scene/
├── output/
│   ├── obj_usd/{oakink,ycb}/
│   └── grasp_collect_no_rot/
│       ├── candidates/pool/
│       ├── merged/
│       ├── state.json
│       └── robot_gt/
└── data_hub/
    ├── meshes/SAM3DMesh/rotated_mesh/
    └── ProcessedData/
        ├── train_fp_rotated/
        └── obj_meshes/{oakink,ycb}/**/scale.json
```
