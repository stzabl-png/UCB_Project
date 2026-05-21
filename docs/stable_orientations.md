# stable_orientations.json

> **Work in progress** — documents the placement / stable-orientation side path only.
> JSON schema, mesh baseline, and batch integration may change.

由 `data/estimate_stable_orientations.py` 生成。mesh 基准与 grasp pipeline 一致：**raw `mesh.ply` + `scale.json`**（须含有效 `scale_factor`），不读已有 `rotation.json`。

批量 `--dataset` 时：**缺少上述文件的物体会自动跳过**（例如 oakink 下 `A01001_0001_*` 子目录）。

## 用法

```bash
conda activate bundlesdf
# 默认跳过已有 JSON；并行 8 进程；不加 --force
python3 data/estimate_stable_orientations.py --dataset oakink --workers 8
python3 data/estimate_stable_orientations.py --dataset ycb --workers 8
python3 data/estimate_stable_orientations.py --obj A01026 --dataset oakink
# 覆盖重算才加 --force
python3 data/estimate_stable_orientations.py --dataset oakink --workers 8 --force
```

## 可视化

```bash
python3 tools/vis_stable_orientations.py --obj A01026 --dataset oakink
python3 tools/vis_stable_orientations.py --obj A01026 --dataset oakink --open   # 生成后打开 overview
```

输出目录：`output/stable_pose_vis/oakink/A01026/`

- `pose_00_identity.png` … 每个朝向一张
- `overview.png` 多宫格总览（建议先看这张）

## 字段

- `orientations[]`：每条含 `id`, `euler_xyz_deg`, `matrix`（`v' = R @ v`）, `method`, `probability`（identity 为 null）
- `id=0`：默认 **identity** 基线（`no_rotation` 管线对照）
- `primary_id`：概率最高的 **stable** 条（非 identity），便于对齐旧 `rotation.json`
- `dedup_stats.skip_z`：因「与已保留姿态仅差绕竖直 Z」而丢弃的条数

## 绕 Z 重复

对称物体上 trimesh 可能返回多个仅差 `R_z` 的稳定解。脚本会**只保留该族中概率最高的一条**；转台多样性可在后续 batch 对选定 `R` 再乘 `R_z(φ)`。
