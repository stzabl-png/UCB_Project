# prepare_affordance_executed — 训练数据（executed + C/B/A + HP）

独立脚本：`tools/prepare_affordance_executed.py`（不修改 `build_dataset.py` / `gen_m5_training_data.py`）。

## 输入

- `output/grasp_collect_no_rot/merged/{obj}_robot_gt_merged.hdf5`（仅 `gripper_tips_trusted` + `executed_panda_hand_at_close`）
- `data_hub/meshes/SAM3DMesh/rotated_mesh/.../mesh.ply` × `scale.json`
- **Human prior（默认）**: `data_hub/ProcessedData/train_fp_rotated/{dataset}/{obj}.hdf5`

## Human prior 坐标系（旋转 + 尺度）

与 `random_grasp_sampler` / `vis_rotated_mesh_hp.py` 一致：

| 步骤 | 内容 |
|------|------|
| **旋转** | `train_fp_rotated` 由 `tools/rotate_training_fp.py` 对 `training_fp` 做 **Rx(+90°)**，与 `rotated_mesh` 同约定 |
| **尺度 OakInk** | 盘上 HP **未** metric → prepare 时 `point_cloud` × `scale.json` |
| **尺度 ycb_dex_*** | 盘上 HP **已是** metric → **不再**乘 scale（mesh 仍按 `scale.json` 乘） |
| **映射** | 对 mesh 表面 4096 采样点，KNN 取最近 HP 点的 `human_prior` 标量 |

对齐检查：`summary.csv` 含 `hp_nn_median_cm`（HP 点到 mesh 表面中位距离）；>2.5cm 会打 QC warning。

可视化核对：

```bash
python3 tools/vis_rotated_mesh_hp.py --obj A01001 --dataset oakink
python3 tools/vis_rotated_mesh_hp.py --obj ycb_dex_10 --dataset dexycb
```

## 接触点方法（C → B → A）

1. **C**：沿指长扫描；相邻表面交点对；选最接近 `finger_width_actual` 的站位  
2. **B**：C 与指宽差 > max(2cm, 35%) → raycast  
3. **A**：解析 fallback  
4. 物体级池化 → KDTree 5mm → `labels`

## 输出

`output/affordance_no_rot_executed/`

- `affordance_train.h5` / `affordance_val.h5`（`human_priors` 默认非零）
- `objects_trainable.txt`, `objects_train_val_split.json`, `dataset_info.json`
- `qc/summary.csv`, `qc/vis/*.png`（`--qc-vis`）

## 用法

```bash
conda activate bundlesdf
cd ~/Project/Affordance2Grasp

python3 tools/prepare_affordance_executed.py
python3 tools/prepare_affordance_executed.py --obj A01001 --qc-vis
python3 tools/prepare_affordance_executed.py --no-hp   # 关闭 HP
bash scripts/run_prepare_affordance_executed.sh
```

## 训练

完整训练说明见 **[`docs/train_affordance.md`](train_affordance.md)**（`model/affordance.train`，soft heatmap + debug 过拟合）。

`human_priors` 在 HDF5 中有字段，但当前 **未** 拼进网络输入；要用于训练需改 Dataset/模型通道数。

```bash
python -m model.affordance.train \
  --dataset_dir output/affordance_no_rot_executed \
  --gpus 0 \
  --batch_size 64
```
