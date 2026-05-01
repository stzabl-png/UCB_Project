# SAM3D 云端重建使用手册

## 服务器信息

| 项目 | 值 |
|------|-----|
| SSH 连接 | `ssh sam3d-gpu` |
| 实例 | 阿里云 PAI DSW |
| GPU | 2× NVIDIA A800 80GB（GPU 0 通常空闲）|
| OSS 挂载 | `/mnt/data/lyh/` |
| SAM3D 代码 | `/root/lyh/sam-3d-objects/` |
| Conda 环境 | `sam3d-objects` |

---

## 目录约定

```
~/input/{dataset}/{obj_name}/
    image.png    ← 选取的最佳 RGB 帧（物体清晰可见）
    0.png        ← SAM2 二值 mask（白色=物体，黑色=背景）

~/output/{dataset}/{obj_name}/
    mesh.ply     ← SAM3D 重建输出 mesh
    splat.ply    ← 3D Gaussian splat（可选，可忽略）
```

---

## 手动标注 Mask（本地，每个新物体做一次）

### 工具说明

| 脚本 | 用途 |
|------|------|
| `tools/annotate_egodex_batch.py` | EgoDex 批量标注，叠加 MegaSAM 深度图辅助 |
| `tools/annotate_egocentric_obj.py` | 通用单序列标注 |

### 使用方法

```bash
conda activate base
cd /home/lyh/Project/Affordance2Grasp

# 标注单个 task（推荐）
python tools/annotate_egodex_batch.py --task slot_batteries
python tools/annotate_egodex_batch.py --task fry_bread

# 从第 N 个 task 开始（断点续标）
python tools/annotate_egodex_batch.py --start 10

# 全量标注所有 task
python tools/annotate_egodex_batch.py
```

### 标注操作键

| 键 | 功能 |
|----|------|
| `← →` | 上/下一帧 |
| `PgUp / PgDn` | 跳 10 帧 |
| `Home / End` | 跳到首/末帧 |
| **左键** | 加前景点（绿色）→ SAM2 实时生成 mask |
| `B` | 切换前景/背景模式（红色点 = 背景） |
| `D` | 切换 MegaSAM 深度图叠加显示 |
| `C` | 清除所有点，重新标注 |
| `Enter` | **保存当前帧 + mask**，进入下一个 task |
| `S` | 跳过该 task（可之后补注） |
| `Q` | 退出，保留已标注进度 |

### 输出路径

```
data_hub/ProcessedData/obj_recon_input/egocentric/{task}/
    image.png    ← 选定帧的 RGB
    0.png        ← SAM2 生成的二值 mask
```

### 查看哪些 task 已标注

```bash
# 检查目标 task 的标注状态
for task in slot_batteries stack_unstack_cups fry_bread build_unstack_lego \
            flip_pages basic_pick_place basic_fold throw_collect_objects \
            assemble_disassemble_furniture_bench_desk; do
    if [ -f "data_hub/ProcessedData/obj_recon_input/egocentric/$task/0.png" ]; then
        echo "✅ $task"
    else
        echo "❌ $task — 需要标注"
    fi
done
```

---

## 快速流程

### Step 1 — 本地：准备输入（运行 prep_sam3d_input.py）

```bash
# 在本地机器
cd /home/lyh/Project/Affordance2Grasp
python tools/prep_sam3d_input.py --dataset egodex

# 打包后 rsync 到云端
rsync -avz /tmp/sam3d_input/ sam3d-gpu:~/input/
```

### Step 2 — 云端：标注 mask（每个新物体做一次）

```bash
# 在云端，用 SAM2 交互式标注
# 或手动用 GIMP/PS 在 image.png 上画 mask 保存为 0.png

# 验证 mask 格式（白色物体，黑色背景）
conda run -n sam3d-objects python -c "
from PIL import Image; import numpy as np
m = np.array(Image.open('~/input/egodex/battery/0.png'))
print('mask shape:', m.shape, 'unique vals:', np.unique(m))
"
```

### Step 3 — 云端：运行 SAM3D 推理

> ⚠️ 非交互式 SSH 不加载 `.bashrc`，`conda` 命令不可用，需要：
> 1. 用环境的完整 Python 路径替代 `conda activate`
> 2. 手动设置 `CONDA_PREFIX`（SAM3D 的 inference.py 依赖此变量设置 CUDA_HOME）

```bash
# 方式 A：通过 SSH 远程执行（推荐）
ssh sam3d-gpu "
cd ~/lyh/sam-3d-objects
CUDA_VISIBLE_DEVICES=0 \
CONDA_PREFIX=~/miniconda3/envs/sam3d-objects \
~/miniconda3/envs/sam3d-objects/bin/python batch_infer.py \
    --input-dir ~/input \
    --output-dir ~/output \
    --dataset egodex 2>&1 | tee ~/output_log.txt
"

# 方式 B：SSH 进入后手动运行（可交互查看进度）
ssh sam3d-gpu
cd ~/lyh/sam-3d-objects
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sam3d-objects
export CUDA_VISIBLE_DEVICES=0
python batch_infer.py --input-dir ~/input --output-dir ~/output --dataset egodex

# 只跑单个 obj（调试用）
python batch_infer.py --input-dir ~/input --output-dir ~/output \
    --dataset egodex --seq battery
```

### Step 4 — 云端→本地：回传 mesh

```bash
# 在本地机器
rsync -avz sam3d-gpu:~/output/egodex/ \
    /home/lyh/Project/Affordance2Grasp/data_hub/ProcessedData/obj_meshes/egocentric/

# 验证
ls /home/lyh/Project/Affordance2Grasp/data_hub/ProcessedData/obj_meshes/egocentric/
```

### Step 5 — 本地：Scale 估计（meter 尺度校正）

```bash
cd /home/lyh/Project/Affordance2Grasp
conda activate hawor
python data/estimate_obj_scale_ego.py --obj battery
# 输出: obj_meshes/egocentric/battery/scale.json
```

---

## SAM3D Python API（单次调用）

```python
import sys
sys.path.append("/root/lyh/sam-3d-objects/notebook")
from inference import Inference, load_image, load_single_mask

# 加载模型（只做一次）
inference = Inference("checkpoints/hf/pipeline.yaml", compile=False)

# 推理
image = load_image("path/to/image.png")
mask  = load_single_mask("path/to/mask_dir", index=0)   # 0 → 0.png
output = inference(image, mask, seed=42)

# 保存 mesh
output["mesh"].export("mesh.ply")
```

---

## egodex 物体注册表（9条序列）

| obj_name | egodex 序列 | 说明 |
|----------|------------|------|
| `bench_desk` | `assemble_disassemble_furniture_bench_desk/15` | 桌腿等组件 |
| `cloth` | `basic_fold/38` | 布料折叠 ⚠️ 可变形 |
| `pick_object` | `basic_pick_place/259` | 泛化拿放物体 |
| `lego` | `build_unstack_lego/9` | 乐高积木 |
| `book` | `flip_pages/14` | 书本翻页 |
| `bread` | `fry_bread/0` | 面包 |
| `battery` | `slot_batteries/1` | 电池 |
| `cup` | `stack_unstack_cups/11` | 杯子 |
| `ball` | `throw_collect_objects/5` | 球类 |

---

## 数据路径汇总（本地）

| 类型 | 路径 |
|------|------|
| RGB 原始帧 | `data_hub/RawData/EgoRawData/egodex/test/{task}/{ep}/extracted_images/` |
| MegaSAM 深度+内参 | `data_hub/ProcessedData/egocentric_depth/egodex/{task}/{ep}/` |
| HaWoR MANO 输出 | `data_hub/ProcessedData/ego_mano/egodex/{task}/{ep}.npz` |
| SAM2 mask 输入 | `data_hub/ProcessedData/obj_recon_input/egocentric/{obj}/0.png` |
| SAM3D mesh 输出 | `data_hub/ProcessedData/obj_meshes/egocentric/{obj}/mesh.ply` |
| FP 物体位姿 | `data_hub/ProcessedData/obj_poses_ego/egodex/{task}/{ep}/ob_in_cam/` |
| 接触 HDF5 | `data_hub/ProcessedData/training_fp_ego/egodex/{obj}.hdf5` |

---

## 常见问题

**Q: GPU 1 被占用怎么办？**
```bash
nvidia-smi   # 查看占用情况
export CUDA_VISIBLE_DEVICES=0   # 强制用 GPU 0
```

**Q: mask 格式要求？**
- 单通道 PNG 或 RGB PNG（白色=物体区域 255，黑色=背景 0）
- 与 image.png 同尺寸
- 文件名必须是 `0.png`（batch_infer.py 的约定）

**Q: 输出 mesh 单位是什么？**
- SAM3D 输出是归一化坐标，不是 metric metres
- 必须用 `estimate_obj_scale_ego.py` 通过 MegaSAM 深度图估计 scale_factor
- scale.json 写入后 FP 和对齐脚本会自动读取
