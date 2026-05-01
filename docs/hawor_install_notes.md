# HaWoR 安装问题总结

> 适用版本: HaWoR (ThunderVVV/HaWoR, CVPR 2025) + PyTorch 2.1.0+cu121 + Python 3.10

## 已知问题 & 解决方案

### 1. `pytorch3d` 从 git 源码安装失败

**错误**
```
ModuleNotFoundError: No module named 'torch'  (in build subprocess)
```

**原因**  
pip 默认启用 build isolation，在隔离子进程中无法看到主环境的 torch。

**解决**  
用 fbaipublicfiles 预编译 wheel 替代（与 git 版本功能等价）：
```bash
pip install pytorch3d \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt210/download.html
grep -v "pytorch3d" requirements.txt > /tmp/req.txt
pip install -r /tmp/req.txt
```

---

### 2. `chumpy` 从 git 源码安装失败

**错误**
```
ModuleNotFoundError: No module named 'pip'  (in build subprocess)
```

**解决**
```bash
pip install chumpy --no-build-isolation
```

然后 **必须** patch chumpy 的 `__init__.py`（numpy 1.24+ 删除了 `np.bool` 等别名）：
```bash
CHUMPY=$(python -c "import importlib.util; print(importlib.util.find_spec('chumpy').origin)")
sed -i 's/from numpy import bool, int, float, complex, object, unicode, str, nan, inf/import numpy as _np; bool=_np.bool_; int=_np.int_; float=_np.float64; complex=_np.complex128; object=_np.object_; unicode=str; nan=_np.nan; inf=_np.inf/' "$CHUMPY"
```

---

### 3. numpy 版本冲突导致 DROID-SLAM 编译/运行崩溃

**错误**
```
A module compiled with NumPy 1.x cannot be run in NumPy 2.x
```

**解决**  
在编译 DROID-SLAM **之前** 固定 numpy 版本，并在安装后保持锁定：
```bash
pip install "numpy==1.26.4" --force-reinstall --no-deps
```

> ⚠️ `pip install -r requirements.txt` 会将 numpy 升级到 2.x，必须在此之后重新固定。

---

### 4. `setuptools` 过新导致 `pkg_resources` 丢失

**错误**
```
ModuleNotFoundError: No module named 'pkg_resources'
```

**原因**  
setuptools ≥ 70 移除了 `pkg_resources`，但 torch 的 `cpp_extension` 依赖它。

**解决**
```bash
pip install "setuptools==69.5.1" --force-reinstall
```

---

### 5. `torch_scatter` CPU-only 版本导致 Segmentation Fault

**错误**
```
Segmentation fault (core dumped)
```
（发生在 `from torch_scatter import scatter_mean` 时）

**原因**  
`pip install -r requirements.txt` 安装了 PyPI 上的通用 `torch-scatter 2.1.2`（无 CUDA 后缀），该版本在有 GPU 的环境中 segfault。

**解决**  
强制安装 PyG 官方 CUDA 专用版本：
```bash
pip install torch-scatter \
  -f https://data.pyg.org/whl/torch-2.1.0+cu121.html \
  --force-reinstall
```

> ⚠️ 如果先安装了 CUDA 版，之后的 `pip install -r requirements.txt` 会覆盖它，必须在所有 requirements 安装完成后再次执行此命令。

---

## 完整安装顺序（最终可用方案）

```bash
conda create -n hawor python=3.10 -y
conda activate hawor
cd /path/to/third_party/hawor

# 1. PyTorch (根据实际 CUDA 版本调整)
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 \
  --extra-index-url https://download.pytorch.org/whl/cu121

# 2. pytorch3d (预编译 wheel)
pip install pytorch3d \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt210/download.html

# 3. 其余 requirements (排除 pytorch3d 和 chumpy)
grep -v "pytorch3d\|chumpy" requirements.txt > /tmp/req.txt
pip install -r /tmp/req.txt

# 4. chumpy (单独，--no-build-isolation)
pip install chumpy --no-build-isolation

# 5. 官方后续步骤
pip install pytorch-lightning==2.2.4 --no-deps
pip install lightning-utilities torchmetrics==1.4.0

# 6. 固定关键版本 (覆盖 requirements 可能带来的升级)
pip install "numpy==1.26.4" --force-reinstall --no-deps
pip install "setuptools==69.5.1" --force-reinstall
pip install torch-scatter \
  -f https://data.pyg.org/whl/torch-2.1.0+cu121.html \
  --force-reinstall

# 7. 编译 DROID-SLAM
cd thirdparty/DROID-SLAM
python setup.py install
cd ../..

# 8. patch chumpy
CHUMPY=$(python -c "import importlib.util; print(importlib.util.find_spec('chumpy').origin)")
sed -i 's/from numpy import bool, int, float, complex, object, unicode, str, nan, inf/import numpy as _np; bool=_np.bool_; int=_np.int_; float=_np.float64; complex=_np.complex128; object=_np.object_; unicode=str; nan=_np.nan; inf=_np.inf/' "$CHUMPY"

# 9. 验证
python demo.py --video_path ./example/video_0.mp4 --vis_mode cam
```

---

## 模型权重

| 文件 | 路径 | 来源 |
|------|------|------|
| `detector.pt` | `weights/external/` | HuggingFace WiLoR |
| `droid.pth` | `weights/external/` | Google Drive (DROID-SLAM官方) |
| `hawor.ckpt` | `weights/hawor/checkpoints/` | HuggingFace ThunderVVV/HaWoR |
| `infiller.pt` | `weights/hawor/checkpoints/` | HuggingFace ThunderVVV/HaWoR |
| `model_config.yaml` | `weights/hawor/` | HuggingFace ThunderVVV/HaWoR |
| `MANO_RIGHT.pkl` | `_DATA/data/mano/` | MANO官网 (需注册) |
| `MANO_LEFT.pkl` | `_DATA/data_left/mano_left/` | MANO官网 (需注册) |
| `metric_depth_vit_large_800k.pth` | `thirdparty/Metric3D/weights/` | Google Drive (Metric3D官方) |

---

## HaWoR 输出格式

运行 `run_hawor_seq.py` 后输出 `.npz`，包含：

| key | shape | 含义 |
|-----|-------|------|
| `right_verts` | (T, 778, 3) | 右手 MANO 顶点（世界坐标） |
| `left_verts` | (T, 778, 3) | 左手 MANO 顶点（世界坐标） |
| `R_w2c` | (T, 3, 3) | 相机旋转矩阵（世界→相机） |
| `t_w2c` | (T, 3) | 相机平移向量（世界→相机） |
| `R_c2w` | (T, 3, 3) | 相机旋转矩阵（相机→世界） |
| `t_c2w` | (T, 3) | 相机平移向量（相机→世界） |
| `img_focal` | scalar | 焦距（像素） |
| `pred_trans` | (2, T, 3) | MANO 根节点平移 [0=左,1=右] |
| `pred_rot` | (2, T, 3) | MANO 根节点旋转（轴角） |
| `pred_hand_pose` | (2, T, 45) | MANO 手部姿态（15关节×3轴角） |
| `pred_betas` | (2, T, 10) | MANO 形状系数 |
| `pred_valid` | (2, T) | 有效帧掩码 |
