# MegaSAM 在 RTX 5090 上的安装与运行

这份文档记录了我们把 `mega-sam` 从 4080 (Ada, sm_89) 迁到 **5090 (Blackwell, sm_120)** 时, 实际跑通的 env stack。partner 在 5090 上遇到的 "之前 4080 没事现在炸了" 几乎可以确定是 **PyTorch / CUDA 不支持 sm_120 架构**。

---

## 1. 问题根因

5090 是 NVIDIA **Blackwell** 架构, compute capability = **12.0** (sm_120)。MegaSAM 仓库 `environment.yml` 默认装的是:

```yaml
- conda-forge::cudatoolkit=11.8
- pytorch::pytorch=2.0.1
- pytorch::torchvision==0.15.2
```

PyTorch 2.0.1 + cu118 build **不带 sm_120 kernel**。在 5090 上跑会出现下面任一现象 (按出现概率排):

| 现象 | 含义 |
|---|---|
| `CUDA error: no kernel image is available for execution on the device` | 90% 的情况, PyTorch 找不到 sm_120 的 kernel |
| `RuntimeError: CUDA error: invalid device function` (DROID-SLAM `droid_backends` 调 CUDA kernel) | 自编译的 ext 不带 sm_120 |
| 输出 NaN / 全 0 的 depth, 但不报错 | 最坑, kernel fallback 但行为错误 |
| `xformers` import 失败 / SwiGLU warning | xformers 必须配 PyTorch 版本 |

要让 5090 工作, 必须满足:
1. **PyTorch** 内置 sm_120 (= PT 2.7+ with cu128, 或者 PT nightly)
2. **CUDA Toolkit** ≥ 12.8 (sm_120 在 12.8 才正式 release)
3. **DROID-SLAM 的 `droid_backends` + `lietorch_backends` 必须用上面的 PyTorch 重新编译**
4. **`torch_scatter` / `xformers`** 必须用匹配 PyTorch 版本的 wheel

---

## 2. 我们这边实测能跑的 env (锁版本)

```text
GPU:        NVIDIA GeForce RTX 5090 (32 GB)
Driver:     580.142
Python:     3.10.20
PyTorch:    2.11.0+cu128
torchvision:0.26.0+cu128
torch_scatter: 2.1.2+pt211cu128
xformers:   0.0.35
CUDA toolkit (build-time): 12.8
cuDNN:      9.19
```

验证当前 PyTorch 是否带 sm_120:

```bash
python -c "import torch; print(torch.cuda.get_arch_list())"
# 期望输出包含 'sm_120':
# ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
```

如果输出里**没有 sm_120**, 这个 PyTorch build 就 100% 不能用, 别折腾别的, 先换 PyTorch。

---

## 3. 安装步骤 (从零)

### 3.1 创建 conda env (用 Python 3.10, 不要新版)

```bash
conda create -n mega_sam python=3.10 -y
conda activate mega_sam
```

### 3.2 装 PyTorch + cu128

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.11.0 torchvision==0.26.0
```

(如果 2.11 wheel 已经被替换, 也可以直接 `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128` 装当前最新 cu128 build; 只要 `get_arch_list()` 含 sm_120 都行)

### 3.3 装 CUDA Toolkit 12.8 (host 端用来编译 DROID-SLAM)

两种办法:

**方案 A — 用系统 nvcc 12.8** (推荐, 编译快):
- 装 CUDA Toolkit 12.8 deb (https://developer.nvidia.com/cuda-12-8-0-download-archive)
- 装完后 `nvcc --version` 应显示 `Cuda compilation tools, release 12.8`

**方案 B — 不装系统 nvcc, 用 PyTorch 自带 cudart**:
- 让 `CUDA_HOME` 指向 conda 里 PyTorch 的 lib (不一定每个 ext 都吃, DROID-SLAM 不行)
- **不推荐**, DROID-SLAM 必须有 nvcc 才能编

### 3.4 装其余 Python 依赖

```bash
pip install opencv-python-headless==4.13.0.92 \
            tqdm einops==0.8.2 scipy matplotlib wandb \
            timm==1.0.26 ninja==1.13.0 \
            huggingface_hub kornia==0.8.2 imageio
```

### 3.5 装 torch_scatter (必须匹配 PT 版本)

```bash
pip install torch_scatter==2.1.2 \
    -f https://data.pyg.org/whl/torch-2.11.0+cu128.html
# 装完应该是 'torch_scatter-2.1.2+pt211cu128'
```

如果上面这条找不到 wheel (PyG 还没发对应版本), 就 fallback 到源码编译:

```bash
TORCH_CUDA_ARCH_LIST="9.0;10.0;12.0+PTX" \
    pip install torch_scatter==2.1.2 --no-build-isolation
```

### 3.6 装 xformers (必须匹配 PT 版本)

```bash
pip install xformers==0.0.35 --index-url https://download.pytorch.org/whl/cu128
```

### 3.7 从源码编译 DROID-SLAM 的 CUDA ext

**这一步是最容易出错的, 也是 partner 大概率卡住的地方。**

```bash
cd <PROJECT_ROOT>/mega-sam/base

# 1. 清掉旧 build (上次在 4080 上编的产物不能用!)
rm -rf build/ *.so

# 2. 重新编译, 务必加 TORCH_CUDA_ARCH_LIST
TORCH_CUDA_ARCH_LIST="9.0;10.0;12.0+PTX" \
    python setup.py build_ext --inplace 2>&1 | tee build_5090.log

# 3. 同步编 lietorch (mega-sam 自带的 fork)
cd thirdparty/lietorch
rm -rf build/ *.so
TORCH_CUDA_ARCH_LIST="9.0;10.0;12.0+PTX" \
    python setup.py install 2>&1 | tee build_5090.log
```

`TORCH_CUDA_ARCH_LIST` 的解释:
- `9.0` = H100 (sm_90, 可选, 留着兼容)
- `10.0` = H200 / B100 (sm_100)
- `12.0+PTX` = **5090 必须有的那一个**; `+PTX` 让二进制能向前兼容

不加 `TORCH_CUDA_ARCH_LIST`, nvcc 默认只编当前 PyTorch 已知的 arch, 但**当 partner 在 4080 上初次编译时, build 产物只含 sm_89**, 拷到 5090 自然炸。

### 3.8 模型 checkpoints

需要下载两个权重 (~3 GB):
```bash
cd <PROJECT_ROOT>/mega-sam
mkdir -p checkpoints Depth-Anything/checkpoints

# 1) MegaSAM final
wget -O checkpoints/megasam_final.pth \
    https://huggingface.co/Mega-SAM/Mega-SAM/resolve/main/megasam_final.pth

# 2) Depth-Anything ViT-L
wget -O Depth-Anything/checkpoints/depth_anything_vitl14.pth \
    https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth
```

UniDepthV2 不需要手动下, 第一次跑会自动从 HF 拉到 `~/.cache/huggingface/`。

---

## 4. 验证安装 (跑之前必看)

```bash
conda activate mega_sam
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.version.cuda)
print('Arch list:', torch.cuda.get_arch_list())
print('Device:', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))

# 必须不报错: DROID-SLAM 自编译 ext
import sys; sys.path.insert(0, '<PROJECT_ROOT>/mega-sam/base')
import droid_backends; print('droid_backends OK')

import lietorch; print('lietorch OK')

import torch_scatter; print('torch_scatter OK:', torch_scatter.__version__)

import xformers; print('xformers OK:', xformers.__version__)

# 小 CUDA kernel smoke test (验证 5090 sm_120 是真的能跑 kernel)
a = torch.randn(1024, 1024, device='cuda')
b = a @ a.T
torch.cuda.synchronize()
print('CUDA matmul on', a.device, 'OK')
"
```

所有行都应该输出 OK。任何一行报错 = 还没装好, 别去跑 batch_megasam。

---

## 5. 跑 EgoDex (我们这边的实际命令)

```bash
cd <PROJECT_ROOT>/mega-sam
conda run -n mega_sam python ../data/batch_megasam.py \
    --dataset egodex --resume \
    2>&1 | tee egodex_megasam.log
```

性能参考 (5090):
- ~23 秒 / episode (60 帧, long-side=640)
- 3050 episodes 全跑完 ~9 小时单卡

GPU 显存峰值 ~10 GB (5090 32GB 够多张并行, 但 batch_megasam.py 当前是单 process)。

---

## 6. 常见坑速查表

| 错误 | 原因 | 解决 |
|---|---|---|
| `no kernel image is available for execution on the device` | PyTorch 没 sm_120 | 装 PT 2.11+ cu128 (§3.2) |
| `invalid device function` (DROID 里) | `droid_backends.so` 在 4080 上编的, 没 sm_120 kernel | 删 `build/`, 用 `TORCH_CUDA_ARCH_LIST="12.0+PTX"` 重编 (§3.7) |
| `libc10.so: cannot open shared object file` | 编 ext 时用的 PyTorch ≠ runtime PyTorch | 把 ext 在**目标 PyTorch 的 env 里**重编 |
| `xformers` import 报 ABI 不匹配 | xformers 版本对不上 PyTorch | 重装匹配版本 (§3.6) |
| Depth/pose 输出全 NaN 但不 crash | 最阴, kernel silently fallback | 检查 `get_arch_list()` 是否含 sm_120; 必要时强制 `torch.cuda.set_device(0)` 后重跑 sanity test |
| `UnicodeDecodeError` 在第一个 episode | meta.yml 编码问题 (一般不在 5090 上出现) | 跟 5090 无关, 检查 dataset 完整性 |
| OOM at SLAM stage | DROID-SLAM peak ~10 GB, 5090 应该够 | 把 `MAX_FRAMES` 从 60 降到 40 |

---

## 7. 为什么 4080 上没事 5090 上炸 — 一句话总结

**4080 是 sm_89 (Ada), 在 PyTorch 2.0.1 的 default arch 列表里**; partner 在 4080 上编 `droid_backends.so` 时 nvcc 默认就编了 sm_89, 自然跑得动。**5090 是 sm_120 (Blackwell), 完全在 PyTorch 2.0.1 之外**, 整个 stack (PyTorch + 自编 ext) 都需要换。

这是个**架构跳代** (从 Ada→Hopper→Blackwell 是两代), 不是普通 driver 更新就能解决, 必须重装。
