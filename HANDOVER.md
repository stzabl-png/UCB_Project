# Affordance2Grasp — 交接手册（Agent 版）
> 写于 2026-05-05，供接手本项目的 Agent 阅读  
> 本文补充 README.md 未覆盖的细节、规则和踩坑记录

---

## 0. 快速定位

| 你要做的事 | 看哪里 |
|-----------|--------|
| 环境部署 | README §3b / §3d / §3e |
| Pipeline 运行命令 | README §Phase 1A / 1B / Egocentric |
| 故障排查 | README §Troubleshooting (T1–T16+) |
| 数据集下载 | README §4 Data layout + 本文 §2 |
| 没写进 README 的规则 | **本文** |

---

## 1. 项目本质（一句话）

> 从人类手部交互视频（DexYCB / HO3D / OakInk / TACO / EgoDex）中提取  
> **接触先验（human contact prior）**，训练 PointNet++ 预测机器人抓取位置。

不是端到端的 imitation learning，不是 RL。  
核心数据链：**RGB-D → 手部姿态 → 物体姿态 → 接触点 → PointNet++ 分类头**

---

## 2. 数据生态（本机实际状态）

### 本机数据量（2026-05-05）

| 数据集 | 路径 | 状态 | 规模 |
|--------|------|------|------|
| DexYCB | `data_hub/RawData/ThirdPersonRawData/dexycb/` | ✅ 完整 | subject-01, 100 序列, 20 物体 |
| HO3D v3 | `data_hub/RawData/ThirdPersonRawData/ho3d_v3/` | ✅ 完整 | train+eval |
| OakInk v1 | `data_hub/RawData/ThirdPersonRawData/oakink_v1/` | ✅ 完整 | 778 序列 |
| TACO Allocentric | `data_hub/RawData/ThirdPersonRawData/taco/Allocentric_RGB_Videos/` | ⚠️ 部分 | 只解压了 10/151 triplet，视频未全量提取帧 |
| TACO Egocentric | `data_hub/RawData/EgoRawData/taco/Egocentric_RGB_Videos/` | ✅ 完整 | 151 triplet |
| EgoDex | `data_hub/RawData/EgoRawData/egodex/test/` | ✅ 完整 | 101 任务类别 |

### 已处理的输出

| 输出目录 | 内容 | 状态 |
|---------|------|------|
| `ProcessedData/third_depth/` | Depth Pro 深度图 | DexYCB ✅ |
| `ProcessedData/third_mano/` | HaPTIC 手部 MANO | DexYCB ✅ |
| `ProcessedData/obj_poses/` | FoundationPose 物体姿态 | DexYCB ✅ |
| `ProcessedData/training_fp/` | 对齐后训练数据 HDF5 | DexYCB ✅ (20 物体) |
| `ProcessedData/human_prior_fp/` | 接触先验 HDF5 | DexYCB ✅ (20 物体) |
| `ProcessedData/obj_recon_input/` | SAM2 种子 mask | 384 个（egocentric 245 + ycb 等）|
| `ProcessedData/egocentric_depth/` | MegaSAM 深度+相机 | 部分 EgoDex |
| `ProcessedData/ego_mano/` | HaWoR 手部 MANO | 部分 EgoDex |

### HuggingFace 仓库（UCBProject 组织）

| 仓库 | 内容 | 访问 |
|------|------|------|
| `Affordance2Grasp-Data` | FoundationPose 权重 + HaPTIC checkpoints + MegaSAM + DepthPro + HaWoR | Private |
| `Affordance2Grasp-Mesh` | DexYCB 20 物体 mesh + 初始 mask | Private |
| `EgoDataMask` | Egocentric 245 个 triplet 初始 mask | Private |
| `Affordance2Grasp-EgoDex` | EgoDex 处理后数据 | Private |
| `Affordance2Grasp-OakInk` | OakInk 处理后数据 | Private |
| `Affordance2Grasp-TACOData` | TACO Ego RGB 视频 (17.7 GB) | Private |
| `ARCTIC-Archive` | ARCTIC 数据备份 | Private |
| `OakInk-Archive` | OakInk 数据备份 | Private |

> ⚠️ **HF Token**: 使用团队共享 token（向 lyh 获取，不在代码库里存储）  
> 注意：请勿将 token 明文写入任何 git 追踪的文件。  
> 位置: HuggingFace Settings → Access Tokens → 新建 write 权限 token

---

## 3. Conda 环境一览（全部精确版本）

> ⚠️ **最重要的规则：不要照抄任何 submodule 自带的 README 安装指引**  
> 所有上游 README 的 torch 版本都和我们的 CUDA 12.1 栈不兼容。

| 环境名 | Python | PyTorch | CUDA | 用途 |
|--------|--------|---------|------|------|
| `depth-pro` | 3.9 | — | — | Phase 1A Step 1 (Depth Pro) |
| `haptic` | 3.10 | 2.1.1+cu121 | 12.1 | Phase 1A Step 2 (HaPTIC) |
| `bundlesdf` | **3.10** | 2.1.1+cu121 | 12.1 | FP / 对齐 / 训练（主环境）|
| `mega_sam` | 3.10 | **2.2.0+cu121** | 12.1 | MegaSAM 深度 (Ego Step 1) |
| `hawor` | 3.10 | **2.1.0+cu121** | 12.1 | HaWoR 手部 (Ego Step 2) |

### 关键版本钉住（必须精确）

```bash
# haptic env
mmpose==0.24.0  mmcv-full==1.3.9  # HaPTIC 对版本极敏感

# bundlesdf env
python=3.10          # 绝对不能用 3.9！nvdiffrast ABI 不兼容
fast-simplification>=0.1.6  # 没有这个对齐步骤会从15分钟变23小时

# hawor env
torch==2.1.0+cu121   # 不是 1.13！HaWoR 官方 README 的版本是错的
torch-scatter==2.1.2+pt21cu121  # 必须 wheel 匹配，不能源码编译
pytorch3d==0.7.5     # py310+cu121+torch2.1 预编译 wheel
pytorch-lightning==2.2.4  # 必须钉住！2.3+ 有 breaking changes
torchmetrics==1.4.0        # 配套 lightning 2.2.x

# mega_sam env
torch==2.2.0+cu121   # 不是 2.0.1！MegaSAM 官方 README 的版本是错的
xformers==0.0.24
# droid_backends 必须从源码编译（无 pip wheel）
cd mega-sam/base/thirdparty/lietorch && pip install -e . --no-build-isolation
```

---

## 4. Pipeline 架构（完整）

### 第三人称 Pipeline（DexYCB / HO3D / OakInk / TACO Alloc）

```
Phase 1A:
  Step 1: depth-pro env → batch_depth_pro.py → third_depth/
  Step 2: haptic env    → batch_haptic.py    → third_mano/
  Step 3: bundlesdf env → batch_fp.py        → obj_poses/
  Step 4: bundlesdf env → batch_align_mano_fp.py → training_fp/ + human_prior_fp/

Phase 1B (只在引入新物体时需要):
  工具标注 → SAM2 → BundleSDF → SAM3D Mesh → scale 校准
```

### 第一人称 Pipeline（EgoDex / TACO Ego）

```
Phase Egocentric:
  Step E0: depth-pro env → batch_depth_pro.py (SLAM 焦距校准)
  Step E1: mega_sam env  → batch_megasam.py  → egocentric_depth/
  Step E2: hawor env     → batch_hawor.py    → ego_mano/
  Step E5: bundlesdf env → batch_obj_pose_ego.py → obj_poses_ego/
  Step E6: bundlesdf env → batch_align_ego.py    → training_fp/ (ego)
```

### 关键调用规则

```bash
# batch_hawor.py 不直接调用 HaWoR，而是：
# 1. 自动把 data/run_hawor_seq.py 同步到 third_party/hawor/
# 2. conda run -n hawor python third_party/hawor/run_hawor_seq.py --hawor_dir ...

# FoundationPose 必须 export FP_ROOT 才能找到 nvdiffrast/mycpp
export FP_ROOT="/path/to/FoundationPose"

# MegaSAM 必须从 mega-sam/ 子目录运行
cd mega-sam
conda run -n mega_sam python ...
```

---

## 5. 不成文规则与约定

### 5.1 路径约定

```python
# config.py 必须正确设置
DATA_HUB   = "/home/lyh/Project/Affordance2Grasp/data_hub"
HAPTIC_DIR = "/path/to/HaPTIC"    # 独立 repo，不是 submodule
FP_ROOT    = "/path/to/FoundationPose"  # 也通过 env var 设置
```

### 5.2 Mask 的两种含义

| 用途 | 路径 | 粒度 | 工具 |
|------|------|------|------|
| SAM3D/BundleSDF 重建 | `obj_recon_input/{dataset}/{seq_id}/0.png` | **每序列 1 张** | `sam2_annotate_by_object.py` + `batch_prepare_frame3.py` |
| FoundationPose 初始化 | `obj_recon_input/{dataset}/{obj}/0.png` | **每物体 1 张** | 同上（但 obj 而非 seq_id） |

EgoDex/TACO Ego 的 FP mask 已上传到 HF `EgoDataMask`（245 个），  
DexYCB 的 FP mask 在 HF `Affordance2Grasp-Mesh`（20 个物体）。

### 5.3 HaPTIC 的三个 quirk

1. **mmpose 版本必须是 0.24.0**，不能用 1.x（API 完全不同）
2. 必须 `os.chdir(HAPTIC_DIR)` 才能找到相对路径的权重
3. `torch.load` 必须加 `weights_only=False`（Python 3.10 + PyTorch 2.1）

### 5.4 FoundationPose 的 quirk

1. 依赖 `nvdiffrast` 和 `mycpp`，必须源码编译
2. 编译时用系统 gcc（不用 conda 的 gcc），否则 glibc 冲突
3. `libstdc++` ABI 问题：在 bundlesdf env 运行时需要：
   ```bash
   export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
   ```
4. `cmake` 必须 < 4.0（cmake ≥ 4.0 移除了 FindBoost）

### 5.5 Depth Pro 的 quirk

1. checkpoint 路径是相对 CWD 的，必须先 `cd third_party/ml-depth-pro/`
2. DexYCB 无 GT 内参，用**两 pass 自校准**：pass1 估算 fx，pass2 固定 fx 重跑
3. 已确定的 DexYCB cam `841412060263` 焦距：`fx ≈ 591.4`

### 5.6 TACO Allocentric 的特殊处理

TACO Allocentric 是 MP4 视频（4096×3000，30fps），**不是预解压的帧图像**。  
必须先用 `tools/extract_taco_frames.py` 解帧，再能运行后续 pipeline。

```bash
# 步骤必须按顺序
# 1. 解帧（89 GB，约 2 小时）
python tools/extract_taco_frames.py --mode pipeline --cam 22139905

# 2. 手动标注 151 triplet 的 seed mask（约 2-3 小时）
python tools/sam2_annotate_by_object.py --dataset taco_allocentric

# 3. 自动生成所有序列的 mask
python data/batch_prepare_frame3.py --dataset taco_allocentric
```

---

## 6. 当前 TODO（按优先级）

### 🔴 立刻要做

| 任务 | 操作 |
|------|------|
| TACO Allocentric 全量解帧 | `extract_taco_frames.py --mode pipeline` |
| TACO Allocentric 手动标注 151 triplet | `sam2_annotate_by_object.py --dataset taco_allocentric` |
| 实验室机器 mega_sam 环境修复 | 重建 env，torch 2.2.0+cu121，编译 lietorch |

### 🟡 近期

| 任务 | 操作 |
|------|------|
| EgoDex 全量 MegaSAM | `batch_megasam.py --dataset egodex`（~50h）|
| EgoDex 全量 HaWoR | `batch_hawor.py --dataset egodex`（~40h）|
| 下载更多 TACO Allocentric cam | 其他 11 个 zip（~400 GB 总计）|

### 🟢 长期

| 任务 | 说明 |
|------|------|
| Phase 4 重训练（加入 TACO 数据）| 等 TACO pipeline 跑完 |
| 新 Baseline 实验 | GraspNet-AnyGrasp / DexGraspVLA / DP |

---

## 7. 实验室机器部署状态（RTX 3090）

**机器配置：**
- Ubuntu 22.04, CUDA 12.1, RTX 3090
- 数据位置：`/media/msc/5TB1/`（5TB 机械硬盘）
- FoundationPose 编译位置：`/media/msc/5TB1/FoundationPose/`

**已完成（DexYCB subject-01, cam 841412060263）：**
```
Phase 1A Step 1 (Depth Pro)    ✅  100/100 序列，fx=591.4
Phase 1A Step 2 (HaPTIC)       ✅  100/100 序列
Phase 1A Step 3 (FP)           ✅  100/100 序列，~6s/seq
Phase 1A Step 4 (Align)        ✅  20 物体，diverged=0
```

**阻塞中（实验室）：**
```
hawor env: pytorch-lightning 版本错误 → 已提供修复命令（README T7.3）
mega_sam env: droid_backends 未编译 → 已提供修复命令（README Section 3d）
```

---

## 8. 训练数据质量说明

### diverged=0 的含义

`batch_align_mano_fp.py` 输出的 `diverged=0` 表示：
- FoundationPose 追踪未漂移（depth sanity check 通过）
- MANO 手部和物体姿态的 scale ratio 在合理范围
- 接触点正常落在 mesh 表面附近

如果 `diverged > 0`，该帧被排除在训练数据之外。DexYCB 20 物体全部 `diverged=0`，数据质量干净。

### contact coverage 指标

```
contact_verts=4978/4980  → 接触顶点覆盖率（越接近满分越好）
cov(>0.01)=76%           → 有弱接触的 mesh 表面比例
cov(>0.5)=16%            → 有强接触的 mesh 表面比例（通常手部直接接触区域）
hp_max=0.840             → 单帧最大接触概率（>0.5 说明有清晰接触）
```

---

## 9. 代码约定（未写进 README）

### 9.1 发现函数的签名

所有 `discover_*` 函数必须遵循统一的签名和 yield 格式：

```python
def discover_xxx(input_dir: str) -> Generator:
    # 对于第三人称 pipeline (batch_prepare_frame3, sam2_annotate):
    yield seq_id: str, img_paths: List[str]
    
    # 对于 sam2_annotate_by_object:
    yield ds_out: str, obj_name: str, frames: List[str]
```

### 9.2 所有 `torch.load` 必须加 `weights_only=False`

```python
# 正确
torch.load(path, weights_only=False)["state_dict"]

# 错误（在 PyTorch 2.6+ 会 crash）
torch.load(path)["state_dict"]
```

已全局修复（commit c9456de）。未来新增代码必须遵守。

### 9.3 submodule 里的自写脚本

如果你在 submodule（如 `third_party/hawor/`）里写了自己的脚本，  
**必须同时在主项目 `data/` 目录放一份**，并在对应 batch 脚本里实现 auto-sync。  
原因：submodule 里的文件不被主项目 git 追踪，clone 后会丢失。

参考：`data/run_hawor_seq.py` + `batch_hawor.py` 里的 `_sync_runner()`。

### 9.4 新数据集的接入 checklist

新数据集要接入 pipeline，必须修改以下文件：
- [ ] `data/batch_depth_pro.py` 或 `batch_megasam.py` (深度)
- [ ] `data/batch_haptic_*.py` 或 `batch_hawor.py` (手部)
- [ ] `data/batch_fp.py` 或 `batch_obj_pose_ego.py` (物体姿态)
- [ ] `data/batch_align_*.py` (接触对齐)
- [ ] `tools/sam2_annotate_by_object.py` (seed mask 标注)
- [ ] `data/batch_prepare_frame3.py` (auto mask 生成)
- [ ] `config.py` (路径配置)

---

## 10. PointNet++ 训练策略摘要

| 参数 | 值 | 原因 |
|------|---|------|
| Architecture | PointNet++ v5 Multi-Task | 分割头 + 力中心回归头 |
| 输入通道 | xyz(3) + normal(3) + human_prior(1) = 7 | human_prior 是输入特征，不是标签 |
| 监督信号 | robot_gt (仿真验证的接触) | 不是直接用 human_prior 监督 |
| 分割损失 | Focal(α=0.75,γ=2.0) + Tversky(α=0.3,β=0.7) | 标签不平衡（约16%接触点）|
| 回归损失权重 λ | 10.0 | 补偿 MSE 数值量级小 |
| 最优阈值 | 0.65-0.75 | 接触不平衡时峰值 F1 不在 0.5 |
| 验证集切分 | 物体级 20%（seed=42）| 防止数据泄漏 |
| gtfree 模式 | ✅ 可以不用 Isaac Sim | 当前 v2 checkpoint F1=0.642 |

---

## 11. 最容易踩的坑（Top 5）

1. **`bundlesdf python=3.9` → 改成 `python=3.10`**  
   nvdiffrast 和 pytorch3d 的 wheel 都是 py310 编译的

2. **hawor/README 写的是 torch 1.13 → 实际用 torch 2.1.0**  
   照着 hawor 官方文档装必然失败

3. **mega-sam/README 写的是 torch 2.0.1 → 实际用 torch 2.2.0**  
   `droid_backends` 从来不是 pip 安装的，必须编译

4. **TACO Allocentric 是 MP4 视频 → 不是图片目录**  
   所有 discover 函数需要 jpg 帧，必须先 `extract_taco_frames.py`

5. **FP_ROOT 环境变量没设 → FoundationPose 所有 step 都会找不到 nvdiffrast**  
   `export FP_ROOT="/path/to/FoundationPose"` 写进 `~/.bashrc`

---

## 12. 文件查找速查

| 我在找... | 位置 |
|----------|------|
| 所有深度估计脚本 | `data/batch_depth_pro.py`, `data/batch_megasam.py` |
| 手部姿态脚本 | `data/batch_haptic_*.py`, `data/batch_hawor.py` |
| 物体姿态脚本 | `data/batch_fp.py`, `data/batch_obj_pose_ego.py` |
| 接触对齐脚本 | `data/batch_align_mano_fp.py`, `data/batch_align_ego.py` |
| SAM2 标注工具 | `tools/sam2_annotate_by_object.py` |
| TACO 帧提取 | `tools/extract_taco_frames.py` |
| 训练脚本 | `model/train.py` |
| 权重下载 | `setup_weights.py` |
| HaWoR 核心 worker | `data/run_hawor_seq.py` (→ auto-sync to hawor submodule) |
| 已知问题全记录 | `README.md` §Troubleshooting |
