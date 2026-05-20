#!/bin/bash
# run_pipeline_gpu1_dexycb.sh
# GPU 1 — DexYCB 完整 Pipeline: 原始视频 → HumanPrior
#
# 注: DexYCB 是 RGB-D 数据集，有传感器深度
#     DepthPro 仅用于估计内参 K (可选)
#     FP 位姿已有 6001 序列，可直接跳到 Step4

set -e
PROJ=/home/vision/Project/Affordance2Grasp
CONDA_ENV_DEPTH=depth-pro
CONDA_ENV_HAPTIC=haptic
CONDA_ENV_FP=bundlesdf
CONDA_ENV_ALIGN=bundlesdf

GPU=1
export CUDA_VISIBLE_DEVICES=$GPU

echo "=========================================="
echo " GPU $GPU — DexYCB Pipeline"
echo " $(date)"
echo "=========================================="

# ──────────────────────────────────────────────
# Step 1: DepthPro (仅估计 K.txt 用于 HaPTIC)
# DexYCB 有传感器深度，DepthPro 主要目的是得到 K
# ──────────────────────────────────────────────
if [[ "$@" != *"--skip-depth"* && "$@" != *"--only-align"* ]]; then
  echo ""
  echo "▶ Step 1/4: DepthPro K 估计 (DexYCB)"
  cd $PROJ
  conda run -n $CONDA_ENV_DEPTH \
    python data/batch_depth_pro.py --dataset dexycb
  echo "✅ Step 1 done → depth/ThirdPerson/Dexycb/"
fi

# ──────────────────────────────────────────────
# Step 2: HaPTIC → mano/ThirdPerson/Dexycb/
# ──────────────────────────────────────────────
if [[ "$@" != *"--only-align"* ]]; then
  echo ""
  echo "▶ Step 2/4: HaPTIC MANO (DexYCB)"
  conda run -n $CONDA_ENV_HAPTIC \
    python data/batch_haptic.py --dataset dexycb
  echo "✅ Step 2 done → mano/ThirdPerson/Dexycb/"

  # ─────────────────────────────────────────────
  # Step 3: FoundationPose → poses/ThirdPerson/Dexycb/
  # DexYCB 已有 6001 序列 ob_in_cam (旧路径)
  # 如果旧数据已存在可以跳过，直接用已有数据
  # ─────────────────────────────────────────────
  echo ""
  echo "▶ Step 3/4: FoundationPose (DexYCB)"
  echo "  注: DexYCB 已有 FP 位姿，检查新路径..."
  
  N_NEW=$(find $PROJ/data_hub/poses/ThirdPerson/Dexycb -name 'ob_in_cam' 2>/dev/null | wc -l)
  N_OLD=$(find $PROJ/data_hub/ProcessedData/obj_poses/dexycb -name 'ob_in_cam' 2>/dev/null | wc -l)
  echo "  新路径已有: $N_NEW 序列  旧路径已有: $N_OLD 序列"

  if [ "$N_NEW" -lt 100 ] && [ "$N_OLD" -gt 100 ]; then
    echo "  ⚡ 检测到旧路径有完整数据，创建符号链接到新路径..."
    mkdir -p $PROJ/data_hub/poses/ThirdPerson/
    ln -sfn $PROJ/data_hub/ProcessedData/obj_poses/dexycb \
            $PROJ/data_hub/poses/ThirdPerson/Dexycb
    echo "  ✅ 符号链接已创建"
  else
    conda run -n $CONDA_ENV_FP \
      python tools/batch_obj_pose.py --dataset dexycb
  fi
  echo "✅ Step 3 done → poses/ThirdPerson/Dexycb/"
fi

# ──────────────────────────────────────────────
# Step 4: AlignManoFP → human_prior_fp/ThirdPerson/Dexycb/
# ──────────────────────────────────────────────
echo ""
echo "▶ Step 4/4: Align MANO×FP → HumanPrior (DexYCB)"
conda run -n $CONDA_ENV_ALIGN \
  python data/batch_align_mano_fp.py --dataset dexycb --n-workers 8
echo "✅ Step 4 done → human_prior_fp/ThirdPerson/Dexycb/"

echo ""
echo "=========================================="
echo "✅ DexYCB Pipeline 完成: $(date)"
echo "=========================================="
