#!/bin/bash
# run_pipeline_gpu0_oakink.sh
# GPU 0 — OakInk 完整 Pipeline: 原始视频 → HumanPrior
#
# 依赖顺序:
#   Step1: DepthPro  (自动)
#   Step2: HaPTIC    (依赖 Step1 K.txt)
#   Step2b: SAM2标注  ⚠️ 需要人工操作 (见下方说明)
#   Step3: FP        (依赖 Step1深度 + Step2b masks)
#   Step4: AlignManoFP (依赖 Step2 + Step3)
#
# 用法:
#   bash run_pipeline_gpu0_oakink.sh
#   bash run_pipeline_gpu0_oakink.sh --skip-depth   # 跳过已完成的 Step1
#   bash run_pipeline_gpu0_oakink.sh --only-align   # 只跑 Step4

set -e
PROJ=/home/vision/Project/Affordance2Grasp
CONDA_ENV_DEPTH=depth-pro
CONDA_ENV_HAPTIC=haptic
CONDA_ENV_FP=bundlesdf
CONDA_ENV_ALIGN=bundlesdf

GPU=0
export CUDA_VISIBLE_DEVICES=$GPU

echo "=========================================="
echo " GPU $GPU — OakInk Pipeline"
echo " $(date)"
echo "=========================================="

# ──────────────────────────────────────────────
# Step 1: DepthPro → depth/ThirdPerson/Oakink/
# ──────────────────────────────────────────────
if [[ "$@" != *"--skip-depth"* && "$@" != *"--only-align"* ]]; then
  echo ""
  echo "▶ Step 1/4: DepthPro (OakInk)"
  cd $PROJ
  conda run -n $CONDA_ENV_DEPTH \
    python data/batch_depth_pro.py --dataset oakink
  echo "✅ Step 1 done → depth/ThirdPerson/Oakink/"
fi

# ──────────────────────────────────────────────
# Step 2: HaPTIC → mano/ThirdPerson/Oakink/
# (使用 Step1 的 K.txt 作为相机内参)
# ──────────────────────────────────────────────
if [[ "$@" != *"--only-align"* ]]; then
  echo ""
  echo "▶ Step 2/4: HaPTIC MANO (OakInk)"
  conda run -n $CONDA_ENV_HAPTIC \
    python data/batch_haptic.py --dataset oakink --only-with-depth-k
  echo "✅ Step 2 done → mano/ThirdPerson/Oakink/"

  # ─────────────────────────────────────────────
  # Step 2b: SAM2 Mask 检查
  # ─────────────────────────────────────────────
  echo ""
  N_MASKS=$(find $PROJ/data_hub/recon_input/ThirdPerson/Oakink -name '*.png' 2>/dev/null | wc -l)
  if [ "$N_MASKS" -lt 10 ]; then
    echo "════════════════════════════════════════════"
    echo "⚠️  recon_input/ThirdPerson/Oakink/ 无 mask (当前: $N_MASKS 张)"
    echo "   需先运行 SAM2 标注:"
    echo "   conda run -n sam2 python tools/sam2_annotate_masks.py --dataset oakink"
    echo "   完成后重新运行: bash run_pipeline_gpu0_oakink.sh --skip-depth"
    echo "════════════════════════════════════════════"
    echo "⏭  跳过 Step3 (FP), 直接到 Step4 (需要已有 FP 位姿)"
    SKIP_FP=1
  fi

  if [ -z "$SKIP_FP" ]; then
    # ─────────────────────────────────────────────
    # Step 3: FoundationPose → poses/ThirdPerson/Oakink/
    # ─────────────────────────────────────────────
    echo ""
    echo "▶ Step 3/4: FoundationPose (OakInk)"
    conda run -n $CONDA_ENV_FP \
      python tools/batch_obj_pose.py --dataset oakink
    echo "✅ Step 3 done → poses/ThirdPerson/Oakink/"
  fi
fi

# ──────────────────────────────────────────────
# Step 4: AlignManoFP → human_prior_fp/ThirdPerson/Oakink/
# ──────────────────────────────────────────────
echo ""
echo "▶ Step 4/4: Align MANO×FP → HumanPrior (OakInk)"
conda run -n $CONDA_ENV_ALIGN \
  python data/batch_align_mano_fp.py --dataset oakink --n-workers 8
echo "✅ Step 4 done → human_prior_fp/ThirdPerson/Oakink/"

echo ""
echo "=========================================="
echo "✅ OakInk Pipeline 完成: $(date)"
echo "=========================================="
