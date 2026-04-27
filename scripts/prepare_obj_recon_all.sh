#!/bin/bash
# ============================================================
#  prepare_obj_recon_all.sh
#  一键对所有第三视角数据集准备 obj_recon (SAM3D) 输入
#
#  用法:
#    bash scripts/prepare_obj_recon_all.sh            # 全量
#    bash scripts/prepare_obj_recon_all.sh --limit 3  # 每数据集3个（测试）
#
#  输出: data_hub/ProcessedData/obj_recon_input/{dataset}/{seq_id}/
#    image.png       — RGBA 输入图（mask 在 alpha 通道）
#    rgb.png         — 原始 RGB
#    depth.npy       — 单帧深度图 float32 (米)
#    K.txt           — 3x3 相机内参
#
#  前置条件:
#    先跑 run_depth_pro_all_third.sh 生成深度图
#  环境: conda activate depth-pro
# ============================================================

set -e

LIMIT=0
if [ "$1" == "--limit" ]; then LIMIT=$2; fi

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$PROJECT_DIR/output/obj_recon_logs"
mkdir -p "$LOG_DIR"

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate depth-pro 2>/dev/null || true

cd "$PROJECT_DIR"

DATASETS=("arctic" "oakink" "ho3d_v3" "dexycb")
LIMIT_ARG=""
if [ "$LIMIT" -gt 0 ]; then
    LIMIT_ARG="--limit $LIMIT"
    echo "========================================"
    echo " 测试模式: 每个数据集处理 $LIMIT 个序列"
else
    echo "========================================"
    echo " 全量模式: 处理所有序列"
fi
echo " 输出: data_hub/ProcessedData/obj_recon_input/"
echo "========================================"

TOTAL_DONE=0

for DS in "${DATASETS[@]}"; do
    echo ""
    echo "────────────────────────────────────────"
    echo " ▶ 数据集: $DS"
    echo "────────────────────────────────────────"
    LOG="$LOG_DIR/obj_recon_${DS}.log"

    python data/prepare_obj_recon_input.py \
        --dataset "$DS" \
        $LIMIT_ARG \
        2>&1 | tee "$LOG"

    DONE=$(ls "$PROJECT_DIR/data_hub/ProcessedData/obj_recon_input/$DS/" 2>/dev/null | wc -l)
    echo "  ✅ $DS 完成: $DONE 个序列"
    TOTAL_DONE=$((TOTAL_DONE + DONE))
done

echo ""
echo "========================================"
echo " ✅ 全部完成，共 $TOTAL_DONE 个序列"
echo " 日志: $LOG_DIR"
echo " 结果: data_hub/ProcessedData/obj_recon_input/"
echo ""
echo " 下一步: 上传到阿里云运行重建"
echo "   rsync -avz data_hub/ProcessedData/obj_recon_input/ \\"
echo "     sam3d-gpu:~/input/"
echo "========================================"
