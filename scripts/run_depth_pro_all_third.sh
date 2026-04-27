#!/bin/bash
# ============================================================
#  run_depth_pro_all_third.sh
#  一键对所有第三视角数据集运行 Depth Pro 深度估计
#
#  用法:
#    bash scripts/run_depth_pro_all_third.sh            # 全量
#    bash scripts/run_depth_pro_all_third.sh --limit 3  # 每数据集3个（测试）
#
#  输出: data_hub/ProcessedData/third_depth/{dataset}/{seq_id}/
#    depths.npz   — 深度图 (N, H, W) float32 单位:米
#    K.txt        — 估计的相机内参 3x3
#    frame_ids.txt— 帧文件名列表
#  环境: conda activate depth-pro
# ============================================================

set -e

LIMIT=0
if [ "$1" == "--limit" ]; then LIMIT=$2; fi

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$PROJECT_DIR/output/depth_logs"
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
echo " 输出: data_hub/ProcessedData/third_depth/"
echo "========================================"

TOTAL_DONE=0

for DS in "${DATASETS[@]}"; do
    echo ""
    echo "────────────────────────────────────────"
    echo " ▶ 数据集: $DS"
    echo "────────────────────────────────────────"
    LOG="$LOG_DIR/depth_${DS}.log"

    python data/batch_depth_pro.py \
        --dataset "$DS" \
        $LIMIT_ARG \
        2>&1 | tee "$LOG"

    DONE=$(ls "$PROJECT_DIR/data_hub/ProcessedData/third_depth/$DS/" 2>/dev/null | wc -l)
    echo "  ✅ $DS 完成: $DONE 个序列"
    TOTAL_DONE=$((TOTAL_DONE + DONE))
done

echo ""
echo "========================================"
echo " ✅ 全部完成，共 $TOTAL_DONE 个序列"
echo " 日志: $LOG_DIR"
echo " 结果: data_hub/ProcessedData/third_depth/"
echo "========================================"
