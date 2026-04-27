#!/bin/bash
# ============================================================
#  run_haptic_all_third.sh
#  一键对所有第三视角数据集运行 HaPTIC MANO 估计
#
#  用法:
#    bash scripts/run_haptic_all_third.sh           # 全量
#    bash scripts/run_haptic_all_third.sh --limit 3 # 每个数据集只跑3个（测试用）
#
#  输出: data_hub/ProcessedData/third_mano/{dataset}/{seq_id}.npz
#  环境: conda activate haptic
# ============================================================

set -e

LIMIT=${2:-0}  # 默认 0 = 全量
if [ "$1" == "--limit" ]; then LIMIT=$2; fi

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$PROJECT_DIR/output/haptic_logs"
mkdir -p "$LOG_DIR"

# 激活环境（如果还没激活）
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate haptic 2>/dev/null || true

cd "$PROJECT_DIR"

DATASETS=("arctic" "oakink" "ho3d_v3" "dexycb" "selfmade")
LIMIT_ARG=""
if [ "$LIMIT" -gt 0 ]; then
    LIMIT_ARG="--limit $LIMIT"
    echo "========================================"
    echo " 测试模式: 每个数据集处理 $LIMIT 个序列"
else
    echo "========================================"
    echo " 全量模式: 处理所有序列"
fi
echo " 输出: data_hub/ProcessedData/third_mano/"
echo "========================================"
echo ""

TOTAL_DONE=0
TOTAL_FAILED=0

for DS in "${DATASETS[@]}"; do
    echo ""
    echo "────────────────────────────────────────"
    echo " ▶ 数据集: $DS"
    echo "────────────────────────────────────────"
    LOG="$LOG_DIR/haptic_${DS}.log"

    python data/batch_haptic.py \
        --dataset "$DS" \
        $LIMIT_ARG \
        2>&1 | tee "$LOG"

    DONE=$(ls "$PROJECT_DIR/data_hub/ProcessedData/third_mano/$DS/" 2>/dev/null | wc -l)
    echo "  ✅ $DS 完成: $DONE 个序列已保存"
    TOTAL_DONE=$((TOTAL_DONE + DONE))
done

echo ""
echo "========================================"
echo " ✅ 全部完成"
echo " 总输出序列: $TOTAL_DONE"
echo " 日志目录: $LOG_DIR"
echo " 结果目录: data_hub/ProcessedData/third_mano/"
echo "========================================"
