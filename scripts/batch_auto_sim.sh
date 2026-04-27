#!/bin/bash
# ============================================================
# Step 2: 对自动生成的抓取跑 Sim 验证 (Isaac Sim 环境)
# ============================================================
# 用法:
#     conda deactivate
#     cd "$(dirname "$(readlink -f "$0")")/.." # project root
#     bash batch_auto_sim.sh
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ="$(cd "$SCRIPT_DIR/.." && pwd)"          # project root
AUTO_GRASP_DIR="${GRASP_DIR:-$PROJ/output/grasps_auto}"
AUTO_GT_DIR="${GT_DIR:-$PROJ/output/robot_gt_auto}"
LOG_DIR="${LOG_DIR:-$PROJ/output/sim_logs_auto}"
SIM_SCRIPT="$PROJ/sim/run_grasp_sim.py"

mkdir -p "$AUTO_GT_DIR" "$LOG_DIR"

export PYTHONUNBUFFERED=1   # 防止 tee 管道吞掉 Python 输出

HDF5_LIST=($(ls "$AUTO_GRASP_DIR"/*_grasps.hdf5 2>/dev/null))
TOTAL=${#HDF5_LIST[@]}

echo "============================================================"
echo "  Auto Grasp Sim Verification"
echo "============================================================"
echo "  Total:      $TOTAL"
echo "  Grasp dir:  $AUTO_GRASP_DIR"
echo "  Result dir: $AUTO_GT_DIR"
echo "============================================================"

SUCCESS=0; FAILED=0; SKIPPED=0

for i in $(seq 0 $((TOTAL-1))); do
    HDF5="${HDF5_LIST[$i]}"
    OBJ_ID=$(basename "$HDF5" _grasps.hdf5)
    N=$((i+1))

    RESULT_FILE="$AUTO_GT_DIR/${OBJ_ID}_robot_gt.hdf5"

    if [ -f "$RESULT_FILE" ]; then
        SKIPPED=$((SKIPPED+1))
        continue
    fi

    echo "  [$N/$TOTAL] $OBJ_ID ..."

    timeout 600 ${ISAAC_SIM_PATH:-/home/lyh/isaac-sim}/python.sh "$SIM_SCRIPT" \
        --hdf5 "$HDF5" \
        --save-result \
        --result-dir "$AUTO_GT_DIR" \
        2>&1 | tee "$LOG_DIR/${OBJ_ID}.log" \
             | grep --line-buffered -E \
               "Attempt [0-9]+|Plan OK|Plan FAILED|cuRobo|GRASP SUCCESS|GRASP|✅|❌|→ 最终|score=|obj Z:|Saved:|结果:|ERROR|Error"

    if [ -f "$RESULT_FILE" ]; then
        RESULT=$(python3 -c "
import h5py
with h5py.File('$RESULT_FILE','r') as f:
    n = int(f.attrs.get('n_successful', 0))
    print('SUCCESS' if n > 0 else 'FAILED')
" 2>/dev/null || echo "UNKNOWN")
        if [ "$RESULT" == "SUCCESS" ]; then
            SUCCESS=$((SUCCESS+1))
        else
            FAILED=$((FAILED+1))
        fi
    else
        FAILED=$((FAILED+1))
    fi
done

echo ""
echo "============================================================"
echo "  DONE  ✅ $SUCCESS  ❌ $FAILED  ⏭️ $SKIPPED  Total: $TOTAL"
echo "  Results: $AUTO_GT_DIR"
echo "============================================================"
