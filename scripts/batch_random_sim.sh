#!/bin/bash
# ============================================================
# 批量 Sim 验证 (v3 高分候选 score≥80, 100 batches)
# ============================================================
# 用法:
#     conda deactivate
#     cd "$(dirname "$(readlink -f "$0")")/.." # project root
#     bash batch_random_sim.sh
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GRASP_DIR="$SCRIPT_DIR/output/grasps_random"
GT_DIR="$SCRIPT_DIR/output/robot_gt_v4_physics"
LOG_DIR="$SCRIPT_DIR/output/sim_logs_v4"

mkdir -p "$GT_DIR" "$LOG_DIR"

HDF5_LIST=($(ls "$GRASP_DIR"/*_grasp.hdf5 2>/dev/null))
TOTAL=${#HDF5_LIST[@]}

echo "============================================================"
echo "  Random Grasp Sim v4 Verification (physics scoring)"
echo "============================================================"
echo "  Total:      $TOTAL"
echo "  Grasp dir:  $GRASP_DIR"
echo "  Result dir: $GT_DIR"
echo "============================================================"

SUCCESS=0; FAILED=0; SKIPPED=0

for i in $(seq 0 $((TOTAL-1))); do
    HDF5="${HDF5_LIST[$i]}"
    OBJ_ID=$(basename "$HDF5" _grasp.hdf5)
    N=$((i+1))

    RESULT_FILE="$GT_DIR/${OBJ_ID}_robot_gt.hdf5"

    if [ -f "$RESULT_FILE" ]; then
        SKIPPED=$((SKIPPED+1))
        continue
    fi

    echo "  [$N/$TOTAL] $OBJ_ID ..."

    timeout 300 ${ISAAC_SIM_PATH:-/opt/isaac-sim}/python.sh "$SCRIPT_DIR/sim/run_grasp_sim.py" \
        --hdf5 "$HDF5" \
        --headless \
        --save-result \
        --result-dir "$GT_DIR" \
        2>&1 | tee "$LOG_DIR/${OBJ_ID}.log" | tail -3

    if [ -f "$RESULT_FILE" ]; then
        RESULT=$(python3 -c "
import h5py
with h5py.File('$RESULT_FILE','r') as f:
    print('SUCCESS' if f.attrs['success'] else 'FAILED')
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
echo "  Sim 完成, 统计结果..."
echo "============================================================"

python3 -c "
import h5py, os, glob

gt_dir = '$GT_DIR'
files = sorted(glob.glob(os.path.join(gt_dir, '*_robot_gt.hdf5')))
total_success_grasps = 0
total_tried = 0
ok_list = []
fail_list = []

for f in files:
    obj_id = os.path.basename(f).replace('_robot_gt.hdf5', '')
    with h5py.File(f, 'r') as h:
        ok = bool(h.attrs.get('success', False))
        n_ok = int(h.attrs.get('n_successful', 0))
        n_t = int(h.attrs.get('n_candidates_tried', 0))
        total_success_grasps += n_ok
        total_tried += n_t
        if ok:
            ok_list.append((obj_id, n_ok, n_t))
        else:
            fail_list.append(obj_id)

print()
print('=== v4 Physics GT 汇总 ===')
print(f'物体总数:     {len(files)}')
print(f'有成功抓取:   {len(ok_list)}')
print(f'全部失败:     {len(fail_list)}')
print(f'成功抓取总数: {total_success_grasps}')
print(f'候选总数:     {total_tried}')
print(f'物体成功率:   {len(ok_list)/max(len(files),1)*100:.1f}%')
print(f'候选成功率:   {total_success_grasps/max(total_tried,1)*100:.1f}%')
print()
print('Top 10 成功最多:')
for obj, n, t in sorted(ok_list, key=lambda x: -x[1])[:10]:
    print(f'  {obj}: {n}/{t} ({n/t*100:.0f}%)')
print()
print('失败物体:')
for obj in fail_list:
    print(f'  ❌ {obj}')
print('============================================================')
"
