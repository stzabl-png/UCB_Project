#!/bin/bash
# ============================================================
# sync_results_from_titan.sh
# TitanX 服务器 → 本机  结果回传
#
# 用法:
#   bash scripts/sync_results_from_titan.sh
#   bash scripts/sync_results_from_titan.sh --dry-run
# ============================================================

TITAN="vision@128.32.164.115"
REMOTE_PROJ="/home/vision/Project/Affordance2Grasp"
LOCAL_PROJ="$(cd "$(dirname "$0")/.." && pwd)"

DRY=""
[ "$1" == "--dry-run" ] && DRY="--dry-run" && echo "⚠️  DRY-RUN"

echo "============================================================"
echo "  TitanX → 本机 结果回传"
echo "  $(date)"
echo "============================================================"

# OakInk robot_gt
echo "  [OakInk robot_gt]"
rsync -avz $DRY --progress \
    "$TITAN:$REMOTE_PROJ/output/robot_gt_oakink/" \
    "$LOCAL_PROJ/output/robot_gt_oakink/"

# DexYCB robot_gt
echo "  [DexYCB robot_gt]"
rsync -avz $DRY --progress \
    "$TITAN:$REMOTE_PROJ/output/robot_gt_dexycb/" \
    "$LOCAL_PROJ/output/robot_gt_dexycb/"

# Sim 日志
echo "  [Sim 日志]"
rsync -avz $DRY \
    "$TITAN:$REMOTE_PROJ/output/sim_logs_oakink/" \
    "$LOCAL_PROJ/output/sim_logs_oakink/"
rsync -avz $DRY \
    "$TITAN:$REMOTE_PROJ/output/sim_logs_dexycb/" \
    "$LOCAL_PROJ/output/sim_logs_dexycb/"

echo ""
echo "============================================================"
echo "  ✅ 回传完成"
echo ""
# 快速统计
python3 -c "
import os, h5py, glob
for tag, d in [('OakInk', '$LOCAL_PROJ/output/robot_gt_oakink'),
               ('DexYCB', '$LOCAL_PROJ/output/robot_gt_dexycb')]:
    files = glob.glob(os.path.join(d, '*_robot_gt.hdf5'))
    total = len(files)
    suc = 0
    for f in files:
        try:
            with h5py.File(f, 'r') as h:
                if int(h.attrs.get('n_successful', 0)) > 0:
                    suc += 1
        except: pass
    print(f'  {tag}: {suc}/{total} 成功  ({total-suc} 失败)')
" 2>/dev/null || echo "  (结果目录为空)"
echo "============================================================"
