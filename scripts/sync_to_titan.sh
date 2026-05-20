#!/bin/bash
# ============================================================
# sync_to_titan.sh
# 本机 → TitanX 服务器 数据同步
#
# 传输内容:
#   1. output/grasps_candidate/     OakInk 候选 HDF5
#   2. output/grasps_candidate_dexycb/  DexYCB 候选 HDF5
#   3. output/obj_usd/              USD 资产 (Isaac Sim 加载)
#   4. sim/                         最新 Sim 脚本
#   5. tools/random_grasp_sampler.py (版本同步)
#
# 用法:
#   bash scripts/sync_to_titan.sh            # 全量同步
#   bash scripts/sync_to_titan.sh --dry-run  # 预览不执行
# ============================================================

set -e

TITAN="vision@128.32.164.115"
REMOTE_PROJ="/home/vision/Project/Affordance2Grasp"
LOCAL_PROJ="$(cd "$(dirname "$0")/.." && pwd)"

DRY=""
if [ "$1" == "--dry-run" ]; then
    DRY="--dry-run"
    echo "⚠️  DRY-RUN 模式，不实际传输"
fi

echo "============================================================"
echo "  本机 → TitanX 数据同步"
echo "  本机:   $LOCAL_PROJ"
echo "  远端:   $TITAN:$REMOTE_PROJ"
echo "  时间:   $(date)"
echo "============================================================"

rsync_run() {
    local src="$1"
    local dst="$2"
    local desc="$3"
    echo ""
    echo "  [$desc]"
    echo "  $src → $TITAN:$dst"
    rsync -avz --progress $DRY \
        --exclude="*.pyc" --exclude="__pycache__" \
        "$src" "$TITAN:$dst"
}

# 1. OakInk 候选 HDF5
rsync_run \
    "$LOCAL_PROJ/output/grasps_candidate/" \
    "$REMOTE_PROJ/output/grasps_candidate/" \
    "OakInk 抓取候选 HDF5"

# 2. DexYCB 候选 HDF5
if [ -d "$LOCAL_PROJ/output/grasps_candidate_dexycb" ]; then
    rsync_run \
        "$LOCAL_PROJ/output/grasps_candidate_dexycb/" \
        "$REMOTE_PROJ/output/grasps_candidate_dexycb/" \
        "DexYCB 抓取候选 HDF5"
fi

# 3. USD 资产 — 仅 OakInk + DexYCB (其他数据集暂不同步)
rsync_run \
    "$LOCAL_PROJ/output/obj_usd/oakink/" \
    "$REMOTE_PROJ/output/obj_usd/oakink/" \
    "USD OakInk"

rsync_run \
    "$LOCAL_PROJ/output/obj_usd/dexycb/" \
    "$REMOTE_PROJ/output/obj_usd/dexycb/" \
    "USD DexYCB"

# 4. Sim 脚本 (保持最新版本)
rsync_run \
    "$LOCAL_PROJ/sim/run_grasp_sim.py" \
    "$REMOTE_PROJ/sim/" \
    "Sim 执行脚本"

rsync_run \
    "$LOCAL_PROJ/sim/object_rotation_overrides.json" \
    "$REMOTE_PROJ/sim/" \
    "物体旋转覆盖配置"

rsync_run \
    "$LOCAL_PROJ/sim/canonical_rotation.json" \
    "$REMOTE_PROJ/sim/" \
    "Canonical rotation"

# 5. 批量 Sim 脚本
rsync_run \
    "$LOCAL_PROJ/scripts/run_sim_titan_gpu0.sh" \
    "$REMOTE_PROJ/scripts/" \
    "GPU0 Sim 脚本"

rsync_run \
    "$LOCAL_PROJ/scripts/run_sim_titan_gpu1.sh" \
    "$REMOTE_PROJ/scripts/" \
    "GPU1 Sim 脚本"

echo ""
echo "============================================================"
echo "  ✅ 同步完成"
echo "  下一步: 在服务器上执行"
echo "    GPU0: bash scripts/run_sim_titan_gpu0.sh"
echo "    GPU1: bash scripts/run_sim_titan_gpu1.sh"
echo "============================================================"
