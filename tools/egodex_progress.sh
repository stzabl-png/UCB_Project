#!/usr/bin/env bash
# ════════════════════════════════════════════════════
# egodex_progress.sh — EgoDex 全量处理 实时进度监控
# 用法: bash tools/egodex_progress.sh
#       (另开一个 terminal 运行，每 10 秒刷新)
# ════════════════════════════════════════════════════

PROJ="/home/lyh/Project/Affordance2Grasp"
EGODEX_RAW="$PROJ/data_hub/RawData/ThirdPersonRawData/egodex/test"
DEPTH_OUT="$PROJ/data_hub/ProcessedData/egocentric_depth/egodex"
OBJ_POSE_OUT="$PROJ/data_hub/ProcessedData/obj_poses_ego/egodex"
PRIOR_OUT="$PROJ/data_hub/human_prior"
VIS_OUT="$PROJ/output/affordance_ego"

TOTAL_SEQ=3051

bar() {
    local done=$1 total=$2 width=40
    local pct=$(( done * 100 / total ))
    local filled=$(( done * width / total ))
    local empty=$(( width - filled ))
    printf "["
    printf '%0.s█' $(seq 1 $filled 2>/dev/null) 2>/dev/null || printf '%*s' $filled '' | tr ' ' '█'
    printf '%*s' $empty '' | tr ' ' '░'
    printf "] %d/%d (%d%%)" $done $total $pct
}

while true; do
    clear
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║        EgoDex Pipeline — Real-time Progress              ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    # Step 0: 帧提取
    s0=$(find "$EGODEX_RAW" -name "extracted_images" -type d 2>/dev/null | \
         while read d; do [ "$(ls "$d" 2>/dev/null | wc -l)" -gt 0 ] && echo 1; done | wc -l)
    echo -n "  Step 0 · 帧提取     "
    bar $s0 $TOTAL_SEQ
    echo ""

    # Step 1: MegaSAM
    s1=$(find "$DEPTH_OUT" -name "depth.npz" 2>/dev/null | wc -l)
    echo -n "  Step 1 · MegaSAM   "
    bar $s1 $TOTAL_SEQ
    # 显示最新处理的序列
    latest_mega=$(find "$DEPTH_OUT" -name "meta.json" 2>/dev/null -printf '%T@ %p\n' | \
                  sort -n | tail -1 | awk '{print $2}' | \
                  sed "s|$DEPTH_OUT/||;s|/meta.json||")
    [ -n "$latest_mega" ] && echo "                       └─ 最新: $latest_mega" || echo ""

    # Step 2: HaWoR
    s2=$(find "$EGODEX_RAW" -name "world_space_res.pth" 2>/dev/null | wc -l)
    echo -n "  Step 2 · HaWoR     "
    bar $s2 $TOTAL_SEQ
    latest_hawor=$(find "$EGODEX_RAW" -name "world_space_res.pth" 2>/dev/null -printf '%T@ %p\n' | \
                   sort -n | tail -1 | awk '{print $2}' | \
                   sed "s|$EGODEX_RAW/||;s|/world_space_res.pth||")
    [ -n "$latest_hawor" ] && echo "                       └─ 最新: $latest_hawor" || echo ""

    # Step 3: 物体 mesh
    n_obj=$(ls "$PROJ/data_hub/ProcessedData/obj_meshes/egocentric/" 2>/dev/null | wc -l)
    n_scale=$(find "$PROJ/data_hub/ProcessedData/obj_meshes/egocentric" -name "scale.json" 2>/dev/null | wc -l)
    echo "  Step 3 · SAM3D mesh  ${n_obj} 个物体  (scale.json: ${n_scale} 个)"

    # Step 6: FoundationPose
    s6=$(find "$OBJ_POSE_OUT" -name "ob_in_cam" -type d 2>/dev/null | wc -l)
    n6_total=$(grep -c "obj_name" "$PROJ/tools/batch_obj_pose_ego.py" 2>/dev/null || echo "?")
    echo "  Step 6 · FP 位姿     ${s6} 条序列完成"

    # Step 7: 接触图
    s7=$(ls "$PRIOR_OUT"/*.hdf5 2>/dev/null | wc -l)
    echo "  Step 7 · 接触图      ${s7} 个物体  →  $PRIOR_OUT"

    # Step 8: 可视化
    s8=$(ls "$VIS_OUT"/*.png 2>/dev/null | wc -l)
    echo "  Step 8 · 可视化      ${s8} 张图    →  $VIS_OUT"

    echo ""
    echo "  ── 等待 10s 后刷新 (Ctrl+C 退出) ──"
    sleep 10
done
