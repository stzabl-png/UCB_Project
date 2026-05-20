#!/bin/bash
# ============================================================
# overnight_pipeline.sh
# 等待候选生成完成 → 自动启动 OakInk + DexYCB Sim
# 隔夜运行，完成后汇总结果
# ============================================================

PROJ="/home/lyh/Project/Affordance2Grasp"
ISAAC_SIM_PATH="/home/lyh/isaac-sim"

echo "============================================================"
echo "  隔夜 Pipeline: 候选生成 → Sim 抓取验证"
echo "  开始时间: $(date)"
echo "============================================================"

# ── 等待候选生成进程结束 ─────────────────────────────────────
echo ""
echo "⏳ 等待 OakInk + DexYCB 候选生成完成..."
wait_count=0
while pgrep -f "random_grasp_sampler" > /dev/null; do
    OAK=$(ls $PROJ/output/grasps_candidate/*_grasp.hdf5 2>/dev/null | wc -l)
    DEX=$(ls $PROJ/output/grasps_candidate_dexycb/*_grasp.hdf5 2>/dev/null | wc -l)
    echo "  [$(date +%H:%M)] OakInk: $OAK/100  DexYCB: $DEX/20"
    sleep 60
    wait_count=$((wait_count+1))
done
echo "✅ 候选生成完成！"
echo ""

# ── 保留第一轮数据，第二轮存到 _r2 目录 ────────────────────────
echo "第二轮结果路径: robot_gt_*_r2  (第一轮数据完好保留)"
mkdir -p $PROJ/output/robot_gt_oakink_r2
mkdir -p $PROJ/output/robot_gt_dexycb_r2
mkdir -p $PROJ/output/sim_logs_oakink_r2
mkdir -p $PROJ/output/sim_logs_dexycb_r2


# ── 串行跑 OakInk Sim ────────────────────────────────────────
echo "============================================================"
echo "  🤖 开始 OakInk Sim 验证 ($(date))"
echo "============================================================"
GRASP_DIR=$PROJ/output/grasps_candidate \
GT_DIR=$PROJ/output/robot_gt_oakink_r2 \
LOG_DIR=$PROJ/output/sim_logs_oakink_r2 \
  bash $PROJ/scripts/run_sim_local_oakink.sh 2>&1 | tee $PROJ/output/sim_logs_oakink_r2/batch.log

# ── 串行跑 DexYCB Sim ────────────────────────────────────────
echo ""
echo "============================================================"
echo "  🤖 开始 DexYCB Sim 验证 ($(date))"
echo "============================================================"
GRASP_DIR=$PROJ/output/grasps_candidate_dexycb \
GT_DIR=$PROJ/output/robot_gt_dexycb_r2 \
LOG_DIR=$PROJ/output/sim_logs_dexycb_r2 \
  bash $PROJ/scripts/run_sim_local_dexycb.sh 2>&1 | tee $PROJ/output/sim_logs_dexycb_r2/batch.log

# ── 汇总结果 ─────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  ✅ 隔夜 Pipeline 完成！$(date)"
echo "============================================================"
python3 -c "
import glob, h5py, os

for tag, d in [
    ('OakInk R2', '$PROJ/output/robot_gt_oakink_r2'),
    ('DexYCB R2', '$PROJ/output/robot_gt_dexycb_r2'),
]:
    files = glob.glob(os.path.join(d, '*_robot_gt.hdf5'))
    suc = [f for f in files if int(h5py.File(f,'r').attrs.get('n_successful',0))>0]
    fail = [f for f in files if int(h5py.File(f,'r').attrs.get('n_successful',0))==0]
    total = len(files)
    print(f'{tag}: ✅{len(suc)} 成功  ❌{len(fail)} 失败  ({total} 完成)')
    if fail:
        print('  失败物体:')
        for f in sorted(fail):
            print(f'    {os.path.basename(f).replace(\"_robot_gt.hdf5\",\"\")}')
" 2>/dev/null
