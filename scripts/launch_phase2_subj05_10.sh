#!/bin/bash
# Phase 2: extract DexYCB subjects 05-10 from full tarball, then Step 1 + Step 3.
# Designed to run AFTER current Step 2 DexYCB (subj 01-04) finishes,
# so all 8 GPUs are free.
set -u
source ~/miniconda3/etc/profile.d/conda.sh

STAGING=~/Affordance2Grasp/data_hub/RawData/staging/dex-ycb-20210415.tar.gz
DEST=~/Affordance2Grasp/data_hub/RawData/ThirdPersonRawData/dexycb
LOGDIR=~/Affordance2Grasp/output/phase2_logs
mkdir -p $LOGDIR

echo "=== [Phase 2] Extract subjects 05-10 ==="
SUBJ_LIST="20200908-subject-05 20200918-subject-06 20200928-subject-07 20201002-subject-08 20201015-subject-09 20201022-subject-10"
mkdir -p $DEST
cd $DEST
for S in $SUBJ_LIST; do
    if [ -d "$S" ]; then
        echo "  $S already exists, skip"
        continue
    fi
    echo "  extracting $S ..."
    tar xf $STAGING --wildcards "$S/*" 2>>$LOGDIR/extract.log || echo "  WARN: $S not in tarball"
done
ls -d $DEST/*/ 2>/dev/null

# ─── Step 1 Depth Pro ──────────────────────────────────────────
echo "=== [Phase 2] Step 1 Depth Pro on new subjects ==="
conda activate depth-pro
cd ~/Affordance2Grasp
CAMS=(841412060263 840412060917 932122060857 836212060125 932122061900 932122062010)
# Re-run depth pro per camera (will skip subj 01-04 sequences that already have depth K)
for i in 0 1 2 3 4 5; do
    GPU=$i
    CAM=${CAMS[$i]}
    CUDA_VISIBLE_DEVICES=$GPU nohup python data/batch_depth_pro.py \
        --dataset dexycb --cam $CAM --two-pass --max-frames 150 \
        > $LOGDIR/depth_cam${GPU}_${CAM}.log 2>&1 &
    echo "Step1 GPU$GPU/$CAM: pid=$!"
done
wait
echo "  Step 1 done."

# ─── Step 3 FoundationPose ─────────────────────────────────────
echo "=== [Phase 2] Step 3 FoundationPose on new subjects ==="
conda activate bundlesdf
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export FP_ROOT=$HOME/Affordance2Grasp/third_party/FoundationPose
NEW_SUBJS=(20200908-subject-05 20200918-subject-06 20200928-subject-07 20201002-subject-08 20201015-subject-09 20201022-subject-10)
for i in 0 1 2 3 4 5; do
    GPU=$i
    SUBJ=${NEW_SUBJS[$i]}
    [ -d ~/Affordance2Grasp/data_hub/RawData/ThirdPersonRawData/dexycb/$SUBJ ] || continue
    CUDA_VISIBLE_DEVICES=$GPU nohup python tools/batch_obj_pose.py --dataset dexycb --seq $SUBJ \
        > $LOGDIR/objpose_${SUBJ}.log 2>&1 &
    echo "Step3 GPU$GPU/$SUBJ: pid=$!"
done
wait
echo "  Step 3 done."

# ─── Step 2 HaPTIC ─────────────────────────────────────────────
echo "=== [Phase 2] Step 2 HaPTIC on new subjects (8-way: 6 subj × split where helpful) ==="
export HAPTIC_DIR=$HOME/Affordance2Grasp/third_party/haptic
# 6 new subjects on 6 GPUs + 2 extras to split heaviest 2
for i in 0 1 2 3 4 5; do
    GPU=$i
    SUBJ=${NEW_SUBJS[$i]}
    [ -d ~/Affordance2Grasp/data_hub/RawData/ThirdPersonRawData/dexycb/$SUBJ ] || continue
    CUDA_VISIBLE_DEVICES=$GPU nohup python data/batch_haptic.py \
        --dataset dexycb --only-with-depth-k --seq $SUBJ --shard 1/2 \
        > $LOGDIR/haptic_${SUBJ}_shard1_2.log 2>&1 &
    echo "Step2 GPU$GPU/$SUBJ shard1/2: pid=$!"
done
# GPUs 6,7 take shard 2/2 of subj-05 and subj-06 (heaviest? or just round-robin)
for i in 0 1; do
    GPU=$((i + 6))
    SUBJ=${NEW_SUBJS[$i]}
    [ -d ~/Affordance2Grasp/data_hub/RawData/ThirdPersonRawData/dexycb/$SUBJ ] || continue
    CUDA_VISIBLE_DEVICES=$GPU nohup python data/batch_haptic.py \
        --dataset dexycb --only-with-depth-k --seq $SUBJ --shard 2/2 \
        > $LOGDIR/haptic_${SUBJ}_shard2_2.log 2>&1 &
    echo "Step2 GPU$GPU/$SUBJ shard2/2: pid=$!"
done
# Actually, every subject deserves split 2/2. But we only have 8 GPUs and 6 subjects — leaving 4 subj single-GPU.
# A second pass after first 2 finish will pick those up.
wait
echo "  Step 2 first pass done."

# Pick up shard 2/2 for the remaining 4 subjects
for i in 2 3 4 5; do
    GPU=$((i - 2))
    SUBJ=${NEW_SUBJS[$i]}
    [ -d ~/Affordance2Grasp/data_hub/RawData/ThirdPersonRawData/dexycb/$SUBJ ] || continue
    CUDA_VISIBLE_DEVICES=$GPU nohup python data/batch_haptic.py \
        --dataset dexycb --only-with-depth-k --seq $SUBJ --shard 2/2 \
        > $LOGDIR/haptic_${SUBJ}_shard2_2.log 2>&1 &
    echo "Step2 GPU$GPU/$SUBJ shard2/2: pid=$!"
done
wait
echo "  Step 2 second pass done."

# ─── Step 4 Alignment (CPU, fast) ──────────────────────────────
echo "=== [Phase 2] Step 4 alignment for both datasets ==="
python data/batch_align_mano_fp.py --dataset dexycb 2>&1 | tee $LOGDIR/align_dexycb.log
python data/batch_align_mano_fp.py --dataset ho3d_v3 2>&1 | tee $LOGDIR/align_ho3d.log

echo "=== Phase 2 complete ==="
