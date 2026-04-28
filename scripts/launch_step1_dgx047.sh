#!/bin/bash
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate depth-pro
cd ~/Affordance2Grasp
LOG=output/depth_logs

CAMS=(841412060263 840412060917 932122060857 836212060125 932122061900 932122062010)

# GPU 0: HO3D-v3
CUDA_VISIBLE_DEVICES=0 nohup python data/batch_depth_pro.py \
    --dataset ho3d_v3 --two-pass --max-frames 150 \
    > $LOG/ho3d.log 2>&1 &
echo "ho3d: gpu0 pid=$!"

# GPUs 1..6: DexYCB cameras
for i in 0 1 2 3 4 5; do
    GPU=$((i+1))
    CAM=${CAMS[$i]}
    CUDA_VISIBLE_DEVICES=$GPU nohup python data/batch_depth_pro.py \
        --dataset dexycb --cam $CAM --two-pass --max-frames 150 \
        > $LOG/dexycb_cam${GPU}_${CAM}.log 2>&1 &
    echo "dexycb cam=$CAM: gpu$GPU pid=$!"
done

wait
echo "All Step 1 processes complete."
