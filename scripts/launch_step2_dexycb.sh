#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bundlesdf
export HAPTIC_DIR=$HOME/Affordance2Grasp/third_party/haptic
cd ~/Affordance2Grasp
mkdir -p output/haptic_logs

# 4 subjects, 4 GPUs, one subject per GPU (~600 seqs each)
declare -A SUBJ
SUBJ[4]=20200709-subject-01
SUBJ[5]=20200813-subject-02
SUBJ[6]=20200820-subject-03
SUBJ[7]=20200903-subject-04

for GPU in 4 5 6 7; do
  S=${SUBJ[$GPU]}
  CUDA_VISIBLE_DEVICES=$GPU nohup python data/batch_haptic.py \
      --dataset dexycb --only-with-depth-k --seq $S \
      > output/haptic_logs/dexycb_gpu${GPU}_${S}.log 2>&1 &
  echo "GPU$GPU/$S: pid=$!"
done

wait
echo "All Step 2 DexYCB processes complete."
