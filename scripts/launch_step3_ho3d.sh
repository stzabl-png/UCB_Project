#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bundlesdf
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export FP_ROOT=$HOME/Affordance2Grasp/third_party/FoundationPose
cd ~/Affordance2Grasp
mkdir -p output/objpose_logs

# Build shard lists (4 shards = 4 GPUs)
python3 - <<'PY'
import os
from natsort import natsorted
base = os.path.expanduser("~/Affordance2Grasp/data_hub/ProcessedData/third_depth/ho3d_v3")
seqs = natsorted(os.listdir(base))
print(f"{len(seqs)} HO3D sequences")
shards = [[] for _ in range(4)]
for i, s in enumerate(seqs):
    shards[i % 4].append(s)
for i, sh in enumerate(shards):
    print(f"GPU{i}: {len(sh)} seqs")
    open(f"/tmp/step3_shard_{i}.txt", "w").write("\n".join(sh))
PY

# Per-GPU worker: process its shard sequentially via --seq
for GPU in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$GPU nohup bash -c "
    while read seq; do
      [ -z \"\$seq\" ] && continue
      python tools/batch_obj_pose.py --dataset ho3d_v3 --seq \"\$seq\"
    done < /tmp/step3_shard_$GPU.txt
  " > output/objpose_logs/ho3d_gpu$GPU.log 2>&1 &
  echo "GPU$GPU: pid=$!"
done

wait
echo "All Step 3 HO3D processes complete."
