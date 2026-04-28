#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bundlesdf
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export FP_ROOT=$HOME/Affordance2Grasp/third_party/FoundationPose
cd ~/Affordance2Grasp
mkdir -p output/objpose_logs

GPUS="${GPUS:-1 2 3}"
N=$(echo $GPUS | wc -w)

python3 - <<PY
import os
from natsort import natsorted
base = os.path.expanduser("~/Affordance2Grasp/data_hub/ProcessedData/third_depth/dexycb")
seqs = natsorted(os.listdir(base))
out_dir = os.path.expanduser("~/Affordance2Grasp/data_hub/ProcessedData/obj_poses/dexycb")
done = set(os.listdir(out_dir)) if os.path.isdir(out_dir) else set()
todo = [s for s in seqs if s not in done]
print(f"{len(seqs)} total, {len(done)} done, {len(todo)} remaining")
N = $N
shards = [[] for _ in range(N)]
for i, s in enumerate(todo):
    shards[i % N].append(s)
gpus = "$GPUS".split()
for i, sh in enumerate(shards):
    print(f"GPU{gpus[i]}: {len(sh)} seqs")
    open(f"/tmp/step3_dex_shard_{gpus[i]}.txt", "w").write("\n".join(sh) + "\n")
PY

for GPU in $GPUS; do
  CUDA_VISIBLE_DEVICES=$GPU nohup bash -c "
    while read seq; do
      [ -z \"\$seq\" ] && continue
      python tools/batch_obj_pose.py --dataset dexycb --seq \"\$seq\"
    done < /tmp/step3_dex_shard_$GPU.txt
  " > output/objpose_logs/dexycb_gpu$GPU.log 2>&1 &
  echo "GPU$GPU: pid=$!"
done

wait
echo "All Step 3 DexYCB processes complete."
