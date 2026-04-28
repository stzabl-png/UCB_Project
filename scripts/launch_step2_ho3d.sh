#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bundlesdf
export HAPTIC_DIR=$HOME/Affordance2Grasp/third_party/haptic
cd ~/Affordance2Grasp
mkdir -p output/haptic_logs

# Build shard lists across N GPUs (default 3: 5,6,7)
GPUS="${GPUS:-5 6 7}"
N=$(echo $GPUS | wc -w)

python3 - <<PY
import os
from natsort import natsorted
base = os.path.expanduser("~/Affordance2Grasp/data_hub/ProcessedData/third_depth/ho3d_v3")
seqs = natsorted(os.listdir(base))
out_dir = os.path.expanduser("~/Affordance2Grasp/data_hub/ProcessedData/third_mano/ho3d_v3")
os.makedirs(out_dir, exist_ok=True)
done = {f.replace(".npz","") for f in os.listdir(out_dir) if f.endswith(".npz")}
todo = [s for s in seqs if s not in done]
print(f"{len(seqs)} total, {len(done)} done, {len(todo)} remaining")
N = $N
shards = [[] for _ in range(N)]
for i, s in enumerate(todo):
    shards[i % N].append(s)
gpus = "$GPUS".split()
for i, sh in enumerate(shards):
    print(f"GPU{gpus[i]}: {len(sh)} seqs")
    open(f"/tmp/step2_shard_{gpus[i]}.txt", "w").write("\n".join(sh))
PY

for GPU in $GPUS; do
  CUDA_VISIBLE_DEVICES=$GPU nohup bash -c "
    while read seq; do
      [ -z \"\$seq\" ] && continue
      python data/batch_haptic.py --dataset ho3d_v3 --only-with-depth-k --seq \"\$seq\"
    done < /tmp/step2_shard_$GPU.txt
  " > output/haptic_logs/ho3d_gpu$GPU.log 2>&1 &
  echo "GPU$GPU: pid=$!"
done

wait
echo "All Step 2 HO3D processes complete."
