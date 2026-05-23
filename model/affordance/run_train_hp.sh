#!/usr/bin/env bash
# Human-prior supervision training (MSE only, PointNet2SegOnly, no FC head).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

python model/train_affordance_hp.py \
  --run-group hp_supervision \
  --run-name hp_mse_full \
  --gpus "${GPU:-0}" \
  --epochs "${EPOCHS:-200}" \
  --patience "${PATIENCE:-30}" \
  --batch_size "${BATCH_SIZE:-64}" \
  --lr 3e-4 \
  --lr-min 1e-5 \
  --warmup-epochs 0 \
  --weight_decay 1e-4 \
  --val_ratio 0.15 \
  --split_seed 42 \
  --augment-mode "${AUG:-full}" \
  --hp-threshold 0.5 \
  --val-vis-max-objects 10 \
  --train-vis-max-objects 10 \
  --head-norm none \
  --dataset_dir output/affordance_no_rot_executed \
  "$@"
