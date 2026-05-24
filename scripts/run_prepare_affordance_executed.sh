#!/usr/bin/env bash
# Run in tmux deployment: bash scripts/run_prepare_affordance_executed.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bundlesdf
LOG_DIR=output/affordance_no_rot_executed/logs
mkdir -p "$LOG_DIR"
echo "=== prepare_affordance_executed $(date -Iseconds) ===" | tee "$LOG_DIR/prepare.log"
WORKERS="${WORKERS:-8}"
python3 tools/prepare_affordance_executed.py \
  --workers "$WORKERS" \
  --write-split \
  --qc-vis \
  --qc-vis-oakink 6 \
  --qc-vis-dexycb 6 \
  --qc-vis-max 16 \
  2>&1 | tee -a "$LOG_DIR/prepare.log"
echo "=== done $(date -Iseconds) ===" | tee -a "$LOG_DIR/prepare.log"
