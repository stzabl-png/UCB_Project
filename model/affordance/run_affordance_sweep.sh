#!/usr/bin/env bash
set -uo pipefail

# ============================================================
# Full-training sweep for PointNet++ affordance prediction
# Location:
#   root_dir/model/affordance/run_affordance_sweep.sh
#
# This script automatically cd's back to root_dir.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

echo "Running from repo root: ${ROOT_DIR}"

# ============================================================
# User-configurable environment variables
# ============================================================
GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-64}"
AUG="${AUG:-weak}"
GROUP="${GROUP:-overnight_affordance_sweep_$(date +%Y%m%d_%H%M%S)}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

FAILED_RUNS=()

COMMON_ARGS=(
  --epochs "${EPOCHS}"
  --patience "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"

  --lr 3e-4
  --lr-min 1e-5
  --weight_decay 1e-4
  --warmup-epochs 0

  --val_ratio 0.15
  --split_seed 42

  --augment-mode "${AUG}"
  --val-vis-max-objects 10

  --heatmap-sigma-ratio 0.03

  --binary-tversky-alpha 0.7
  --binary-tversky-beta 0.3
  --binary-neg-ratio 2
  --soft-background-weight 0.5

  --lambda-binary 1.0
  --lambda-aff 0.3
  --lambda-peak 0.0
  --lambda-center-heatmap 0.0
  --lambda-center-head 0.0
  --lambda-consistency 0.0
  --lambda-smooth 0.0

  --gpus "${GPU}"
)

run_exp () {
  local NAME="$1"
  shift

  echo
  echo "============================================================"
  echo "Running: ${NAME}"
  echo "Group:   ${GROUP}"
  echo "AUG:     ${AUG}"
  echo "EPOCHS:  ${EPOCHS}"
  echo "GPU:     ${GPU}"
  echo "Extra:   $*"
  echo "============================================================"
  echo

  if python model/train_affordance.py \
      --run-group "${GROUP}" \
      --run-name "${NAME}" \
      "${COMMON_ARGS[@]}" \
      "$@"; then
    echo
    echo "✅ Finished: ${NAME}"
  else
    echo
    echo "❌ Failed: ${NAME}"
    FAILED_RUNS+=("${NAME}")
    if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
      exit 1
    fi
  fi
}

# ============================================================
# Stage A: architecture ablation
# Fixed stable loss:
#   L = 1.0 * L_binary + 0.3 * L_aff
#   no peak, no center loss
#
# Tests:
#   FC branch off/on
#   head normalization none/groupnorm
# ============================================================

run_exp "A01_noFC_normNone_base" \
  --head-norm none

run_exp "A02_noFC_normGN_base" \
  --head-norm groupnorm

run_exp "A03_FCbranch_normNone_base" \
  --predict-force-center \
  --head-norm none

run_exp "A04_FCbranch_normGN_base" \
  --predict-force-center \
  --head-norm groupnorm


# ============================================================
# Stage B: lambda_aff sweep
#
# Purpose:
#   Test how much soft heatmap supervision helps.
#   Lower lambda_aff may be sharper/more binary.
#   Higher lambda_aff may be smoother but may increase FP.
# ============================================================

run_exp "B01_noFC_normNone_aff01" \
  --head-norm none \
  --lambda-aff 0.1

run_exp "B02_noFC_normGN_aff01" \
  --head-norm groupnorm \
  --lambda-aff 0.1

run_exp "B03_noFC_normNone_aff05" \
  --head-norm none \
  --lambda-aff 0.5

run_exp "B04_noFC_normGN_aff05" \
  --head-norm groupnorm \
  --lambda-aff 0.5

run_exp "B05_noFC_normNone_aff08" \
  --head-norm none \
  --lambda-aff 0.8

run_exp "B06_noFC_normGN_aff08" \
  --head-norm groupnorm \
  --lambda-aff 0.8


# ============================================================
# Stage C: negative sampling ratio sweep
#
# Purpose:
#   binary-neg-ratio controls negative pressure in binary loss.
#   neg=1 gives higher recall but more FP.
#   neg=3 gives stronger FP control but may suppress positives.
# ============================================================

run_exp "C01_noFC_normNone_neg1" \
  --head-norm none \
  --binary-neg-ratio 1

run_exp "C02_noFC_normGN_neg1" \
  --head-norm groupnorm \
  --binary-neg-ratio 1

run_exp "C03_noFC_normNone_neg3" \
  --head-norm none \
  --binary-neg-ratio 3

run_exp "C04_noFC_normGN_neg3" \
  --head-norm groupnorm \
  --binary-neg-ratio 3


# ============================================================
# Stage D: Tversky FP/FN tradeoff
#
# Purpose:
#   alpha controls FP penalty.
#   beta controls FN penalty.
#
# Current default:
#   alpha=0.7, beta=0.3
#
# More strict:
#   alpha=0.8, beta=0.2
# ============================================================

run_exp "D01_noFC_normNone_tversky08_02" \
  --head-norm none \
  --binary-tversky-alpha 0.8 \
  --binary-tversky-beta 0.2

run_exp "D02_noFC_normGN_tversky08_02" \
  --head-norm groupnorm \
  --binary-tversky-alpha 0.8 \
  --binary-tversky-beta 0.2


# ============================================================
# Stage E: soft-background-weight sweep
#
# Purpose:
#   Higher soft-background-weight makes soft heatmap loss punish background more.
#   This may reduce FP but can make training conservative.
# ============================================================

run_exp "E01_noFC_normNone_softBG025" \
  --head-norm none \
  --soft-background-weight 0.25

run_exp "E02_noFC_normGN_softBG025" \
  --head-norm groupnorm \
  --soft-background-weight 0.25

run_exp "E03_noFC_normNone_softBG100" \
  --head-norm none \
  --soft-background-weight 1.0

run_exp "E04_noFC_normGN_softBG100" \
  --head-norm groupnorm \
  --soft-background-weight 1.0


# ============================================================
# Stage F: small peak loss sweep
#
# Purpose:
#   Previous large peak caused all-positive tendency.
#   Here we test very small positive-only peak weights.
#   If it helps recall without collapse, it may be useful.
# ============================================================

run_exp "F01_noFC_normNone_peak002" \
  --head-norm none \
  --lambda-peak 0.02

run_exp "F02_noFC_normGN_peak002" \
  --head-norm groupnorm \
  --lambda-peak 0.02

run_exp "F03_noFC_normNone_peak005" \
  --head-norm none \
  --lambda-peak 0.05

run_exp "F04_noFC_normGN_peak005" \
  --head-norm groupnorm \
  --lambda-peak 0.05


# ============================================================
# Stage G: FC branch + supervised FC head
#
# Purpose:
#   Test whether auxiliary FC head supervision improves representation.
#
# Important:
#   --predict-force-center controls architecture.
#   --lambda-center-head controls supervision.
#   Heatmap-center loss remains off here.
# ============================================================

run_exp "G01_FChead_normNone_lfc001" \
  --predict-force-center \
  --head-norm none \
  --lambda-center-head 0.01

run_exp "G02_FChead_normGN_lfc001" \
  --predict-force-center \
  --head-norm groupnorm \
  --lambda-center-head 0.01

run_exp "G03_FChead_normNone_lfc005" \
  --predict-force-center \
  --head-norm none \
  --lambda-center-head 0.05

run_exp "G04_FChead_normGN_lfc005" \
  --predict-force-center \
  --head-norm groupnorm \
  --lambda-center-head 0.05


# ============================================================
# Stage H: heatmap-derived center loss, exploratory
#
# Purpose:
#   Test whether forcing the predicted affordance center toward fc_gt helps.
#
# Warning:
#   This can distort heatmaps if fc_gt is noisy or not the same as contact-region centroid.
#   Keep weight small.
# ============================================================

run_exp "H01_noFC_normNone_centerHM005" \
  --head-norm none \
  --lambda-center-heatmap 0.05

run_exp "H02_noFC_normGN_centerHM005" \
  --head-norm groupnorm \
  --lambda-center-heatmap 0.05

run_exp "H03_FCbranch_normGN_centerHM005" \
  --predict-force-center \
  --head-norm groupnorm \
  --lambda-center-heatmap 0.05


# ============================================================
# Summary
# ============================================================

echo
echo "============================================================"
echo "All scheduled experiments finished."
echo "Group: ${GROUP}"
echo "Results should be under:"
echo "  output/affordance_no_rot_executed/training_runs/${GROUP}/"
echo "============================================================"

if [[ "${#FAILED_RUNS[@]}" -gt 0 ]]; then
  echo
  echo "Some runs failed:"
  for r in "${FAILED_RUNS[@]}"; do
    echo "  - ${r}"
  done
  echo
  exit 1
else
  echo
  echo "All runs completed successfully."
fi