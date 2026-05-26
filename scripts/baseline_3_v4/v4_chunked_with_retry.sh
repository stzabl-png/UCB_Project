#!/usr/bin/env bash
# Per-object chunked-retry wrapper for baseline_3 v4 collector.
#
# Each ep with --yaw-aug ON produces 4 attempts (orig + yaw90/180/270),
# each printing exactly ONE terminal marker ("GRASPED + LIFTED" /
# "object Z … not lifted" / "plan sequence failed" / "abort ep").
#
# Handles 3 chunk-end cases:
#   A. chunk fully complete                       → no retry
#   B. sanity FAIL mid-chunk (EARLY-EXIT printed) → parse "remaining N eps", retry the tail
#   C. process crash with no EARLY-EXIT marker    → count terminal markers vs LIMIT, retry diff
#
# Usage:
#   bash v4_chunked_with_retry.sh OBJ_CODE CHUNK_SIZE OUT_DIR
#
# Env vars (all overridable):
#   PY            python interpreter with env_isaaclab (IsaacSim 5.1 + cuRobo 0.8)
#   PROJ          UCB_Project root
#   RESULTS_FILE  per-run results log (DONE_RETRY etc. appended here)
#   N_TOTAL       total source eps in episodes_g/ per object (default 50)
#   NO_YAW_AUG=1  disable yaw aug → 1 attempt/ep instead of 4
set -u

# --- Configurable paths (override via env) -----------------------------------
PY=${PY:-/home/accelerator/miniforge3/envs/env_isaaclab/bin/python}
PROJ=${PROJ:-/home/accelerator/UCB_Project}
RESULTS=${RESULTS_FILE:-/tmp/v4_results.txt}
N_TOTAL=${N_TOTAL:-50}
# -----------------------------------------------------------------------------

NN=$1; CHUNK=${2:-5}; OUT=${3:-Baseline1/data/episodes_b3_v4_2par_c5}
cd "$PROJ"; mkdir -p "$OUT"

# v4 collector default --yaw-aug=True → 4 attempts/ep (orig + 3 yaws).
# Pass NO_YAW_AUG=1 to use 1 attempt/ep instead.
if [[ "${NO_YAW_AUG:-0}" == "1" ]]; then
    ATTEMPTS_PER_EP=1
    YAW_FLAG="--no-yaw-aug"
else
    ATTEMPTS_PER_EP=4
    YAW_FLAG=""   # default ON, no flag needed
fi

run_chunk() {
    local NN=$1 START=$2 LIMIT=$3 TAG=$4
    local LOG=/tmp/v4_obj_${NN}_${TAG}.out
    "$PY" -u sim/run_grasp_sim_baseline3_v4.py \
        --object "ycb_dex_${NN}" --headless \
        --start "$START" --limit "$LIMIT" \
        --out-dir "$OUT" \
        $YAW_FLAG \
        > "$LOG" 2>&1

    # case B: sanity FAIL → parse "remaining N eps" emitted by collector
    if grep -q "EARLY-EXIT" "$LOG"; then
        REMAINING=$(grep -oE "remaining [0-9]+ eps" "$LOG" | head -1 | grep -oE "[0-9]+")
        if [[ -n "$REMAINING" && "$REMAINING" -gt 0 ]]; then
            echo "$REMAINING"
            return
        fi
    fi

    # case C: process died without sanity (OOM/crash/kernel-kill).
    # Each (ep, yaw attempt) prints ONE terminal marker. Compute MISSING by:
    #   completed_eps = floor(markers / ATTEMPTS_PER_EP)
    #   MISSING       = LIMIT - completed_eps
    MARKERS=$(grep -cE "GRASPED \+ LIFTED|object Z .* not lifted|plan sequence failed|abort ep" "$LOG")
    COMPLETED_EPS=$(( MARKERS / ATTEMPTS_PER_EP ))
    if [[ "$COMPLETED_EPS" -lt "$LIMIT" ]]; then
        MISSING=$(( LIMIT - COMPLETED_EPS ))
        echo "$MISSING"
        return
    fi

    # case A: clean finish
    echo "0"
}

for ((S=0; S<N_TOTAL; S+=CHUNK)); do
    L=$((N_TOTAL-S<CHUNK ? N_TOTAL-S : CHUNK))
    SKIPPED=$(run_chunk "$NN" "$S" "$L" "c${S}")
    if [[ "$SKIPPED" -gt 0 ]]; then
        SKIP_START=$((S + L - SKIPPED))
        SKIP_END=$((S + L))
        for ((R=SKIP_START; R<SKIP_END; R++)); do
            echo "  [retry] $NN ep $R (skipped by sanity/crash in chunk $S)" >&2
            run_chunk "$NN" "$R" "1" "r${R}" > /dev/null
        done
    fi
done

# Aggregate stats across ALL logs for this object (chunks + retries) for the
# DONE_RETRY line. Note: orig + yaw variants both counted by glob below.
SAVED_ORIG=$(ls "$OUT"/*ycb_dex_${NN}.hdf5 2>/dev/null | wc -l)
SAVED_YAW=$(ls "$OUT"/*ycb_dex_${NN}_yaw*.hdf5 2>/dev/null | wc -l)
SAVED=$(( SAVED_ORIG + SAVED_YAW ))

G=$(cat /tmp/v4_obj_${NN}_*.out 2>/dev/null | grep -cE "GRASPED \+ LIFTED")
NL=$(cat /tmp/v4_obj_${NN}_*.out 2>/dev/null | grep -cE "object Z .* not lifted")
PF=$(cat /tmp/v4_obj_${NN}_*.out 2>/dev/null | grep -c "plan sequence failed")
SAN=$(cat /tmp/v4_obj_${NN}_*.out 2>/dev/null | grep -c "sanity check FAIL")
AB=$(cat /tmp/v4_obj_${NN}_*.out 2>/dev/null | grep -c "abort ep")
SF=$(cat /tmp/v4_obj_${NN}_*.out 2>/dev/null | grep -c "object settle FAIL")
WARN=$(cat /tmp/v4_obj_${NN}_*.out 2>/dev/null | grep -ciE "Invalid PhysX")
RTRY=$(ls /tmp/v4_obj_${NN}_r*.out 2>/dev/null | wc -l)
echo "[$(date +%H:%M:%S)] DONE_RETRY ycb_dex_${NN}  saved=${SAVED} (orig=${SAVED_ORIG} yaw=${SAVED_YAW})  GRASPED=${G}  not_lifted=${NL}  plan_fail=${PF}  sanity=${SAN}  abort=${AB}  settle_fail=${SF}  PhysX_warns=${WARN}  retries=${RTRY}" >> "$RESULTS"
