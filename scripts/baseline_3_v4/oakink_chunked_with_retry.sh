#!/usr/bin/env bash
# Per-OakInk-object chunked-retry wrapper for baseline_3 v4 collector.
#
# OakInk variant of v4_chunked_with_retry.sh — separate file per the
# "Oakink相关处理脚本单独文件夹" rule (see memory).
#
# Differences vs DexYCB v4_chunked_with_retry.sh:
#   - obj id is alphanumeric (A01001, S10018, ...) not 2-digit numeric
#   - source eps come from oakink retarget dir, not Baseline1/data/episodes_g/
#   - file naming uses the OakInk obj_id substring (no "ycb_dex_" prefix)
#   - N_TOTAL auto-detected from src dir (OakInk obj sessions: 2..16, varies)
#
# Same chunk/retry semantics:
#   A. chunk fully complete                       → no retry
#   B. sanity FAIL mid-chunk (EARLY-EXIT printed) → parse "remaining N eps", retry tail
#   C. process crash with no EARLY-EXIT marker    → count terminal markers vs LIMIT, retry diff
#
# Usage:
#   bash oakink_chunked_with_retry.sh OBJ_ID CHUNK_SIZE OUT_DIR
#
# Env vars (all overridable):
#   PY            python interpreter with env_isaaclab (IsaacSim 5.1 + cuRobo 0.8)
#   PROJ          UCB_Project root
#   SRC_DIR       OakInk retarget input dir
#   RESULTS_FILE  per-run results log (DONE_RETRY etc. appended here)
#   NO_YAW_AUG=1  disable yaw aug → 1 attempt/ep instead of 4
set -u

PY=${PY:-/home/accelerator/miniforge3/envs/env_isaaclab/bin/python}
PROJ=${PROJ:-/home/accelerator/UCB_Project}
SRC_DIR=${SRC_DIR:-Baseline1/data/episodes_oakink_v3_full89_cam0only_2026-05-26}
RESULTS=${RESULTS_FILE:-/tmp/oakink_results.txt}

OBJ=$1; CHUNK=${2:-5}; OUT=${3:-Baseline1/data/episodes_b3_v4_oakink}
cd "$PROJ"; mkdir -p "$OUT"

# Auto-detect total source eps for this obj (OakInk obj have varying session counts)
N_TOTAL=$(ls "${SRC_DIR}"/*__${OBJ}_*__0.hdf5 2>/dev/null | wc -l)
if [[ "$N_TOTAL" -eq 0 ]]; then
    # Fall back to looser pattern (some obj_ids have no trailing _Y_Z subseq)
    N_TOTAL=$(ls "${SRC_DIR}"/oakink__${OBJ}_*.hdf5 2>/dev/null | wc -l)
fi
if [[ "$N_TOTAL" -eq 0 ]]; then
    echo "ERROR: no source eps for $OBJ in $SRC_DIR" >&2
    exit 1
fi

if [[ "${NO_YAW_AUG:-0}" == "1" ]]; then
    ATTEMPTS_PER_EP=1
    YAW_FLAG="--no-yaw-aug"
else
    ATTEMPTS_PER_EP=4
    YAW_FLAG=""
fi

run_chunk() {
    local OBJ=$1 START=$2 LIMIT=$3 TAG=$4
    local LOG=/tmp/oakink_obj_${OBJ}_${TAG}.out
    "$PY" -u sim/run_grasp_sim_baseline3_v4.py \
        --episodes "${SRC_DIR}/*.hdf5" \
        --object "${OBJ}" --headless \
        --start "$START" --limit "$LIMIT" \
        --out-dir "$OUT" \
        $YAW_FLAG \
        > "$LOG" 2>&1

    if grep -q "EARLY-EXIT" "$LOG"; then
        REMAINING=$(grep -oE "remaining [0-9]+ eps" "$LOG" | head -1 | grep -oE "[0-9]+")
        if [[ -n "$REMAINING" && "$REMAINING" -gt 0 ]]; then
            echo "$REMAINING"; return
        fi
    fi

    MARKERS=$(grep -cE "GRASPED \+ LIFTED|object Z .* not lifted|plan sequence failed|abort ep|\[resume-skip\]" "$LOG")
    COMPLETED_EPS=$(( MARKERS / ATTEMPTS_PER_EP ))
    if [[ "$COMPLETED_EPS" -lt "$LIMIT" ]]; then
        echo $(( LIMIT - COMPLETED_EPS )); return
    fi
    echo "0"
}

echo "[$(date +%H:%M:%S)] START $OBJ  n_total=$N_TOTAL  chunk=$CHUNK" >> "$RESULTS"

for ((S=0; S<N_TOTAL; S+=CHUNK)); do
    L=$((N_TOTAL-S<CHUNK ? N_TOTAL-S : CHUNK))
    SKIPPED=$(run_chunk "$OBJ" "$S" "$L" "c${S}")
    if [[ "$SKIPPED" -gt 0 ]]; then
        SKIP_START=$((S + L - SKIPPED))
        SKIP_END=$((S + L))
        for ((R=SKIP_START; R<SKIP_END; R++)); do
            echo "  [retry] $OBJ ep $R (skipped by sanity/crash in chunk $S)" >&2
            run_chunk "$OBJ" "$R" "1" "r${R}" > /dev/null
        done
    fi
done

# OakInk file naming: oakink__<obj_id>_<subseq>__<ts>__<subj>__<cam>[_yaw{N}].hdf5
# Glob: anything with "_${OBJ}_" or "_${OBJ}__" in the filename, of obj=cid for this obj.
# Use substring match like collector does (obj_id is unique among 89 obj_ids).
SAVED_ORIG=$(ls "$OUT"/oakink__${OBJ}_*.hdf5 2>/dev/null | grep -v "_yaw" | wc -l)
SAVED_YAW=$(ls "$OUT"/oakink__${OBJ}_*_yaw*.hdf5 2>/dev/null | wc -l)
SAVED=$(( SAVED_ORIG + SAVED_YAW ))

G=$(cat /tmp/oakink_obj_${OBJ}_*.out 2>/dev/null | grep -cE "GRASPED \+ LIFTED")
NL=$(cat /tmp/oakink_obj_${OBJ}_*.out 2>/dev/null | grep -cE "object Z .* not lifted")
PF=$(cat /tmp/oakink_obj_${OBJ}_*.out 2>/dev/null | grep -c "plan sequence failed")
SAN=$(cat /tmp/oakink_obj_${OBJ}_*.out 2>/dev/null | grep -c "sanity check FAIL")
AB=$(cat /tmp/oakink_obj_${OBJ}_*.out 2>/dev/null | grep -c "abort ep")
SF=$(cat /tmp/oakink_obj_${OBJ}_*.out 2>/dev/null | grep -c "object settle FAIL")
WARN=$(cat /tmp/oakink_obj_${OBJ}_*.out 2>/dev/null | grep -ciE "Invalid PhysX")
RTRY=$(ls /tmp/oakink_obj_${OBJ}_r*.out 2>/dev/null | wc -l)
echo "[$(date +%H:%M:%S)] DONE_RETRY ${OBJ}  n_total=${N_TOTAL}  saved=${SAVED} (orig=${SAVED_ORIG} yaw=${SAVED_YAW})  GRASPED=${G}  not_lifted=${NL}  plan_fail=${PF}  sanity=${SAN}  abort=${AB}  settle_fail=${SF}  PhysX_warns=${WARN}  retries=${RTRY}" >> "$RESULTS"
