#!/usr/bin/env bash
# Resume-capable 2-parallel orchestrator for OakInk 89-obj baseline_3 v4 sim collection.
# OakInk variant of v4_full15_queue_resume.sh.
#
# Skips any object already DONE_RETRY in the results log → safe to re-run after
# crash / rsync / partial completion.
#
# Required env vars (override as needed):
#   PROJ          UCB_Project root          (default: /home/accelerator/UCB_Project)
#   PY            env_isaaclab python       (default: /home/accelerator/miniforge3/envs/env_isaaclab/bin/python)
#   SRC_DIR       OakInk retarget input dir (default: Baseline1/data/episodes_oakink_v3_full74_pinch-middle_cam0_2026-05-26)
#   OUT           output dir for hdf5       (default: Baseline1/data/episodes_b3_v4_oakink89_<TODAY>)
#   RESULTS       results log path          (default: /tmp/oakink_full89_results_<TODAY>.txt)
#   PAR           concurrent IsaacSim procs (default: 2 — same risk profile as DexYCB on 5090)
#
# Usage:
#   bash scripts/baseline_3_v4/oakink_full89_queue_resume.sh
set -u

PROJ=${PROJ:-/home/accelerator/UCB_Project}
PY=${PY:-/home/accelerator/miniforge3/envs/env_isaaclab/bin/python}
SRC_DIR=${SRC_DIR:-Baseline1/data/episodes_oakink_v3_full74_pinch-middle_cam0_2026-05-26}
PAR=${PAR:-2}
cd "$PROJ"

TODAY=$(date +%Y-%m-%d)
OUT=${OUT:-Baseline1/data/episodes_b3_v4_oakink89_${TODAY}}
mkdir -p "$OUT"

RESULTS=${RESULTS:-/tmp/oakink_full89_results_${TODAY}.txt}

if [[ -n "$(ls -A "$OUT" 2>/dev/null)" && ! -f "$RESULTS" ]]; then
    echo "ERROR: OUT=$OUT is non-empty but RESULTS=$RESULTS missing." >&2
    echo "       Either ship the matching RESULTS log alongside OUT, or empty OUT first." >&2
    exit 1
fi

touch "$RESULTS"
export RESULTS_FILE="$RESULTS"
export PY PROJ SRC_DIR

# Object list: 89 OakInk obj_ids from class_id_map.json (use=true), sorted alphabetically.
# Auto-generated from manifest to avoid drift; if manifest expands, just re-run this script
# (already-collected obj are skipped via DONE_RETRY).
OBJS=( $("$PY" -c "
import json
m = json.load(open('Baseline1/oakink/class_id_map.json'))
print(' '.join(sorted(o for o,i in m['objects'].items() if i.get('use'))))
") )

if [[ ${#OBJS[@]} -ne 74 ]]; then
    echo "WARNING: manifest has ${#OBJS[@]} use=true objects (expected 74)" >&2
fi

GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# MANIFEST.md (overwritten each run with up-to-date metadata)
cat > "$OUT/MANIFEST.md" <<EOF
# OakInk baseline_3 v4 collection — last update $(date "+%Y-%m-%d %H:%M:%S %Z")

- git: $GIT_SHA
- collector: sim/run_grasp_sim_baseline3_v4.py
- per-obj wrapper:   scripts/baseline_3_v4/oakink_chunked_with_retry.sh (chunk-5, retry)
- orchestrator:      scripts/baseline_3_v4/oakink_full89_queue_resume.sh (resume-aware, ${PAR}-parallel)
- input dir:  $SRC_DIR  ($(ls $SRC_DIR/*.hdf5 2>/dev/null | wc -l) source ep, cam=0 + subj=0/1)
- yaw config: 3 yaws/source (orig + 90 + 180 + 270 = 4 attempts/ep)
- object mass: 0.05 kg (hardcoded in collector)
- objects (${#OBJS[@]}): see Baseline1/oakink/class_id_map.json (use=true)
- results log: $RESULTS

File naming inside OUT:
  oakink__<obj_id>_<sub>__<ts>__<subj>__<cam>.hdf5          original yaw
  oakink__<obj_id>_<sub>__<ts>__<subj>__<cam>_yaw{90,180,270}.hdf5  augmented
EOF

echo "[$(date +%H:%M:%S)] RESUME_START ${#OBJS[@]} obj ${PAR}-parallel chunk-5 yaw=all3  OUT=$OUT  GIT=$GIT_SHA  SRC=$SRC_DIR" >> "$RESULTS"

launched=0; skipped=0
for OBJ in "${OBJS[@]}"; do
    if grep -q "DONE_RETRY ${OBJ} " "$RESULTS" 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] SKIP ${OBJ} (already DONE_RETRY)" >> "$RESULTS"
        skipped=$((skipped + 1))
        continue
    fi

    # Clean stale per-object logs
    rm -f /tmp/oakink_obj_${OBJ}_*.out /tmp/oakink_orchestrator_${OBJ}.out

    # Concurrency cap
    while [[ $(jobs -rp | wc -l) -ge $PAR ]]; do sleep 10; done
    echo "[$(date +%H:%M:%S)] LAUNCH ${OBJ}" >> "$RESULTS"
    bash scripts/baseline_3_v4/oakink_chunked_with_retry.sh "$OBJ" 5 "$OUT" \
        > /tmp/oakink_orchestrator_${OBJ}.out 2>&1 &
    launched=$((launched + 1))
done
wait

echo "[$(date +%H:%M:%S)] QUEUE_RESUME_DONE  launched=${launched}  skipped=${skipped}" >> "$RESULTS"

{
echo ""
echo "=== FINAL PER-OBJ SUMMARY ($(date "+%Y-%m-%d %H:%M:%S")) ==="
for OBJ in "${OBJS[@]}"; do
    O=$(ls "$OUT"/oakink__${OBJ}_*.hdf5 2>/dev/null | grep -v "_yaw" | wc -l)
    Y=$(ls "$OUT"/oakink__${OBJ}_*_yaw*.hdf5 2>/dev/null | wc -l)
    printf "  %-8s orig=%2d  yaw=%2d  total=%2d\n" "$OBJ" "$O" "$Y" "$((O+Y))"
done
T=$(ls "$OUT"/oakink__*.hdf5 2>/dev/null | wc -l)
echo "  TOTAL: $T trajectories"

echo ""
echo "=== AGGREGATE FAILURE BREAKDOWN (sum of all DONE_RETRY rows in $RESULTS) ==="
awk '
/DONE_RETRY/ {
    for (i=1; i<=NF; i++) {
        split($i, kv, "=")
        if (kv[1] ~ /^(n_total|saved|GRASPED|not_lifted|plan_fail|sanity|abort|settle_fail|PhysX_warns|retries)$/) {
            sum[kv[1]] += kv[2]
        }
    }
    n_obj++
}
END {
    printf "  obj completed:      %d\n", n_obj
    printf "  total attempted:    %d (n_total summed)\n", sum["n_total"]
    printf "  saved (success):    %d\n", sum["saved"]
    printf "  GRASPED markers:    %d\n", sum["GRASPED"]
    printf "  not_lifted:         %d  (Z Δ < 3cm)\n", sum["not_lifted"]
    printf "  plan_fail:          %d  (cuRobo could not plan a path)\n", sum["plan_fail"]
    printf "  sanity_fail:        %d  (parent process PhysX-poisoned)\n", sum["sanity"]
    printf "  abort:              %d  (collector early-exit)\n", sum["abort"]
    printf "  settle_fail:        %d  (obj drift >30cm after spawn)\n", sum["settle_fail"]
    printf "  PhysX_warns:        %d  (Invalid PhysX warnings)\n", sum["PhysX_warns"]
    printf "  retries:            %d  (chunked-wrapper retries)\n", sum["retries"]
    if (sum["n_total"]>0) printf "  success rate:       %.1f%%  (%d / %d)\n", 100.0*sum["saved"]/sum["n_total"], sum["saved"], sum["n_total"]
}' "$RESULTS"
} >> "$RESULTS"
