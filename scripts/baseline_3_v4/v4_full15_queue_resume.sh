#!/usr/bin/env bash
# Resume-capable 2-parallel orchestrator for baseline_3 v4 sim collection.
# Skips any object already DONE_RETRY in the results log → safe to re-run on
# a partner machine after rsync-ing partial results.
#
# Required env vars (override these for partner setup):
#   PROJ          UCB_Project root          (default: /home/accelerator/UCB_Project)
#   PY            env_isaaclab python       (default: /home/accelerator/miniforge3/envs/env_isaaclab/bin/python)
#   OUT           output dir for hdf5       (default: Baseline1/data/episodes_b3_v4_15obj_3yaw_<TODAY>)
#   RESULTS       results log path          (default: /tmp/v4_full15_results_<TODAY>.txt)
#   PAR           concurrent IsaacSim procs (default: 2). On A6000 (48 GB) you can
#                 try 3 if VRAM headroom is comfortable — but cuRobo subprocess
#                 contention historically caused 65 % plan_fail at 4-par on RTX 5090.
#                 A6000 has 50 % fewer CUDA cores so cuRobo contention may be WORSE
#                 there even with more VRAM. Default 2 is safe; bump to 3 only after
#                 watching the first 1-2 objects' plan_fail rate in RESULTS.
#
# Resume usage (partner side, after copying our data into matching paths):
#   PROJ=$HOME/UCB_Project \
#   PY=$HOME/miniconda3/envs/env_isaaclab/bin/python \
#   OUT=Baseline1/data/episodes_b3_v4_15obj_3yaw_2026-05-25 \
#   RESULTS=/tmp/v4_full15_results_2026-05-25.txt \
#   PAR=2 \
#     bash scripts/baseline_3_v4/v4_full15_queue_resume.sh
set -u

PROJ=${PROJ:-/home/accelerator/UCB_Project}
PY=${PY:-/home/accelerator/miniforge3/envs/env_isaaclab/bin/python}
PAR=${PAR:-2}
cd "$PROJ"

TODAY=$(date +%Y-%m-%d)
OUT=${OUT:-Baseline1/data/episodes_b3_v4_15obj_3yaw_${TODAY}}
mkdir -p "$OUT"

RESULTS=${RESULTS:-/tmp/v4_full15_results_${TODAY}.txt}

# RESUME safety: if OUT already has data, refuse unless RESULTS log is present too
# (so we know what's already done). Without RESULTS we'd re-collect overwriting hdf5.
if [[ -n "$(ls -A "$OUT" 2>/dev/null)" && ! -f "$RESULTS" ]]; then
    echo "ERROR: OUT=$OUT is non-empty but RESULTS=$RESULTS missing." >&2
    echo "       Either ship the matching RESULTS log alongside OUT, or empty OUT first." >&2
    exit 1
fi

# Touch RESULTS (don't truncate — we need the prior DONE_RETRY entries).
touch "$RESULTS"

# Exported for the per-object wrapper (consumed via $RESULTS_FILE there).
export RESULTS_FILE="$RESULTS"
export PY PROJ

# Object list — 15 entries; partner can comment out specific ones to skip.
OBJS=(
  06   # tuna             — collected on dev box
  04   # tomato           — collected
  07   # pudding          — collected
  08   # gelatin          — collected
  18   # marker           — collected
  09   # potted_meat      — collected
  12   # bleach
  05   # mustard
  03   # sugar
  15   # drill
  02   # cracker_box
  11   # pitcher_base
  14   # mug
  19   # large_clamp      — NOTE: no DexYCB raw → src ep dir has 0 ep → collector silently emits nothing
  20   # extra_large_clamp
)

GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Idempotent MANIFEST refresh.
cat > "$OUT/MANIFEST.md" <<EOF
# baseline_3 v4 collection run — last update $(date "+%Y-%m-%d %H:%M:%S %Z")

- git: $GIT_SHA
- collector: sim/run_grasp_sim_baseline3_v4.py
- wrapper:   scripts/baseline_3_v4/v4_chunked_with_retry.sh (chunk-5, retry)
- orchestrator: scripts/baseline_3_v4/v4_full15_queue_resume.sh (RESUME-aware, 2-parallel)
- yaw config: 3 yaws/source (orig + 90 + 180 + 270 = 4 attempts/ep)
- object mass: 0.05 kg (hardcoded in collector, see Baseline1/RETRAIN_V4_FULL12.md history)
- objects (${#OBJS[@]}): $(IFS=,; echo "${OBJS[*]}")
- results log: $RESULTS

File naming inside OUT:
  <source_ep>__ycb_dex_NN.hdf5          original yaw
  <source_ep>__ycb_dex_NN_yaw{90,180,270}.hdf5   augmented
EOF

echo "[$(date +%H:%M:%S)] RESUME_START 15 obj 2-parallel chunk-5 yaw=all3  OUT=$OUT  GIT=$GIT_SHA" >> "$RESULTS"

launched=0; skipped=0
for NN in "${OBJS[@]}"; do
    if grep -q "DONE_RETRY ycb_dex_${NN} " "$RESULTS" 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] SKIP ycb_dex_${NN} (already DONE_RETRY)" >> "$RESULTS"
        skipped=$((skipped + 1))
        continue
    fi

    # Clean stale per-object logs so wrapper's stats aren't contaminated.
    rm -f /tmp/v4_obj_${NN}_*.out /tmp/v4_orchestrator_${NN}.out

    # Concurrency cap (default 2; override via PAR=N).
    while [[ $(jobs -rp | wc -l) -ge $PAR ]]; do sleep 10; done
    echo "[$(date +%H:%M:%S)] LAUNCH ycb_dex_${NN}" >> "$RESULTS"
    bash scripts/baseline_3_v4/v4_chunked_with_retry.sh "$NN" 5 "$OUT" \
        > /tmp/v4_orchestrator_${NN}.out 2>&1 &
    launched=$((launched + 1))
done
wait

echo "[$(date +%H:%M:%S)] QUEUE_RESUME_DONE  launched=${launched}  skipped=${skipped}" >> "$RESULTS"

{
echo ""
echo "=== FINAL PER-OBJ SUMMARY (after resume) ==="
for NN in "${OBJS[@]}"; do
    O=$(ls "$OUT"/*ycb_dex_${NN}.hdf5 2>/dev/null | wc -l)
    Y=$(ls "$OUT"/*ycb_dex_${NN}_yaw*.hdf5 2>/dev/null | wc -l)
    echo "  ycb_dex_${NN}: orig=${O}  yaw=${Y}  total=$((O+Y))"
done
T=$(ls "$OUT"/*.hdf5 2>/dev/null | wc -l)
echo "  TOTAL: $T trajectories"
} >> "$RESULTS"
