#!/usr/bin/env bash
# Batch DP3 eval — handles multi-class episode sets safely.
#
# WHY: sim/eval_dp3_baseline3.py loads ONE obj USD per process (IsaacSim PhysX
# refuses to swap rigid-body prims mid-run: "Failed to get rigid body velocities
# from backend"). So we group eps by (dataset, obj_id, ycb_class_id) and launch
# one eval subprocess per group, sharing the same DP3 inference server.
#
# Usage:
#   bash scripts/eval_combined/04_batch_eval.sh \
#       --ckpt <path> \
#       --episodes-glob '<pattern>' \
#       --tag <name> \
#       [--n-per-class N | --total N] \
#       [--max-chunks 5] [--port 8765]
#
# Examples:
#   # eval new ckpt on 16 DexYCB + 16 OakInk train eps
#   bash scripts/eval_combined/04_batch_eval.sh \
#       --ckpt Baseline1/dp3_runs/combined_dexycb162_oakink207/epoch=2000-*.ckpt \
#       --episodes-glob '/tmp/eval_combined_train_dexycb16/*.hdf5' \
#       --tag e2000_dexycb16
#   bash scripts/eval_combined/04_batch_eval.sh \
#       --ckpt Baseline1/dp3_runs/combined_dexycb162_oakink207/epoch=2000-*.ckpt \
#       --episodes-glob '/tmp/eval_combined_train_oakink16/*.hdf5' \
#       --tag e2000_oakink16
#
# Outputs (one per tag):
#   output/dp3_eval_combined_<tag>/per_class/<obj_label>/eval_<ts>.json
#   output/dp3_eval_combined_<tag>/summary.json        ← aggregated
#   replay_video_check/eval_combined_<tag>/<obj_label>/*.mp4
#   /tmp/dp3_batch_eval_<tag>.log
set -euo pipefail

# ---- args ----
CKPT=""; GLOB=""; TAG=""; MAX_CHUNKS=5; PORT=8765
N_PER_CLASS=""; TOTAL=""
PAR=1; RESUME=0; N_SERVERS=1
while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt)           CKPT=$2; shift 2;;
        --episodes-glob)  GLOB=$2; shift 2;;
        --tag)            TAG=$2; shift 2;;
        --n-per-class)    N_PER_CLASS=$2; shift 2;;
        --total)          TOTAL=$2; shift 2;;
        --max-chunks)     MAX_CHUNKS=$2; shift 2;;
        --port)           PORT=$2; shift 2;;
        --par)            PAR=$2; shift 2;;
        --n-servers)      N_SERVERS=$2; shift 2;;
        --resume)         RESUME=1; shift 1;;
        *) echo "unknown arg: $1" >&2; exit 1;;
    esac
done
[[ -z "$CKPT" || -z "$GLOB" || -z "$TAG" ]] && {
    echo "usage: --ckpt <path> --episodes-glob <pattern> --tag <name>" >&2
    exit 1
}
CKPT=$(ls $CKPT 2>/dev/null | head -1)   # expand globs in CKPT
[[ -f "$CKPT" ]] || { echo "ERROR: ckpt not found: $CKPT" >&2; exit 1; }

PROJ=${PROJ:-/home/accelerator/UCB_Project}
PY_DP3=${PY_DP3:-/home/accelerator/miniforge3/envs/dp3/bin/python}
PY_ISAAC=${PY_ISAAC:-/home/accelerator/miniforge3/envs/env_isaaclab/bin/python}
cd "$PROJ"

RESULT_DIR="output/dp3_eval_combined_${TAG}"
VIDEO_ROOT="replay_video_check/eval_combined_${TAG}"
LOG="/tmp/dp3_batch_eval_${TAG}.log"
PERCLASS_DIR="$RESULT_DIR/per_class"
STAGE_ROOT="/tmp/dp3_batch_eval_${TAG}_stage"
if [[ "$RESUME" -eq 1 && -d "$PERCLASS_DIR" ]]; then
    echo "==> RESUME mode: preserving existing $PERCLASS_DIR (will skip classes with eval_*.json)"
else
    rm -rf "$RESULT_DIR" "$VIDEO_ROOT" "$STAGE_ROOT"
fi
mkdir -p "$PERCLASS_DIR" "$VIDEO_ROOT" "$STAGE_ROOT"

# ---- 1. group eps by (dataset, obj_id, cid) ----
echo "==> grouping eps from glob: $GLOB"
"$PY_DP3" - <<EOF
import h5py, glob, os, shutil, json, random
from collections import defaultdict
files = sorted(glob.glob("$GLOB"))
print(f"  found {len(files)} hdf5")
groups = defaultdict(list)
for f in files:
    with h5py.File(f, "r") as h:
        cid = int(h.attrs["ycb_class_id"])
        ds  = str(h.attrs.get("dataset", "dexycb"))
        obj_id = str(h.attrs.get("obj_id", ""))
    if ds == "oakink":
        label = f"oakink_{obj_id}"
    else:
        label = f"dexycb_{cid:02d}"
    groups[label].append(f)
# Stage per-class
random.seed(0)
manifest = {}
n_per_class = ${N_PER_CLASS:-0}
total = ${TOTAL:-0}
for label, fs in sorted(groups.items()):
    if n_per_class > 0:
        fs = random.sample(fs, min(n_per_class, len(fs)))
    d = "$STAGE_ROOT/" + label
    os.makedirs(d, exist_ok=True)
    for f in fs: shutil.copy(f, d)
    manifest[label] = len(fs)
# apply --total budget (proportionally truncate)
if total > 0:
    cur_total = sum(manifest.values())
    if cur_total > total:
        # randomly drop to fit budget
        all_staged = []
        for label in manifest:
            d = "$STAGE_ROOT/" + label
            all_staged.extend([(label, os.path.join(d, n)) for n in os.listdir(d)])
        keep = set(random.sample(all_staged, total))
        for label, path in all_staged:
            if (label, path) not in keep: os.remove(path)
        manifest = {l: len(os.listdir("$STAGE_ROOT/"+l)) for l in manifest if os.listdir("$STAGE_ROOT/"+l)}
print(f"  → {len(manifest)} classes, {sum(manifest.values())} total eps")
for l, n in sorted(manifest.items()):
    print(f"    {l}: {n} ep")
with open("$STAGE_ROOT/manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
EOF

# ---- 2. start N DP3 inference servers (background, round-robin distribution) ----
SERVER_PIDS=()
SERVER_PORTS=()
for ((s=0; s<N_SERVERS; s++)); do
    SPORT=$((PORT + s))
    SLOG="/tmp/dp3_server_${TAG}_p${SPORT}.log"
    echo "==> starting DP3 server #$s (port $SPORT) → $SLOG"
    setsid nohup "$PY_DP3" Baseline1/eval/dp3_inference_server.py \
        --ckpt "$CKPT" --port "$SPORT" \
        > "$SLOG" 2>&1 < /dev/null &
    SPID=$!
    disown $SPID
    SERVER_PIDS+=($SPID)
    SERVER_PORTS+=($SPORT)
    echo "  pid=$SPID"
done

cleanup() {
    for pid in "${SERVER_PIDS[@]}"; do
        echo "==> killing server pgrp=-$pid"
        kill -TERM -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 2
    for pid in "${SERVER_PIDS[@]}"; do
        kill -KILL -"$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    done
    # belt-and-suspenders: sweep stragglers matching any of our ports
    for sport in "${SERVER_PORTS[@]}"; do
        pkill -f "dp3_inference_server.py.*--port $sport" 2>/dev/null || true
    done
}
trap cleanup EXIT

# Wait for /info on each server then warmup /predict on each
for sport in "${SERVER_PORTS[@]}"; do
    echo "  waiting for /info on $sport ..."
    for i in $(seq 1 60); do
        if curl -s "http://127.0.0.1:$sport/info" > /dev/null 2>&1; then
            echo "  ✓ /info ready on $sport (${i}*2s)"
            break
        fi
        sleep 2
        if [[ $i -eq 60 ]]; then
            echo "ERROR: server /info on $sport not ready in 120s" >&2
            tail -20 "/tmp/dp3_server_${TAG}_p${sport}.log" >&2
            exit 1
        fi
    done
    # warmup /predict to catch GPU-dead servers
    echo "  warmup /predict on $sport ..."
    if ! "$PY_DP3" - <<EOF
import requests, numpy as np, sys
pc = np.zeros((2, 4096, 3), dtype=np.float32).tolist()
ap = np.zeros((2, 8), dtype=np.float32).tolist()
try:
    r = requests.post("http://127.0.0.1:$sport/predict", json={"point_cloud":pc, "agent_pos":ap}, timeout=60)
    assert r.status_code == 200, f"status={r.status_code} body={r.text[:200]}"
    j = r.json()
    assert "action" in j, f"missing action key: {list(j.keys())}"
    print(f"  ✓ warmup OK on $sport — got action shape {np.array(j['action']).shape}")
except Exception as e:
    print(f"  ✗ warmup FAIL on $sport: {e}", file=sys.stderr)
    sys.exit(1)
EOF
    then
        echo "ERROR: server warmup /predict on $sport failed" >&2
        tail -20 "/tmp/dp3_server_${TAG}_p${sport}.log" >&2
        exit 1
    fi
done
echo "  ✓ all $N_SERVERS servers ready (ports: ${SERVER_PORTS[*]})"

# ---- 3. per-class eval (parallel-N, resume-aware) ----
echo "==> running per-class evals → $RESULT_DIR  (par=$PAR, resume=$RESUME)"

eval_one_class() {
    local CLASS_DIR=$1
    local LABEL=$(basename "$CLASS_DIR")
    local N=$(ls "$CLASS_DIR"/*.hdf5 2>/dev/null | wc -l)
    [[ "$N" -eq 0 ]] && return

    local CLASS_RESULT_DIR="$PERCLASS_DIR/$LABEL"
    local CLASS_VIDEO_DIR="$VIDEO_ROOT/$LABEL"

    # Resume: skip if any eval_*.json already exists
    if [[ "$RESUME" -eq 1 ]] && ls "$CLASS_RESULT_DIR"/eval_*.json > /dev/null 2>&1; then
        local J=$(ls -t "$CLASS_RESULT_DIR"/eval_*.json | head -1)
        local STR=$("$PY_DP3" -c "import json; d=json.load(open('$J')); print(f'{d[\"n_success\"]}/{d[\"n_total\"]}')" 2>/dev/null)
        echo "  ⤴ skip [$LABEL] (cached): $STR"
        return
    fi
    mkdir -p "$CLASS_RESULT_DIR" "$CLASS_VIDEO_DIR"

    local SPORT=$2   # the server port assigned by orchestrator
    local CLASS_LOG="/tmp/dp3_batch_eval_${TAG}_${LABEL}.log"
    echo "  ▶ launch [$LABEL] $N eps  (server:$SPORT) → $CLASS_LOG"
    "$PY_ISAAC" -u sim/eval_dp3_baseline3.py \
        --episodes-glob "$CLASS_DIR/*.hdf5" \
        --n-rollouts "$N" \
        --max-chunks "$MAX_CHUNKS" \
        --server-url "http://127.0.0.1:$SPORT" \
        --headless \
        --video "$CLASS_VIDEO_DIR" \
        --video-all \
        --result-dir "$CLASS_RESULT_DIR" \
        --seed 0 \
        > "$CLASS_LOG" 2>&1
    local JSON=$(ls -t "$CLASS_RESULT_DIR"/eval_*.json 2>/dev/null | head -1)
    if [[ -f "$JSON" ]]; then
        local STR=$("$PY_DP3" -c "import json; d=json.load(open('$JSON')); print(f'{d[\"n_success\"]}/{d[\"n_total\"]} = {100*d[\"n_success\"]/max(d[\"n_total\"],1):.0f}%')")
        echo "  ✓ done   [$LABEL] $STR"
    else
        echo "  ✗ FAIL   [$LABEL] (no JSON — check $CLASS_LOG)"
    fi
}
export -f eval_one_class
export PERCLASS_DIR VIDEO_ROOT TAG MAX_CHUNKS PY_DP3 PY_ISAAC RESUME

# Launch with concurrency cap = $PAR. Round-robin server port assignment.
LAUNCH_IDX=0
for CLASS_DIR in "$STAGE_ROOT"/*/; do
    [[ -d "$CLASS_DIR" ]] || continue
    while [[ $(jobs -rp | wc -l) -ge $PAR ]]; do sleep 5; done
    SPORT_I=${SERVER_PORTS[$((LAUNCH_IDX % N_SERVERS))]}
    eval_one_class "$CLASS_DIR" "$SPORT_I" &
    LAUNCH_IDX=$((LAUNCH_IDX + 1))
done
wait

# ---- 4. aggregate summary ----
echo
echo "==> aggregating summary"
"$PY_DP3" - <<EOF
import json, glob, os
per_class = []
total_success = 0; total_attempted = 0
for class_dir in sorted(glob.glob("$PERCLASS_DIR/*/")):
    label = os.path.basename(class_dir.rstrip("/"))
    jsons = sorted(glob.glob(os.path.join(class_dir, "*.json")))
    if not jsons: continue
    d = json.load(open(jsons[-1]))
    n_s, n_t = d["n_success"], d["n_total"]
    total_success += n_s; total_attempted += n_t
    per_class.append({
        "label": label,
        "n_success": n_s, "n_total": n_t,
        "success_rate": (100*n_s/max(n_t,1)),
        "per_ep": d.get("results", []),
    })
summary = {
    "tag": "$TAG",
    "ckpt": "$CKPT",
    "n_classes": len(per_class),
    "total_success": total_success,
    "total_attempted": total_attempted,
    "overall_rate": (100*total_success/max(total_attempted,1)),
    "per_class": per_class,
}
with open("$RESULT_DIR/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n  ==> OVERALL: {total_success}/{total_attempted} = {summary['overall_rate']:.1f}%")
print(f"  per-class breakdown:")
for pc in per_class:
    print(f"    {pc['label']:<22}  {pc['n_success']:>2}/{pc['n_total']:<2}  {pc['success_rate']:>5.1f}%")
print(f"\n  summary: $RESULT_DIR/summary.json")
print(f"  per-class JSONs: $PERCLASS_DIR/<label>/")
print(f"  videos: $VIDEO_ROOT/<label>/")
EOF
