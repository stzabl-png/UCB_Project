#!/usr/bin/env bash
# oakink_rescue_from_old_index.sh — copy OLD thumb+index hdf5 into the new
# thumb+middle dir for filenames that the new pipeline did NOT produce.
#
# Rationale: thumb+middle has higher overall success rate (~2×) but a small
# number of (src_ep, yaw) combos that thumb+index could grasp now fail. To
# maximize training-data coverage / DP3 generalization, fill those gaps with
# the corresponding OLD thumb+index hdf5.
#
# Trade-off: introduces a small mixed finger-convention noise in the EE target
# (~2.8cm shift between conventions). Accepted because (a) the gap-fills are
# minority, (b) DP3 PC→EE training tolerates target-noise of that magnitude.
#
# Logic: rescue = {basename ∈ OLD/*.hdf5} - {basename ∈ NEW/*.hdf5}
# (no use=true filter — extra coverage from use=false obj is welcome.)
#
# Usage:
#   bash scripts/baseline_3_v4/oakink_rescue_from_old_index.sh                # dry-run
#   bash scripts/baseline_3_v4/oakink_rescue_from_old_index.sh --apply        # really copy
#   bash scripts/baseline_3_v4/oakink_rescue_from_old_index.sh --apply --old DIR --new DIR
set -u
OLD="Baseline1/data/episodes_b3_v4_oakink89_2026-05-26"
NEW="Baseline1/data/episodes_b3_v4_oakink74_pinch-middle_2026-05-27"
APPLY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --old)   OLD="$2"; shift 2 ;;
        --new)   NEW="$2"; shift 2 ;;
        --apply) APPLY=1;  shift ;;
        -h|--help)
            sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

[[ -d "$OLD" ]] || { echo "ERROR: OLD dir not found: $OLD" >&2; exit 1; }
[[ -d "$NEW" ]] || { echo "ERROR: NEW dir not found: $NEW" >&2; exit 1; }

if [[ "$APPLY" -eq 1 ]] && pgrep -f "oakink_full89_queue_resume\|run_grasp_sim_baseline3_v4" >/dev/null 2>&1; then
    echo "ERROR: collection batch still running. Wait for it to finish before --apply." >&2
    echo "       (rescue must run AFTER batch completes — otherwise rescued files might" >&2
    echo "        conflict with in-flight saves.)" >&2
    exit 1
fi

echo "═══════════════════════════════════════════════════════════════"
echo " rescue-from-old-index  (APPLY=$APPLY)"
echo "═══════════════════════════════════════════════════════════════"
echo "  OLD: $OLD ($(ls "$OLD"/*.hdf5 2>/dev/null | wc -l) hdf5)"
echo "  NEW: $NEW ($(ls "$NEW"/*.hdf5 2>/dev/null | wc -l) hdf5)"
echo

python3 <<PYEOF
import os, glob, shutil, json, datetime
from collections import Counter

old = "$OLD"; new = "$NEW"; apply = bool($APPLY)

old_files = set(os.path.basename(f) for f in glob.glob(f"{old}/*.hdf5"))
new_files = set(os.path.basename(f) for f in glob.glob(f"{new}/*.hdf5"))
rescue = sorted(old_files - new_files)
overlap = old_files & new_files

m = json.load(open('Baseline1/oakink/class_id_map.json'))
def oid(f): return f.split("__")[1].split("_")[0]
ut_set = set(o for o,i in m['objects'].items() if i.get('use'))

cnt_obj = Counter(oid(f) for f in rescue)
cnt_yaw = Counter()
for f in rescue:
    if "_yaw" in f:
        suff = "_yaw" + f.split("_yaw")[1].split(".")[0]
        cnt_yaw[suff] += 1
    else:
        cnt_yaw["_orig"] += 1

print(f"  overlap (in BOTH OLD & NEW, untouched): {len(overlap)}")
print(f"  RESCUE candidates (OLD-only):           {len(rescue)}")
print(f"    from current use=true obj:  {sum(1 for f in rescue if oid(f) in ut_set)}")
print(f"    from current use=false obj: {sum(1 for f in rescue if oid(f) not in ut_set)}  (extra coverage)")
print()
print("  per-yaw breakdown of rescue:")
for k in sorted(cnt_yaw):
    print(f"    {k:10s}  {cnt_yaw[k]}")
print()
print(f"  top obj contributing to rescue:")
for o, n in cnt_obj.most_common(20):
    flag = "" if o in ut_set else "  (use=false)"
    name = m['objects'].get(o,{}).get('name','?')
    print(f"    {o:8s}  {n:3d} ep   name={name}{flag}")
if len(cnt_obj) > 20:
    print(f"    ... and {len(cnt_obj)-20} more obj")

if not apply:
    print()
    print("=== DRY-RUN — nothing copied. Re-run with --apply to actually copy. ===")
else:
    print()
    print(f"=== APPLY: copying {len(rescue)} hdf5 OLD → NEW ===")
    n_ok = n_fail = 0
    for f in rescue:
        src = os.path.join(old, f); dst = os.path.join(new, f)
        try:
            shutil.copy2(src, dst)   # preserve mtime
            n_ok += 1
        except Exception as e:
            print(f"  FAIL {f}: {e}")
            n_fail += 1
    print(f"  copied: {n_ok}   failed: {n_fail}")
    # rescue log
    log = os.path.join(new, "RESCUE_LOG.md")
    with open(log, "w") as fp:
        fp.write(f"# Rescue log — {datetime.datetime.now().isoformat(timespec='seconds')}\n\n")
        fp.write(f"- script: scripts/baseline_3_v4/oakink_rescue_from_old_index.sh\n")
        fp.write(f"- OLD source: {old}\n")
        fp.write(f"- NEW target: {new}\n")
        fp.write(f"- copied: {n_ok}   failed: {n_fail}\n")
        fp.write(f"- finger convention: rescued ep use thumb+INDEX retarget (not thumb+middle)\n")
        fp.write(f"  T_obj_grasp differs by ~2.8cm at the grasp pose; downstream training\n")
        fp.write(f"  accepts this mixed-convention noise for coverage benefit.\n")
        fp.write(f"\n## rescued files ({n_ok}):\n")
        for f in rescue:
            tag = "" if oid(f) in ut_set else "  (use=false — extra coverage)"
            fp.write(f"- {f}{tag}\n")
    print(f"  rescue log → {log}")
    final = len(glob.glob(f"{new}/*.hdf5"))
    print(f"  NEW dir now: {final} hdf5  (was {len(new_files)})")
PYEOF
