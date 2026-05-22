#!/usr/bin/env bash
# run_dp3_twophase.sh — two-phase DP3 eval (currently sugar / tomato).
#
# Phase 1: run the DP3 policy CLOSED-LOOP just to COLLECT the trajectory it produces;
#          gt_replay_ikpd_v2.py --dp3 writes /tmp/dp3_traj_ycb_dex_NN.hdf5 and exits.
# Phase 2: replay that collected trajectory through gt_replay's normal global-continuity
#          IK chain (smooth joints — no per-chunk IK branch-switching), with --grasp-lift,
#          and ffmpeg the captured frames to an mp4.
#
# PREREQUISITE: the DP3 inference server MUST already be running on :8765 — Phase 1
# queries it (gt_replay --dp3 talks to http://127.0.0.1:8765). This script does NOT
# start it.

set -u

PY=/home/accelerator/miniforge3/envs/env_isaaclab/bin/python
PROJ=/home/accelerator/UCB_Project
cd "$PROJ" || exit 1

VID_DIR=/tmp/dp3_vid
mkdir -p "$VID_DIR" replay_video_check

# objname  subject                  session          objid
OBJECTS=(
  "sugar   20200709-subject-01      20200709_142517  ycb_dex_03"
  "tomato  20201015-subject-09      20201015_143403  ycb_dex_04"
  # "banana 20200709-subject-01      20200709_145401  ycb_dex_10"   # re-enable later
)

for row in "${OBJECTS[@]}"; do
  read -r OBJNAME SUBJECT SESS OBJID <<< "$row"
  echo "=================================================================="
  echo "  $OBJNAME  subject=$SUBJECT  session=$SESS  object=$OBJID"
  echo "=================================================================="

  SRC_TRAJ="Baseline1/data/episodes_g/dexycb__${SUBJECT}__${SESS}__840412060917__${OBJID}.hdf5"

  # ── Phase 1: DP3 closed-loop collect → /tmp/dp3_traj_${OBJID}.hdf5 ──────────
  echo "[Phase 1] DP3 collect ($OBJNAME) -> /tmp/dp3_traj_${OBJID}.hdf5"
  timeout 1500 "$PY" -u sim/gt_replay_ikpd_v2.py \
    --session "$SESS" --object "$OBJID" \
    --traj "$SRC_TRAJ" \
    --dp3 --headless \
    > "/tmp/dp3_tp_${OBJNAME}_p1.log" 2>&1
  P1_RC=$?
  echo "[Phase 1] rc=$P1_RC  (log: /tmp/dp3_tp_${OBJNAME}_p1.log)"
  if [[ $P1_RC -ne 0 || ! -f "/tmp/dp3_traj_${OBJID}.hdf5" ]]; then
    echo "[Phase 1] FAILED for $OBJNAME — skipping Phase 2"
    continue
  fi

  # ── Phase 2: replay the collected trajectory (smooth IK) + grasp-lift + video ─
  rm -f "$VID_DIR"/*.png
  echo "[Phase 2] replay collected trajectory ($OBJNAME)"
  timeout 1500 "$PY" -u sim/gt_replay_ikpd_v2.py \
    --session "$SESS" --object "$OBJID" \
    --traj "/tmp/dp3_traj_${OBJID}.hdf5" \
    --drive pd --grasp-lift --grasp-collision --headless \
    --video "$VID_DIR" --video-every 2 \
    > "/tmp/dp3_tp_${OBJNAME}_p2.log" 2>&1
  P2_RC=$?
  echo "[Phase 2] rc=$P2_RC  (log: /tmp/dp3_tp_${OBJNAME}_p2.log)"
  if [[ $P2_RC -ne 0 ]]; then
    echo "[Phase 2] FAILED for $OBJNAME — skipping ffmpeg"
    continue
  fi

  # ── encode the captured frames to mp4 ──────────────────────────────────────
  ffmpeg -y -framerate 20 -i "$VID_DIR/f_%05d.png" \
    -c:v libx264 -pix_fmt yuv420p "replay_video_check/dp3_twophase_${OBJNAME}.mp4"
  echo "[ffmpeg] -> replay_video_check/dp3_twophase_${OBJNAME}.mp4"
done

echo "=================================================================="
echo "  done — videos in replay_video_check/dp3_twophase_*.mp4"
echo "=================================================================="
