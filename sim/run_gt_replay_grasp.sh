#!/usr/bin/env bash
# run_gt_replay_grasp.sh — replay the retargeted-human TRAINING trajectory directly
# (cuRobo whole-trajectory continuity IK + PD drive + dynamic collidable object +
# close-gripper + lift). Tests whether the retarget-from-video trajectory ITSELF
# grasps the object — no DP3 policy, no inference server, pure replay of the
# training-episode HDF5. Isolates "retargeting quality" from "DP3 policy quality".
set -u
PY=/home/accelerator/miniforge3/envs/env_isaaclab/bin/python
PROJ=/home/accelerator/UCB_Project
cd "$PROJ" || exit 1
VID_DIR=/tmp/dp3_vid
mkdir -p "$VID_DIR" replay_video_check
EP=Baseline1/data/episodes_g

# objname       session          objid        episode-hdf5 (retargeted human trajectory)
OBJECTS=(
  "sugar        20200709_142517  ycb_dex_03   $EP/dexycb__20200709-subject-01__20200709_142517__840412060917__ycb_dex_03.hdf5"
  "tomato       20201015_143403  ycb_dex_04   $EP/dexycb__20201015-subject-09__20201015_143403__840412060917__ycb_dex_04.hdf5"
  "mustard      20200709_143211  ycb_dex_05   $EP/dexycb__20200709-subject-01__20200709_143211__840412060917__ycb_dex_05.hdf5"
  "large_marker 20200709_152506  ycb_dex_18   $EP/dexycb__20200709-subject-01__20200709_152506__840412060917__ycb_dex_18.hdf5"
)

for row in "${OBJECTS[@]}"; do
  read -r OBJNAME SESS OBJID TRAJ <<< "$row"
  echo "=================================================================="
  echo "  $OBJNAME  session=$SESS  object=$OBJID"
  echo "=================================================================="
  if [[ ! -f "$TRAJ" ]]; then
    echo "  MISSING traj: $TRAJ — skip"; continue
  fi
  rm -f "$VID_DIR"/*.png
  timeout 1500 "$PY" -u sim/gt_replay_ikpd_v2.py \
    --session "$SESS" --object "$OBJID" --traj "$TRAJ" \
    --drive pd --grasp-lift --grasp-collision --headless \
    --video "$VID_DIR" --video-every 2 \
    > "/tmp/gtreplay_${OBJNAME}.log" 2>&1
  RC=$?
  echo "  rc=$RC  (log: /tmp/gtreplay_${OBJNAME}.log)"
  if [[ $RC -ne 0 ]]; then
    echo "  FAILED — skip ffmpeg"; continue
  fi
  ffmpeg -y -framerate 20 -i "$VID_DIR/f_%05d.png" \
    -c:v libx264 -pix_fmt yuv420p "replay_video_check/gtreplay_grasp_${OBJNAME}.mp4" >/dev/null 2>&1
  echo "  -> replay_video_check/gtreplay_grasp_${OBJNAME}.mp4"
done
echo "=================================================================="
echo "  done — videos in replay_video_check/gtreplay_grasp_*.mp4"
echo "=================================================================="
