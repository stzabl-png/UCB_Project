#!/usr/bin/env bash
# Open the IsaacSim GUI to INSPECT baseline_3's scene for ONE episode.
# Builds the scene (Franka / table / object at the retarget pose) and holds the
# GUI open — no grasp, no batch collection.
#
#   bash sim/inspect_b3.sh              # default: ycb_dex_05 (mustard), first episode
#   bash sim/inspect_b3.sh ycb_dex_06   # another object
#   bash sim/inspect_b3.sh ycb_dex_05 7 # object + --start 7 (the 8th episode)
#
# Close the IsaacSim window (or Ctrl-C) to exit.
OBJ="${1:-ycb_dex_05}"
START="${2:-0}"
cd /home/accelerator/UCB_Project || exit 1
DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority \
  /home/accelerator/miniforge3/envs/env_isaaclab/bin/python \
  sim/run_grasp_sim_baseline3.py --object "$OBJ" --start "$START" --inspect
