#!/usr/bin/env bash
# multi_angle_replay.sh — replay the 2 successful retargeted trajectories
# (sugar + tomato) from 4 camera angles each, for paper figures.
#
# Inputs : partner_trial_data/<obj>/trajectory.hdf5
# Outputs: tem_replay_video_and_frames/<obj>/<angle>/{frames/*.png, replay.mp4, replay.log}
#
# 4 camera angles per object:
#   iso   — 3/4 isometric (full scene)
#   side  — robot's right side, gripper-eye-level (close-up of close+lift)
#   front — opposite end of table, looking back toward Franka
#   top   — bird's-eye (xy positioning)

set -u
PY=/home/accelerator/miniforge3/envs/env_isaaclab/bin/python
PROJ=/home/accelerator/UCB_Project
cd "$PROJ" || exit 1
OUT_ROOT=tem_replay_video_and_frames
mkdir -p "$OUT_ROOT"

# (folder-name : object-id : session-id : extra-args) for the 2 known successful trajectories
# Reference success videos:
#   replay_video_check/gtreplay_grasp_sugar_0p2kg.mp4    sugar lightened to 0.2 kg
#   replay_video_check/gtreplay_grasp_mustard_realmass.mp4   mustard at real 0.603 kg
OBJECTS=(
    "sugar_box_ycb_dex_03:ycb_dex_03:20200709_142517:--object-mass 0.2"
    "mustard_bottle_ycb_dex_05:ycb_dex_05:20200709_143211:"
)

# (angle-name : camera-eye xyz : camera-target xyz)
# Scene anchors: Franka at world (0.20,-0.05,0.80) yaw 90°; table at (0,1,0.75);
# object spawned at (0, 0.55, ~0.84). Camera units in metres, world frame.
ANGLES=(
    "iso:1.5 1.5 1.5:0.0 0.4 0.85"
    "side:1.5 0.55 0.95:0.0 0.55 0.85"
    "front:0.0 2.5 1.1:0.0 0.4 0.85"
    "top:0.0 0.5 2.5:0.0 0.5 0.85"
)

T0=$SECONDS
for o in "${OBJECTS[@]}"; do
    IFS=':' read -r NAME OBJID SESS EXTRA <<< "$o"
    TRAJ="partner_trial_data/${NAME}/trajectory.hdf5"
    [[ -f "$TRAJ" ]] || { echo "❌ missing $TRAJ — skip $NAME"; continue; }

    for a in "${ANGLES[@]}"; do
        IFS=':' read -r ANAME EYE TARGET <<< "$a"
        OUT="$OUT_ROOT/${NAME}/${ANAME}"
        FRAMES="$OUT/frames"
        rm -rf "$FRAMES"; mkdir -p "$FRAMES"

        echo "############################################################"
        echo "  $NAME / $ANAME   eye=($EYE)  target=($TARGET)  extra=[$EXTRA]   [$((SECONDS-T0))s]"
        echo "############################################################"

        "$PY" -u sim/gt_replay_ikpd_v2.py \
            --session "$SESS" --object "$OBJID" --traj "$TRAJ" \
            --camera-eye $EYE --camera-target $TARGET \
            --drive pd --grasp-lift --grasp-collision \
            --headless --video "$FRAMES" --video-every 2 \
            $EXTRA \
            > "$OUT/replay.log" 2>&1
        RC=$?

        N_FRAMES=$(ls "$FRAMES"/*.png 2>/dev/null | wc -l)
        # Parse object dz from the replay log to know whether this run actually grasped
        DZ_LINE=$(grep -E "object dz" "$OUT/replay.log" 2>/dev/null | tail -1)
        if echo "$DZ_LINE" | grep -q "GRASPED + LIFTED"; then
            GRASP_TAG="✓ grasped"
        elif echo "$DZ_LINE" | grep -q "not lifted"; then
            GRASP_TAG="✗ NOT lifted"
        else
            GRASP_TAG="? unknown"
        fi
        if [[ $RC -eq 0 && $N_FRAMES -gt 0 ]]; then
            ffmpeg -y -framerate 20 -i "$FRAMES/f_%05d.png" \
                -c:v libx264 -pix_fmt yuv420p "$OUT/replay.mp4" \
                > "$OUT/ffmpeg.log" 2>&1
            FF_RC=$?
            if [[ $FF_RC -eq 0 ]]; then
                echo "  ✅ $N_FRAMES frames → $OUT/replay.mp4  [$GRASP_TAG]"
            else
                echo "  ⚠️  ffmpeg failed (rc=$FF_RC), see $OUT/ffmpeg.log"
            fi
        else
            echo "  ❌ replay failed (rc=$RC, n_frames=$N_FRAMES), see $OUT/replay.log"
        fi
    done
done

# ── README in the output dir ─────────────────────────────────────────────────
cat > "$OUT_ROOT/README.md" <<EOF
# tem_replay_video_and_frames

Multi-angle IsaacSim replays of the 2 successfully-retargeted DexYCB → Franka
trajectories (sugar box + tomato soup can). Generated for paper-figure use.

## Layout
\`\`\`
tem_replay_video_and_frames/
├── sugar_box_ycb_dex_03/
│   ├── iso/    {frames/*.png, replay.mp4}
│   ├── side/   {frames/*.png, replay.mp4}
│   ├── front/  {frames/*.png, replay.mp4}
│   └── top/    {frames/*.png, replay.mp4}
└── tomato_soup_can_ycb_dex_04/
    └── (same 4 angles)
\`\`\`

## Camera angles (world frame, metres)
| name  | eye               | target          | what's shown |
|-------|-------------------|-----------------|--------------|
| iso   | (1.5, 1.5, 1.5)   | (0, 0.4, 0.85)  | 3/4 isometric overview |
| side  | (1.5, 0.55, 0.95) | (0, 0.55, 0.85) | close-up from robot's right, gripper-eye-level |
| front | (0, 2.5, 1.1)     | (0, 0.4, 0.85)  | looking back toward Franka from object side |
| top   | (0, 0.5, 2.5)     | (0, 0.5, 0.85)  | bird's-eye (XY positioning) |

## Trajectory source
- Source HDF5: \`partner_trial_data/<obj>/trajectory.hdf5\`
- Sim driver:  \`sim/gt_replay_ikpd_v2.py --drive pd --grasp-lift --grasp-collision\`
- Replay generator: \`sim/multi_angle_replay.sh\`

Frames are captured at 1 / \`--video-every\`=2 sim steps; mp4 encoded at 20 fps.
EOF

echo
echo "############################################################"
echo "DONE in $((SECONDS-T0))s — outputs in $OUT_ROOT/"
echo "############################################################"
find "$OUT_ROOT" -name "replay.mp4" | sort | xargs ls -lh 2>/dev/null
