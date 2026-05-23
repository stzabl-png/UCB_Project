# tem_replay_video_and_frames

Multi-angle IsaacSim replays of the 2 successfully-retargeted DexYCB → Franka
trajectories (sugar box + tomato soup can). Generated for paper-figure use.

## Layout
```
tem_replay_video_and_frames/
├── sugar_box_ycb_dex_03/
│   ├── iso/    {frames/*.png, replay.mp4}
│   ├── side/   {frames/*.png, replay.mp4}
│   ├── front/  {frames/*.png, replay.mp4}
│   └── top/    {frames/*.png, replay.mp4}
└── tomato_soup_can_ycb_dex_04/
    └── (same 4 angles)
```

## Camera angles (world frame, metres)
| name  | eye               | target          | what's shown |
|-------|-------------------|-----------------|--------------|
| iso   | (1.5, 1.5, 1.5)   | (0, 0.4, 0.85)  | 3/4 isometric overview |
| side  | (1.5, 0.55, 0.95) | (0, 0.55, 0.85) | close-up from robot's right, gripper-eye-level |
| front | (0, 2.5, 1.1)     | (0, 0.4, 0.85)  | looking back toward Franka from object side |
| top   | (0, 0.5, 2.5)     | (0, 0.5, 0.85)  | bird's-eye (XY positioning) |

## Trajectory source
- Source HDF5: `partner_trial_data/<obj>/trajectory.hdf5`
- Sim driver:  `sim/gt_replay_ikpd_v2.py --drive pd --grasp-lift --grasp-collision`
- Replay generator: `sim/multi_angle_replay.sh`

Frames are captured at 1 / `--video-every`=2 sim steps; mp4 encoded at 20 fps.
