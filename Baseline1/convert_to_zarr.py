#!/usr/bin/env python3
"""
Baseline1 — concatenate episode HDF5 files → DP3 zarr (ReplayBuffer format).

Same on-disk layout as Baseline2/convert_to_zarr.py so both DP baselines train
with an identical task config:
    data/point_cloud  (N_total, 4096, 3)  float32
    data/state        (N_total, 8)        float32   [x,y,z, qw,qx,qy,qz, gripper]
    data/action       (N_total, 8)        float32   = state shifted by 1
    meta/episode_ends (n_episodes,)        int64

The per-episode `finger_angles` / `wrist_pose` datasets (GT MANO metadata) are
ignored here — DP3 only consumes point_cloud / state / action.

Usage:
    python Baseline1/convert_to_zarr.py \\
        --input_dir   Baseline1/data/episodes \\
        --output_zarr Baseline1/data/human_dp_baseline.zarr
"""
import argparse, os, glob
import numpy as np
import h5py
import zarr

ap = argparse.ArgumentParser()
ap.add_argument("--input_dir",   required=True, help="dir with *.hdf5 episode files")
ap.add_argument("--output_zarr", default="Baseline1/data/human_dp_baseline.zarr")
ap.add_argument("--pattern",     default="*.hdf5", help="glob pattern for episode files")
args = ap.parse_args()


def main():
    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not files:
        print(f"❌ no {args.pattern} files in {args.input_dir}"); return
    print(f"Found {len(files)} episode files")

    all_pc, all_state, all_action, ep_ends = [], [], [], []
    cum = 0
    n_bad = 0
    for path in files:
        try:
            with h5py.File(path, "r") as f:
                pc  = f["point_cloud"][:]
                st  = f["state"][:]
                ac  = f["action"][:]
        except Exception as e:
            print(f"  ⚠️  {os.path.basename(path)}: read error {e}"); n_bad += 1; continue
        T = len(st)
        if not (len(pc) == T == len(ac)) or T < 1:
            print(f"  ⚠️  {os.path.basename(path)}: shape mismatch / empty (pc={len(pc)} st={T} ac={len(ac)})")
            n_bad += 1; continue
        all_pc.append(pc); all_state.append(st); all_action.append(ac)
        cum += T; ep_ends.append(cum)

    if not ep_ends:
        print("❌ no usable episodes"); return

    all_pc     = np.concatenate(all_pc,     axis=0).astype(np.float32)
    all_state  = np.concatenate(all_state,  axis=0).astype(np.float32)
    all_action = np.concatenate(all_action, axis=0).astype(np.float32)
    ep_ends    = np.array(ep_ends, dtype=np.int64)

    print(f"\nepisodes : {len(ep_ends)}  ({n_bad} skipped)")
    print(f"steps    : {len(all_state)}")
    print(f"point_cloud {all_pc.shape}  state {all_state.shape}  action {all_action.shape}")
    print(f"gripper range: [{all_state[:,7].min():.3f}, {all_state[:,7].max():.3f}]  "
          f"xyz extent (m): {np.round(all_state[:,:3].max(0) - all_state[:,:3].min(0), 3)}")

    root = zarr.open(args.output_zarr, mode="w")
    data = root.require_group("data")
    data.create_dataset("point_cloud", data=all_pc,
                        chunks=(100, all_pc.shape[1], all_pc.shape[2]), dtype=np.float32)
    data.create_dataset("state",  data=all_state,  chunks=(2000, all_state.shape[1]),  dtype=np.float32)
    data.create_dataset("action", data=all_action, chunks=(2000, all_action.shape[1]), dtype=np.float32)
    meta = root.require_group("meta")
    meta.create_dataset("episode_ends", data=ep_ends, dtype=np.int64)

    print(f"\n✅ wrote {args.output_zarr}")
    print(f"   verify: python -c \"import zarr; r=zarr.open('{args.output_zarr}'); "
          f"print(r['data/point_cloud'].shape, r['data/state'].shape, r['meta/episode_ends'][:5])\"")


if __name__ == "__main__":
    main()
