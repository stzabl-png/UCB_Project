#!/usr/bin/env python3
"""
Baseline2 — Convert episode HDF5 files to DP3 zarr format.

Usage:
    python Baseline2/convert_to_zarr.py \\
        --input_dir Baseline2/data/episodes \\
        --output_zarr Baseline2/data/robot_dp_baseline.zarr
"""
import argparse, os, glob
import numpy as np
import h5py
import zarr
from termcolor import cprint

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir",   type=str, required=True,
                    help="Directory containing *_ep*.hdf5 files")
parser.add_argument("--output_zarr", type=str, default="Baseline2/data/robot_dp_baseline.zarr",
                    help="Output zarr path")
args = parser.parse_args()


def main():
    ep_files = sorted(glob.glob(os.path.join(args.input_dir, "*.hdf5")))
    if not ep_files:
        cprint(f"❌ No HDF5 files in {args.input_dir}", "red")
        return

    cprint(f"Found {len(ep_files)} episodes", "cyan")

    # ── Collect all episodes ────────────────────────────────
    all_pc     = []   # list of (T, 4096, 3)
    all_state  = []   # list of (T, 8)
    all_action = []   # list of (T, 8)
    ep_ends    = []   # cumulative episode end indices

    cum = 0
    for path in ep_files:
        with h5py.File(path, "r") as f:
            pc     = f["point_cloud"][:]   # (T, 4096, 3)
            state  = f["state"][:]          # (T, 8)
            action = f["action"][:]         # (T, 8)

        T = len(state)
        assert len(pc) == T and len(action) == T, \
            f"Shape mismatch in {path}"

        all_pc.append(pc)
        all_state.append(state)
        all_action.append(action)
        cum += T
        ep_ends.append(cum)
        cprint(f"  {os.path.basename(path)}: {T} steps", "white")

    # ── Concatenate ────────────────────────────────────────
    all_pc     = np.concatenate(all_pc,     axis=0)   # (N_total, 4096, 3)
    all_state  = np.concatenate(all_state,  axis=0)   # (N_total, 8)
    all_action = np.concatenate(all_action, axis=0)   # (N_total, 8)
    ep_ends    = np.array(ep_ends, dtype=np.int64)

    cprint(f"\nTotal steps:    {len(all_state)}", "cyan")
    cprint(f"Total episodes: {len(ep_ends)}", "cyan")
    cprint(f"point_cloud:    {all_pc.shape}", "cyan")
    cprint(f"state:          {all_state.shape}", "cyan")
    cprint(f"action:         {all_action.shape}", "cyan")

    # ── Write zarr (DP3 ReplayBuffer format) ───────────────
    # DP3 ReplayBuffer expects:
    #   data/point_cloud  (N, 4096, 3)
    #   data/state        (N, D)
    #   data/action       (N, D)
    #   meta/episode_ends (n_episodes,)
    root = zarr.open(args.output_zarr, mode="w")

    data = root.require_group("data")
    data.create_dataset("point_cloud",
                        data=all_pc.astype(np.float32),
                        chunks=(100, all_pc.shape[1], all_pc.shape[2]),
                        dtype=np.float32)
    data.create_dataset("state",
                        data=all_state.astype(np.float32),
                        chunks=(1000, all_state.shape[1]),
                        dtype=np.float32)
    data.create_dataset("action",
                        data=all_action.astype(np.float32),
                        chunks=(1000, all_action.shape[1]),
                        dtype=np.float32)

    meta = root.require_group("meta")
    meta.create_dataset("episode_ends", data=ep_ends, dtype=np.int64)

    cprint(f"\n✅ Saved zarr: {args.output_zarr}", "green")
    cprint(f"   Verify:", "green")
    cprint(f"     zarr.open('{args.output_zarr}')['data/point_cloud'].shape", "white")


if __name__ == "__main__":
    main()
