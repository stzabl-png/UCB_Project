#!/usr/bin/env python3
"""Deterministic 80/20 train/test split for baseline_3 v4 hdf5 episodes.

Stratified by ycb_class_id so each object's eps are split independently — keeps
test set's per-object representation proportional. Reproducible via --seed.

Usage:
    python Baseline1/scripts/split_train_test.py \
        --input-dir   Baseline1/data/episodes_b3_v4_full12_yaw \
        --output-dir  Baseline1/data/dp3_full12_yaw \
        --test-ratio  0.20 \
        --seed        42

Output layout:
    <output-dir>/train/<ep>.hdf5
    <output-dir>/test/<ep>.hdf5
Files are copied (not symlinked) so the source dir can be archived separately.
"""
import argparse
import glob
import os
import re
import shutil
from collections import defaultdict

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--input-dir", required=True,
                help="dir containing *.hdf5 episode files from the collector")
ap.add_argument("--output-dir", required=True,
                help="train/test/ subdirs are written here")
ap.add_argument("--test-ratio", type=float, default=0.20)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--pattern", default="*.hdf5")
args = ap.parse_args()


def class_id(path):
    """Parse '..._ycb_dex_NN[_yawXXX].hdf5' → int(NN)."""
    m = re.search(r"ycb_dex_(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def main():
    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not files:
        raise SystemExit(f"no {args.pattern} in {args.input_dir}")

    by_cid = defaultdict(list)
    for f in files:
        by_cid[class_id(f)].append(f)

    rng = np.random.default_rng(args.seed)
    train_files, test_files = [], []
    print(f"Stratified 80/20 split, seed={args.seed}, test_ratio={args.test_ratio}:")
    for cid in sorted(by_cid):
        eps = sorted(by_cid[cid])
        n = len(eps)
        n_test = max(1, int(round(n * args.test_ratio))) if n >= 2 else 0
        idx = rng.permutation(n)
        test_idx = set(idx[:n_test].tolist())
        for i, ep in enumerate(eps):
            (test_files if i in test_idx else train_files).append(ep)
        print(f"  cid={cid:02d}: total={n:3d}  train={n - n_test:3d}  test={n_test:3d}")
    print(f"TOTAL  train={len(train_files)}  test={len(test_files)}")

    train_dir = os.path.join(args.output_dir, "train")
    test_dir = os.path.join(args.output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for src in train_files:
        shutil.copy2(src, os.path.join(train_dir, os.path.basename(src)))
    for src in test_files:
        shutil.copy2(src, os.path.join(test_dir, os.path.basename(src)))
    print(f"Wrote → {train_dir}  ({len(train_files)} files)")
    print(f"Wrote → {test_dir}   ({len(test_files)} files)")


if __name__ == "__main__":
    main()
