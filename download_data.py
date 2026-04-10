#!/usr/bin/env python3
"""
Download training data and checkpoints from HuggingFace.

Usage:
    pip install huggingface-hub
    python download_data.py
"""

import os
import argparse


def main():
    parser = argparse.ArgumentParser(description="Download Affordance2Grasp data from HuggingFace")
    parser.add_argument("--repo", default="UCBProject/Affordance2Grasp-Data",
                        help="HuggingFace dataset repo ID")
    parser.add_argument("--subset", choices=["all", "train", "inference"],
                        default="all",
                        help="What to download: all, train-only, or inference-only")
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface-hub not installed.")
        print("  pip install huggingface-hub")
        return

    project_dir = os.path.dirname(os.path.abspath(__file__))

    if args.subset == "inference":
        # Minimal: just checkpoint + meshes
        patterns = ["checkpoints/**", "data_hub/meshes/**", "data_hub/human_prior/**"]
    elif args.subset == "train":
        # Training: dataset + human_prior + training data
        patterns = [
            "checkpoints/**",
            "data_hub/**",
            "dataset/**",
        ]
    else:
        # Everything
        patterns = None

    print(f"Downloading from {args.repo} ({args.subset})...")
    snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        local_dir=project_dir,
        allow_patterns=patterns,
    )
    print("✅ Download complete!")
    print(f"   Data: {os.path.join(project_dir, 'data_hub')}")
    print(f"   Checkpoints: {os.path.join(project_dir, 'checkpoints')}")


if __name__ == "__main__":
    main()
