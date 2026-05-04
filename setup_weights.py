#!/usr/bin/env python3
"""
setup_weights.py — Download all model weights from HuggingFace
==============================================================
Run once after cloning the repo to pull all required model weights.

Usage:
    python setup_weights.py            # download all weights
    python setup_weights.py --tool fp  # download only FoundationPose weights

Tools:
    fp       — FoundationPose  (248 MB)
    hawor    — HaWoR           (3.6 GB, checkpoints + external)
    haptic   — HaPTIC vitpose  (3.8 GB)
    megasam  — MegaSAM         (21 MB)
    depthpro — Apple Depth Pro (1.9 GB, avoids Apple CDN firewall issues)

Notes:
    - MANO model weights require manual registration at https://mano.is.tue.mpg.de/
      Place MANO_RIGHT.pkl and MANO_LEFT.pkl under third_party/haptic/assets/mano/
    - Requires: pip install huggingface_hub
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("❌ huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)

REPO_ID   = "UCBProject/Affordance2Grasp-Data"
REPO_TYPE = "dataset"
PROJECT   = Path(__file__).parent


def download_and_place(patterns, local_prefix, dest):
    """Download files matching patterns and move them to dest."""
    tmp = PROJECT / ".hf_tmp"
    tmp.mkdir(exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        allow_patterns=patterns,
        local_dir=str(tmp),
    )
    src = tmp / local_prefix
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        if dest.exists():
            shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
        shutil.move(str(src), str(dest))
    shutil.rmtree(tmp, ignore_errors=True)


def download_fp():
    print("\n📥 FoundationPose weights (~248 MB)...")
    fp_root = PROJECT / "third_party" / "FoundationPose" / "weights"
    fp_root.mkdir(parents=True, exist_ok=True)
    for folder in ["2023-10-28-18-33-37", "2024-01-11-20-02-45"]:
        dest = fp_root / folder
        if dest.exists():
            print(f"   ⏭️  {folder} already exists, skipping")
            continue
        download_and_place(
            patterns=[f"FoundationPose/weights/{folder}/*"],
            local_prefix=f"FoundationPose/weights/{folder}",
            dest=dest,
        )
        print(f"   ✅ {folder}")
    print("✅ FoundationPose done")


def download_hawor():
    print("\n📥 HaWoR weights (~3.6 GB, this may take a while)...")
    hawor_root = PROJECT / "third_party" / "hawor" / "weights"

    # checkpoints (main model)
    ckpt_dest = hawor_root / "hawor" / "checkpoints"
    if not (ckpt_dest / "hawor.ckpt").exists():
        print("   Downloading hawor/checkpoints...")
        download_and_place(
            patterns=["HaWoR/weights/hawor/checkpoints/hawor.ckpt",
                      "HaWoR/weights/hawor/checkpoints/infiller.pt",
                      "HaWoR/weights/hawor/checkpoints/model_config.yaml"],
            local_prefix="HaWoR/weights/hawor/checkpoints",
            dest=ckpt_dest,
        )
    else:
        print("   ⏭️  hawor/checkpoints already exists, skipping")

    # external (detector + droid)
    ext_dest = hawor_root / "external"
    if not (ext_dest / "detector.pt").exists():
        print("   Downloading external weights...")
        download_and_place(
            patterns=["HaWoR/weights/external/*"],
            local_prefix="HaWoR/weights/external",
            dest=ext_dest,
        )
    else:
        print("   ⏭️  external already exists, skipping")

    print("✅ HaWoR done")


def download_haptic():
    print("\n📥 HaPTIC vitpose weights (~3.8 GB, this may take a while)...")
    dest = PROJECT / "third_party" / "haptic" / "_DATA" / "vitpose_ckpts"
    if dest.exists() and any(dest.iterdir()):
        print("   ⏭️  HaPTIC vitpose already exists, skipping")
        return
    download_and_place(
        patterns=["HaPTIC/vitpose_ckpts/*"],
        local_prefix="HaPTIC/vitpose_ckpts",
        dest=dest,
    )
    print("✅ HaPTIC done")


def download_megasam():
    print("\n📥 MegaSAM checkpoint (~21 MB)...")
    dest = PROJECT / "mega-sam" / "checkpoints" / "megasam_final.pth"
    if dest.exists():
        print("   ⏭️  megasam_final.pth already exists, skipping")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    download_and_place(
        patterns=["MegaSAM/checkpoints/megasam_final.pth"],
        local_prefix="MegaSAM/checkpoints",
        dest=dest.parent,
    )
    print("✅ MegaSAM done")


def download_depthpro():
    print("\n📥 Depth Pro checkpoint (~1.9 GB)...")
    dest = PROJECT / "third_party" / "ml-depth-pro" / "checkpoints" / "depth_pro.pt"
    if dest.exists():
        print("   ⏭️  depth_pro.pt already exists, skipping")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    download_and_place(
        patterns=["DepthPro/checkpoints/depth_pro.pt"],
        local_prefix="DepthPro/checkpoints",
        dest=dest.parent,
    )
    print("✅ Depth Pro done")


TOOLS = {
    "fp":       download_fp,
    "hawor":    download_hawor,
    "haptic":   download_haptic,
    "megasam":  download_megasam,
    "depthpro": download_depthpro,
}


def main():
    parser = argparse.ArgumentParser(description="Download model weights from HuggingFace")
    parser.add_argument("--tool", choices=list(TOOLS.keys()),
                        help="Download weights for a specific tool only")
    args = parser.parse_args()

    print("=" * 60)
    print("  Affordance2Grasp — Weight Setup")
    print(f"  Source: {REPO_ID}")
    print("=" * 60)

    if args.tool:
        TOOLS[args.tool]()
    else:
        for fn in TOOLS.values():
            fn()

    print("\n" + "=" * 60)
    print("  All weights downloaded.")
    print()
    print("  ⚠️  MANO requires manual download (license restriction):")
    print("     1. Register at https://mano.is.tue.mpg.de/")
    print("     2. Download MANO_RIGHT.pkl and MANO_LEFT.pkl")
    print("     3. Place under: third_party/haptic/assets/mano/")
    print("=" * 60)


if __name__ == "__main__":
    main()
