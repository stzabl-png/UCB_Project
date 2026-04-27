"""
Universal Depth Pro batch inference — generates metric depth maps for all datasets.

Usage:
  conda activate depth-pro
  python data/batch_depth_pro.py --dataset arctic    # ARCTIC cam1
  python data/batch_depth_pro.py --dataset oakink
  python data/batch_depth_pro.py --dataset ho3d_v3
  python data/batch_depth_pro.py --dataset dexycb
  python data/batch_depth_pro.py --dataset arctic --limit 3  # test 3 sequences
  bash scripts/run_depth_pro_all_third.sh --limit 3           # all datasets

Output per sequence:
  data_hub/ProcessedData/third_depth/{dataset}/{seq_id}/
    depths.npz      — depth maps, shape (N, H, W), float32, metric metres
    K.txt           — 3x3 camera intrinsic matrix (estimated by Depth Pro)
    frame_ids.txt   — original frame filenames in order

This output is the SAM3D input package (upload to cloud for mesh reconstruction).
"""

import os, sys, argparse, numpy as np, torch
from glob import glob
from natsort import natsorted
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# Depth Pro lives in third_party/ml-depth-pro
DEPTH_PRO_DIR = os.path.join(config.PROJECT_DIR, "third_party", "ml-depth-pro", "src")
sys.path.insert(0, DEPTH_PRO_DIR)

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_BASE = os.path.join(config.DATA_HUB, "ProcessedData", "third_depth")
RAW_BASE = os.path.join(config.DATA_HUB, "RawData", "ThirdPersonRawData")


# ── Dataset discoverers (same as batch_haptic.py) ────────────────────────────

def discover_arctic(input_dir):
    for subj in natsorted(os.listdir(input_dir)):
        subj_dir = os.path.join(input_dir, subj)
        if not os.path.isdir(subj_dir): continue
        for seq in natsorted(os.listdir(subj_dir)):
            cam_dir = os.path.join(subj_dir, seq, "1")
            if not os.path.isdir(cam_dir): continue
            imgs = natsorted(glob(os.path.join(cam_dir, "*.jpg")) +
                             glob(os.path.join(cam_dir, "*.png")))
            if imgs:
                yield f"{subj}__{seq}", imgs


def discover_oakink(input_dir):
    for seq_id in natsorted(os.listdir(input_dir)):
        seq_dir = os.path.join(input_dir, seq_id)
        if not os.path.isdir(seq_dir): continue
        imgs = natsorted(glob(os.path.join(seq_dir, "*", "north_west_color_*.png")) +
                         glob(os.path.join(seq_dir, "north_west_color_*.png")))
        if imgs:
            yield seq_id, imgs


def discover_ho3d(input_dir):
    for split in ["train", "evaluation"]:
        split_dir = os.path.join(input_dir, split)
        if not os.path.isdir(split_dir): continue
        for seq_id in natsorted(os.listdir(split_dir)):
            rgb_dir = os.path.join(split_dir, seq_id, "rgb")
            if not os.path.isdir(rgb_dir): continue
            imgs = natsorted(glob(os.path.join(rgb_dir, "*.jpg")))
            if imgs:
                yield f"{split}__{seq_id}", imgs


def discover_dexycb(input_dir, cam_filter=None):
    """Yield (seq_id, img_paths) for DexYCB.
    cam_filter: if set, only yield sequences matching this camera serial.
    """
    for subj in natsorted(os.listdir(input_dir)):
        subj_dir = os.path.join(input_dir, subj)
        if not os.path.isdir(subj_dir): continue
        for dt in natsorted(os.listdir(subj_dir)):
            dt_dir = os.path.join(subj_dir, dt)
            if not os.path.isdir(dt_dir): continue
            for serial in natsorted(os.listdir(dt_dir)):
                if cam_filter and serial not in cam_filter:
                    continue
                cam_dir = os.path.join(dt_dir, serial)
                if not os.path.isdir(cam_dir): continue
                imgs = natsorted(glob(os.path.join(cam_dir, "color_*.jpg")))
                if imgs:
                    yield f"{subj}__{dt}__{serial}", imgs


DISCOVERERS = {
    "arctic":  (discover_arctic,  "arctic"),
    "oakink":  (discover_oakink,  "oakink_v1"),
    "ho3d_v3": (discover_ho3d,    "ho3d_v3"),
    "dexycb":  (discover_dexycb,  "dexycb"),
}


# ── Depth Pro inference ───────────────────────────────────────────────────────

def load_depth_pro_model():
    import depth_pro
    model, transform = depth_pro.create_model_and_transforms(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        precision=torch.float16
    )
    model.eval()
    return model, transform


def run_depth_pro(model, transform, img_paths, max_frames=150):
    """Run Depth Pro on a list of image paths.

    Returns:
        depths:    np.ndarray (N, H, W) float32, metric depth in metres
        K:         np.ndarray (3, 3) float32, estimated camera intrinsics
        frame_ids: list of str, original filenames
    """
    import depth_pro

    # Subsample long sequences
    if len(img_paths) > max_frames:
        step = len(img_paths) // max_frames
        img_paths = img_paths[::step][:max_frames]

    depth_list = []
    fx_list = []
    H_ref, W_ref = None, None

    device = next(model.parameters()).device

    for p in tqdm(img_paths, desc="  frames", leave=False):
        image, _, f_px = depth_pro.load_rgb(p)
        image_t = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model.infer(image_t, f_px=f_px)

        depth = pred["depth"].squeeze().cpu().float().numpy()   # (H, W)
        focallen = float(pred["focallength_px"])

        depth_list.append(depth)
        fx_list.append(focallen)
        if H_ref is None:
            H_ref, W_ref = depth.shape

    depths = np.stack(depth_list).astype(np.float32)          # (N, H, W)
    fx = float(np.median(fx_list))   # median focal length across frames
    cx, cy = W_ref / 2.0, H_ref / 2.0
    K = np.array([[fx,  0, cx],
                  [ 0, fx, cy],
                  [ 0,  0,  1]], dtype=np.float32)

    frame_ids = [os.path.basename(p) for p in img_paths]
    return depths, K, frame_ids


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import json as _json
    CAM_CFG = os.path.join(os.path.dirname(__file__), "dexycb_camera_config.json")

    parser = argparse.ArgumentParser(description="Universal Depth Pro batch inference")
    parser.add_argument("--dataset",    required=True, choices=list(DISCOVERERS.keys()))
    parser.add_argument("--input-dir",  default=None)
    parser.add_argument("--seq",        default=None, help="Substring match for sequence ID")
    parser.add_argument("--limit",      type=int, default=0, help="Process first N sequences only")
    parser.add_argument("--max-frames", type=int, default=150,
                        help="Max frames per sequence (default 150 to save memory)")
    parser.add_argument("--cam",        default=None,
                        help="DexYCB camera serial(s) to process, comma-separated. "
                             "Defaults to 'selected' list in dexycb_camera_config.json")
    args = parser.parse_args()

    # Resolve DexYCB camera filter
    cam_filter = None
    if args.dataset == "dexycb":
        if args.cam:
            cam_filter = [c.strip() for c in args.cam.split(",")]
        elif os.path.exists(CAM_CFG):
            cfg = _json.load(open(CAM_CFG))
            cam_filter = cfg.get("selected", None)
        if cam_filter:
            print(f"DexYCB camera filter: {cam_filter}")

    discover_fn, data_folder = DISCOVERERS[args.dataset]
    input_dir  = args.input_dir or os.path.join(RAW_BASE, data_folder)
    output_dir = os.path.join(OUT_BASE, args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    # Build sequence list (DexYCB supports camera filter)
    if args.dataset == "dexycb":
        sequences = list(discover_fn(input_dir, cam_filter=cam_filter))
    else:
        sequences = list(discover_fn(input_dir))
    if args.seq:
        sequences = [(sid, imgs) for sid, imgs in sequences if args.seq in sid]
    if args.limit > 0:
        sequences = sequences[:args.limit]
    print(f"Found {len(sequences)} sequences\n")

    print("Loading Depth Pro model...")
    model, transform = load_depth_pro_model()
    print(f"✅ Model loaded (device: {next(model.parameters()).device})\n")

    done, skipped, failed = 0, 0, 0

    for seq_id, img_paths in tqdm(sequences, desc=f"DepthPro/{args.dataset}"):
        seq_out = os.path.join(output_dir, seq_id)
        depths_path = os.path.join(seq_out, "depths.npz")

        if os.path.exists(depths_path):
            tqdm.write(f"  ⏭  {seq_id}: cached")
            skipped += 1
            continue

        try:
            depths, K, frame_ids = run_depth_pro(model, transform, img_paths, args.max_frames)

            os.makedirs(seq_out, exist_ok=True)
            np.savez_compressed(depths_path, depths=depths)
            np.savetxt(os.path.join(seq_out, "K.txt"), K, fmt="%.6f")
            with open(os.path.join(seq_out, "frame_ids.txt"), "w") as f:
                f.write("\n".join(frame_ids))

            tqdm.write(f"  ✅ {seq_id}: depths {depths.shape}  K fx={K[0,0]:.1f}")
            done += 1

        except Exception as e:
            tqdm.write(f"  ❌ {seq_id}: {e}")
            failed += 1

        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"✅ Done: {done}  ⏭ Skipped: {skipped}  ❌ Failed: {failed}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
