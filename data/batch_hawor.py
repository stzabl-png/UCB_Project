"""
Universal HaWoR batch inference — supports all first-person (egocentric) datasets.

Usage:
  conda activate hawor
  python data/batch_hawor.py --dataset arctic
  python data/batch_hawor.py --dataset fpha
  python data/batch_hawor.py --dataset fpha --seq Subject_1  # single subject
  python data/batch_hawor.py --dataset arctic --seq s05

Output:
  data_hub/ProcessedData/ego_mano/{dataset}/{seq_id}.npz
    right_verts: (N, 778, 3)   right hand MANO vertices (world space)
    left_verts:  (N, 778, 3)   left  hand MANO vertices (world space)
    R_w2c: (N, 3, 3), t_w2c: (N, 3)   camera extrinsics
    img_focal: float

Note: For frame-based datasets (FPHA), frames are temporarily encoded to
      MP4 via ffmpeg before being passed to HaWoR.
"""

import os, sys, shutil, tempfile, subprocess, argparse, numpy as np, torch
from glob import glob
from natsort import natsorted
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

sys.path.insert(0, config.HAWOR_DIR)
sys.path.insert(0, os.path.join(config.HAWOR_DIR, 'thirdparty', 'DROID-SLAM', 'droid_slam'))

from data.megasam_utils import load_megasam_K, load_megasam_K_fx

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_BASE = os.path.join(config.DATA_HUB, "ProcessedData", "ego_mano")
RAW_BASE = os.path.join(config.DATA_HUB, "RawData", "EgoRawData")

HAWOR_CKPT     = os.path.join(config.HAWOR_DIR, 'weights/hawor/checkpoints/hawor.ckpt')
HAWOR_INFILLER = os.path.join(config.HAWOR_DIR, 'weights/hawor/checkpoints/infiller.pt')

# Known SLAM failure sequences (dataset-specific)
SLAM_SKIP = {
    "arctic": {("s09", "scissors_use_01"), ("s10", "scissors_use_02")},
    "fpha":   set(),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def frames_to_video(img_paths, fps=15, out_path=None):
    """Convert a sorted list of frames to MP4 using ffmpeg (lossless-ish)."""
    tmp_dir = tempfile.mkdtemp(prefix="hawor_frames_")
    # Symlink frames with sequential names so ffmpeg pattern works
    ext = os.path.splitext(img_paths[0])[1]
    for i, p in enumerate(img_paths):
        os.symlink(os.path.abspath(p), os.path.join(tmp_dir, f"{i:06d}{ext}"))

    if out_path is None:
        out_path = os.path.join(tmp_dir, "input.mp4")

    res = subprocess.run([
        "ffmpeg", "-y", "-r", str(fps),
        "-i", os.path.join(tmp_dir, f"%06d{ext}"),
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        out_path
    ], capture_output=True, text=True)

    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {res.stderr[-500:]}")

    return out_path, tmp_dir


def run_hawor_on_video(video_path, seq_name, out_path, dataset):
    """Run full HaWoR pipeline on one video file, save result to out_path."""
    from scripts.scripts_test_video.detect_track_video import detect_track_video
    from scripts.scripts_test_video.hawor_video import hawor_motion_estimation, hawor_infiller
    from scripts.scripts_test_video.hawor_slam import hawor_slam
    from hawor.utils.process import run_mano, run_mano_left
    from lib.eval_utils.custom_utils import load_slam_cam

    class Args:
        pass
    args = Args()
    args.video_path   = video_path
    args.input_type   = 'file'
    args.img_focal    = load_megasam_K_fx(seq_name)
    args.checkpoint       = HAWOR_CKPT
    args.infiller_weight  = HAWOR_INFILLER
    args.vis_mode     = 'cam'

    megasam_K = load_megasam_K(seq_name)
    if args.img_focal:
        print(f"  [MegaSAM] focal={args.img_focal:.1f}px for {seq_name}")

    _orig = os.getcwd()
    os.chdir(config.HAWOR_DIR)
    try:
        start_idx, end_idx, seq_folder, imgfiles = detect_track_video(args)
        frame_chunks_all, img_focal = hawor_motion_estimation(args, start_idx, end_idx, seq_folder)

        slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
        if not os.path.exists(slam_path):
            existing = natsorted(glob(os.path.join(seq_folder, "SLAM/hawor_slam_w_scale_*.npz")))
            if existing:
                slam_path = existing[-1]
            else:
                hawor_slam(args, start_idx, end_idx)
        R_w2c_sl, t_w2c_sl, R_c2w_sl, t_c2w_sl = load_slam_cam(slam_path)

        pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = hawor_infiller(
            args, start_idx, end_idx, frame_chunks_all)
    finally:
        os.chdir(_orig)

    N = pred_trans.shape[1]
    R_x = np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=np.float32)

    right_verts = run_mano(pred_trans[1:2,:N], pred_rot[1:2,:N],
                           pred_hand_pose[1:2,:N], betas=pred_betas[1:2,:N])
    right_verts = np.einsum('ij,tnj->tni', R_x, right_verts['vertices'][0].cpu().numpy())

    left_verts = run_mano_left(pred_trans[0:1,:N], pred_rot[0:1,:N],
                               pred_hand_pose[0:1,:N], betas=pred_betas[0:1,:N])
    left_verts = np.einsum('ij,tnj->tni', R_x, left_verts['vertices'][0].cpu().numpy())

    R_c2w = np.einsum('ij,njk->nik', R_x, R_c2w_sl.cpu().numpy())
    t_c2w = np.einsum('ij,nj->ni', R_x, t_c2w_sl.cpu().numpy())
    R_w2c = np.transpose(R_c2w, (0,2,1))
    t_w2c = -np.einsum('nij,nj->ni', R_w2c, t_c2w)

    save_kw = dict(right_verts=right_verts, left_verts=left_verts,
                   R_w2c=R_w2c[:N], t_w2c=t_w2c[:N], img_focal=img_focal,
                   pred_trans=pred_trans.cpu().numpy(),
                   pred_rot=pred_rot.cpu().numpy(),
                   pred_hand_pose=pred_hand_pose.cpu().numpy(),
                   pred_betas=pred_betas.cpu().numpy())
    if megasam_K is not None:
        save_kw['megasam_K'] = megasam_K

    np.savez_compressed(out_path, **save_kw)


# ── Dataset-specific discoverers ──────────────────────────────────────────────

def discover_arctic_ego(input_dir):
    """ARCTIC cam0: {subj}/{seq}/0/*.jpg  → needs temp video"""
    for subj in natsorted(os.listdir(input_dir)):
        subj_dir = os.path.join(input_dir, subj)
        if not os.path.isdir(subj_dir): continue
        for seq in natsorted(os.listdir(subj_dir)):
            cam_dir = os.path.join(subj_dir, seq, "0")
            if not os.path.isdir(cam_dir): continue
            imgs = natsorted(glob(os.path.join(cam_dir, "*.jpg")) +
                             glob(os.path.join(cam_dir, "*.png")))
            if imgs:
                yield f"{subj}__{seq}", imgs, "frames"


def discover_fpha(input_dir):
    """FPHA: {Subject_N}/{action}/frame_XXXXXX.jpeg"""
    for subj in natsorted(os.listdir(input_dir)):
        subj_dir = os.path.join(input_dir, subj)
        if not os.path.isdir(subj_dir): continue
        for action in natsorted(os.listdir(subj_dir)):
            action_dir = os.path.join(subj_dir, action)
            if not os.path.isdir(action_dir): continue
            imgs = natsorted(
                glob(os.path.join(action_dir, "*.jpeg")) +
                glob(os.path.join(action_dir, "*.jpg")) +
                glob(os.path.join(action_dir, "*.png"))
            )
            if imgs:
                yield f"{subj}__{action}", imgs, "frames"


DISCOVERERS = {
    "arctic": (discover_arctic_ego, "arctic"),
    "fpha":   (discover_fpha,       "fpha"),
}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Universal HaWoR batch inference")
    parser.add_argument("--dataset", required=True, choices=list(DISCOVERERS.keys()))
    parser.add_argument("--input-dir", default=None,
                        help="Override input directory")
    parser.add_argument("--seq", default=None,
                        help="Process only sequences matching this substring")
    parser.add_argument("--fps", type=int, default=15,
                        help="FPS for temp video encoding (default 15)")
    parser.add_argument("--max-frames", type=int, default=500,
                        help="Max frames per sequence (0=all)")
    args = parser.parse_args()

    discover_fn, data_folder = DISCOVERERS[args.dataset]
    input_dir  = args.input_dir or os.path.join(RAW_BASE, data_folder)
    output_dir = os.path.join(OUT_BASE, args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f" Dataset   : {args.dataset}")
    print(f" Input     : {input_dir}")
    print(f" Output    : {output_dir}")
    print(f"{'='*60}")

    if not os.path.isdir(input_dir):
        print(f"❌ Input not found: {input_dir}")
        return

    sequences = list(discover_fn(input_dir))
    if args.seq:
        sequences = [(sid, imgs, t) for sid, imgs, t in sequences if args.seq in sid]
    print(f"Found {len(sequences)} sequences\n")

    skip_set = SLAM_SKIP.get(args.dataset, set())
    done, skipped, failed = 0, 0, 0

    for seq_id, img_paths, src_type in tqdm(sequences, desc=f"HaWoR/{args.dataset}"):
        out_path = os.path.join(output_dir, f"{seq_id}.npz")

        if os.path.exists(out_path):
            tqdm.write(f"  ⏭  {seq_id}: cached")
            skipped += 1
            continue

        # Check skip list
        parts = seq_id.split("__")
        if len(parts) >= 2 and (parts[0], parts[1]) in skip_set:
            tqdm.write(f"  ⚠️  {seq_id}: known SLAM failure, skipping")
            skipped += 1
            continue

        # Subsample long sequences
        if args.max_frames > 0 and len(img_paths) > args.max_frames:
            step = len(img_paths) // args.max_frames
            img_paths = img_paths[::step][:args.max_frames]

        tmp_dir = None
        try:
            # Convert frames → temp video if needed
            video_path, tmp_dir = frames_to_video(img_paths, fps=args.fps)
            tqdm.write(f"  🎬 {seq_id}: {len(img_paths)} frames → temp video")

            run_hawor_on_video(video_path, seq_id, out_path, args.dataset)
            tqdm.write(f"  ✅ {seq_id} → {out_path}")
            done += 1

        except Exception as e:
            tqdm.write(f"  ❌ {seq_id}: {e}")
            failed += 1

        finally:
            if tmp_dir and os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)
            torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"✅ Done: {done}  ⏭ Skipped: {skipped}  ❌ Failed: {failed}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
