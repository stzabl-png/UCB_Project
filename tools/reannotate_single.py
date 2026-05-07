#!/usr/bin/env python3
"""
reannotate_single.py — 对单个任务重新标注 mask（覆盖原有 0.png）
==================================================================
复用 sam2_server.py，给定已有的 image.png 让你重新点击标注。

用法:
  conda activate base
  python tools/reannotate_single.py \
      --mask_dir data_hub/ProcessedData/obj_recon_input/egocentric/add_remove_lid

操作:
  左键(FG)  / 右键(BG)   加正/负点
  B                      切换 FG/BG 模式（左键的作用）
  C                      清除所有点，重新开始
  ENTER                  保存 0.png 并退出
  Q                      放弃退出（不保存）

完成后用 --upload 上传到 HuggingFace:
  python tools/reannotate_single.py \
      --mask_dir data_hub/ProcessedData/obj_recon_input/egocentric/add_remove_lid \
      --upload UCBProject/EgoDataMask
"""

import os
import sys
import argparse
import json
import subprocess
import socket
import time
import numpy as np
import cv2
from pathlib import Path

# ── SAM2 server path ─────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
SAM2_SRV   = str(SCRIPT_DIR / "sam2_server.py")


def _find_hawor_python():
    if "HAWOR_PY" in os.environ:
        return os.environ["HAWOR_PY"]
    for candidate in [
        os.path.expanduser("~/anaconda3/envs/hawor/bin/python"),
        os.path.expanduser("~/miniconda3/envs/hawor/bin/python"),
    ]:
        if os.path.exists(candidate):
            return candidate
    return sys.executable  # fallback to current python


# ── SAM2 client (same as sam2_annotate_by_object.py) ─────────────────────────
class SAM2Client:
    def __init__(self, python_exe=None):
        exe = python_exe or _find_hawor_python()
        print(f"Starting SAM2 server via {exe} ...")
        self.proc = subprocess.Popen(
            [exe, SAM2_SRV],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1,
        )
        resp = json.loads(self.proc.stdout.readline())
        if resp.get("status") != "ready":
            raise RuntimeError(f"SAM2 server failed: {resp}")
        print("✅ SAM2 server ready")

    def _call(self, req: dict) -> dict:
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    def set_image(self, img_path: str):
        return self._call({"cmd": "set_image", "path": img_path})

    def predict(self, fg_pts, bg_pts):
        return self._call({"cmd": "predict",
                           "fg": fg_pts, "bg": bg_pts})

    def quit(self):
        try:
            self._call({"cmd": "quit"})
        except Exception:
            pass
        self.proc.terminate()


# ── Overlay helper ────────────────────────────────────────────────────────────
FG_COLOR = (0, 255, 0)   # green dot = foreground
BG_COLOR = (0, 0, 255)   # red   dot = background
OVERLAY_ALPHA = 0.35


def draw_overlay(base_bgr, mask_bool, fg_pts, bg_pts, mode):
    vis = base_bgr.copy()
    if mask_bool is not None:
        overlay = vis.copy()
        overlay[mask_bool] = (0, 200, 100)
        vis = cv2.addWeighted(vis, 1 - OVERLAY_ALPHA, overlay, OVERLAY_ALPHA, 0)
    for (x, y) in fg_pts:
        cv2.circle(vis, (x, y), 8, FG_COLOR, -1)
        cv2.circle(vis, (x, y), 9, (0, 0, 0), 1)
    for (x, y) in bg_pts:
        cv2.circle(vis, (x, y), 8, BG_COLOR, -1)
        cv2.circle(vis, (x, y), 9, (0, 0, 0), 1)
    mode_txt = "Mode: FG (left-click)" if mode == "fg" else "Mode: BG (left-click)"
    cv2.putText(vis, mode_txt, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    cv2.putText(vis, "B=toggle  C=clear  ENTER=save  Q=quit",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    return vis


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_dir", required=True,
                        help="Directory containing image.png (and existing 0.png)")
    parser.add_argument("--upload", default=None,
                        help="HuggingFace repo to upload result, e.g. UCBProject/EgoDataMask")
    parser.add_argument("--repo_subdir", default=None,
                        help="Subdir in HF repo (auto-detected from mask_dir by default)")
    args = parser.parse_args()

    mask_dir  = Path(args.mask_dir)
    image_png = mask_dir / "image.png"
    mask_png  = mask_dir / "0.png"

    if not image_png.exists():
        print(f"❌ {image_png} not found")
        sys.exit(1)

    # Show existing mask info
    if mask_png.exists():
        existing = np.array(__import__('PIL').Image.open(str(mask_png)))
        print(f"Existing mask: {mask_png.name}  nonzero={( existing > 0).sum()}/{existing.size} "
              f"({(existing > 0).mean()*100:.1f}%)")
    else:
        print("No existing mask found — creating new.")

    # Load reference image
    img_bgr = cv2.imread(str(image_png))
    if img_bgr is None:
        print(f"❌ Cannot read {image_png}")
        sys.exit(1)
    h, w = img_bgr.shape[:2]

    # Start SAM2
    sam2 = SAM2Client()
    sam2.set_image(str(image_png))

    # State
    mode     = "fg"
    fg_pts   = []
    bg_pts   = []
    cur_mask = None
    WIN      = f"Re-annotate: {mask_dir.name}"

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, min(w, 1600), min(h * min(w, 1600) // w, 900))

    def mouse_cb(event, x, y, flags, param):
        nonlocal fg_pts, bg_pts, cur_mask
        if event == cv2.EVENT_LBUTTONDOWN:
            if mode == "fg":
                fg_pts.append((x, y))
            else:
                bg_pts.append((x, y))
            resp = sam2.predict(fg_pts, bg_pts)
            if "mask" in resp:
                cur_mask = np.array(resp["mask"], dtype=bool)
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            bg_pts.append((x, y))
            resp = sam2.predict(fg_pts, bg_pts)
            if "mask" in resp:
                cur_mask = np.array(resp["mask"], dtype=bool)
            redraw()

    def redraw():
        vis = draw_overlay(img_bgr, cur_mask, fg_pts, bg_pts, mode)
        cv2.imshow(WIN, vis)

    cv2.setMouseCallback(WIN, mouse_cb)
    redraw()

    saved = False
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord('b') or key == ord('B'):
            mode = "bg" if mode == "fg" else "fg"
            redraw()
        elif key == ord('c') or key == ord('C'):
            fg_pts.clear(); bg_pts.clear(); cur_mask = None
            sam2.set_image(str(image_png))   # reset SAM2 state
            redraw()
        elif key == 13:  # ENTER
            if cur_mask is None:
                print("⚠️  No mask yet — click at least one FG point first")
                continue
            # Save 0.png
            from PIL import Image
            out = Image.fromarray((cur_mask.astype(np.uint8) * 255))
            out.save(str(mask_png))
            nz = cur_mask.sum()
            print(f"✅ Saved {mask_png}  nonzero={nz}/{cur_mask.size} ({nz/cur_mask.size*100:.1f}%)")
            saved = True
            break
        elif key == ord('q') or key == ord('Q'):
            print("Aborted — no changes saved.")
            break
        # also break on window close
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    sam2.quit()

    # Upload to HuggingFace
    if saved and args.upload:
        repo_id = args.upload
        # Determine subdir in HF repo
        if args.repo_subdir:
            subdir = args.repo_subdir
        else:
            # Auto-detect: look for 'egocentric' in path
            parts = mask_dir.parts
            try:
                idx = next(i for i, p in enumerate(parts)
                           if p in ("egocentric", "ycb", "oakink", "taco", "arctic"))
                subdir = "/".join(parts[idx:])
            except StopIteration:
                subdir = f"egocentric/{mask_dir.name}"

        print(f"\n📤 Uploading to {repo_id} → {subdir}/")
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            for fname in ["0.png", "image.png"]:
                local = mask_dir / fname
                if local.exists():
                    api.upload_file(
                        path_or_fileobj=str(local),
                        path_in_repo=f"{subdir}/{fname}",
                        repo_id=repo_id,
                        repo_type="dataset",
                        commit_message=f"Re-annotate {mask_dir.name}/{fname}",
                    )
                    print(f"  ✅ uploaded {subdir}/{fname}")
            print("✅ Upload complete")
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            print(f"   Manual upload:")
            print(f"   huggingface-cli upload {repo_id} {mask_png} {subdir}/0.png --repo-type dataset")


if __name__ == "__main__":
    main()
