#!/usr/bin/env python3
"""
annotate_obj_mask_recon.py — 交互式物体帧选择 + Mask 标注

功能:
  对每个第三视角序列，显示所有帧，让用户：
  1. 用 ← → 翻帧，找到物体清晰可见的帧
  2. 在选定帧上拖拽 bbox 框选物体
  3. ENTER 确认 → 保存到 obj_recon_input/

操作说明:
  ← →        上一帧 / 下一帧
  Page Up/Dn  跳 10 帧
  Home / End  跳到首帧 / 末帧
  左键拖拽    画 bbox
  ENTER       保存当前帧 + mask
  R           重画 bbox
  S           跳过该序列
  Q           退出（已标注的全部保留）

用法:
  conda activate base
  python tools/annotate_obj_mask_recon.py                    # 全部数据集
  python tools/annotate_obj_mask_recon.py --dataset oakink   # 单数据集
  python tools/annotate_obj_mask_recon.py --redo             # 覆盖已有标注

输出:
  data_hub/ProcessedData/obj_recon_input/{dataset}/{seq_id}/
    image.png        ← 选定帧 RGB（SAM3D 输入）
    0.png            ← 二值 mask（SAM3D mask）
    source_frame.txt ← 来源帧路径 + 帧号
"""

import os, sys, argparse
import cv2, numpy as np
from glob import glob
from natsort import natsorted
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

OUT_BASE = os.path.join(config.DATA_HUB, "ProcessedData", "obj_recon_input")
RAW_BASE = os.path.join(config.DATA_HUB, "RawData", "ThirdPersonRawData")
WIN      = "Frame Select & Mask Annotate"

# ── Dataset discoverers ───────────────────────────────────────────────────────

def discover_arctic(input_dir):
    seqs = []
    for subj in natsorted(os.listdir(input_dir)):
        subj_dir = os.path.join(input_dir, subj)
        if not os.path.isdir(subj_dir): continue
        for seq in natsorted(os.listdir(subj_dir)):
            cam_dir = os.path.join(subj_dir, seq, "1")
            if not os.path.isdir(cam_dir): continue
            imgs = natsorted(glob(os.path.join(cam_dir, "*.jpg")) +
                             glob(os.path.join(cam_dir, "*.png")))
            if imgs: seqs.append((f"{subj}__{seq}", imgs))
    return seqs

def discover_oakink(input_dir):
    seqs = []
    for seq_id in natsorted(os.listdir(input_dir)):
        seq_dir = os.path.join(input_dir, seq_id)
        if not os.path.isdir(seq_dir): continue
        imgs = natsorted(glob(os.path.join(seq_dir, "*", "north_west_color_*.png")) +
                         glob(os.path.join(seq_dir, "north_west_color_*.png")))
        if imgs: seqs.append((seq_id, imgs))
    return seqs

def discover_ho3d(input_dir):
    seqs = []
    for split in ["train", "evaluation"]:
        split_dir = os.path.join(input_dir, split)
        if not os.path.isdir(split_dir): continue
        for seq_id in natsorted(os.listdir(split_dir)):
            rgb_dir = os.path.join(split_dir, seq_id, "rgb")
            if not os.path.isdir(rgb_dir): continue
            imgs = natsorted(glob(os.path.join(rgb_dir, "*.jpg")))
            if imgs: seqs.append((f"{split}__{seq_id}", imgs))
    return seqs

def discover_dexycb(input_dir):
    seqs = []
    for subj in natsorted(os.listdir(input_dir)):
        subj_dir = os.path.join(input_dir, subj)
        if not os.path.isdir(subj_dir): continue
        for dt in natsorted(os.listdir(subj_dir)):
            dt_dir = os.path.join(subj_dir, dt)
            if not os.path.isdir(dt_dir): continue
            for serial in natsorted(os.listdir(dt_dir)):
                cam_dir = os.path.join(dt_dir, serial)
                if not os.path.isdir(cam_dir): continue
                imgs = natsorted(glob(os.path.join(cam_dir, "color_*.jpg")))
                if imgs: seqs.append((f"{subj}__{dt}__{serial}", imgs))
    return seqs

DISCOVERERS = {
    "arctic":  (discover_arctic,  "arctic"),
    "oakink":  (discover_oakink,  "oakink_v1"),
    "ho3d_v3": (discover_ho3d,    "ho3d_v3"),
    "dexycb":  (discover_dexycb,  "dexycb"),
}

# ── Mouse state ───────────────────────────────────────────────────────────────
bbox_state = {'drawing': False, 'x0':0,'y0':0,'x1':0,'y1':0, 'done': False}

def mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        bbox_state.update(drawing=True, x0=x,y0=y,x1=x,y1=y, done=False)
    elif event == cv2.EVENT_MOUSEMOVE and bbox_state['drawing']:
        bbox_state.update(x1=x, y1=y)
    elif event == cv2.EVENT_LBUTTONUP:
        bbox_state.update(drawing=False, x1=x, y1=y, done=True)


def draw_frame(frame, fi, total_frames, seq_idx, total_seqs, ds, seq_id, scale):
    disp = frame.copy()
    h, w = disp.shape[:2]

    # Top bar
    cv2.rectangle(disp, (0, 0), (w, 60), (15, 18, 30), -1)
    cv2.putText(disp, f'[Seq {seq_idx+1}/{total_seqs}] {ds}/{seq_id}',
                (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220,220,220), 1)
    cv2.putText(disp, f'Frame {fi+1}/{total_frames}  |  ←→:帧  PgUp/Dn:±10  Home/End  |  拖拽bbox  ENTER:保存  R:重画  S:跳过  Q:退出',
                (6, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100,220,100), 1)

    # Progress bar
    prog = int(w * (fi / max(total_frames-1, 1)))
    cv2.rectangle(disp, (0, 58), (prog, 62), (0,200,80), -1)
    cv2.rectangle(disp, (0, 58), (w, 62), (60,60,60), 1)

    # Bbox
    if bbox_state['done'] or bbox_state['drawing']:
        x0 = min(bbox_state['x0'], bbox_state['x1'])
        y0 = min(bbox_state['y0'], bbox_state['y1'])
        x1 = max(bbox_state['x0'], bbox_state['x1'])
        y1 = max(bbox_state['y0'], bbox_state['y1'])
        cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 255, 80), 2)
        if bbox_state['done']:
            cv2.putText(disp, f'bbox OK — 按 ENTER 保存，R 重画',
                        (x0, max(y0-6, 70)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,80), 1)

    return disp


# ── Per-sequence annotation ───────────────────────────────────────────────────
def annotate_sequence(ds, seq_id, img_paths, seq_idx, total_seqs):
    """Returns True if saved, False if skipped."""
    out_dir   = os.path.join(OUT_BASE, ds, seq_id)
    img_out   = os.path.join(out_dir, "image.png")
    mask_out  = os.path.join(out_dir, "0.png")

    # Load first frame to get display scale
    sample = cv2.imread(img_paths[0])
    if sample is None:
        print(f"  ⚠️  Cannot read {img_paths[0]}")
        return False
    H, W = sample.shape[:2]
    DISP_H = 720
    scale  = DISP_H / H

    total_frames = len(img_paths)
    fi = 0  # current frame index

    # Cache loaded frames (thread-free, simple)
    frame_cache = {}

    def get_frame(idx):
        if idx not in frame_cache:
            f = cv2.imread(img_paths[idx])
            if f is not None:
                frame_cache[idx] = cv2.resize(f, (int(W*scale), DISP_H))
        return frame_cache.get(idx)

    # Reset bbox
    bbox_state.update(drawing=False, done=False, x0=0,y0=0,x1=0,y1=0)
    cv2.resizeWindow(WIN, int(W*scale), DISP_H)

    print(f"\n  📽  {ds}/{seq_id}  ({total_frames} frames)")

    while True:
        frame_d = get_frame(fi)
        if frame_d is None:
            fi = (fi + 1) % total_frames
            continue

        disp = draw_frame(frame_d, fi, total_frames, seq_idx, total_seqs, ds, seq_id, scale)
        cv2.imshow(WIN, disp)
        key = cv2.waitKey(30) & 0xFF

        # Navigation
        if key == 81 or key == 2:    # ← (Linux arrow left = 81, macOS = 2)
            fi = max(0, fi - 1)
        elif key == 83 or key == 3:  # →
            fi = min(total_frames - 1, fi + 1)
        elif key == 85:              # PgUp
            fi = max(0, fi - 10)
        elif key == 86:              # PgDn
            fi = min(total_frames - 1, fi + 10)
        elif key == 80:              # Home
            fi = 0
        elif key == 87:              # End
            fi = total_frames - 1

        # Bbox reset
        elif key in (ord('r'), ord('R')):
            bbox_state.update(done=False, drawing=False)

        # Skip
        elif key in (ord('s'), ord('S')):
            print(f"  ⏭  Skipped")
            return False

        # Quit
        elif key in (ord('q'), ord('Q')):
            return None  # signal quit-all

        # Confirm
        elif key in (13, 10):  # ENTER
            if not bbox_state['done']:
                print("  ⚠️  先画一个 bbox（拖动鼠标）")
                continue

            # Get bbox in original coords
            x0 = int(min(bbox_state['x0'], bbox_state['x1']) / scale)
            y0 = int(min(bbox_state['y0'], bbox_state['y1']) / scale)
            x1 = int(max(bbox_state['x0'], bbox_state['x1']) / scale)
            y1 = int(max(bbox_state['y0'], bbox_state['y1']) / scale)
            x0,x1 = max(0,x0), min(W,x1)
            y0,y1 = max(0,y0), min(H,y1)

            if x1-x0 < 5 or y1-y0 < 5:
                print("  ⚠️  Bbox 太小，重画")
                continue

            # Save
            os.makedirs(out_dir, exist_ok=True)

            # image.png → plain RGB of selected frame
            Image.open(img_paths[fi]).convert("RGB").save(img_out)

            # 0.png → binary mask
            mask = np.zeros((H, W), dtype=np.uint8)
            mask[y0:y1, x0:x1] = 255
            cv2.imwrite(mask_out, mask)

            # source_frame.txt
            with open(os.path.join(out_dir, "source_frame.txt"), "w") as f:
                f.write(f"{img_paths[fi]}\nframe_idx={fi}\nbbox=[{x0},{y0},{x1},{y1}]\n")

            cov = (mask > 0).mean() * 100
            print(f"  ✅ Saved frame {fi+1}/{total_frames}  bbox=[{x0},{y0},{x1},{y1}]  mask={cov:.1f}%")
            return True


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, choices=list(DISCOVERERS.keys()))
    parser.add_argument("--seq",     default=None)
    parser.add_argument("--redo",    action="store_true")
    args = parser.parse_args()

    # Collect all sequences
    all_seqs = []
    datasets = [args.dataset] if args.dataset else list(DISCOVERERS.keys())
    for ds in datasets:
        fn, folder = DISCOVERERS[ds]
        input_dir = os.path.join(RAW_BASE, folder)
        if not os.path.isdir(input_dir):
            print(f"  ⚠️  {ds}: {input_dir} not found, skipping")
            continue
        seqs = fn(input_dir)
        if args.seq:
            seqs = [(s, p) for s, p in seqs if args.seq in s]
        for seq_id, img_paths in seqs:
            all_seqs.append((ds, seq_id, img_paths))

    print(f"Found {len(all_seqs)} sequences across {len(datasets)} dataset(s)\n")

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, mouse_cb)

    done, skipped = 0, 0

    for idx, (ds, seq_id, img_paths) in enumerate(all_seqs):
        out_dir  = os.path.join(OUT_BASE, ds, seq_id)
        img_done = os.path.join(out_dir, "image.png")
        msk_done = os.path.join(out_dir, "0.png")

        if os.path.exists(img_done) and os.path.exists(msk_done) and not args.redo:
            print(f"  ⏭  {ds}/{seq_id}: already annotated (--redo to redo)")
            done += 1
            continue

        result = annotate_sequence(ds, seq_id, img_paths, idx, len(all_seqs))
        if result is None:   # Q pressed → quit all
            print(f"\n  Quit. {done} saved, {skipped} skipped.")
            break
        elif result:
            done += 1
        else:
            skipped += 1

    cv2.destroyAllWindows()

    print(f"\n{'='*60}")
    print(f"✅ 标注完成: {done} 保存 / {skipped} 跳过")
    print(f"输出: {OUT_BASE}")
    print(f"\n上传云端:")
    print(f"  rsync -avz data_hub/ProcessedData/obj_recon_input/ sam3d-gpu:~/input/")
    print(f"\n云端重建:")
    print(f"  CUDA_VISIBLE_DEVICES=0 python batch_infer.py --input-dir ~/input --output-dir ~/output")


if __name__ == "__main__":
    main()
