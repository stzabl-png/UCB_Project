#!/usr/bin/env python3
"""
annotate_ego_masks.py — 为 EgoDex / TACO Ego 任务手工标注物体 mask（SAM2 点击）

自动发现所有还没有 mask 的任务，一个一个展示给你标注。

输出:  obj_recon_input/egocentric/{task_name}/
         0.png       ← 二值 mask（255=物体，0=背景）
         image.png   ← 对应 RGB 帧

操作:
  ← →        上 / 下一帧（MP4 拖帧）
  PgUp/Dn    跳 30 帧
  Home/End   跳到首 / 末帧
  左键       加 FG 点（绿色）
  B          切换 FG / BG 模式（蓝色 = BG）
  C          清除所有点
  ENTER      保存当前帧 + mask → 进入下一个任务
  S          跳过该任务
  Q          退出（已保存的保留）

用法:
  conda activate base
  cd /home/lyh/Project/Affordance2Grasp
  python tools/annotate_ego_masks.py                    # EgoDex + TACO Ego 全部
  python tools/annotate_ego_masks.py --dataset egodex   # 只跑 EgoDex
  python tools/annotate_ego_masks.py --dataset taco     # 只跑 TACO
  python tools/annotate_ego_masks.py --redo             # 覆盖已有 mask
"""

import os, sys, json, argparse, subprocess, tempfile
import numpy as np
import cv2
from natsort import natsorted
from PIL import Image
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ── Paths ─────────────────────────────────────────────────────────────────────
EGO_RAW   = os.path.join(config.DATA_HUB, "RawData", "EgoRawData")
OUT_BASE  = os.path.join(config.DATA_HUB, "ProcessedData", "obj_recon_input", "egocentric")
EGODEX_ROOT = os.path.join(EGO_RAW, "egodex", "test")
TACO_ROOT   = os.path.join(EGO_RAW, "taco", "Egocentric_RGB_Videos")

# SAM2 server (uses hawor env which has SAM2)
HAWOR_PY  = os.path.join(os.environ.get("CONDA_PREFIX", "").replace("/envs/base",""),
                          "envs", "hawor", "bin", "python")
if not os.path.exists(HAWOR_PY):
    import shutil
    HAWOR_PY = shutil.which("python") or sys.executable
SAM2_SRV  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam2_server.py")
WIN       = "EgoMask Annotator"
DISP_H    = 720


# ── Task discovery ────────────────────────────────────────────────────────────

def discover_missing(redo=False):
    """Yield (task_name, mp4_path, dataset) for tasks without a mask."""
    tasks = []
    existing = set(os.listdir(OUT_BASE)) if os.path.isdir(OUT_BASE) else set()

    # EgoDex
    if os.path.isdir(EGODEX_ROOT):
        for task in sorted(os.listdir(EGODEX_ROOT)):
            task_dir = os.path.join(EGODEX_ROOT, task)
            if not os.path.isdir(task_dir):
                continue
            if not redo and task in existing and os.path.exists(
                    os.path.join(OUT_BASE, task, "0.png")):
                continue
            mp4s = sorted(glob(os.path.join(task_dir, "*.mp4")))
            if mp4s:
                tasks.append((task, mp4s[0], "egodex"))

    # TACO Ego
    if os.path.isdir(TACO_ROOT):
        for task in sorted(os.listdir(TACO_ROOT)):
            task_dir = os.path.join(TACO_ROOT, task)
            if not os.path.isdir(task_dir):
                continue
            if not redo and task in existing and os.path.exists(
                    os.path.join(OUT_BASE, task, "0.png")):
                continue
            # TACO: task/{episode_id}/color.mp4
            mp4s = sorted(glob(os.path.join(task_dir, "*", "color.mp4")))
            if mp4s:
                tasks.append((task, mp4s[0], "taco"))

    return tasks


# ── MP4 frame extractor ───────────────────────────────────────────────────────

class MP4Frames:
    """Lazy frame reader for a single MP4 file."""
    def __init__(self, mp4_path: str):
        self.path = mp4_path
        cap = cv2.VideoCapture(mp4_path)
        self.total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        self._cap = None
        self._last_fi = -1

    def __len__(self):
        return self.total

    def get(self, fi: int):
        """Return BGR frame at index fi."""
        fi = max(0, min(fi, self.total - 1))
        if self._cap is None or self._last_fi != fi - 1:
            if self._cap is None:
                self._cap = cv2.VideoCapture(self.path)
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = self._cap.read()
        if ret:
            self._last_fi = fi
            return frame
        return None

    def save_frame(self, fi: int, path: str):
        frame = self.get(fi)
        if frame is not None:
            cv2.imwrite(path, frame)
            return True
        return False

    def close(self):
        if self._cap:
            self._cap.release()
            self._cap = None


# ── SAM2 client ───────────────────────────────────────────────────────────────

class SAM2Client:
    def __init__(self):
        print("Starting SAM2 server (hawor env)...")
        self.proc = subprocess.Popen(
            [HAWOR_PY, SAM2_SRV],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=sys.stderr, text=True, bufsize=1)
        line = self.proc.stdout.readline()
        resp = json.loads(line)
        if resp.get("status") != "ready":
            raise RuntimeError(f"SAM2 server failed: {resp}")
        print("✅ SAM2 server ready")

    def _send(self, obj):
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    def set_image(self, path): return self._send({"cmd": "set_image", "path": path})
    def predict(self, fg, bg): return self._send({"cmd": "predict", "fg": fg, "bg": bg})
    def quit(self):
        try: self._send({"cmd": "quit"})
        except: pass
        self.proc.wait(timeout=5)


# ── Mouse state ───────────────────────────────────────────────────────────────

ms = {'fg': [], 'bg': [], 'changed': False, 'mode': 'fg'}

def mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        scale = param['scale']
        ox, oy = int(x / scale), int(y / scale)
        ms['fg' if ms['mode'] == 'fg' else 'bg'].append([ox, oy])
        ms['changed'] = True


def draw_ui(disp_frame, fi, total, task_idx, total_tasks, task, ds, mask_img, scale):
    out = disp_frame.copy()
    dH, dW = out.shape[:2]

    # Mask overlay
    if mask_img is not None:
        mx = cv2.resize(mask_img, (dW, dH), interpolation=cv2.INTER_NEAREST)
        ov = out.copy()
        ov[mx > 0] = (0, 80, 255)
        out = cv2.addWeighted(out, 0.6, ov, 0.4, 0)
        cnts, _ = cv2.findContours(mx, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (0, 255, 100), 2)

    # FG / BG points
    for px, py in ms['fg']:
        cv2.circle(out, (int(px*scale), int(py*scale)), 8, (0, 255, 80), -1)
        cv2.circle(out, (int(px*scale), int(py*scale)), 8, (255, 255, 255), 2)
    for px, py in ms['bg']:
        cv2.circle(out, (int(px*scale), int(py*scale)), 8, (30, 30, 220), -1)
        cv2.circle(out, (int(px*scale), int(py*scale)), 8, (255, 255, 255), 2)

    # Header bar
    cv2.rectangle(out, (0, 0), (dW, 64), (12, 14, 22), -1)
    cv2.putText(out, f'[{task_idx+1}/{total_tasks}]  {ds} / {task}',
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    mode_c = (0, 255, 80) if ms['mode'] == 'fg' else (30, 80, 240)
    mode_t = '● FG (绿)' if ms['mode'] == 'fg' else '● BG (蓝)'
    cv2.putText(out, mode_t, (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, mode_c, 1)
    info = f'Frame {fi+1}/{total}  FG:{len(ms["fg"])} BG:{len(ms["bg"])}'
    if mask_img is not None:
        info += f'  mask={(mask_img>0).mean()*100:.1f}%'
    cv2.putText(out, info, (130, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 200, 150), 1)
    cv2.putText(out, 'B:FG/BG  C:清除  ←→:帧  PgUp/Dn:±30  Home/End  ENTER:保存  S:跳过  Q:退出',
                (dW - 560, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (100, 130, 100), 1)

    # Progress
    prog = int(dW * fi / max(total - 1, 1))
    cv2.rectangle(out, (0, 61), (prog, 64), (0, 180, 80), -1)
    cv2.rectangle(out, (0, 61), (dW, 64), (40, 40, 40), 1)
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="both",
                        choices=["egodex", "taco", "both"])
    parser.add_argument("--redo", action="store_true",
                        help="Re-annotate tasks that already have masks")
    parser.add_argument("--start", type=int, default=0,
                        help="Start from this task index (for resuming)")
    args = parser.parse_args()

    os.makedirs(OUT_BASE, exist_ok=True)

    tasks = discover_missing(redo=args.redo)

    # Filter by dataset
    if args.dataset == "egodex":
        tasks = [(t, p, ds) for t, p, ds in tasks if ds == "egodex"]
    elif args.dataset == "taco":
        tasks = [(t, p, ds) for t, p, ds in tasks if ds == "taco"]

    tasks = tasks[args.start:]
    total_tasks = len(tasks)

    if total_tasks == 0:
        print("✅ 所有任务已有 mask！用 --redo 重新标注")
        return

    print(f"\n待标注: {total_tasks} 个任务 (从第 {args.start+1} 个开始)")
    print(f"输出:   {OUT_BASE}\n")

    sam2 = SAM2Client()
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    with tempfile.TemporaryDirectory(prefix="ego_ann_") as tmpdir:
        for task_idx, (task, mp4_path, ds) in enumerate(tasks):
            print(f"\n[{task_idx+1}/{total_tasks}] {ds} / {task}")
            print(f"  MP4: {mp4_path}")

            out_dir  = os.path.join(OUT_BASE, task)
            img_out  = os.path.join(out_dir, "image.png")
            mask_out = os.path.join(out_dir, "0.png")

            reader = MP4Frames(mp4_path)
            total_frames = len(reader)
            print(f"  Frames: {total_frames}")

            fi = 0   # start from first frame
            current_fi = -1
            current_frame_path = None
            current_mask = None
            ms['fg'] = []; ms['bg'] = []
            ms['changed'] = False; ms['mode'] = 'fg'
            frame_d = None
            scale = 1.0

            action = None

            while True:
                fi = max(0, min(fi, total_frames - 1))

                # Load new frame
                if fi != current_fi:
                    bgr = reader.get(fi)
                    if bgr is None:
                        fi = current_fi or 0
                        continue
                    H, W = bgr.shape[:2]
                    scale = DISP_H / H
                    frame_d = cv2.resize(bgr, (int(W * scale), DISP_H))
                    cv2.resizeWindow(WIN, int(W * scale), DISP_H)
                    cv2.setMouseCallback(WIN, mouse_cb, {'scale': scale})

                    # Save temp frame for SAM2
                    current_frame_path = os.path.join(tmpdir, "frame.jpg")
                    cv2.imwrite(current_frame_path, bgr)
                    sam2.set_image(current_frame_path)

                    current_fi = fi
                    current_mask = None
                    ms['fg'] = []; ms['bg'] = []
                    ms['changed'] = False

                # SAM2 inference when points change
                if ms['changed'] and ms['fg']:
                    resp = sam2.predict(ms['fg'], ms['bg'])
                    ms['changed'] = False
                    if resp.get("status") == "ok" and resp.get("mask_path"):
                        current_mask = cv2.imread(resp["mask_path"], 0)
                    else:
                        current_mask = None

                disp = draw_ui(frame_d, fi, total_frames, task_idx, total_tasks,
                               task, ds, current_mask, scale)
                cv2.imshow(WIN, disp)
                key = cv2.waitKey(30) & 0xFF

                if   key in (81, 2):  fi -= 1
                elif key in (83, 3):  fi += 1
                elif key == 85:       fi = max(0, fi - 30)
                elif key == 86:       fi = min(total_frames - 1, fi + 30)
                elif key == 80:       fi = 0
                elif key == 87:       fi = total_frames - 1
                elif key in (ord('b'), ord('B')):
                    ms['mode'] = 'bg' if ms['mode'] == 'fg' else 'fg'
                elif key in (ord('c'), ord('C')):
                    ms['fg'] = []; ms['bg'] = []
                    ms['mode'] = 'fg'; current_mask = None
                elif key in (ord('s'), ord('S')):
                    print(f"  ⏭  跳过 {task}")
                    action = 'skip'; break
                elif key in (ord('q'), ord('Q'), 27):
                    action = 'quit'; break
                elif key in (13, 10):   # ENTER
                    if current_mask is None or not ms['fg']:
                        print("  ⚠️  先点击物体前景点（左键）"); continue
                    # Save
                    os.makedirs(out_dir, exist_ok=True)
                    bgr_orig = reader.get(fi)
                    rgb_orig = cv2.cvtColor(bgr_orig, cv2.COLOR_BGR2RGB)
                    Image.fromarray(rgb_orig).save(img_out)
                    cv2.imwrite(mask_out, current_mask)
                    cov = (current_mask > 0).mean() * 100
                    print(f"  ✅ {task}: frame={fi+1}  "
                          f"fg={len(ms['fg'])}  mask={cov:.1f}%  → {out_dir}/")
                    action = 'next'; break

            reader.close()

            if action == 'quit':
                print(f"\n退出 ({task_idx+1}/{total_tasks})"); break

    cv2.destroyAllWindows()
    sam2.quit()

    done = sum(1 for t, _, _ in tasks
               if os.path.exists(os.path.join(OUT_BASE, t, "0.png")))
    print(f"\n{'='*60}")
    print(f"✅ 完成: {done}/{total_tasks}")
    print(f"输出:   {OUT_BASE}")
    print(f"\n标注完后上传到 HF:")
    print(f"  python data/prepare_ego_masks.py --upload")


if __name__ == "__main__":
    main()
