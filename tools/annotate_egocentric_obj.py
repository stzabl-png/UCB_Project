#!/usr/bin/env python3
"""
annotate_egocentric_obj.py — 为自中心视频序列标注物体 mask 并上传 SAM3D

复用 sam2_annotate_by_object.py 的 SAM2Client + GUI 架构
输出格式:  /tmp/sam3d_upload/{obj_name}/frame.jpg
           /tmp/sam3d_upload/{obj_name}/mask.png

操作:
  ← →       上/下一帧
  PgUp/Dn   跳 10 帧
  Home/End  跳到首/末帧
  左键      加 FG 点（绿色）
  B         切换 FG/BG 模式（红色 = BG）
  C         清除所有点
  ENTER     保存当前帧 + mask → 下一个物体
  S         跳过该物体
  Q         退出

用法:
  conda activate base
  cd /home/lyh/Project/Affordance2Grasp
  python tools/annotate_egocentric_obj.py

  # 标完后 rsync 到云端:
  rsync -avz /tmp/sam3d_upload/ root@<SAM3D_IP>:/mnt/data/lyh/sam3d_input/
"""

import os, sys, json, shutil, subprocess
import numpy as np
import cv2
from natsort import natsorted
from PIL import Image
from glob import glob

# ── Paths ─────────────────────────────────────────────────────────────────────
UPLOAD_DIR = "/tmp/sam3d_upload"
HAWOR_PY   = "/home/lyh/anaconda3/envs/hawor/bin/python"
SAM2_SRV   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam2_server.py")
WIN        = "EgoCentric Object Annotator"
DISP_H     = 720

# ── 目标序列: (物体名, 提帧目录, 帧步长) ─────────────────────────────────────
TARGETS = [
    {
        "obj_name":   "assemble_tile",           # SAM3D 物体目录名
        "frames_dir": "/home/lyh/Project/Affordance2Grasp/data_hub/RawData/ThirdPersonRawData"
                      "/egodex/test/assemble_disassemble_tiles/1/extracted_images",
        "start_frame": 0,                         # 建议从哪帧开始看（物体出现帧）
    },
    {
        "obj_name":   "orange",
        "frames_dir": "/home/lyh/Project/Affordance2Grasp/data_hub/RawData/ThirdPersonRawData"
                      "/ph2d/1407-picking_orange_tj_2025-03-12_16-42-19"
                      "/processed_episode_3/extracted_images",
        "start_frame": 50,
    },
]

# ── SAM2 server client (same as sam2_annotate_by_object.py) ───────────────────
class SAM2Client:
    def __init__(self):
        print("Starting SAM2 server...")
        self.proc = subprocess.Popen(
            [HAWOR_PY, SAM2_SRV],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=sys.stderr, text=True, bufsize=1)
        resp = json.loads(self.proc.stdout.readline())
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


def draw_ui(frame_d, fi, total, obj_idx, total_objs, obj_name, mask_img, scale):
    out = frame_d.copy()
    dH, dW = out.shape[:2]

    # Mask overlay
    if mask_img is not None:
        mx = cv2.resize(mask_img, (dW, dH), interpolation=cv2.INTER_NEAREST)
        ov = out.copy(); ov[mx > 0] = (0, 80, 255)
        out = cv2.addWeighted(out, 0.6, ov, 0.4, 0)
        cnts, _ = cv2.findContours(mx, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (0, 255, 100), 2)

    # FG / BG points
    for px, py in ms['fg']:
        cv2.circle(out, (int(px*scale), int(py*scale)), 8, (0, 255, 80), -1)
        cv2.circle(out, (int(px*scale), int(py*scale)), 8, (255,255,255), 2)
    for px, py in ms['bg']:
        cv2.circle(out, (int(px*scale), int(py*scale)), 8, (0, 60, 220), -1)
        cv2.circle(out, (int(px*scale), int(py*scale)), 8, (255,255,255), 2)

    # Header
    cv2.rectangle(out, (0,0), (dW, 64), (12,14,22), -1)
    cv2.putText(out, f'[{obj_idx+1}/{total_objs}] {obj_name}',
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 1)
    mode_c = (0,255,80) if ms['mode']=='fg' else (0,80,220)
    mode_t = '● FG (绿)' if ms['mode']=='fg' else '● BG (蓝)'
    cv2.putText(out, mode_t, (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, mode_c, 1)
    info = f'Frame {fi+1}/{total}  FG:{len(ms["fg"])} BG:{len(ms["bg"])}'
    if mask_img is not None:
        info += f'  mask={(mask_img>0).mean()*100:.1f}%'
    cv2.putText(out, info, (120, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,200,150), 1)
    cv2.putText(out, 'B:FG/BG  C:清除  ←→:帧  PgUp/Dn:±10  ENTER:保存  S:跳过  Q:退出',
                (dW-490, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100,130,100), 1)

    # Progress bar
    prog = int(dW * fi / max(total-1, 1))
    cv2.rectangle(out, (0,61), (prog, 64), (0,180,80), -1)
    cv2.rectangle(out, (0,61), (dW, 64), (40,40,40), 1)
    return out


def main():
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # Check frames dirs exist
    for t in TARGETS:
        if not os.path.isdir(t["frames_dir"]):
            print(f"❌ frames_dir not found: {t['frames_dir']}")
            sys.exit(1)

    sam2 = SAM2Client()
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    for obj_idx, target in enumerate(TARGETS):
        obj_name   = target["obj_name"]
        frames_dir = target["frames_dir"]
        start_fi   = target["start_frame"]

        out_dir = os.path.join(UPLOAD_DIR, obj_name)
        img_out  = os.path.join(out_dir, "frame.jpg")
        mask_out = os.path.join(out_dir, "mask.png")

        if os.path.exists(img_out) and os.path.exists(mask_out):
            print(f"[SKIP] {obj_name} already annotated at {out_dir}")
            continue

        all_frames = natsorted(glob(os.path.join(frames_dir, "*.jpg")) +
                               glob(os.path.join(frames_dir, "*.png")))
        if not all_frames:
            print(f"❌ No frames found in {frames_dir}"); continue

        print(f"\n[{obj_idx+1}/{len(TARGETS)}] {obj_name}  ({len(all_frames)} frames)")
        print(f"  → 从第 {start_fi+1} 帧开始，用 ← → 找到物体最清晰的帧，点击标注后 ENTER")

        fi = min(start_fi, len(all_frames)-1)
        current_path = None
        current_mask = None
        H = W = scale = None
        frame_d = None
        ms['fg'] = []; ms['bg'] = []; ms['changed'] = False; ms['mode'] = 'fg'

        action = None
        while True:
            fi = max(0, min(fi, len(all_frames)-1))
            fpath = all_frames[fi]

            # Load new frame
            if fpath != current_path:
                img_pil = Image.open(fpath).convert("RGB")
                H, W    = img_pil.height, img_pil.width
                scale   = DISP_H / H
                frame_d = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                frame_d = cv2.resize(frame_d, (int(W*scale), DISP_H))
                cv2.resizeWindow(WIN, int(W*scale), DISP_H)
                cv2.setMouseCallback(WIN, mouse_cb, {'scale': scale})
                sam2.set_image(fpath)
                current_path = fpath
                current_mask = None
                ms['fg'] = []; ms['bg'] = []; ms['changed'] = False

            # Run SAM2 when points change
            if ms['changed'] and ms['fg']:
                resp = sam2.predict(ms['fg'], ms['bg'])
                ms['changed'] = False
                if resp.get("status") == "ok" and resp.get("mask_path"):
                    current_mask = cv2.imread(resp["mask_path"], 0)
                else:
                    current_mask = None

            disp = draw_ui(frame_d, fi, len(all_frames), obj_idx, len(TARGETS),
                           obj_name, current_mask, scale)
            cv2.imshow(WIN, disp)
            key = cv2.waitKey(30) & 0xFF

            if   key in (81, 2):   fi -= 1          # ←
            elif key in (83, 3):   fi += 1          # →
            elif key == 85:        fi = max(0, fi-10)          # PgUp
            elif key == 86:        fi = min(len(all_frames)-1, fi+10)  # PgDn
            elif key == 80:        fi = 0            # Home
            elif key == 87:        fi = len(all_frames)-1      # End
            elif key in (ord('b'), ord('B')):
                ms['mode'] = 'bg' if ms['mode']=='fg' else 'fg'
            elif key in (ord('c'), ord('C')):
                ms['fg']=[]; ms['bg']=[]; ms['mode']='fg'; current_mask=None
            elif key in (ord('s'), ord('S')): action='skip'; break
            elif key in (ord('q'), ord('Q'), 27): action='quit'; break
            elif key in (13, 10):   # ENTER
                if current_mask is None or not ms['fg']:
                    print("  ⚠️  先点击物体前景点"); continue
                # Save
                os.makedirs(out_dir, exist_ok=True)
                # Save original-resolution image for SAM3D
                shutil.copy2(fpath, img_out)
                # Save mask at same resolution as image
                orig_mask = cv2.resize(current_mask, (W, H), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(mask_out, orig_mask)
                cov = (orig_mask>0).mean()*100
                print(f"  ✅ {obj_name}: frame={fi+1}  "
                      f"fg={len(ms['fg'])}  mask={cov:.1f}%  → {out_dir}/")
                action = 'next'; break

        if action == 'quit':
            print(f"\n退出（{obj_idx+1}/{len(TARGETS)}）"); break
        elif action == 'skip':
            print(f"  ⏭  跳过 {obj_name}")

    cv2.destroyAllWindows()
    sam2.quit()

    # Summary
    done = [t for t in TARGETS
            if os.path.exists(os.path.join(UPLOAD_DIR, t["obj_name"], "frame.jpg"))]
    print(f"\n{'='*60}")
    print(f"完成 {len(done)}/{len(TARGETS)} 个物体标注")
    print(f"上传目录: {UPLOAD_DIR}")
    print()
    for t in done:
        od = os.path.join(UPLOAD_DIR, t["obj_name"])
        print(f"  {t['obj_name']:25s} → {od}/frame.jpg + mask.png")
    print()
    print("下一步 — 上传到 SAM3D 云端:")
    print(f"  rsync -avz {UPLOAD_DIR}/ root@<SAM3D_IP>:/mnt/data/lyh/sam3d_input/")
    print(f"  # 然后在云端运行 SAM3D（见 batch_infer.py）")


if __name__ == "__main__":
    main()
