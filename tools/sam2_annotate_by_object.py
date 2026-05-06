#!/usr/bin/env python3
"""
sam2_annotate_by_object.py — 按物体标注（每物体只需标一次）

每个物体给你完整序列的全部帧，你自己选最清晰的帧 → SAM2 点击标注 → 保存

输出结构（每物体一个目录）:
  obj_recon_input/arctic/{obj_name}/image.png + 0.png
  obj_recon_input/oakink/{obj_name}/image.png + 0.png
  obj_recon_input/ycb/{ycb_name}/image.png + 0.png   ← HO3D + DexYCB 共用

操作:
  ← →       上/下一帧
  PgUp/Dn   跳 10 帧
  Home/End  跳到首/末帧
  左键      在当前模式(FG/BG)下加点
  B         切换 FG/BG 模式
  C         清除所有点
  ENTER     保存当前帧 + mask → 下一个物体
  S         跳过该物体
  Q         退出

用法:
  conda activate base
  python tools/sam2_annotate_by_object.py                  # 全部
  python tools/sam2_annotate_by_object.py --dataset arctic # 单数据集
  python tools/sam2_annotate_by_object.py --redo           # 覆盖已有
"""

import os, sys, json, argparse, subprocess, re
import numpy as np
import cv2
from natsort import natsorted
from PIL import Image
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_BASE       = os.path.join(config.DATA_HUB, "ProcessedData", "obj_recon_input")
RAW_BASE       = os.path.join(config.DATA_HUB, "RawData", "ThirdPersonRawData")
TACO_ALLOC_DIR = os.path.join(RAW_BASE, "taco", "Allocentric_RGB_Videos")
HAWOR_PY  = "/home/lyh/anaconda3/envs/hawor/bin/python"
SAM2_SRV  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam2_server.py")
WIN       = "SAM2 Object Annotator"
DISP_H    = 720

# ── HO3D: sequence prefix → YCB object name ──────────────────────────────────
HO3D_OBJ = {
    'ABF': '003_cracker_box', 'BB': '011_banana',
    'GPMF': '010_potted_meat_can', 'GSF': '010_potted_meat_can',
    'MC': '003_cracker_box', 'MDF': '035_power_drill',
    'ND': '035_power_drill', 'SB': '021_bleach_cleanser',
    'ShSu': '004_sugar_box', 'SiBF': '003_cracker_box',
    'SiS': '052_extra_large_clamp', 'SM': '006_mustard_bottle',
    'SMu': '006_mustard_bottle', 'SS': '004_sugar_box',
}

# ── Object discoverers ────────────────────────────────────────────────────────
def discover_arctic(raw_dir):
    """Yield (obj_name, frames_list) — one representative seq per object."""
    seqs_by_obj = {}
    for subj in natsorted(os.listdir(raw_dir)):
        subj_dir = os.path.join(raw_dir, subj)
        if not os.path.isdir(subj_dir): continue
        for seq in natsorted(os.listdir(subj_dir)):
            m = re.match(r'^(.+?)_(grab|use)_', seq)
            if not m: continue
            obj = m.group(1)
            if obj in seqs_by_obj: continue   # already have one
            cam_dir = os.path.join(subj_dir, seq, "1")
            frames = natsorted(glob(os.path.join(cam_dir, "*.jpg")) +
                               glob(os.path.join(cam_dir, "*.png")))
            if frames:
                seqs_by_obj[obj] = frames
    for obj, frames in sorted(seqs_by_obj.items()):
        yield "arctic", obj, frames


def discover_oakink(raw_dir):
    """One seq per OakInk object code."""
    seqs_by_obj = {}
    for seq_id in natsorted(os.listdir(raw_dir)):
        obj_code = seq_id.split('_')[0]
        if obj_code in seqs_by_obj: continue
        seq_dir = os.path.join(raw_dir, seq_id)
        frames = natsorted(glob(os.path.join(seq_dir, "*", "north_west_color_*.png")) +
                           glob(os.path.join(seq_dir, "north_west_color_*.png")))
        if frames:
            seqs_by_obj[obj_code] = frames
    for obj, frames in sorted(seqs_by_obj.items()):
        yield "oakink", obj, frames


def discover_ho3d(raw_dir):
    """One seq per YCB object, shared namespace 'ycb'."""
    seqs_by_obj = {}
    for split in ["train", "evaluation"]:
        split_dir = os.path.join(raw_dir, split)
        if not os.path.isdir(split_dir): continue
        for seq_id in natsorted(os.listdir(split_dir)):
            prefix = re.sub(r'\d+$', '', seq_id)
            ycb_name = HO3D_OBJ.get(prefix)
            if not ycb_name or ycb_name in seqs_by_obj: continue
            rgb_dir = os.path.join(split_dir, seq_id, "rgb")
            frames = natsorted(glob(os.path.join(rgb_dir, "*.jpg")))
            if frames:
                seqs_by_obj[ycb_name] = frames
    for obj, frames in sorted(seqs_by_obj.items()):
        yield "ycb", obj, frames


def discover_dexycb(raw_dir):
    """One capture per YCB object from subject-01. 20 total."""
    # DexYCB subject-01: 100 captures, 5 captures per object, 20 objects
    # Use first capture of each group-of-5 as representative
    subj_dir = os.path.join(raw_dir, "20200709-subject-01")
    if not os.path.isdir(subj_dir): return
    captures = natsorted(os.listdir(subj_dir))
    # 20 objects × 5 captures = 100 (take one per group)
    for i, cap in enumerate(captures):
        if i % 5 != 0: continue   # every 5th = new object
        obj_idx = i // 5 + 1
        obj_name = f"ycb_dex_{obj_idx:02d}"  # placeholder until mapped
        frames = []
        cap_dir = os.path.join(subj_dir, cap)
        for serial in natsorted(os.listdir(cap_dir)):
            serial_dir = os.path.join(cap_dir, serial)
            if not os.path.isdir(serial_dir): continue
            imgs = natsorted(glob(os.path.join(serial_dir, "color_*.jpg")))
            if imgs: frames = imgs; break
        if frames:
            yield "ycb", obj_name, frames


def discover_taco_allocentric(raw_dir):
    """
    TACO Allocentric: groups by triplet name, one representative session per triplet.
    Yields (ds_out, triplet_name, frames) for each unique triplet.
    Requires frames to be pre-extracted via:
        python tools/extract_taco_frames.py --mode pipeline --cam <serial>
    Output saved to: obj_recon_input/taco_allocentric/{triplet}/0.png
    """
    base = TACO_ALLOC_DIR   # raw_dir arg is ignored for TACO (path is hardcoded)
    if not os.path.isdir(base):
        return

    for triplet in natsorted(os.listdir(base)):
        triplet_path = os.path.join(base, triplet)
        if not os.path.isdir(triplet_path):
            continue

        # Find first session that has extracted jpg frames
        best_frames = []
        for session in natsorted(os.listdir(triplet_path)):
            session_path = os.path.join(triplet_path, session)
            if not os.path.isdir(session_path):
                continue
            for cam_serial in natsorted(os.listdir(session_path)):
                cam_dir = os.path.join(session_path, cam_serial)
                if not os.path.isdir(cam_dir):
                    continue
                frames = natsorted(glob(os.path.join(cam_dir, "*.jpg")))
                if frames:
                    best_frames = frames
                    break
            if best_frames:
                break

        if best_frames:
            yield "taco_allocentric", triplet, best_frames


DISCOVERERS = {
    "arctic":           (discover_arctic,           "arctic"),
    "oakink":           (discover_oakink,           "oakink_v1"),
    "ho3d_v3":          (discover_ho3d,             "ho3d_v3"),
    "dexycb":           (discover_dexycb,           "dexycb"),
    "taco_allocentric": (discover_taco_allocentric, "taco"),  # uses TACO_ALLOC_DIR directly
}

# ── SAM2 client ───────────────────────────────────────────────────────────────
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
        ox, oy = int(x / param['scale']), int(y / param['scale'])
        ms['fg' if ms['mode'] == 'fg' else 'bg'].append([ox, oy])
        ms['changed'] = True


def draw_ui(frame_d, fi, total, obj_idx, total_objs, ds, obj_name,
            mask_img, scale, waiting):
    out = frame_d.copy()
    dH, dW = out.shape[:2]

    # Mask overlay
    if mask_img is not None:
        mx = cv2.resize(mask_img, (dW, dH), interpolation=cv2.INTER_NEAREST)
        ov = out.copy(); ov[mx > 0] = (0, 80, 255)
        out = cv2.addWeighted(out, 0.6, ov, 0.4, 0)
        cnts, _ = cv2.findContours(mx, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (0, 255, 100), 2)

    # Points
    for px, py in ms['fg']:
        cv2.circle(out, (int(px*scale), int(py*scale)), 7, (0, 255, 80), -1)
        cv2.circle(out, (int(px*scale), int(py*scale)), 7, (255,255,255), 2)
    for px, py in ms['bg']:
        cv2.circle(out, (int(px*scale), int(py*scale)), 7, (0, 0, 220), -1)
        cv2.circle(out, (int(px*scale), int(py*scale)), 7, (255,255,255), 2)

    # Header
    cv2.rectangle(out, (0,0), (dW, 62), (12,14,22), -1)
    cv2.putText(out, f'[{obj_idx+1}/{total_objs}]  {ds} / {obj_name}',
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220,220,220), 1)

    mode_c = (0,255,80) if ms['mode']=='fg' else (0,80,220)
    mode_t = '● FG' if ms['mode']=='fg' else '● BG'
    cv2.putText(out, mode_t, (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, mode_c, 1)
    info = f'Frame {fi+1}/{total}  FG:{len(ms["fg"])} BG:{len(ms["bg"])}'
    if mask_img is not None:
        info += f'  mask={(mask_img>0).mean()*100:.1f}%'
    if waiting: info += '  [SAM2...]'
    cv2.putText(out, info, (60, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150,200,150), 1)
    cv2.putText(out, 'B:FG/BG  C:清  ←→:帧  PgUp/Dn:±10  Home/End  ENTER:保存  S:跳  Q:退',
                (dW-520, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100,130,100), 1)

    # Frame progress
    prog = int(dW * fi / max(total-1, 1))
    cv2.rectangle(out, (0,59), (prog, 63), (0,180,80), -1)
    cv2.rectangle(out, (0,59), (dW, 63), (40,40,40), 1)
    return out


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, choices=list(DISCOVERERS.keys()))
    parser.add_argument("--redo",    action="store_true")
    parser.add_argument("--start",   type=int, default=0)
    args = parser.parse_args()

    sam2 = SAM2Client()

    # Collect all objects
    all_objs = []   # (ds_out, obj_name, frames)
    datasets = [args.dataset] if args.dataset else list(DISCOVERERS.keys())
    for ds_key in datasets:
        fn, folder = DISCOVERERS[ds_key]
        raw_dir = os.path.join(RAW_BASE, folder)
        if not os.path.isdir(raw_dir):
            print(f"  ⚠️  {ds_key}: {raw_dir} not found, skipping"); continue
        for ds_out, obj_name, frames in fn(raw_dir):
            out_dir = os.path.join(OUT_BASE, ds_out, obj_name)
            img_out = os.path.join(out_dir, "image.png")
            if not args.redo and os.path.exists(img_out):
                continue   # already annotated
            all_objs.append((ds_out, obj_name, frames))

    total_objs = len(all_objs)
    print(f"\n物体总数: {total_objs}")
    if total_objs == 0:
        print("全部已标注！用 --redo 重新标注"); sam2.quit(); return

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    obj_idx = args.start
    while obj_idx < total_objs:
        ds_out, obj_name, all_frames = all_objs[obj_idx]
        out_dir  = os.path.join(OUT_BASE, ds_out, obj_name)
        img_out  = os.path.join(out_dir, "image.png")
        mask_out = os.path.join(out_dir, "0.png")

        fi = 2   # start at frame 3
        current_path = None
        current_mask = None
        H = W = scale = None
        frame_d = None
        ms['fg'] = []; ms['bg'] = []; ms['changed'] = False; ms['mode'] = 'fg'

        print(f"\n  [{obj_idx+1}/{total_objs}] {ds_out}/{obj_name}  ({len(all_frames)} frames)")
        action = None

        while True:
            fi = max(0, min(fi, len(all_frames)-1))
            fpath = all_frames[fi]

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

            if ms['changed'] and ms['fg']:
                resp = sam2.predict(ms['fg'], ms['bg'])
                ms['changed'] = False
                if resp.get("status") == "ok" and resp.get("mask_path"):
                    current_mask = cv2.imread(resp["mask_path"], 0)
                else:
                    current_mask = None

            disp = draw_ui(frame_d, fi, len(all_frames), obj_idx, total_objs,
                           ds_out, obj_name, current_mask, scale, False)
            cv2.imshow(WIN, disp)
            key = cv2.waitKey(30) & 0xFF

            if   key in (81, 2):  fi -= 1
            elif key in (83, 3):  fi += 1
            elif key == 85:       fi = max(0, fi-10)
            elif key == 86:       fi = min(len(all_frames)-1, fi+10)
            elif key == 80:       fi = 0
            elif key == 87:       fi = len(all_frames)-1
            elif key in (ord('b'),ord('B')):
                ms['mode'] = 'bg' if ms['mode']=='fg' else 'fg'
                print(f"  Mode: {ms['mode'].upper()}")
            elif key in (ord('c'),ord('C')):
                ms['fg']=[]; ms['bg']=[]; ms['mode']='fg'; current_mask=None
            elif key in (ord('s'),ord('S')): action='skip'; break
            elif key in (ord('q'),ord('Q'),27): action='quit'; break
            elif key in (13, 10):
                if current_mask is None:
                    print("  ⚠️  先点击物体设置前景点"); continue
                os.makedirs(out_dir, exist_ok=True)
                Image.open(fpath).convert("RGB").save(img_out)
                cv2.imwrite(mask_out, current_mask)
                with open(os.path.join(out_dir, "source_frame.txt"), "w") as f:
                    f.write(f"{fpath}\nframe_idx={fi}\nobj={obj_name}\nds={ds_out}\n")
                cov = (current_mask>0).mean()*100
                print(f"  ✅ {ds_out}/{obj_name}: frame={fi+1}  "
                      f"fg={len(ms['fg'])}  mask={cov:.1f}%")
                action = 'next'; break

        if action == 'quit':
            print(f"\n  Quit at {obj_idx+1}/{total_objs}"); break
        elif action == 'skip':
            print(f"  ⏭  {ds_out}/{obj_name}")
        obj_idx += 1

    cv2.destroyAllWindows()
    sam2.quit()
    print(f"\n✅ 标注完成: {obj_idx}/{total_objs}")
    print(f"输出: {OUT_BASE}")


if __name__ == "__main__":
    main()
