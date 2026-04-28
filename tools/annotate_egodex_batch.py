#!/usr/bin/env python3
"""
annotate_egodex_batch.py — EgoDex 全量物体批量标注工具

在 annotate_egocentric_obj.py 基础上扩展：
  • 自动发现 101 个 task 目录，每个 task 取一个代表性序列做标注
  • 叠加 MegaSAM depth map（热力图）辅助判断物体深度
  • 标注完成后自动写入 sequence_registry.json，供规模化 FP/接触图使用

操作键（与原工具相同）:
  ← →       上/下一帧
  PgUp/Dn   跳 10 帧
  左键      加 FG 点（绿色）
  B         切换 FG/BG 模式（红色 = BG）
  D         切换深度叠加显示
  C         清除所有点
  ENTER     保存 frame + mask，注册序列
  S         跳过此 task（之后可补注）
  Q         退出（保留已标注进度）

用法:
  conda activate base
  cd /home/lyh/Project/Affordance2Grasp
  python tools/annotate_egodex_batch.py

  # 只标注某个 task:
  python tools/annotate_egodex_batch.py --task add_remove_lid

  # 从第 N 个 task 继续:
  python tools/annotate_egodex_batch.py --start 10
"""

import os, sys, json, shutil, subprocess, argparse
import numpy as np
import cv2
from natsort import natsorted
from PIL import Image
from glob import glob
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJ        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_HUB    = os.path.join(PROJ, "data_hub")
EGODEX_ROOT = os.path.join(DATA_HUB, "RawData", "ThirdPersonRawData", "egodex", "test")
DEPTH_BASE  = os.path.join(DATA_HUB, "ProcessedData", "egocentric_depth", "egodex")
MESH_BASE   = os.path.join(DATA_HUB, "ProcessedData", "obj_meshes", "egocentric")
OBJ_INPUT   = os.path.join(DATA_HUB, "ProcessedData", "obj_recon_input", "egocentric")
UPLOAD_DIR  = "/tmp/sam3d_upload"
REGISTRY_F  = os.path.join(PROJ, "tools", "egodex_sequence_registry.json")

HAWOR_PY = "/home/lyh/anaconda3/envs/hawor/bin/python"
SAM2_SRV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam2_server.py")
WIN      = "EgoDex Batch Annotator"
DISP_H   = 720

# ── 自动发现：task → obj_name 约定（task 名即 obj 名，下划线规范化）─────────
def task_to_obj_name(task: str) -> str:
    """
    EgoDex task 名 → obj_name:
      assemble_disassemble_tiles → assemble_tile
      add_remove_lid             → container_lid
    对于未知 task，默认用 task 名（用户可在标注时确认）。
    """
    # 已知映射
    OVERRIDE = {
        "assemble_disassemble_tiles":       "assemble_tile",
        "assemble_disassemble_furniture_bench_chair":  "furniture_chair",
        "assemble_disassemble_furniture_bench_desk":   "furniture_desk",
        "assemble_disassemble_furniture_bench_drawer": "furniture_drawer",
        "add_remove_lid":                   "container_lid",
        "arrange_topple_dominoes":          "domino",
        "build_jenga_tower":                "jenga_block",
        "card_stacking":                    "card",
        "flip_book_pages":                  "book",
    }
    return OVERRIDE.get(task, task.replace("-", "_"))


def discover_tasks():
    """返回所有 (task, seq_subj, frames_dir) 的列表，每个 task 取最长的序列。"""
    tasks_found = []
    for task in natsorted(os.listdir(EGODEX_ROOT)):
        task_dir = os.path.join(EGODEX_ROOT, task)
        if not os.path.isdir(task_dir):
            continue
        # 找所有 extracted_images 子目录
        img_dirs = natsorted(glob(f"{task_dir}/*/extracted_images/"))
        if not img_dirs:
            continue
        # 选帧数最多的序列作为标注代表
        best = max(img_dirs, key=lambda d: len(glob(f"{d}*.jpg")))
        subj = Path(best).parent.name   # e.g. "1"
        n    = len(glob(f"{best}*.jpg"))
        tasks_found.append({
            "task":       task,
            "subj":       subj,
            "frames_dir": best,
            "n_frames":   n,
            "obj_name":   task_to_obj_name(task),
            "seq_id":     f"{task}/{subj}",
        })
    return tasks_found


def load_depth_colormap(task, subj, frame_idx, W, H):
    """叠加 MegaSAM 深度图（如果存在）。返回 (H,W,3) BGR 或 None。"""
    npz = os.path.join(DEPTH_BASE, task, subj, "depth.npz")
    if not os.path.exists(npz):
        return None
    try:
        d = np.load(npz)["depths"]    # (N, Hd, Wd)
        fi = min(frame_idx, d.shape[0] - 1)
        dm = d[fi]                     # (Hd, Wd) float32 metres
        dm_vis = np.clip(dm / dm.max(), 0, 1) if dm.max() > 0 else dm
        dm_u8  = (dm_vis * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(dm_u8, cv2.COLORMAP_TURBO)
        return cv2.resize(heatmap, (W, H))
    except Exception:
        return None


# ── SAM2 Client ───────────────────────────────────────────────────────────────
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


# ── Registry I/O ──────────────────────────────────────────────────────────────
def load_registry():
    if os.path.exists(REGISTRY_F):
        with open(REGISTRY_F) as f:
            return json.load(f)
    return {}

def save_registry(reg):
    with open(REGISTRY_F, "w") as f:
        json.dump(reg, f, indent=2)
    print(f"  Registry → {REGISTRY_F}  ({len(reg)} entries)")


# ── UI ────────────────────────────────────────────────────────────────────────
ms = {'fg': [], 'bg': [], 'changed': False, 'mode': 'fg'}

def mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        s = param['scale']
        ox, oy = int(x / s), int(y / s)
        ms['fg' if ms['mode'] == 'fg' else 'bg'].append([ox, oy])
        ms['changed'] = True


def draw_ui(base, fi, total, task_idx, total_tasks, info, mask_img,
            depth_map, show_depth, scale):
    out  = base.copy()
    dH, dW = out.shape[:2]

    # Depth overlay
    if show_depth and depth_map is not None:
        depth_r = cv2.resize(depth_map, (dW, dH))
        out = cv2.addWeighted(out, 0.55, depth_r, 0.45, 0)

    # Mask overlay
    if mask_img is not None:
        mx = cv2.resize(mask_img, (dW, dH), interpolation=cv2.INTER_NEAREST)
        ov = out.copy(); ov[mx > 0] = (0, 80, 255)
        out = cv2.addWeighted(out, 0.6, ov, 0.4, 0)
        cnts, _ = cv2.findContours(mx, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (0, 255, 100), 2)

    # Points
    for px, py in ms['fg']:
        cv2.circle(out, (int(px*scale), int(py*scale)), 8, (0, 255, 80), -1)
        cv2.circle(out, (int(px*scale), int(py*scale)), 8, (255,255,255), 2)
    for px, py in ms['bg']:
        cv2.circle(out, (int(px*scale), int(py*scale)), 8, (0, 60, 220), -1)
        cv2.circle(out, (int(px*scale), int(py*scale)), 8, (255,255,255), 2)

    # Header bar
    cv2.rectangle(out, (0,0), (dW, 68), (12,14,22), -1)
    obj_name = info.get("obj_name", "")
    task     = info.get("task", "")
    cv2.putText(out, f'[{task_idx+1}/{total_tasks}] {task}  →  obj: {obj_name}',
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220,220,220), 1)
    mode_c = (0,255,80) if ms['mode']=='fg' else (0,80,220)
    mode_t = '● FG (绿)' if ms['mode']=='fg' else '● BG (蓝)'
    cv2.putText(out, mode_t, (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, mode_c, 1)
    stat = f'Frame {fi+1}/{total}  FG:{len(ms["fg"])} BG:{len(ms["bg"])}'
    if mask_img is not None:
        stat += f'  mask={(mask_img>0).mean()*100:.0f}%'
    depth_label = f'  D:深度{"ON" if show_depth else "OFF"}'
    cv2.putText(out, stat + depth_label,
                (130, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (150,200,150), 1)
    hint = 'B:FG/BG  C:清除  D:深度  ←→:帧  PgUp/Dn:±10  ENTER:保存  S:跳过  Q:退出'
    cv2.putText(out, hint, (dW-530, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100,130,100), 1)

    # Progress bar
    prog = int(dW * fi / max(total-1, 1))
    cv2.rectangle(out, (0,65), (prog, 68), (0,180,80), -1)
    cv2.rectangle(out, (0,65), (dW, 68), (40,40,40), 1)
    return out


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="EgoDex 全量物体标注工具")
    parser.add_argument("--task",  default=None, help="只处理此 task 子串")
    parser.add_argument("--start", type=int, default=0, help="从第 N 个 task 开始")
    args = parser.parse_args()

    tasks = discover_tasks()
    if args.task:
        tasks = [t for t in tasks if args.task in t["task"]]
    tasks = tasks[args.start:]

    registry = load_registry()
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # Skip already-annotated
    pending = [t for t in tasks
               if not os.path.exists(os.path.join(OBJ_INPUT, t["obj_name"], "0.png"))]
    print(f"\n  Tasks total : {len(tasks)}")
    print(f"  Already done: {len(tasks) - len(pending)}")
    print(f"  Pending     : {len(pending)}")

    if not pending:
        print("✅ 所有 task 已标注完成！")
        return

    sam2 = SAM2Client()
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    show_depth = True

    for task_idx, t in enumerate(pending):
        task       = t["task"]
        subj       = t["subj"]
        seq_id     = t["seq_id"]
        obj_name   = t["obj_name"]
        frames_dir = t["frames_dir"]

        all_frames = natsorted(glob(os.path.join(frames_dir, "*.jpg")) +
                               glob(os.path.join(frames_dir, "*.png")))
        if not all_frames:
            print(f"  ⚠️  No frames: {frames_dir}"); continue

        print(f"\n[{task_idx+1}/{len(pending)}] {task}  ({len(all_frames)} 帧)")
        print(f"  seq_id: {seq_id}  obj_name: {obj_name}")

        fi = max(0, len(all_frames) // 5)   # 从1/5处开始（物体更可能出现）
        current_path = None
        current_mask = None
        depth_map    = None
        H = W = scale = None
        frame_d = None
        ms['fg'] = []; ms['bg'] = []; ms['changed'] = False; ms['mode'] = 'fg'

        action = None
        while True:
            fi = max(0, min(fi, len(all_frames)-1))
            fpath = all_frames[fi]

            # Load frame when changed
            if fpath != current_path:
                img_pil = Image.open(fpath).convert("RGB")
                H, W    = img_pil.height, img_pil.width
                scale   = DISP_H / H
                dW_px   = int(W * scale)
                frame_d = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                frame_d = cv2.resize(frame_d, (dW_px, DISP_H))
                cv2.resizeWindow(WIN, dW_px, DISP_H)
                cv2.setMouseCallback(WIN, mouse_cb, {'scale': scale})
                sam2.set_image(fpath)
                current_path = fpath
                current_mask = None
                depth_map    = load_depth_colormap(task, subj, fi, dW_px, DISP_H)
                ms['fg'] = []; ms['bg'] = []; ms['changed'] = False

            # SAM2 predict
            if ms['changed'] and ms['fg']:
                resp = sam2.predict(ms['fg'], ms['bg'])
                ms['changed'] = False
                if resp.get("status") == "ok" and resp.get("mask_path"):
                    current_mask = cv2.imread(resp["mask_path"], 0)
                else:
                    current_mask = None

            disp = draw_ui(frame_d, fi, len(all_frames),
                           task_idx, len(pending), t,
                           current_mask, depth_map, show_depth, scale)
            cv2.imshow(WIN, disp)
            key = cv2.waitKey(30) & 0xFF

            if   key in (81, 2):   fi -= 1
            elif key in (83, 3):   fi += 1
            elif key == 85:        fi = max(0, fi-10)
            elif key == 86:        fi = min(len(all_frames)-1, fi+10)
            elif key == 80:        fi = 0
            elif key == 87:        fi = len(all_frames)-1
            elif key in (ord('b'), ord('B')):
                ms['mode'] = 'bg' if ms['mode']=='fg' else 'fg'
            elif key in (ord('d'), ord('D')):
                show_depth = not show_depth
            elif key in (ord('c'), ord('C')):
                ms['fg']=[]; ms['bg']=[]; ms['mode']='fg'; current_mask=None
            elif key in (ord('s'), ord('S')): action='skip'; break
            elif key in (ord('q'), ord('Q'), 27): action='quit'; break
            elif key in (13, 10):  # ENTER
                if current_mask is None or not ms['fg']:
                    print("  ⚠️  先点击物体前景点"); continue
                # ── Save mask → OBJ_INPUT ──
                out_dir  = os.path.join(OBJ_INPUT, obj_name)
                os.makedirs(out_dir, exist_ok=True)
                orig_mask = cv2.resize(current_mask, (W, H), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(os.path.join(out_dir, "0.png"), orig_mask)
                # Save reference image
                shutil.copy2(fpath, os.path.join(out_dir, "image.png"))
                cov = (orig_mask>0).mean()*100
                print(f"  ✅ {obj_name}: frame={fi+1}  mask={cov:.1f}%  → {out_dir}/")

                # ── Also save to /tmp/sam3d_upload for SAM3D upload ──
                upload_dir = os.path.join(UPLOAD_DIR, obj_name)
                os.makedirs(upload_dir, exist_ok=True)
                shutil.copy2(fpath, os.path.join(upload_dir, "frame.jpg"))
                cv2.imwrite(os.path.join(upload_dir, "mask.png"), orig_mask)

                # ── Register sequence ──
                rgb_dir   = str(Path(frames_dir))
                depth_dir = os.path.join(DEPTH_BASE, task, subj)
                pose_dir  = os.path.join(DATA_HUB, "ProcessedData", "obj_poses_ego",
                                         "egodex", seq_id.replace("/", "__"))
                registry[f"egodex__{seq_id}"] = {
                    "dataset":   "egodex",
                    "seq_id":    seq_id,
                    "obj_name":  obj_name,
                    "rgb_dir":   rgb_dir,
                    "depth_dir": depth_dir,
                    "pose_dir":  pose_dir,
                    "hawor_dir": str(Path(frames_dir).parent),
                    "annotated_frame": fi,
                }
                save_registry(registry)
                action = 'next'; break

        if action == 'quit':
            print(f"\n退出（已完成 {task_idx}/{len(pending)}）")
            break
        elif action == 'skip':
            print(f"  ⏭  跳过 {task}")

    cv2.destroyAllWindows()
    sam2.quit()

    # Summary
    done_keys = [k for k in registry if k.startswith("egodex__")]
    print(f"\n{'='*60}")
    print(f"Registry: {len(done_keys)} egodex 序列已注册")
    print(f"\n下一步:")
    print(f"  1. 上传 SAM3D:")
    print(f"     rsync -avz {UPLOAD_DIR}/ root@<SAM3D_IP>:/mnt/data/lyh/sam3d_input/")
    print(f"  2. 下载 mesh 后放到:")
    print(f"     {MESH_BASE}/{{obj_name}}/mesh.ply")
    print(f"  3. 估计 scale:")
    print(f"     conda run -n hawor python data/estimate_obj_scale_ego.py")
    print(f"  4. 运行 FP + 接触图（基于 egodex_sequence_registry.json）")


if __name__ == "__main__":
    main()
