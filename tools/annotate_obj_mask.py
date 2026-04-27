#!/usr/bin/env python3
"""
annotate_obj_mask.py — 交互式物体 mask 生成

对每个序列的最佳 OBJ 帧，手动框选物体 bbox → SAM2 生成精确 mask

操作:
  鼠标左键拖拽 → 画 bbox
  ENTER       → 确认当前 bbox，运行 SAM2
  R           → 重新画
  S           → 跳过此物体
  Q           → 退出

用法 (base env 有 Qt GUI):
  python tools/annotate_obj_mask.py

输出:
  data/trimmed/<seq>/<seq>_obj/mask_best.png     ← SAM2 生成的 mask
  data/trimmed/<seq>/<seq>_obj/bbox.json         ← bbox 记录
  /tmp/sam3d_upload/<obj>/frame.jpg              ← 上传帧
  /tmp/sam3d_upload/<obj>/mask.png               ← 上传 mask
"""
import os, sys, json, glob, shutil, subprocess
import cv2
import numpy as np

TRIMMED    = '/home/lyh/Project/Affordance2Grasp/data/trimmed'
SAM2_ROOT  = '/home/lyh/Project/sam2'
UPLOAD_DIR = '/tmp/sam3d_upload'
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── 找所有序列 ──────────────────────────────────────────────────────────────
seqs = sorted([os.path.basename(d) for d in glob.glob(f'{TRIMMED}/s05_*_grab_01')])
print(f'Found {len(seqs)} sequences\n')

# ── 全局状态 (for mouse callback) ──────────────────────────────────────────
state = {'drawing': False, 'x0': 0, 'y0': 0, 'x1': 0, 'y1': 0, 'done': False}

def mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        state.update(drawing=True, x0=x, y0=y, x1=x, y1=y, done=False)
    elif event == cv2.EVENT_MOUSEMOVE and state['drawing']:
        state.update(x1=x, y1=y)
    elif event == cv2.EVENT_LBUTTONUP:
        state.update(drawing=False, x1=x, y1=y, done=True)


def run_sam2_with_bbox(img_path, bbox, out_dir):
    """Run SAM2 on single image with bbox prompt → returns mask array."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cmd = [
        'python', f'{SAM2_ROOT}/scripts/amg.py',  # or whatever SAM2 script
    ]
    # If SAM2 doesn't have an easy CLI, fall back to our generate_masks_sam2.py
    script = '/home/lyh/Project/Affordance2Grasp/tools/generate_masks_sam2.py'
    if os.path.exists(script):
        cmd = [
            'python', script,
            '--input_dir', out_dir,
            '--ref_frame', '0',
            '--bbox', str(x1), str(y1), str(x2), str(y2),
            '--sam2_root', SAM2_ROOT,
        ]
        print(f'  Running SAM2: bbox=[{x1},{y1},{x2},{y2}]')
        r = subprocess.run(cmd, capture_output=True, text=True,
                           cwd='/home/lyh/Project/Affordance2Grasp')
        if r.returncode != 0:
            print(f'  SAM2 error: {r.stderr[-300:]}')
            return None
        # Find generated mask
        masks = sorted(glob.glob(f'{out_dir}/masks/*') +
                       glob.glob(f'{out_dir}/*.png'))
        if masks:
            return cv2.imread(masks[0], 0)
    return None


def draw_ui(img, bbox_done, bbox, obj_name, seq, n_done, n_total):
    disp = img.copy()
    h, w = disp.shape[:2]

    # Header
    cv2.rectangle(disp, (0,0), (w, 44), (15,18,30), -1)
    cv2.putText(disp, f'[{n_done+1}/{n_total}] {obj_name} — {seq}',
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220,220,220), 1)
    cv2.putText(disp, 'DRAG:bbox  ENTER:confirm  R:redo  S:skip  Q:quit',
                (6, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140,200,140), 1)

    # Current bbox
    if state['done'] or state['drawing']:
        x0, y0 = min(state['x0'], state['x1']), min(state['y0'], state['y1'])
        x1, y1 = max(state['x0'], state['x1']), max(state['y0'], state['y1'])
        cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 255, 80), 2)
        cv2.putText(disp, f'bbox [{x0},{y0},{x1},{y1}]',
                    (x0, max(y0-6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,80), 1)

    # Bottom hint
    lb = np.full((28, w, 3), 14, np.uint8)
    cv2.putText(lb, f'Choose the TIGHTEST bbox around the object only (no background)',
                (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160,200,160), 1)
    return np.vstack([disp, lb])


results = {}   # seq → {bbox, img_path, mask_path}
WIN = 'Annotate Object Mask'

for ni, seq in enumerate(seqs):
    obj_name = seq.replace('s05_','').replace('_grab_01','')

    # Check if already done
    bbox_f = f'{TRIMMED}/{seq}/s05_{seq.replace("s05_","")}_obj/bbox.json'
    # simpler path
    obj_dirs = glob.glob(f'{TRIMMED}/{seq}/*_obj')
    if not obj_dirs:
        print(f'[SKIP] {seq}: no _obj dir')
        continue
    obj_dir  = obj_dirs[0]
    bbox_f   = f'{obj_dir}/bbox.json'

    if os.path.exists(bbox_f):
        saved = json.load(open(bbox_f))
        print(f'[SKIP] {obj_name}: bbox already saved = {saved["bbox"]}')
        results[seq] = saved
        continue

    obj_imgs = sorted(glob.glob(f'{obj_dir}/*.jpg'))
    if not obj_imgs:
        print(f'[SKIP] {seq}: no frames'); continue

    # Best frame: pick 2nd/3rd frame (first might have motion blur)
    best_img = obj_imgs[min(2, len(obj_imgs)-1)]
    frame    = cv2.imread(best_img)
    H, W     = frame.shape[:2]
    DISP_H   = 720
    scale    = DISP_H / H
    frame_d  = cv2.resize(frame, (int(W*scale), DISP_H))

    # Reset state
    state.update(drawing=False, done=False, x0=0, y0=0, x1=0, y1=0)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, frame_d.shape[1], frame_d.shape[0]+28)
    cv2.setMouseCallback(WIN, mouse_cb)

    action = None
    while True:
        ui = draw_ui(frame_d, state['done'], None, obj_name, seq, ni, len(seqs))
        cv2.imshow(WIN, ui)
        key = cv2.waitKey(20) & 0xFF

        if key == 13 or key == 10:   # ENTER
            if state['done']:
                action = 'confirm'; break
            else:
                print('  [!] Draw a bbox first')
        elif key == ord('r') or key == ord('R'):
            state.update(done=False, drawing=False)
        elif key == ord('s') or key == ord('S'):
            action = 'skip'; break
        elif key == ord('q') or key == ord('Q'):
            action = 'quit'; break

    if action == 'quit':
        cv2.destroyAllWindows()
        print('Quit.'); break
    if action == 'skip':
        print(f'  [SKIP] {obj_name}')
        continue

    # Scale bbox back to original image coords
    x0_d = min(state['x0'], state['x1'])
    y0_d = min(state['y0'], state['y1'])
    x1_d = max(state['x0'], state['x1'])
    y1_d = max(state['y0'], state['y1'])
    bbox_orig = [int(x0_d/scale), int(y0_d/scale),
                 int(x1_d/scale), int(y1_d/scale)]
    print(f'  {obj_name}: bbox = {bbox_orig}')

    # Save bbox
    meta = {'seq': seq, 'obj': obj_name, 'bbox': bbox_orig,
            'frame': os.path.basename(best_img)}
    json.dump(meta, open(bbox_f, 'w'), indent=2)

    # Simple rectangular mask (SAM2 will refine)
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[bbox_orig[1]:bbox_orig[3], bbox_orig[0]:bbox_orig[2]] = 255
    mask_path = f'{obj_dir}/mask_bbox.png'
    cv2.imwrite(mask_path, mask)

    # Also run SAM2 for refined mask (needs hawor/sam2 env)
    print(f'  Running SAM2 mask refinement...')
    script = '/home/lyh/Project/Affordance2Grasp/tools/generate_masks_sam2.py'
    if os.path.exists(script):
        r = subprocess.run([
                sys.executable, script,
                '--input_dir', obj_dir,
                '--ref_frame', '0',
                '--bbox'] + [str(v) for v in bbox_orig] +
                ['--sam2_root', SAM2_ROOT],
            capture_output=True, text=True,
            cwd='/home/lyh/Project/Affordance2Grasp')
        if r.returncode == 0:
            print('  ✓ SAM2 mask generated')
        else:
            print(f'  [SAM2 failed] using bbox mask instead')
            print(r.stderr[-200:])

    # Prepare upload dir
    up_obj = f'{UPLOAD_DIR}/{obj_name}'
    os.makedirs(up_obj, exist_ok=True)
    shutil.copy2(best_img, f'{up_obj}/frame.jpg')
    shutil.copy2(mask_path, f'{up_obj}/mask.png')

    results[seq] = meta
    print(f'  ✓ Upload ready: {up_obj}/')

cv2.destroyAllWindows()

# Summary
print(f'\n{"="*60}')
print(f'完成 {len(results)}/{len(seqs)} 个序列的 mask 标注')
print(f'上传目录: {UPLOAD_DIR}')
print()
print('下一步:')
print(f'  rsync -avz {UPLOAD_DIR}/ root@<SERVER>:/mnt/data/lyh/sam3d_input/')
print(f'  (然后在云服务器上运行 SAM3D)')
