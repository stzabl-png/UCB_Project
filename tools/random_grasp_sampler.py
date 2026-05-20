#!/usr/bin/env python3
"""
M2 Random Grasp Sampler v2
==========================
内部采样 + ±XYZ 6方向 + Raycast 居中 + 评分系统 + HP引导
采样策略: 50% Human-Prior 引导 + 50% 纯随机

迭代生成: 每批20点×6方向 → 评分 → 不够20个>60分 → 再来一批
最终输出: top 20 高质量候选 (分数>60)

用法:
    # Run from project root
    python3 tools/random_grasp_sampler.py --obj A01001           # OakInk 单个物体
    python3 tools/random_grasp_sampler.py --all                  # OakInk 全部物体
    python3 tools/random_grasp_sampler.py --arctic               # ARCTIC 全部 10 个物体
    python3 tools/random_grasp_sampler.py --arctic --obj scissors # ARCTIC 单个物体
"""
import os, sys, glob, argparse, time
import numpy as np
import trimesh
import h5py
from scipy.spatial.transform import Rotation

import json
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HP_DIR        = os.path.join(PROJ, 'data_hub', 'ProcessedData', 'training_fp')  # 主要来源
INFER_HP_DIR  = os.path.join(PROJ, 'data_hub', 'human_prior_infer')
OUTPUT_DIR    = os.path.join(PROJ, 'output', 'grasps_candidate')
INFER_OUT_DIR = os.path.join(PROJ, 'output', 'grasps_infer')

# ── 统一 Mesh 来源 ─────────────────────────────────────────────────────────────
OBJ_MESHES_DIR  = os.path.join(PROJ, 'data_hub', 'ProcessedData', 'obj_meshes')
OBJ_MESHES_DATASETS = ['oakink', 'ycb', 'arctic', 'dexycb', 'egocentric', 'ho3d_v3']
TRAINING_FP_DIR = os.path.join(PROJ, 'data_hub', 'ProcessedData', 'training_fp')
# Canonical rotation file (用于 SAM3D mesh 朝向修正)
CANONICAL_ROT_JSON = os.path.join(PROJ, 'sim', 'canonical_rotation.json')

# ARCTIC legacy — 先把项目根目录加入 sys.path，再 import config
import sys as _sys; _sys.path.insert(0, PROJ)
import config as _cfg
ARCTIC_ROOT = _cfg.ARCTIC_ROOT
ARCTIC_OBJS = ('box capsulemachine espressomachine ketchup microwave '
               'mixer notebook phone scissors waffleiron').split()
ARCTIC_MESH_DIR = os.path.join(ARCTIC_ROOT, 'meta', 'object_vtemplates')
MAX_GRIPPER_OPEN = 0.08

# 与 convert_arctic_to_usd.py 保持一致的规范化旋转
# Key = orig arctic obj name, Value = 3×3 R, applied as: verts = (R @ verts.T).T
ARCTIC_CANONICAL_ROT = {
    'ketchup': np.array([[ 0, 0,-1],
                          [ 0, 1, 0],
                          [ 1, 0, 0]], dtype=np.float64),  # 长轴 X→Z, 竖立
    'phone':   np.array([[ 0, 0,-1],
                          [ 0, 1, 0],
                          [ 1, 0, 0]], dtype=np.float64),  # 长轴 X→Z, 竖立
}
MIN_GRIPPER_WIDTH = 0.005
# MAX_GRIPPER_OPEN = 0.08  已在上方第45行定义 (Franka 最大开口 8cm)
N_POINTS_PER_BATCH  = 20     # 每批采样点数
N_APPROACH_PER_PT   = 6      # 每个内部点随机采样的 approach 方向数
TARGET_HIGH_QUALITY = 50     # 目标高质量候选数
SCORE_THRESHOLD     = 70.0   # 高质量门槛 (R3 提高到 70)
SKIP_AFTER_BATCHES  = 8      # 超过此批次仍无高质量候选 → 标记为难抓物体并跳过
MAX_BATCHES         = 40     # 最大迭代批次


def sample_approach_dirs(n: int, z_max_cos: float = 0.3) -> list:
    """
    在 canonical 坐标系（Z=竖直向上）中均匀采样 approach 方向。
    
    约定: approach 向量指向夹爪进入物体的方向
      - approach.z < 0  → 从上往下 (top-down)  ✅ 允许
      - approach.z ≈ 0  → 水平侧向              ✅ 允许
      - approach.z > z_max_cos → 从下往上 (桌下穿入) ❌ 禁止
    
    z_max_cos=0.3 表示排除与+Z夹角<72°的向上方向（即wrist在物体正下方）。
    """
    dirs = []
    while len(dirs) < n:
        v = np.random.randn(3).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-8
        if v[2] <= z_max_cos:   # 排除从桌底穿入的向上方向
            dirs.append(v)
    return dirs


def load_canonical_rotations():
    """加载 canonical_rotation.json (SAM3D mesh → 正确朝向的旋转)."""
    if os.path.exists(CANONICAL_ROT_JSON):
        with open(CANONICAL_ROT_JSON) as f:
            d = json.load(f)
        return {k: v for k, v in d.items() if not k.startswith('_')}
    return {}


def find_obj_mesh(obj_id, dataset=None):
    """在 obj_meshes/{dataset}/{obj_id}/ 中查找 mesh.ply + scale.json.

    Returns:
        mesh_path   (str | None)
        scale_factor (float)  — 1.0 if scale.json not found
        dataset     (str | None) — 所属数据集名
    """
    datasets = [dataset] if dataset else OBJ_MESHES_DATASETS
    for ds in datasets:
        mesh_path  = os.path.join(OBJ_MESHES_DIR, ds, obj_id, 'mesh.ply')
        scale_path = os.path.join(OBJ_MESHES_DIR, ds, obj_id, 'scale.json')
        if not os.path.exists(mesh_path):
            continue
        scale_factor = 1.0
        if os.path.exists(scale_path):
            with open(scale_path) as f:
                scale_factor = float(json.load(f)['scale_factor'])
        return mesh_path, scale_factor, ds
    return None, 1.0, None


def list_dataset_objs(dataset):
    """列出 obj_meshes/{dataset}/ 中有 mesh.ply 的物体 ID 列表."""
    ds_dir = os.path.join(OBJ_MESHES_DIR, dataset)
    if not os.path.isdir(ds_dir):
        return []
    return sorted(
        o for o in os.listdir(ds_dir)
        if os.path.exists(os.path.join(ds_dir, o, 'mesh.ply'))
    )


def load_human_prior(obj_id, hp_dir=None, dataset=None):
    """
    加载 HumanPrior。搜索顺序:
      1. ProcessedData/training_fp/{dataset}/{obj_id}.hdf5   (新格式 ★推荐)
      2. ProcessedData/training_fp/oakink/{obj_id}.hdf5
      3. {hp_dir}/{obj_id}.hdf5  (legacy)
      4. {hp_dir}/oakink_{obj_id}.hdf5
    """
    # 候选路径列表
    candidates = []

    # 优先搜索 training_fp 各数据集子目录
    for ds in ([dataset] if dataset else ['oakink', 'ycb', 'dexycb', 'arctic']):
        candidates.append(os.path.join(TRAINING_FP_DIR, ds, f'{obj_id}.hdf5'))

    # Legacy fallback
    if hp_dir is None:
        hp_dir = HP_DIR
    candidates += [
        os.path.join(hp_dir, f'{obj_id}.hdf5'),
        os.path.join(hp_dir, f'oakink_{obj_id}.hdf5'),
        os.path.join(hp_dir, f'arctic_{obj_id}.hdf5'),
        os.path.join(hp_dir, f'grab_{obj_id}.hdf5'),
    ]

    for path in candidates:
        if os.path.exists(path):
            with h5py.File(path, 'r') as f:
                return f['point_cloud'][()].astype(np.float32), f['human_prior'][()].astype(np.float32)
    return None, None


def sample_points(mesh, hp_pc, hp_labels, n_total, has_hp):
    """50% Human-Prior-guided + 50% 纯随机采样.
    
    HP-guided: 在 human_prior > 0.3 的顶点附近 ±5mm jitter, 找 mesh 内部点.
    随机:      在 bbox 内均匀随机采样, 取 mesh 内部点.
    """
    points = []

    # ── 50% HP-guided ──────────────────────────────────────────
    n_hp = n_total // 2 if has_hp else 0
    if n_hp > 0 and hp_pc is not None and hp_labels is not None:
        high_mask = hp_labels > 0.3
        if high_mask.sum() > 0:
            hp_pts = hp_pc[high_mask]
            weights = hp_labels[high_mask]
            weights = weights / (weights.sum() + 1e-8)
            # 按 prior 概率加权采样 HP 顶点 (允许重复)
            chosen_idx = np.random.choice(len(hp_pts), size=n_hp * 10, replace=True, p=weights)
            chosen = hp_pts[chosen_idx]
            # 加 ±5mm jitter
            jitter = np.random.randn(*chosen.shape) * 0.005
            candidates = chosen + jitter
            # 只保留在 mesh 内部的点
            inside = mesh.contains(candidates)
            for p in candidates[inside][:n_hp]:
                points.append(p.astype(np.float32))

    n_hp_got = len(points)

    # ── 50% 纯随机 (补足至 n_total) ────────────────────────────
    n_rand = n_total - n_hp_got
    if n_rand > 0:
        bbox_min, bbox_max = mesh.bounds[0], mesh.bounds[1]
        all_pts = np.random.uniform(bbox_min, bbox_max, size=(n_rand * 20, 3))
        inside = mesh.contains(all_pts)
        for p in all_pts[inside][:n_rand]:
            points.append(p.astype(np.float32))

    return points


def choose_finger_dir(approach):
    up = np.array([0, 0, 1], dtype=np.float32)
    if abs(np.dot(approach, up)) > 0.9:
        return np.array([1, 0, 0], dtype=np.float32)
    else:
        finger = np.cross(approach, up)
        return (finger / (np.linalg.norm(finger) + 1e-8)).astype(np.float32)


def make_rotation_matrix(approach, finger_dir):
    z = approach / (np.linalg.norm(approach) + 1e-8)
    x = finger_dir / (np.linalg.norm(finger_dir) + 1e-8)
    y = np.cross(z, x)
    y = y / (np.linalg.norm(y) + 1e-8)
    x = np.cross(y, z)
    x = x / (np.linalg.norm(x) + 1e-8)
    R = np.column_stack([x, y, z]).astype(np.float32)
    if np.linalg.det(R) < 0:
        R = np.column_stack([-x, y, z]).astype(np.float32)
    return R


def score_candidate(mesh, width, approach, finger_dir, grasp_center,
                    contact_L, contact_R, z_min, z_max, mesh_rc=None):
    """
    物理评分 v5.1 (mesh_rc: 用简化 mesh 做法线查询，大幅加速高面数物体)
    """
    score_mesh = mesh_rc if mesh_rc is not None else mesh

    # === 1. 反力分 (Antipodal, 35%) ===
    closest_L, _, tri_L = score_mesh.nearest.on_surface([contact_L])
    closest_R, _, tri_R = score_mesh.nearest.on_surface([contact_R])
    normal_L = score_mesh.face_normals[tri_L[0]]
    normal_R = score_mesh.face_normals[tri_R[0]]
    antipodal_dot = -np.dot(normal_L, finger_dir) * np.dot(normal_R, finger_dir)
    antipodal_score = float(np.clip(antipodal_dot, 0, 1))

    # === 2. 中心轴对齐分 (Axis Alignment, 25%) ===
    # 物体竖直中轴: 过 XY 重心、方向为世界 Z 轴的直线
    # 越靠近中轴 → 抓取越对称，物体不易侧翻
    centroid_xy = mesh.centroid[:2]
    gc_xy = np.array(grasp_center[:2], dtype=np.float64)
    dist_to_axis = float(np.linalg.norm(gc_xy - centroid_xy))
    extents = mesh.bounds[1] - mesh.bounds[0]
    xy_radius = float(max(extents[0], extents[1]) / 2.0 + 1e-8)
    axis_score = float(np.clip(1.0 - dist_to_axis / xy_radius, 0, 1))

    # === 3. 宽度分 (Width, 20%) ===
    ws = float(np.clip(1.0 - abs(width - 0.035) / 0.045, 0, 1))

    # === 4. Franka 可达性分 (Reachability, 20%) — 连续评分 ===
    # +Y 正前方最可达, 从下方 (-Z) 最难
    # 用 approach 方向与理想方向的余弦相似度做连续评分
    app = np.array(approach, dtype=np.float32)
    app = app / (np.linalg.norm(app) + 1e-8)
    # 理想方向混合: 0.6×+Y + 0.4×-Z (正面偏顶部)
    ideal = np.array([0.0, 0.6, -0.4], dtype=np.float32)
    ideal /= np.linalg.norm(ideal)
    cos_sim = float(np.dot(app, ideal))         # [-1, 1]
    reach_score = float(np.clip((cos_sim + 1) / 2, 0, 1))  # → [0, 1]

    # 合计: 0.35 + 0.25 + 0.20 + 0.20 = 1.00 → × 100
    return (0.35 * antipodal_score +
            0.25 * axis_score +
            0.20 * ws +
            0.20 * reach_score) * 100


def _local_flatness(mesh, point, radius=0.01):
    """计算接触点附近的表面平整度 (法线一致性)."""
    # 找附近的面
    center = np.array(point)
    face_centers = mesh.triangles_center
    dists = np.linalg.norm(face_centers - center, axis=1)
    nearby = dists < radius
    if np.sum(nearby) < 2:
        nearby = dists < radius * 3  # 扩大搜索
    if np.sum(nearby) < 2:
        return 0.5  # 默认中等
    normals = mesh.face_normals[nearby]
    # 法线一致性: 所有法线的平均方向 vs 各法线的 cos 相似度
    mean_n = normals.mean(axis=0)
    mean_n = mean_n / (np.linalg.norm(mean_n) + 1e-8)
    cos_sims = np.dot(normals, mean_n)
    return float(np.clip(np.mean(cos_sims), 0, 1))


def check_finger_reachable(mesh, grasp_center, approach, max_finger_depth=0.04):
    """检查手指能否从 approach 方向到达抓取中心.
    
    从 grasp_center 沿 -approach (向外) 射线 → 打到物体表面
      距离 ≤ 4cm (手指长度) → 手指够得到 ✅
      距离 > 4cm → 手指伸不到 ❌
    """
    hits, _, _ = mesh.ray.intersects_location([grasp_center], [-approach])
    if len(hits) == 0:
        return True  # 没打到表面 = 抓取中心在物体外部边缘, 一定够得到
    
    dists = np.linalg.norm(hits - grasp_center, axis=1)
    nearest_dist = np.min(dists)
    
    return nearest_dist <= max_finger_depth


def generate_one_batch(mesh, points, z_min, z_max, mesh_rc=None):
    """从一批采样点生成候选并评分."""
    PALM_CLEARANCE  = 0.010   # 手掌到近端面最小间距 1cm
    FRANKA_FINGER_D = 0.040   # Franka 指深 4cm

    rc = mesh_rc if mesh_rc is not None else mesh   # 简化 mesh 用于 raycast
    candidates = []
    for pt in points:
        for approach in sample_approach_dirs(N_APPROACH_PER_PT):
            finger_dir = choose_finger_dir(approach)

            hits_pos, _, _ = rc.ray.intersects_location([pt], [finger_dir])
            hits_neg, _, _ = rc.ray.intersects_location([pt], [-finger_dir])

            if len(hits_pos) == 0 or len(hits_neg) == 0:
                continue

            d_pos = np.linalg.norm(hits_pos - pt, axis=1)
            d_neg = np.linalg.norm(hits_neg - pt, axis=1)
            nearest_pos = hits_pos[np.argmin(d_pos)]
            nearest_neg = hits_neg[np.argmin(d_neg)]

            width = np.linalg.norm(nearest_pos - nearest_neg)
            if width > MAX_GRIPPER_OPEN or width < MIN_GRIPPER_WIDTH:
                continue

            grasp_center = ((nearest_pos + nearest_neg) / 2.0).astype(np.float32)

            # ── 手指深度检查: d_near ≤ 夹爪活动段长度(4cm) ─────────────────
            # Franka TCP_OFFSET=10.5cm, 固定段6.5cm, 活动段4cm
            # 手掌面在 grasp_center - approach*4cm
            # d_near > 4cm → 手掌面会顶进物体 → 丢弃
            FINGER_ACTIVE_DEPTH = 0.040   # = TCP_OFFSET(10.5) - palm_fixed(6.5)
            hits_near, _, _ = rc.ray.intersects_location(
                [grasp_center], [-approach]
            )
            if len(hits_near):
                d_near = float(np.min(
                    np.linalg.norm(hits_near - grasp_center, axis=1)
                ))
                if d_near > FINGER_ACTIVE_DEPTH:
                    continue   # 手掌会顶物体，丢弃
            else:
                d_near = 0.0

            R = make_rotation_matrix(approach, finger_dir)
            gripper_width = float(np.clip(width + 0.005, 0.01, MAX_GRIPPER_OPEN))
            score = score_candidate(
                mesh, width, approach, finger_dir,
                grasp_center, nearest_neg, nearest_pos,
                z_min, z_max, mesh_rc=rc
            )

            candidates.append({
                'name':               '',
                'position':           grasp_center,   # 接触中点 (Sim 再减 TCP_OFFSET=0.105)
                'grasp_point':        grasp_center,   # 同上，保留字段
                'rotation':           R,
                'gripper_width':      gripper_width,
                'approach':           approach.copy(),
                'finger_dir':         finger_dir.copy(),
                'contact_L':          nearest_neg.astype(np.float32),
                'contact_R':          nearest_pos.astype(np.float32),
                'score':              score,
                'cross_section_width': float(width),
                'd_near':             d_near,
            })
    return candidates


def generate_candidates_iterative(mesh, obj_id, hp_dir=None, mesh_rc=None):
    """迭代生成候选, 直到有 TARGET_HIGH_QUALITY 个分数 > SCORE_THRESHOLD."""
    hp_pc, hp_labels = load_human_prior(obj_id, hp_dir=hp_dir)
    has_hp = hp_pc is not None and np.any(hp_labels > 0.5)

    # 尺寸预检: 物体最小边 > 2× 夹持器开口 → 极难抓, 快速跳过
    extents = mesh.bounding_box.extents
    min_ext = extents.min()
    if min_ext > 2 * MAX_GRIPPER_OPEN:
        print(f"  [SKIP LARGE] 最小边 {min_ext*100:.1f}cm > {2*MAX_GRIPPER_OPEN*100:.0f}cm, 跳过")
        return []

    z_min, z_max = mesh.bounds[0][2], mesh.bounds[1][2]
    all_candidates = []
    
    for batch in range(MAX_BATCHES):
        pts = sample_points(mesh, hp_pc, hp_labels, N_POINTS_PER_BATCH, has_hp)
        new_cands = generate_one_batch(mesh, pts, z_min, z_max, mesh_rc=mesh_rc)
        all_candidates.extend(new_cands)

        # 统计高质量候选
        high_quality = [c for c in all_candidates if c['score'] >= SCORE_THRESHOLD]
        hp_ratio = "50%HP+50%rnd" if has_hp else "100%rnd"
        print(f"    batch {batch+1}: +{len(new_cands)} 候选, "
              f"高质量≥{SCORE_THRESHOLD:.0f}分: {len(high_quality)}/{TARGET_HIGH_QUALITY} ({hp_ratio})")

        if len(high_quality) >= TARGET_HIGH_QUALITY:
            break

        # 快速放弃: 超过 SKIP_AFTER_BATCHES 批仍无高质量 → 物体难抓, 跳过
        if batch + 1 >= SKIP_AFTER_BATCHES and len(high_quality) == 0:
            print(f"  [SKIP] {batch+1} 批次 ({(batch+1)*N_POINTS_PER_BATCH} 个随机位置) 均无 ≥{SCORE_THRESHOLD:.0f} 分候选"
                  f" → 标记为难抓物体")
            return []   # 返回空 → 调用方写 .skip 标记
    
    # 按分数排序, 取 top TARGET_HIGH_QUALITY
    all_candidates.sort(key=lambda c: -c['score'])
    selected = all_candidates[:TARGET_HIGH_QUALITY]
    
    # 重命名
    for i, c in enumerate(selected):
        c['name'] = f'raycast_{i}'
    
    if selected:
        print(f"  → 最终选出 {len(selected)} 个候选 "
              f"(分数: {selected[0]['score']:.1f} ~ {selected[-1]['score']:.1f})")
    else:
        print(f"  ⚠️ 无有效候选 (物体可能太大，超出夹爪 {MAX_GRIPPER_OPEN*100:.0f}cm 张开)")
    
    return selected


def save_candidates_hdf5(candidates, obj_id, mesh_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'{obj_id}_grasp.hdf5')
    
    with h5py.File(path, 'w') as f:
        m = f.create_group('metadata')
        m.attrs['obj_id'] = obj_id
        m.attrs['mesh_path'] = os.path.abspath(mesh_path)
        m.attrs['method'] = 'raycast_scored_v2'
        # ★ canonical_rotation_applied=True 表示生成时已把 rotation.json 应用到 mesh 顶点
        # 因此 grasp 坐标已在 canonical 系里，与 USD/Sim 完全一致，Sim 无需任何补偿
        # mesh_prerotation_euler 必须存 [0,0,0]，否则 execute_grasp 会错误地再转一次
        _dataset_m = 'dexycb' if obj_id.startswith('ycb_') else 'oakink'
        import sys as _sys2; import os as _os2
        _tdir = _os2.path.dirname(_os2.path.abspath(__file__))
        if _tdir not in _sys2.path: _sys2.path.insert(0, _tdir)
        from mesh_utils import get_canonical_euler as _gce
        _euler = _gce(obj_id, _dataset_m)
        m.attrs['mesh_prerotation_euler'] = [0.0, 0.0, 0.0]   # ← 不需要补偿
        m.attrs['canonical_rotation_applied'] = any(abs(e) > 0.5 for e in _euler)
        m.attrs['canonical_euler_info'] = _euler               # 仅供参考，不被 sim 使用
        
        cg = f.create_group('candidates')
        cg.attrs['n_candidates'] = len(candidates)
        for i, c in enumerate(candidates):
            ci = cg.create_group(f'candidate_{i}')
            ci.create_dataset('position', data=c['position'])
            ci.create_dataset('grasp_point', data=c['grasp_point'])
            ci.create_dataset('rotation', data=c['rotation'])
            ci.attrs['name'] = c['name']
            ci.attrs['score'] = c['score']
            ci.attrs['gripper_width'] = c['gripper_width']
            ci.attrs['cross_section_width'] = c.get('cross_section_width', 0)
            ci.attrs['d_near'] = c.get('d_near', -1.0)
        
        if candidates:
            best = candidates[0]
            g = f.create_group('grasp')
            g.create_dataset('position', data=best['position'])
            g.create_dataset('grasp_point', data=best['grasp_point'])
            g.create_dataset('rotation', data=best['rotation'])
            quat_xyzw = Rotation.from_matrix(best['rotation']).as_quat()
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            g.create_dataset('quaternion_wxyz', data=quat_wxyz.astype(np.float32))
            g.attrs['gripper_width'] = best['gripper_width']
        
        aff = f.create_group('affordance')
        aff.attrs['n_contact'] = 0
    return path


def visualize_candidates(mesh, candidates, obj_id):
    import open3d as o3d
    
    geometries = []
    
    N_VIS = 30000
    vis_pc, _ = trimesh.sample.sample_surface(mesh, N_VIS)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vis_pc)
    pcd.paint_uniform_color([0.75, 0.75, 0.82])
    geometries.append(pcd)
    
    for i, c in enumerate(candidates):
        center = c['grasp_point']
        
        sphere_L = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
        sphere_L.translate(c['contact_L'])
        sphere_L.paint_uniform_color([0.9, 0.1, 0.1])
        geometries.append(sphere_L)
        
        sphere_R = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
        sphere_R.translate(c['contact_R'])
        sphere_R.paint_uniform_color([0.1, 0.3, 0.9])
        geometries.append(sphere_R)
        
        center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.004)
        center_sphere.translate(center)
        center_sphere.paint_uniform_color([1.0, 0.85, 0.0])
        geometries.append(center_sphere)
        
        pts = np.array([c['contact_L'], c['contact_R']])
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector(pts)
        line.lines = o3d.utility.Vector2iVector([[0, 1]])
        line.colors = o3d.utility.Vector3dVector([[0.6, 0.6, 0.6]])
        geometries.append(line)
        
        arrow_end = center - c['approach'] * 0.05
        arrow_line = o3d.geometry.LineSet()
        arrow_line.points = o3d.utility.Vector3dVector([center, arrow_end])
        arrow_line.lines = o3d.utility.Vector2iVector([[0, 1]])
        arrow_line.colors = o3d.utility.Vector3dVector([[0.2, 0.8, 0.2]])
        geometries.append(arrow_line)
    
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    geometries.append(coord)
    
    print(f"\n  🔍 Open3D: {obj_id} (top {len(candidates)} 候选)")
    print(f"     🔴 红=左接触  🔵 蓝=右接触  🟡 黄=中心  🟢 绿=approach")
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Top {len(candidates)} — {obj_id}", width=1200, height=800)
    for g in geometries:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.background_color = np.array([1, 1, 1])
    opt.point_size = 2.0
    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(description='Grasp Sampler v2 (Scored + Iterative)')
    parser.add_argument('--obj',     help='单个物体 ID (自动在 obj_meshes/ 所有数据集中查找)')
    parser.add_argument('--all',     action='store_true', help='批量处理 (默认 oakink, 配合 --dataset 使用)')
    parser.add_argument('--dataset', default=None, help='指定数据集: oakink / ycb / arctic / dexycb / egocentric')
    parser.add_argument('--arctic',  action='store_true', help='ARCTIC 10个物体 (mm→m 自动缩放)')
    parser.add_argument('--infer',   action='store_true', help='纯推理模式: 从 human_prior_infer/ 读取, 输出到 grasps_infer/')
    parser.add_argument('--force',   action='store_true', help='强制重新生成（覆盖已有）')
    parser.add_argument('--vis',     action='store_true')
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()

    # 推理模式：切换目录
    _hp_dir  = INFER_HP_DIR if args.infer else HP_DIR
    _out_dir = args.output_dir or (INFER_OUT_DIR if args.infer else OUTPUT_DIR)
    os.makedirs(_out_dir, exist_ok=True)

    # ── 构建 obj_list：5元组 (obj_id, mesh_path, scale, hp_name, hp_dir) ──
    obj_list = []

    if args.arctic:
        objs = [args.obj] if args.obj else ARCTIC_OBJS
        for obj in objs:
            mp = os.path.join(ARCTIC_ROOT, 'meta', 'object_vtemplates', obj, 'mesh_tex.obj')
            arctic_id = f'arctic_{obj}'
            obj_list.append((arctic_id, mp, 1.0 / 1000.0, obj, _hp_dir))

    elif args.obj:
        # ── 统一从 obj_meshes/ 查找 ──────────────────────────────────────
        mesh_path, scale_factor, ds = find_obj_mesh(args.obj, dataset=args.dataset)
        if mesh_path is None:
            print(f'❌ obj_meshes/ 中未找到: {args.obj}')
            print(f'   搜索路径: {OBJ_MESHES_DIR}')
            return
        print(f'   mesh: {mesh_path}  scale={scale_factor:.6f}  dataset={ds}')
        obj_list = [(args.obj, mesh_path, scale_factor, args.obj, _hp_dir)]

    elif args.all or args.dataset:
        # ── 按数据集批量处理 ──────────────────────────────────────────────
        target_ds = [args.dataset] if args.dataset else ['oakink']
        for ds in target_ds:
            for obj_id in list_dataset_objs(ds):
                mesh_path, scale_factor, _ = find_obj_mesh(obj_id, dataset=ds)
                if mesh_path:
                    obj_list.append((obj_id, mesh_path, scale_factor, obj_id, _hp_dir))
        print(f'数据集 {target_ds}: {len(obj_list)} 个物体')

    else:
        print("用法:")
        print("  python3 tools/random_grasp_sampler.py --obj A16013          # 单个物体")
        print("  python3 tools/random_grasp_sampler.py --all                 # OakInk 全部")
        print("  python3 tools/random_grasp_sampler.py --all --dataset ycb   # YCB 全部")
        print("  python3 tools/random_grasp_sampler.py --arctic              # ARCTIC (mm→m)")
        return

    mode = 'ARCTIC' if args.arctic else f'obj_meshes/{getattr(args,"dataset","oakink") or "oakink"}'
    print('=' * 60)
    print(f'  Grasp Sampler v2 [{mode}] (50%HP + 50%rnd)')
    print(f'  Target: {TARGET_HIGH_QUALITY} candidates ≥ {SCORE_THRESHOLD} pts')
    print('=' * 60)

    # 预加载 canonical rotation
    canonical_rotations = load_canonical_rotations()

    generated = 0
    # 支持5元组 (obj_id, mesh_path, scale, hp_name, hp_dir) 和旧4元组
    for idx, entry in enumerate(obj_list):
        if len(entry) == 5:
            obj_id, mesh_path, scale, hp_name, hp_dir_use = entry
        else:
            obj_id, mesh_path, scale, hp_name = entry
            hp_dir_use = _hp_dir

        print(f'\n[{idx+1}/{len(obj_list)}] {obj_id}')

        skip_path = os.path.join(_out_dir, f'{obj_id}.skip')
        out_path  = os.path.join(_out_dir, f'{obj_id}_grasp.hdf5')

        if not args.force:
            if os.path.exists(skip_path):
                print(f' ⏭️ [SKIP标记] 已知难抓物体')
                continue
            if os.path.exists(out_path):
                print(' ⏭️ (已生成)')
                continue

        if not os.path.exists(mesh_path):
            print(f' ❌ mesh 不存在: {mesh_path}')
            continue

        mesh = trimesh.load(mesh_path, force='mesh')

        # ── Step 1: 应用 scale_factor → 米制 ─────────────────────────────
        if scale != 1.0:
            mesh.vertices *= scale

        # ── Step 2: 应用 canonical rotation (与 USD/Sim 保持完全一致) ────────
        # 来源: obj_meshes/{dataset}/{obj_id}/rotation.json (同 convert_obj_usd.py)
        import sys as _sys
        _tools_dir = os.path.dirname(os.path.abspath(__file__))
        if _tools_dir not in _sys.path:
            _sys.path.insert(0, _tools_dir)
        from mesh_utils import get_canonical_euler as _get_euler
        _dataset = 'dexycb' if obj_id.startswith('ycb_') else ('arctic' if args.arctic else 'oakink')
        rot_euler = _get_euler(obj_id, _dataset)
        if any(abs(e) > 0.5 for e in rot_euler):
            from scipy.spatial.transform import Rotation as _R
            R_mat = _R.from_euler('xyz', rot_euler, degrees=True).as_matrix()
            mesh.vertices = (R_mat @ mesh.vertices.T).T
            print(f'     [canonical rot (rotation.json): {[round(e,1) for e in rot_euler]}°]')
        else:
            print(f'     [canonical rot: identity]')

        if not mesh.is_watertight:
            trimesh.repair.fill_holes(mesh)
            trimesh.repair.fix_normals(mesh)

        ext = mesh.bounding_box.extents * 100
        print(f'  尺寸: {ext[0]:.1f}×{ext[1]:.1f}×{ext[2]:.1f} cm  ({len(mesh.faces):,} 面)')

        # ── Step 3: 简化 mesh 用于 raycast（加速 ~18×，几何精度足够）───────
        # 原始高精度 PLY (~500K面) 做 raycast 极慢；简化到 5000 面精度已足够
        # mesh.contains() 用原始 mesh（速度差不多），raycast 用简化 mesh
        SIMPLIFY_TARGET = 5000
        mesh_rc = None   # 默认: 不需要简化时直接用原始 mesh
        if len(mesh.faces) > SIMPLIFY_TARGET * 2:
            t_s = time.time()
            mesh_rc = mesh.simplify_quadric_decimation(face_count=SIMPLIFY_TARGET)
            if not mesh_rc.is_watertight:
                trimesh.repair.fix_normals(mesh_rc)
            print(f'  → 简化为 {len(mesh_rc.faces):,} 面 (raycast用, {time.time()-t_s:.2f}s)')

        # HP 从指定目录读取（支持 training_fp/ 直接读）
        candidates = generate_candidates_iterative(mesh, hp_name, hp_dir=hp_dir_use,
                                                   mesh_rc=mesh_rc)

        if candidates:
            path = save_candidates_hdf5(candidates, obj_id, mesh_path, _out_dir)
            print(f'  ✅ → {os.path.basename(path)} ({len(candidates)} 候选)')
            generated += 1
        else:
            # 无有效候选 → 写 .skip 标记，下次直接跳过
            open(skip_path, 'w').write(f'SKIP: {SKIP_AFTER_BATCHES} batches, 0 candidates >= {SCORE_THRESHOLD}\n')
            print(f'  ⬛ → {obj_id}.skip (难抓物体，已标记)')

        if args.vis and candidates:
            visualize_candidates(mesh, candidates, obj_id)

    print(f"\n{'='*60}")
    print(f'  完成! 生成 {generated}/{len(obj_list)} 个物体的候选')
    print(f'  输出: {_out_dir}')
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
