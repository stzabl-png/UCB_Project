#!/usr/bin/env python3
"""
**WORK IN PROGRESS** — placement sampler (stable_orientations); HDF5 schema, vis outputs,
and interaction with Sim/mesh_prerotation are not finalized.

random_grasp_sampler_placement.py — Grasp sampler with stable_orientations placement
==================================================================================
- Sample on mesh rotated by R_placement (work frame); world-frame approach dirs.
- Store candidate poses in identity object frame (R_placement^T applied before save).
- Writes mesh_prerotation/ with placement_id (Sim reads same group as legacy).

用法:
    python3 tools/random_grasp_sampler_placement.py --obj A01026 --dataset oakink
    python3 tools/random_grasp_sampler_placement.py --obj A01026 --dataset oakink \\
        --placement random --placement-seed 42 --force
    python3 tools/random_grasp_sampler_placement.py --dataset oakink --all

    # 调试: 输出 raw 分开图 + placement work/identity 对比图
    python3 tools/random_grasp_sampler_placement.py --obj A01026 --dataset oakink \\
        --placement random --target 5 --vis-sample --force
    # → output/sampler_vis/{obj}_raw_mesh_hp.png
    # → output/sampler_vis/{obj}_placement{id}_debug.png
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time

import h5py
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, "tools"))

import random_grasp_sampler as rgs
from mesh_utils import (
    infer_dataset,
    list_ready_objects,
    placement_seed,
    resolve_placement,
    rotate_points,
    rotate_vertices,
    write_mesh_prerotation_hdf5,
)

SAMPLER_VERSION = "placement_v1"
SIMPLIFY_TARGET = 5000
SAMPLER_VIS_DIR = os.path.join(PROJ, "output", "sampler_vis")
HP_VIS_THRESH = rgs.HP_CONTACT_LABEL_THRESH


def _candidates_to_identity_frame(candidates: list[dict], R_placement: np.ndarray) -> None:
    """p_local = R^T @ p_work; R_grasp_local = R^T @ R_grasp_work."""
    R_inv = np.asarray(R_placement, dtype=np.float64).T
    for c in candidates:
        c["position"] = (R_inv @ np.asarray(c["position"], dtype=np.float32)).astype(np.float32)
        c["grasp_point"] = (R_inv @ np.asarray(c["grasp_point"], dtype=np.float32)).astype(np.float32)
        c["rotation"] = (R_inv @ np.asarray(c["rotation"], dtype=np.float32)).astype(np.float32)
        if "approach" in c:
            c["approach"] = (R_inv @ np.asarray(c["approach"], dtype=np.float32)).astype(np.float32)
        if "finger_dir" in c:
            c["finger_dir"] = (R_inv @ np.asarray(c["finger_dir"], dtype=np.float32)).astype(np.float32)
        for key in ("contact_L", "contact_R"):
            if key in c and c[key] is not None:
                c[key] = (R_inv @ np.asarray(c[key], dtype=np.float32)).astype(np.float32)


def _extents_cm(points: np.ndarray) -> tuple[float, float, float]:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return 0.0, 0.0, 0.0
    ext = pts.max(axis=0) - pts.min(axis=0)
    return float(ext[0] * 100), float(ext[1] * 100), float(ext[2] * 100)


def _set_axes_equal_3d(ax, bounds_list: list[np.ndarray], pad: float = 0.08):
    """Union bounds of several (2,3) arrays; axes in metres."""
    bmin = np.min([b[0] for b in bounds_list], axis=0)
    bmax = np.max([b[1] for b in bounds_list], axis=0)
    c = (bmin + bmax) / 2.0
    r = float(np.max(bmax - bmin)) * (0.5 + pad)
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)


def _draw_sampler_panel(
    ax,
    mesh: trimesh.Trimesh,
    hp_pc: np.ndarray | None,
    hp_labels: np.ndarray | None,
    cand: dict | None,
    *,
    mesh_label: str,
    hp_label: str,
    grasp_style: str = "solid",
    hp_thresh: float = HP_VIS_THRESH,
    show_world_z: bool = False,
):
    surf_pts, _ = trimesh.sample.sample_surface(mesh, 5000)
    ax.scatter(
        surf_pts[:, 0], surf_pts[:, 1], surf_pts[:, 2],
        c="#5ab4d4", s=2.0, alpha=0.5, linewidths=0, zorder=2, label=mesh_label,
    )

    if hp_pc is not None and hp_labels is not None:
        labels = np.asarray(hp_labels, dtype=np.float32).reshape(-1)
        mask = labels > 0.05
        if np.any(mask):
            pts = np.asarray(hp_pc[mask], dtype=np.float64)
            vals = labels[mask]
            if len(pts) > 2000:
                idx = np.random.default_rng(0).choice(len(pts), 2000, replace=False)
                pts, vals = pts[idx], vals[idx]
            ax.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2],
                c=vals, cmap="hot", vmin=0, vmax=1,
                s=5 + 22 * vals, alpha=0.8, linewidths=0, zorder=3, label=hp_label,
            )
            hi = vals > hp_thresh
            if np.any(hi):
                ax.scatter(
                    pts[hi, 0], pts[hi, 1], pts[hi, 2],
                    c="none", edgecolors="#ff5fa8", s=40, linewidths=0.7,
                    zorder=4, label=f"HP>{hp_thresh}",
                )

    if cand is not None:
        pos = np.asarray(cand["grasp_point"], dtype=np.float64)
        approach = np.asarray(cand.get("approach", cand["rotation"][:, 2]), dtype=np.float64)
        approach /= np.linalg.norm(approach) + 1e-8
        finger = np.asarray(cand.get("finger_dir", cand["rotation"][:, 0]), dtype=np.float64)
        finger /= np.linalg.norm(finger) + 1e-8
        hw = float(cand.get("gripper_width", 0.05)) / 2.0
        c_l = np.asarray(cand.get("contact_L", pos - hw * finger), dtype=np.float64)
        c_r = np.asarray(cand.get("contact_R", pos + hw * finger), dtype=np.float64)
        ls = "--" if grasp_style == "dashed" else "-"
        alpha = 0.75 if grasp_style == "dashed" else 1.0
        ax.scatter(*c_l, c="#e74c3c", s=45, marker="s", zorder=7, alpha=alpha)
        ax.scatter(*c_r, c="#3498db", s=45, marker="s", zorder=7, alpha=alpha)
        ax.scatter(*pos, c="white", s=60, zorder=8, edgecolors="#2ecc71", linewidths=1.0,
                   alpha=alpha, label="grasp")
        ax.plot([c_l[0], c_r[0]], [c_l[1], c_r[1]], [c_l[2], c_r[2]],
                c="#17becf", lw=1.6, zorder=6, linestyle=ls, alpha=alpha)
        ax.quiver(
            pos[0], pos[1], pos[2],
            approach[0], approach[1], approach[2],
            length=0.03, color="#2ecc71", arrow_length_ratio=0.35,
            linewidth=1.8, zorder=7, alpha=alpha,
        )

    if show_world_z:
        o = mesh.centroid
        ax.quiver(o[0], o[1], o[2], 0, 0, 0.04, color="#aaa",
                  arrow_length_ratio=0.2, linewidth=1.2, zorder=1)
        ax.text(o[0], o[1], o[2] + 0.045, "+Z", color="#aaa", fontsize=6)

    ax.set_xlabel("X", color="#aaa", fontsize=8)
    ax.set_ylabel("Y", color="#aaa", fontsize=8)
    ax.set_zlabel("Z", color="#aaa", fontsize=8)
    ax.tick_params(colors="#555", labelsize=6)
    ax.view_init(elev=22, azim=128)


def visualize_raw_mesh_hp_split(
    mesh_path: str,
    obj_id: str,
    dataset: str,
    hp_name: str,
    hp_dir: str,
    out_path: str,
    *,
    scale_factor: float = 1.0,
    hp_thresh: float = HP_VIS_THRESH,
) -> str:
    """
    三栏分开、无任何旋转:
      1) mesh.ply 原文件顶点
      2) training_fp point_cloud 原文件坐标
      3) mesh.ply × scale.json（batch_align 建 HP 时用的尺度；当前 HP 文件未必已 scale）
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    mesh_raw = trimesh.load(mesh_path, force="mesh")
    mesh_scaled = mesh_raw.copy()
    if abs(scale_factor - 1.0) > 1e-8:
        mesh_scaled.vertices = mesh_scaled.vertices * float(scale_factor)

    hp_raw, hp_labels, _ = rgs.load_human_prior(hp_name, hp_dir=hp_dir, dataset=dataset)

    fig = plt.figure(figsize=(16, 5.5), facecolor="#1a1a2e")
    axes = [fig.add_subplot(131 + i, projection="3d", facecolor="#1a1a2e") for i in range(3)]

    panels: list[tuple[trimesh.Trimesh | np.ndarray, str, str, bool]] = [
        (mesh_raw, "mesh.ply file", "RAW mesh (file vertices)", False),
        (mesh_scaled, f"mesh × scale={scale_factor:.4f}", "mesh × scale.json only", False),
    ]
    if hp_raw is not None:
        panels.insert(1, (hp_raw, "training_fp file", "RAW HP (training_fp)", True))

    all_bounds: list[np.ndarray] = []
    for ax, (geom, subtitle, title, is_hp) in zip(axes[: len(panels)], panels):
        if is_hp:
            labels = np.asarray(hp_labels, dtype=np.float32).reshape(-1)
            pts = np.asarray(geom, dtype=np.float64)
            mask = labels > 0.05
            pts = pts[mask] if np.any(mask) else pts
            vals = labels[mask] if np.any(mask) else labels
            if len(pts) > 2500:
                idx = np.random.default_rng(0).choice(len(pts), 2500, replace=False)
                pts, vals = pts[idx], vals[idx]
            ax.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2],
                c=vals, cmap="hot", vmin=0, vmax=1, s=5 + 22 * vals, alpha=0.85,
            )
            ext = _extents_cm(geom)
            all_bounds.append(np.array([geom.min(axis=0), geom.max(axis=0)]))
        else:
            m: trimesh.Trimesh = geom  # type: ignore[assignment]
            pts, _ = trimesh.sample.sample_surface(m, 6000)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="#5ab4d4", s=2.0, alpha=0.6)
            ext = _extents_cm(pts)
            all_bounds.append(m.bounds)
        ax.set_title(
            f"{title}\n{subtitle}\next(cm)=[{ext[0]:.1f},{ext[1]:.1f},{ext[2]:.1f}]",
            color="#ddd", fontsize=7, pad=5,
        )
        ax.set_xlabel("X", color="#aaa", fontsize=7)
        ax.set_ylabel("Y", color="#aaa", fontsize=7)
        ax.set_zlabel("Z", color="#aaa", fontsize=7)
        ax.tick_params(colors="#555", labelsize=5)
        ax.view_init(elev=22, azim=128)
        _set_axes_equal_3d(ax, all_bounds)

    em = _extents_cm(trimesh.sample.sample_surface(mesh_raw, 2000)[0])
    es = _extents_cm(trimesh.sample.sample_surface(mesh_scaled, 2000)[0])
    print(f"  [vis-raw] mesh file      ext(cm)={em}")
    print(f"  [vis-raw] mesh×scale    ext(cm)={es}")
    if hp_raw is not None:
        eh = _extents_cm(hp_raw)
        print(f"  [vis-raw] HP file       ext(cm)={eh}")
        if max(em) > 1e-3:
            axis_ratio = np.array(eh) / (np.array(em) + 1e-8)
            print(f"  [vis-raw] HP/mesh axis ratio (file): {axis_ratio.round(2)}")
        if max(es) > 1e-3:
            axis_ratio_s = np.array(eh) / (np.array(es) + 1e-8)
            print(f"  [vis-raw] HP/mesh axis ratio (HP vs scaled mesh): {axis_ratio_s.round(2)}")

    warn = ""
    if hp_raw is not None and max(np.abs(np.array(eh) - np.array(es))) > 8.0:
        warn = "⚠ HP file looks UNSCALED vs mesh×scale — regenerate training_fp or scale HP in loader"

    fig.suptitle(
        f"{obj_id} ({dataset})  separate RAW sources (no rotation)\n{warn}",
        color="#fcc" if warn else "#ccc",
        fontsize=9,
        y=1.02,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    return out_path


def visualize_sampler_work_frame(
    mesh_work: trimesh.Trimesh,
    hp_work: np.ndarray | None,
    hp_labels: np.ndarray | None,
    cand_work: dict,
    mesh_identity: trimesh.Trimesh,
    hp_identity: np.ndarray | None,
    cand_identity: dict | None,
    obj_id: str,
    placement_rec: dict,
    out_path: str,
    *,
    scale: float = 1.0,
    hp_thresh: float = HP_VIS_THRESH,
) -> str:
    """
    双面板调试图:
      左 — work 系 (采样用): mesh×scale×R_place, HP×scale×R_place, 候选 work
      右 — identity 系: mesh×scale, HP×scale (未 R_place), 候选 R^T@work (虚线)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    fig = plt.figure(figsize=(15, 7), facecolor="#1a1a2e")
    ax_w = fig.add_subplot(121, projection="3d", facecolor="#1a1a2e")
    ax_i = fig.add_subplot(122, projection="3d", facecolor="#1a1a2e")

    pid = placement_rec.get("placement_id", "?")
    mb_w = _extents_cm(trimesh.sample.sample_surface(mesh_work, 2000)[0])
    mb_i = _extents_cm(trimesh.sample.sample_surface(mesh_identity, 2000)[0])
    hp_w_note = "no HP"
    hp_i_note = "no HP"
    if hp_work is not None:
        ex, ey, ez = _extents_cm(hp_work)
        hp_w_note = f"HP ext(cm)=[{ex:.1f},{ey:.1f},{ez:.1f}]"
    if hp_identity is not None:
        ex, ey, ez = _extents_cm(hp_identity)
        hp_i_note = f"HP ext(cm)=[{ex:.1f},{ey:.1f},{ez:.1f}]"

    _draw_sampler_panel(
        ax_w, mesh_work, hp_work, hp_labels, cand_work,
        mesh_label="mesh (scale+R_place)",
        hp_label="HP (scale+R_place)",
        grasp_style="solid",
        hp_thresh=hp_thresh,
        show_world_z=True,
    )
    ax_w.set_title(
        f"WORK (sampler raycast)\n{obj_id} pid={pid}  {cand_work.get('name','')} "
        f"score={cand_work.get('score',0):.1f}\n"
        f"mesh(cm)=[{mb_w[0]:.1f},{mb_w[1]:.1f},{mb_w[2]:.1f}]  {hp_w_note}",
        color="#ddd", fontsize=7.5, pad=6,
    )

    _draw_sampler_panel(
        ax_i, mesh_identity, hp_identity, hp_labels, cand_identity,
        mesh_label="mesh (scale, identity)",
        hp_label="HP (scale, no R_place)",
        grasp_style="dashed" if cand_identity else "solid",
        hp_thresh=hp_thresh,
    )
    ax_i.set_title(
        f"IDENTITY (reference)\nscale={scale:.4f}  (HP raw×scale only)\n"
        f"mesh(cm)=[{mb_i[0]:.1f},{mb_i[1]:.1f},{mb_i[2]:.1f}]  {hp_i_note}",
        color="#ddd", fontsize=7.5, pad=6,
    )

    bounds = [mesh_work.bounds, mesh_identity.bounds]
    if hp_work is not None and len(hp_work):
        bounds.append(np.array([hp_work.min(0), hp_work.max(0)]))
    if hp_identity is not None and len(hp_identity):
        bounds.append(np.array([hp_identity.min(0), hp_identity.max(0)]))
    _set_axes_equal_3d(ax_w, bounds)
    _set_axes_equal_3d(ax_i, bounds)

    ax_w.legend(loc="upper left", fontsize=5.5, facecolor="#2a2a3e", edgecolor="#555",
                labelcolor="#ccc")
    ax_i.legend(loc="upper left", fontsize=5.5, facecolor="#2a2a3e", edgecolor="#555",
                labelcolor="#ccc")

    fig.suptitle(
        f"{obj_id} placement debug  |  R_place from stable_orientations id={pid}",
        color="#ccc", fontsize=10, y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    return out_path


def save_candidates_hdf5_placement(
    candidates: list[dict],
    obj_id: str,
    mesh_path: str,
    output_dir: str,
    placement_rec: dict,
    *,
    dataset: str,
    placement_seed_value: int | None = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{obj_id}_grasp.hdf5")
    ds = infer_dataset(obj_id, dataset)
    R = np.asarray(placement_rec["matrix"], dtype=np.float64)
    is_identity = bool(np.allclose(R, np.eye(3), atol=1e-5))

    with h5py.File(path, "w") as f:
        m = f.create_group("metadata")
        m.attrs["obj_id"] = obj_id
        m.attrs["mesh_path"] = os.path.abspath(mesh_path)
        m.attrs["method"] = "raycast_scored_placement_v1"
        m.attrs["sampler_version"] = SAMPLER_VERSION
        m.attrs["dataset"] = ds
        m.attrs["placement_id"] = int(placement_rec["placement_id"])
        m.attrs["placement_method"] = str(placement_rec["method"])
        m.attrs["placement_source"] = str(placement_rec.get("source", ""))
        if placement_rec.get("probability") is not None:
            m.attrs["placement_probability"] = float(placement_rec["probability"])
        if placement_seed_value is not None:
            m.attrs["placement_seed"] = int(placement_seed_value)
        m.attrs["no_rotation"] = bool(is_identity)

        write_mesh_prerotation_hdf5(m, placement_rec)

        cg = f.create_group("candidates")
        cg.attrs["n_candidates"] = len(candidates)
        for i, c in enumerate(candidates):
            ci = cg.create_group(f"candidate_{i}")
            ci.create_dataset("position", data=c["position"])
            ci.create_dataset("grasp_point", data=c["grasp_point"])
            ci.create_dataset("rotation", data=c["rotation"])
            ci.attrs["name"] = c["name"]
            ci.attrs["score"] = c["score"]
            ci.attrs["gripper_width"] = c["gripper_width"]
            ci.attrs["cross_section_width"] = c.get("cross_section_width", 0)
            ci.attrs["d_near"] = c.get("d_near", -1.0)
            write_mesh_prerotation_hdf5(ci, placement_rec)

        if candidates:
            best = candidates[0]
            g = f.create_group("grasp")
            write_mesh_prerotation_hdf5(g, placement_rec)
            g.create_dataset("position", data=best["position"])
            g.create_dataset("grasp_point", data=best["grasp_point"])
            g.create_dataset("rotation", data=best["rotation"])
            quat_xyzw = Rotation.from_matrix(best["rotation"]).as_quat()
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            g.create_dataset("quaternion_wxyz", data=quat_wxyz.astype(np.float32))
            g.attrs["gripper_width"] = best["gripper_width"]

        aff = f.create_group("affordance")
        aff.attrs["n_contact"] = 0

    return path


def process_one_object(
    obj_id: str,
    mesh_path: str,
    scale: float,
    hp_name: str,
    hp_dir: str,
    output_dir: str,
    *,
    dataset: str = "oakink",
    force: bool = False,
    placement_mode: str = "random",
    placement_id: int | None = None,
    placement_seed_value: int | None = None,
    round_idx: int | None = None,
    target_n: int = rgs.TARGET_HIGH_QUALITY,
    score_threshold: float = rgs.SCORE_THRESHOLD,
    arctic: bool = False,
    require_hp_contact: bool = rgs.REQUIRE_HP_CONTACT_DEFAULT,
    vis_sample: bool = False,
    vis_sample_path: str | None = None,
    vis_cand_index: int = 0,
) -> tuple[str | None, str | None]:
    skip_path = os.path.join(output_dir, f"{obj_id}.skip")
    out_path = os.path.join(output_dir, f"{obj_id}_grasp.hdf5")

    if not force:
        if os.path.exists(skip_path):
            return None, "skip_marked"
        if os.path.exists(out_path):
            return out_path, "exists"

    if not os.path.exists(mesh_path):
        return None, "no_mesh"

    ds = infer_dataset(obj_id, dataset if not arctic else "arctic")

    if placement_seed_value is None and round_idx is not None:
        placement_seed_value = placement_seed(ds, obj_id, round_idx)

    if placement_mode == "id":
        if placement_id is None:
            return None, "placement_id_required"
        R_place, place_rec = resolve_placement(
            obj_id, ds, placement_id=placement_id,
        )
    else:
        seed = placement_seed_value if placement_seed_value is not None else 0
        R_place, place_rec = resolve_placement(obj_id, ds, seed=seed)

    mesh = trimesh.load(mesh_path, force="mesh")
    if scale != 1.0:
        mesh.vertices *= scale
    mesh_identity = mesh.copy()

    hp_pc, hp_labels, _ = rgs.load_human_prior(hp_name, hp_dir=hp_dir, dataset=ds)
    hp_identity = None
    if hp_pc is not None:
        if scale != 1.0:
            hp_pc = (np.asarray(hp_pc, dtype=np.float64) * scale).astype(np.float32)
        hp_identity = np.asarray(hp_pc, dtype=np.float32).copy()
        hp_pc = rotate_points(hp_pc, R_place)

    rotate_vertices(mesh, R_place)

    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_normals(mesh)

    ext = mesh.bounding_box.extents * 100
    pid = place_rec["placement_id"]
    print(
        f"     [placement id={pid} method={place_rec['method']} "
        f"seed={placement_seed_value}]"
    )
    print(f"  尺寸: {ext[0]:.1f}×{ext[1]:.1f}×{ext[2]:.1f} cm  ({len(mesh.faces):,} 面)")

    mesh_rc = None
    if len(mesh.faces) > SIMPLIFY_TARGET * 2:
        t_s = time.time()
        mesh_rc = mesh.simplify_quadric_decimation(face_count=SIMPLIFY_TARGET)
        if not mesh_rc.is_watertight:
            trimesh.repair.fix_normals(mesh_rc)
        print(f"  → 简化为 {len(mesh_rc.faces):,} 面 (raycast用, {time.time()-t_s:.2f}s)")

    if require_hp_contact:
        print("  [scoring] require ≥1 contact in human_prior region (label>0.3)")
    candidates = rgs.generate_candidates_iterative(
        mesh,
        hp_name,
        hp_dir=hp_dir,
        mesh_rc=mesh_rc,
        target_n=target_n,
        score_threshold=score_threshold,
        require_hp_contact=require_hp_contact,
        hp_pc=hp_pc,
        hp_labels=hp_labels,
    )

    if candidates:
        if vis_sample:
            vi = int(np.clip(vis_cand_index, 0, len(candidates) - 1))
            pid = int(place_rec["placement_id"])
            raw_path = os.path.join(SAMPLER_VIS_DIR, f"{obj_id}_raw_mesh_hp.png")
            visualize_raw_mesh_hp_split(
                mesh_path, obj_id, ds, hp_name, hp_dir, raw_path,
                scale_factor=scale,
            )
            print(f"  [vis-raw] → {raw_path}")

            vpath = vis_sample_path or os.path.join(
                SAMPLER_VIS_DIR,
                f"{obj_id}_placement{pid:02d}_debug.png",
            )
            cand_work = copy.deepcopy(candidates[vi])
            cand_id = copy.deepcopy(cand_work)
            _candidates_to_identity_frame([cand_id], R_place)
            visualize_sampler_work_frame(
                mesh,
                hp_pc,
                hp_labels,
                cand_work,
                mesh_identity,
                hp_identity,
                cand_id,
                obj_id,
                place_rec,
                vpath,
                scale=scale,
            )
            print(f"  [vis-sample] → {vpath}  (cand #{vi}; L=work R=identity)")
        _candidates_to_identity_frame(candidates, R_place)
        path = save_candidates_hdf5_placement(
            candidates,
            obj_id,
            mesh_path,
            output_dir,
            place_rec,
            dataset=ds,
            placement_seed_value=placement_seed_value,
        )
        print(f"  ✅ → {os.path.basename(path)} ({len(candidates)} 候选, placement_id={pid})")
        return path, None

    open(skip_path, "w").write(
        f"SKIP: {rgs.max_sampler_batches(target_n)} sampler batches exhausted, 0 candidates >= {score_threshold}\n"
    )
    print(f"  ⬛ → {obj_id}.skip (难抓物体，已标记)")
    return None, "no_candidates"


def main():
    parser = argparse.ArgumentParser(
        description="Grasp sampler with stable_orientations placement (identity-local poses)",
    )
    parser.add_argument("--obj", help="单个物体 ID")
    parser.add_argument("--all", action="store_true", help="批量处理 --dataset 下全部 ready 物体")
    parser.add_argument("--dataset", default="oakink")
    parser.add_argument("--arctic", action="store_true")
    parser.add_argument("--infer", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--vis-sample",
        action="store_true",
        help="保存调试 PNG: raw 分开图 + work/identity 对比 (见 --vis-sample-out)",
    )
    parser.add_argument(
        "--vis-sample-out",
        default=None,
        help="调试图路径；默认 output/sampler_vis/{obj}_placement{id}_work.png",
    )
    parser.add_argument(
        "--vis-cand-index",
        type=int,
        default=0,
        help="--vis-sample 时画第几个候选 (按分数排序后, 默认 0=最高分)",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--target", type=int, default=rgs.TARGET_HIGH_QUALITY)
    parser.add_argument("--score-threshold", type=float, default=rgs.SCORE_THRESHOLD)
    parser.add_argument(
        "--placement",
        choices=("random", "id"),
        default="random",
        help="random: 均匀抽 stable 库; id: 固定 --placement-id",
    )
    parser.add_argument("--placement-id", type=int, default=None)
    parser.add_argument(
        "--placement-seed",
        type=int,
        default=None,
        help="复现 placement 抽样 (batch 用 placement_seed(dataset,obj,round))",
    )
    parser.add_argument(
        "--round-idx",
        type=int,
        default=None,
        help="与 --placement-seed 二选一: 自动 seed=hash(dataset,obj,round)",
    )
    parser.add_argument(
        "--no-hp-contact-required",
        action="store_true",
        help="关闭硬性要求：至少一个接触点在 human_prior 区域 (默认开启)",
    )
    args = parser.parse_args()

    _hp_dir = rgs.INFER_HP_DIR if args.infer else rgs.HP_DIR
    _out_dir = args.output_dir or (rgs.INFER_OUT_DIR if args.infer else rgs.OUTPUT_DIR)
    os.makedirs(_out_dir, exist_ok=True)

    obj_list: list[tuple] = []

    if args.arctic:
        objs = [args.obj] if args.obj else rgs.ARCTIC_OBJS
        for obj in objs:
            mp = os.path.join(rgs.ARCTIC_ROOT, "meta", "object_vtemplates", obj, "mesh_tex.obj")
            arctic_id = f"arctic_{obj}"
            obj_list.append((arctic_id, mp, 1.0 / 1000.0, obj, _hp_dir))
    elif args.obj:
        mesh_path, scale_factor, ds, _apply_scale = rgs.find_obj_mesh(
            args.obj, dataset=args.dataset, use_legacy_assets=True,
        )
        if mesh_path is None:
            print(f"❌ obj_meshes/ 中未找到: {args.obj}")
            return
        print(f"   mesh: {mesh_path}  scale={scale_factor:.6f}  dataset={ds}")
        obj_list = [(args.obj, mesh_path, scale_factor, args.obj, _hp_dir)]
    elif args.all or args.dataset:
        target_ds = args.dataset or "oakink"
        for obj_id in list_ready_objects(target_ds):
            mesh_path, scale_factor, _, _apply_scale = rgs.find_obj_mesh(
                obj_id, dataset=target_ds, use_legacy_assets=True,
            )
            if mesh_path:
                obj_list.append((obj_id, mesh_path, scale_factor, obj_id, _hp_dir))
        print(f"数据集 {target_ds}: {len(obj_list)} 个 ready 物体")
    else:
        parser.print_help()
        return

    print("=" * 60)
    print("  Grasp Sampler placement_v1 (stable_orientations)")
    print(f"  Target: {args.target} candidates ≥ {args.score_threshold}")
    print(f"  Placement: {args.placement}")
    print(
        f"  HP contact required: {not args.no_hp_contact_required} "
        f"(label>{rgs.HP_CONTACT_LABEL_THRESH})"
    )
    print("=" * 60)

    generated = 0
    _ds = args.dataset or ("arctic" if args.arctic else "oakink")
    for idx, entry in enumerate(obj_list):
        if len(entry) == 5:
            obj_id, mesh_path, scale, hp_name, hp_dir_use = entry
        else:
            obj_id, mesh_path, scale, hp_name = entry
            hp_dir_use = _hp_dir

        print(f"\n[{idx+1}/{len(obj_list)}] {obj_id}")

        out_path, reason = process_one_object(
            obj_id,
            mesh_path,
            scale,
            hp_name,
            hp_dir_use,
            _out_dir,
            dataset=_ds,
            force=args.force,
            placement_mode=args.placement,
            placement_id=args.placement_id,
            placement_seed_value=args.placement_seed,
            round_idx=args.round_idx,
            target_n=args.target,
            score_threshold=args.score_threshold,
            arctic=args.arctic,
            require_hp_contact=not args.no_hp_contact_required,
            vis_sample=args.vis_sample,
            vis_sample_path=args.vis_sample_out,
            vis_cand_index=args.vis_cand_index,
        )
        if reason == "skip_marked":
            print(" ⏭️ [SKIP标记] 已知难抓物体")
        elif reason == "exists":
            print(" ⏭️ (已生成)")
        elif reason == "no_mesh":
            print(f" ❌ mesh 不存在: {mesh_path}")
        elif reason == "placement_id_required":
            print(" ❌ --placement id 需要 --placement-id")
        elif out_path:
            generated += 1

    print(f"\n{'='*60}")
    print(f"  完成! 生成 {generated}/{len(obj_list)} 个物体的候选")
    print(f"  输出: {_out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
