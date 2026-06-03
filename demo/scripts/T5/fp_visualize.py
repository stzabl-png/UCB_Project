"""T5 vis: 2×2 layout to explain pose — mask, bbox, mesh on black, 3D camera view."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_T4_DIR = Path(__file__).resolve().parents[1] / "T4"
if str(_T4_DIR) not in sys.path:
    sys.path.insert(0, str(_T4_DIR))
from scale_common import (  # noqa: E402
    depth_mask_to_pointcloud,
    load_depth_m,
    load_mask_bool,
    preprocess_mask,
    project_cam_to_image,
)


def _load_mesh_trimesh(mesh_path: Path):
    import trimesh

    mesh = trimesh.load(str(mesh_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    return mesh


def mesh_to_cam(verts_mesh: np.ndarray, T_cam_mesh: np.ndarray) -> np.ndarray:
    ones = np.ones((len(verts_mesh), 1), dtype=np.float64)
    return (T_cam_mesh @ np.hstack([verts_mesh, ones]).T).T[:, :3]


def simplify_mesh_for_vis(mesh, max_faces: int = 8000):
    import trimesh

    n = len(mesh.faces)
    if n <= max_faces:
        return mesh
    try:
        import fast_simplification

        ratio = 1.0 - min(max_faces / n, 0.9999)
        pts, faces = fast_simplification.simplify(
            mesh.vertices, mesh.faces, target_reduction=ratio
        )
        return trimesh.Trimesh(vertices=pts, faces=faces, process=False)
    except Exception:
        idx = np.linspace(0, n - 1, max_faces, dtype=int)
        return trimesh.Trimesh(
            vertices=mesh.vertices, faces=mesh.faces[idx], process=False
        )


def _viridis_bgr_lut(n: int = 256) -> np.ndarray:
    """BGR uint8 LUT — must match matplotlib colorbar."""
    import matplotlib.cm as cm

    rgba = (cm.get_cmap("viridis")(np.linspace(0, 1, n))[:, :3] * 255).astype(np.uint8)
    return rgba[:, ::-1]  # RGB → BGR


def _z_to_bgr(z: float, z_min: float, z_max: float, lut: np.ndarray) -> tuple[int, int, int]:
    t = float(np.clip((z - z_min) / (z_max - z_min + 1e-9), 0.0, 1.0))
    idx = int(t * (len(lut) - 1))
    c = lut[idx]
    return int(c[0]), int(c[1]), int(c[2])


def _face_facing_camera(v_cam: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Camera +Z forward (OpenCV). Visible faces have normal pointing toward origin → nz < 0.
    """
    v0 = v_cam[faces[:, 0]]
    v1 = v_cam[faces[:, 1]]
    v2 = v_cam[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    return n[:, 2] < 0


def _rasterize_front_faces_zbuf(
    H: int,
    W: int,
    v_cam: np.ndarray,
    faces: np.ndarray,
    K: np.ndarray,
    lut: np.ndarray,
    z_min: float,
    z_max: float,
    *,
    draw_edges: bool = True,
    edge_color: tuple[int, int, int] = (200, 200, 200),
) -> np.ndarray:
    """Z-buffered front faces only; colors = viridis(camera Z)."""
    import cv2

    out = np.zeros((H, W, 3), dtype=np.uint8)
    out[:] = (28, 28, 32)
    zbuf = np.full((H, W), np.inf, dtype=np.float64)

    facing = _face_facing_camera(v_cam, faces)
    z_centers = v_cam[faces].mean(axis=1)[:, 2]
    valid = facing & (z_centers > 0.02)
    if not valid.any():
        return out

    order = np.where(valid)[0]
    order = order[np.argsort(z_centers[order])]  # far → near (painter fallback)

    for fi in order:
        tri = faces[fi]
        pts = v_cam[tri]
        if pts[:, 2].min() < 0.02:
            continue
        u, v, ok = project_cam_to_image(pts, K)
        if ok.sum() < 3:
            continue
        uv = np.stack([u, v], axis=1).astype(np.float64)
        zv = pts[:, 2]

        u0, u1 = max(0, int(np.floor(uv[:, 0].min()))), min(W - 1, int(np.ceil(uv[:, 0].max())))
        v0, v1 = max(0, int(np.floor(uv[:, 1].min()))), min(H - 1, int(np.ceil(uv[:, 1].max())))
        if u1 < u0 or v1 < v0:
            continue

        # Barycentric raster (camera-space Z interpolation)
        v0p, v1p, v2p = uv[0], uv[1], uv[2]
        area = (v1p[0] - v0p[0]) * (v2p[1] - v0p[1]) - (v2p[0] - v0p[0]) * (v1p[1] - v0p[1])
        if abs(area) < 1e-6:
            continue

        for py in range(v0, v1 + 1):
            for px in range(u0, u1 + 1):
                w0 = (
                    (v1p[0] - px) * (v2p[1] - py) - (v2p[0] - px) * (v1p[1] - py)
                ) / area
                w1 = (
                    (v2p[0] - px) * (v0p[1] - py) - (v0p[0] - px) * (v2p[1] - py)
                ) / area
                w2 = 1.0 - w0 - w1
                if w0 < -1e-4 or w1 < -1e-4 or w2 < -1e-4:
                    continue
                z = w0 * zv[0] + w1 * zv[1] + w2 * zv[2]
                if z < zbuf[py, px]:
                    zbuf[py, px] = z
                    out[py, px] = _z_to_bgr(z, z_min, z_max, lut)

    if draw_edges:
        edge_layer = out.copy()
        for fi in order:
            tri = faces[fi]
            pts = v_cam[tri]
            u, v, ok = project_cam_to_image(pts, K)
            if not ok.all():
                continue
            poly = np.round(np.stack([u, v], axis=1)).astype(np.int32)
            cv2.polylines(
                edge_layer, [poly], isClosed=True, color=edge_color, thickness=1, lineType=cv2.LINE_AA
            )
        out = cv2.addWeighted(edge_layer, 0.55, out, 0.45, 0)

    return out


def _project_triangle(
    pts_cam: np.ndarray, K: np.ndarray, H: int, W: int
) -> tuple[np.ndarray | None, float]:
    if pts_cam[:, 2].min() < 0.02:
        return None, 0.0
    u, v, ok = project_cam_to_image(pts_cam, K)
    if not ok.all():
        return None, 0.0
    uv = np.stack([u, v], axis=1).astype(np.float64)
    if (uv[:, 0] < -W * 0.1).all() or (uv[:, 0] > W * 1.1).all():
        return None, 0.0
    if (uv[:, 1] < -H * 0.1).all() or (uv[:, 1] > H * 1.1).all():
        return None, 0.0
    return np.round(uv).astype(np.int32), float(pts_cam[:, 2].mean())


def _prepare_mesh_cam(
    mesh_path: Path, T_cam_mesh: np.ndarray, max_faces: int = 8000
) -> tuple[np.ndarray, np.ndarray]:
    mesh = simplify_mesh_for_vis(_load_mesh_trimesh(mesh_path), max_faces=max_faces)
    v_cam = mesh_to_cam(np.asarray(mesh.vertices, dtype=np.float64), T_cam_mesh)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    return v_cam, faces


def overlay_mask_on_rgb(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vis = rgb.astype(np.float32).copy()
    m = mask > 0
    tint = np.array([80.0, 200.0, 120.0], dtype=np.float32)
    vis[m] = vis[m] * 0.55 + tint * 0.45
    return np.clip(vis, 0, 255).astype(np.uint8)


def _z_range_front_faces(v_cam: np.ndarray, faces: np.ndarray) -> tuple[float, float]:
    facing = _face_facing_camera(v_cam, faces)
    z = v_cam[faces].mean(axis=1)[:, 2]
    z = z[facing & (z > 0.02)]
    if len(z) < 10:
        z = v_cam[v_cam[:, 2] > 0.02, 2]
    return float(np.percentile(z, 5)), float(np.percentile(z, 95))


def render_mesh_occluded(
    H: int,
    W: int,
    v_cam: np.ndarray,
    faces: np.ndarray,
    K: np.ndarray,
    *,
    background: np.ndarray | None = None,
    alpha: float = 1.0,
) -> tuple[np.ndarray, float, float]:
    """Front faces only + z-buffer; returns (image, z_min, z_max) for colorbar."""
    lut = _viridis_bgr_lut()
    z_min, z_max = _z_range_front_faces(v_cam, faces)
    solid = _rasterize_front_faces_zbuf(
        H, W, v_cam, faces, K, lut, z_min, z_max, draw_edges=True
    )
    if background is None:
        return solid, z_min, z_max

    import cv2

    base = background.copy()
    if base.dtype != np.uint8:
        base = np.clip(base, 0, 255).astype(np.uint8)
    gray = np.all(solid == (28, 28, 32), axis=2)
    out = cv2.addWeighted(solid, alpha, base, 1.0 - alpha, 0)
    out[gray] = base[gray]
    return out, z_min, z_max


def draw_mesh_on_rgb(
    rgb: np.ndarray,
    v_cam: np.ndarray,
    faces: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """Front-facing mesh over RGB (alignment check)."""
    out, _, _ = render_mesh_occluded(
        rgb.shape[0], rgb.shape[1], v_cam, faces, K, background=rgb, alpha=0.72
    )
    return out


def draw_bbox_panel(
    rgb: np.ndarray,
    T_cam_mesh: np.ndarray,
    mesh_path: Path,
    K: np.ndarray,
    fp_root: Path,
) -> np.ndarray:
    """FP oriented bbox axes (OBB ≠ mesh coordinate axes). Use for SAM3D row only."""
    import cv2

    fp_root = fp_root.resolve()
    if str(fp_root) not in sys.path:
        sys.path.insert(0, str(fp_root))
    from estimater import draw_posed_3d_box, draw_xyz_axis
    import trimesh

    mesh = _load_mesh_trimesh(mesh_path)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
    center_pose = T_cam_mesh @ np.linalg.inv(to_origin)

    img = rgb.copy()
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    vis = draw_posed_3d_box(
        K, img=img, ob_in_cam=center_pose, bbox=bbox, line_color=(0, 255, 0)
    )
    vis = draw_xyz_axis(
        vis,
        ob_in_cam=center_pose,
        scale=0.1,
        K=K,
        thickness=3,
        transparency=0,
        is_input_rgb=True,
    )
    return vis


def draw_mesh_frame_axes_panel(
    rgb: np.ndarray,
    T_cam_mesh: np.ndarray,
    mesh_path: Path,
    K: np.ndarray,
    fp_root: Path,
    *,
    axis_scale: float = 0.12,
    at_centroid: bool = True,
) -> np.ndarray:
    """
    Draw mesh coordinate axes (columns of R_cam_mesh): R=X, G=Y, B=Z.
    For base-aligned mesh, these are parallel to robot base axes (not OBB).
    """
    import cv2

    fp_root = fp_root.resolve()
    if str(fp_root) not in sys.path:
        sys.path.insert(0, str(fp_root))
    from estimater import draw_xyz_axis

    T = np.asarray(T_cam_mesh, dtype=np.float64).reshape(4, 4)
    if at_centroid:
        mesh = _load_mesh_trimesh(mesh_path)
        c = np.asarray(mesh.vertices, dtype=np.float64).mean(axis=0)
        T_show = T @ np.array(
            [
                [1, 0, 0, c[0]],
                [0, 1, 0, c[1]],
                [0, 0, 1, c[2]],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
    else:
        T_show = T

    img = rgb.copy()
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return draw_xyz_axis(
        img,
        ob_in_cam=T_show,
        scale=axis_scale,
        K=K,
        thickness=3,
        transparency=0,
        is_input_rgb=True,
    )


def pose_summary_line(T_cam_mesh: np.ndarray, v_cam: np.ndarray) -> str:
    """Short text: where mesh sits in camera frame (Z forward)."""
    c = v_cam.mean(axis=0)
    t = T_cam_mesh[:3, 3]
    try:
        from scipy.spatial.transform import Rotation as Rot

        euler = Rot.from_matrix(T_cam_mesh[:3, :3]).as_euler("xyz", degrees=True)
        rot_s = f"euler_xyz=[{euler[0]:+.0f},{euler[1]:+.0f},{euler[2]:+.0f}]°"
    except Exception:
        rot_s = "euler=?"
    return (
        f"mesh in camera frame: center≈({c[0]:.2f},{c[1]:.2f},{c[2]:.2f})m  "
        f"T translation=({t[0]:.2f},{t[1]:.2f},{t[2]:.2f})m  {rot_s}  "
        f"(+X right, +Y down, +Z forward)"
    )


def _plot_3d_camera_scene(
    ax,
    v_cam: np.ndarray,
    faces: np.ndarray,
    depth_pts: np.ndarray | None,
) -> None:
    """3D view in camera coordinates: green=depth, red=mesh wireframe."""
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    if depth_pts is not None and len(depth_pts) > 0:
        n = min(len(depth_pts), 3500)
        idx = np.linspace(0, len(depth_pts) - 1, n, dtype=int)
        p = depth_pts[idx]
        ax.scatter(
            p[:, 0], p[:, 1], p[:, 2], c="#2ecc71", s=1.5, alpha=0.5, label="depth (mask)"
        )

    # Mesh edges (subsample faces)
    step = max(1, len(faces) // 2500)
    segments = []
    for fi in faces[::step]:
        tri = v_cam[fi]
        if tri[:, 2].min() < 0.02:
            continue
        for a, b in ((0, 1), (1, 2), (2, 0)):
            segments.append([tri[a], tri[b]])
    if segments:
        lc = Line3DCollection(
            segments, colors="#ff6b6b", linewidths=0.35, alpha=0.85
        )
        ax.add_collection3d(lc)

    c = v_cam.mean(axis=0)
    ax.scatter([c[0]], [c[1]], [c[2]], c="yellow", s=40, marker="o", label="mesh center")
    L = 0.08
    ax.quiver(0, 0, 0, L, 0, 0, color="r", linewidth=1.5, arrow_length_ratio=0.2)
    ax.quiver(0, 0, 0, 0, L, 0, color="g", linewidth=1.5, arrow_length_ratio=0.2)
    ax.quiver(0, 0, 0, 0, 0, L, color="b", linewidth=1.5, arrow_length_ratio=0.2)

    pts_all = v_cam[v_cam[:, 2] > 0.05]
    if depth_pts is not None and len(depth_pts) > 0:
        pts_all = np.vstack([pts_all, depth_pts[: min(len(depth_pts), 2000)]])
    if len(pts_all) < 10:
        pts_all = v_cam
    ctr = pts_all.mean(axis=0)
    span = max(float(np.ptp(pts_all, axis=0).max()), 0.12) * 0.55
    ax.set_xlim(ctr[0] - span, ctr[0] + span)
    ax.set_ylim(ctr[1] - span, ctr[1] + span)
    ax.set_zlim(ctr[2] - span, ctr[2] + span)
    ax.set_xlabel("X →")
    ax.set_ylabel("Y ↓")
    ax.set_zlabel("Z forward")
    ax.view_init(elev=22, azim=-58)
    ax.legend(loc="upper left", fontsize=6, markerscale=2)


def _plot_3d_base_scene(
    ax,
    v_cam: np.ndarray,
    faces: np.ndarray,
    T_base_cam: np.ndarray,
    depth_pts_cam: np.ndarray | None,
    *,
    T_base_mesh: np.ndarray | None = None,
) -> None:
    """Mesh + depth in robot base; v_cam must already be T_cam_mesh @ v_local."""
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    T_base_cam = np.asarray(T_base_cam, dtype=np.float64).reshape(4, 4)
    ones = np.ones((len(v_cam), 1))
    v_base = (T_base_cam @ np.hstack([v_cam, ones]).T).T[:, :3]

    if depth_pts_cam is not None and len(depth_pts_cam) > 0:
        d1 = np.hstack([depth_pts_cam, np.ones((len(depth_pts_cam), 1))])
        d_base = (T_base_cam @ d1.T).T[:, :3]
        n = min(len(d_base), 2500)
        idx = np.linspace(0, len(d_base) - 1, n, dtype=int)
        ax.scatter(
            d_base[idx, 0],
            d_base[idx, 1],
            d_base[idx, 2],
            c="#2ecc71",
            s=1.2,
            alpha=0.45,
            label="depth→base",
        )

    step = max(1, len(faces) // 2500)
    segments = []
    for fi in faces[::step]:
        tri = v_base[fi]
        for a, b in ((0, 1), (1, 2), (2, 0)):
            segments.append([tri[a], tri[b]])
    if segments:
        ax.add_collection3d(
            Line3DCollection(segments, colors="#6eb5ff", linewidths=0.35, alpha=0.9)
        )

    c = v_base.mean(axis=0)
    ax.scatter([c[0]], [c[1]], [c[2]], c="yellow", s=35, marker="o")
    L = 0.1
    if T_base_mesh is not None:
        R_mb = np.asarray(T_base_mesh, dtype=np.float64).reshape(4, 4)[:3, :3]
        for col, color in enumerate(("r", "g", "b")):
            d = R_mb[:, col] * L
            ax.quiver(
                c[0], c[1], c[2], d[0], d[1], d[2],
                color=color,
                linewidth=1.2,
                arrow_length_ratio=0.2,
            )
    else:
        ax.quiver(c[0], c[1], c[2], L, 0, 0, color="r", linewidth=1.2, arrow_length_ratio=0.2)
        ax.quiver(c[0], c[1], c[2], 0, L, 0, color="g", linewidth=1.2, arrow_length_ratio=0.2)
        ax.quiver(c[0], c[1], c[2], 0, 0, L, color="b", linewidth=1.2, arrow_length_ratio=0.2)

    span = max(float(np.ptp(v_base, axis=0).max()), 0.15) * 0.6
    ax.set_xlim(c[0] - span, c[0] + span)
    ax.set_ylim(c[1] - span, c[1] + span)
    ax.set_zlim(c[2] - span, c[2] + span)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.set_xlabel("base X")
    ax.set_ylabel("base Y")
    ax.set_zlabel("base Z")
    ax.view_init(elev=24, azim=-52)


def draw_aligned_mesh_axes_on_rgb(
    rgb: np.ndarray,
    v_cam: np.ndarray,
    faces: np.ndarray,
    K: np.ndarray,
    T_cam_mesh: np.ndarray,
    mesh_path: Path,
    fp_root: Path,
) -> np.ndarray:
    """RGB + semi-transparent mesh + mesh/base axes at centroid."""
    over = draw_mesh_on_rgb(rgb, v_cam, faces, K)
    return draw_mesh_frame_axes_panel(
        over, T_cam_mesh, mesh_path, K, fp_root, at_centroid=True
    )


def save_foundationpose_comparison(
    rgb: np.ndarray,
    K: np.ndarray,
    out_path: Path,
    *,
    fp_root: Path,
    session_id: str = "",
    est_iter: int = 5,
    mask: np.ndarray | None = None,
    depth_m: np.ndarray | None = None,
    sam3d_T_cam_mesh: np.ndarray,
    sam3d_mesh_path: Path,
    aligned_T_cam_mesh: np.ndarray,
    aligned_mesh_path: Path,
    T_base_mesh_aligned: np.ndarray,
    align_R_residual: float = 0.0,
    T_base_cam: np.ndarray | None = None,
) -> Path:
    """
    2×3+1: top = SAM3D/FP frame; bottom = aligned RGB+axes + 3D base (no duplicate z-mesh panel).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    H, W = rgb.shape[:2]
    depth_pts = None
    if depth_m is not None and mask is not None:
        m, _, _ = preprocess_mask(mask)
        if m.shape != depth_m.shape:
            import cv2

            m = (
                cv2.resize(
                    m.astype(np.uint8), (depth_m.shape[1], depth_m.shape[0])
                )
                > 0
            )
        depth_pts = depth_mask_to_pointcloud(depth_m, m, K)

    # Row 0 — SAM3D / FP frame
    v0, f0 = _prepare_mesh_cam(sam3d_mesh_path, sam3d_T_cam_mesh)
    p_mask = overlay_mask_on_rgb(rgb, mask) if mask is not None else rgb.copy()
    p_bbox0 = draw_bbox_panel(rgb, sam3d_T_cam_mesh, sam3d_mesh_path, K, fp_root)
    p_black0, z0_min, z0_max = render_mesh_occluded(H, W, v0, f0, K)
    p_ov0 = draw_mesh_on_rgb(rgb, v0, f0, K)

    # Row 1 — base-aligned (same physical pose as row 0; only mesh frame relabeled)
    v1, f1 = _prepare_mesh_cam(aligned_mesh_path, aligned_T_cam_mesh)
    p_align = draw_aligned_mesh_axes_on_rgb(
        rgb, v1, f1, K, aligned_T_cam_mesh, aligned_mesh_path, fp_root
    )

    sid = session_id or "session"
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    fig.suptitle(
        f"T5 — {sid}  est_iter={est_iter}  |  base-align ‖R-I‖_F={align_R_residual:.4f}\n"
        f"Top: SAM3D/FP mesh frame  |  Bottom: base-aligned mesh (same 3D pose, mesh axes ∥ robot base)",
        fontsize=9,
        y=1.0,
    )

    gs = fig.add_gridspec(2, 4, hspace=0.14, wspace=0.05)
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax02 = fig.add_subplot(gs[0, 2])
    ax03 = fig.add_subplot(gs[0, 3], projection="3d")
    ax10 = fig.add_subplot(gs[1, 0:2])
    ax13 = fig.add_subplot(gs[1, 2:4], projection="3d")

    for ax, img, title in [
        (ax00, p_mask, "① RGB + mask"),
        (ax01, p_bbox0, "② FP bbox (SAM3D mesh frame)\n    R=X G=Y B=Z object axes"),
        (ax02, p_black0, "③ SAM3D mesh @ T_cam (camera Z color)"),
        (
            ax10,
            p_align,
            "⑤ Base-aligned: mesh ∩ RGB + axes\n"
            "    R=X G=Y B=Z = robot base (same pose as ②–③)",
        ),
    ]:
        ax.imshow(img)
        ax.set_title(title, fontsize=8, loc="left")
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.axis("off")

    sm0 = plt.cm.ScalarMappable(cmap="viridis", norm=Normalize(vmin=z0_min, vmax=z0_max))
    fig.colorbar(sm0, ax=ax02, fraction=0.046, pad=0.02, shrink=0.7).set_label("cam Z (m)", fontsize=6)

    _plot_3d_camera_scene(ax03, v0, f0, depth_pts)
    ax03.set_title("④ 3D camera: depth + SAM3D mesh", fontsize=8, loc="left")

    if T_base_cam is not None:
        _plot_3d_base_scene(
            ax13,
            v1,
            f1,
            T_base_cam,
            depth_pts,
            T_base_mesh=T_base_mesh_aligned,
        )
    else:
        ax13.text2D(
            0.5,
            0.5,
            "missing T_base_cam\n(input/calib/extrinsics.json)",
            transform=ax13.transAxes,
            ha="center",
            va="center",
            fontsize=9,
        )
    ax13.set_title(
        "⑥ 3D base: depth + aligned mesh\n    R/G/B = robot base axes at centroid",
        fontsize=8,
        loc="left",
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    return out_path


def comparison_from_session(
    session_dir: Path,
    *,
    fp_root: Path | None = None,
) -> Path:
    import json

    _SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
    if str(_SCRIPTS_ROOT) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS_ROOT))
    from _session_io import resolve_session_dirs  # noqa: E402

    _t5 = Path(__file__).resolve().parent
    if str(_t5) not in sys.path:
        sys.path.insert(0, str(_t5))
    from fp_common import default_fp_root, load_K, load_rgb_np  # noqa: E402

    dirs = resolve_session_dirs(session_dir=session_dir)
    fp_root = fp_root or default_fp_root()
    rgb = load_rgb_np(dirs.input_rel("rgb", "left_rgb.png"))
    K = load_K(dirs.input_dir)
    mask = load_mask_bool(dirs.output_rel("segment", "mask.png"))
    depth_m = load_depth_m(dirs.input_rel("depth", "depth.npy"))

    reg = dirs.output_rel("register")
    fp_cam = json.loads((reg / "T_cam_mesh_fp.json").read_text())
    alg_cam = json.loads((reg / "T_cam_mesh.json").read_text())
    alg_base = json.loads((reg / "T_base_mesh.json").read_text())
    T_fp = np.asarray(fp_cam["T_cam_mesh"], dtype=np.float64)
    T_alg = np.asarray(alg_cam["T_cam_mesh"], dtype=np.float64)
    T_base = np.asarray(alg_base["T_base_mesh"], dtype=np.float64)
    from fp_common import resolve_mesh_path  # noqa: E402

    mesh_fp = resolve_mesh_path(dirs, fp_cam.get("mesh_file", "output/mesh/object_scaled.glb"))
    mesh_alg = resolve_mesh_path(
        dirs, alg_cam.get("mesh_file", "output/mesh/object_base_aligned.glb")
    )
    meta_path = dirs.output_rel("register", "foundationpose_meta.json")
    est_iter = 5
    align_R = 0.0
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        est_iter = int(meta.get("est_iter", 5))
        align_R = float(meta.get("mesh_align", 0.0))
    from fp_common import load_T_base_cam  # noqa: E402

    return save_foundationpose_comparison(
        rgb,
        K,
        dirs.output_rel("vis", "T5_foundationpose_overlay.png"),
        fp_root=fp_root,
        session_id=dirs.session_id,
        est_iter=est_iter,
        mask=mask,
        depth_m=depth_m,
        sam3d_T_cam_mesh=T_fp,
        sam3d_mesh_path=mesh_fp,
        aligned_T_cam_mesh=T_alg,
        aligned_mesh_path=mesh_alg,
        T_base_mesh_aligned=T_base,
        align_R_residual=align_R,
        T_base_cam=load_T_base_cam(dirs.input_dir),
    )
