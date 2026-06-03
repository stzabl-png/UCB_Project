"""Headless SAM3D mesh preview: triangle mesh + object-frame axes (matplotlib)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Mesh frame = SAM3D native; axes are orthonormal basis columns of this matrix.
MESH_FRAME_R = np.eye(3, dtype=np.float64)


def mesh_frame_origin(verts: np.ndarray) -> np.ndarray:
    """Origin for axis triad: world origin if inside AABB, else vertex centroid."""
    vmin, vmax = verts.min(axis=0), verts.max(axis=0)
    if np.all(vmin <= 0) and np.all(vmax >= 0):
        return np.zeros(3, dtype=np.float64)
    return verts.mean(axis=0)


def axis_length(verts: np.ndarray) -> float:
    ext = float((verts.max(axis=0) - verts.min(axis=0)).max())
    return max(ext * 0.35, 1e-4)


def draw_mesh_frame_axes(ax, origin: np.ndarray, length: float, R: np.ndarray = MESH_FRAME_R) -> None:
    """RGB = XYZ unit axes of the mesh (object) coordinate frame."""
    colors = ("#e74c3c", "#2ecc71", "#3498db")
    labels = ("X", "Y", "Z")
    for i in range(3):
        d = R[:, i] * length
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            d[0],
            d[1],
            d[2],
            color=colors[i],
            linewidth=2.2,
            arrow_length_ratio=0.12,
        )
        tip = origin + d
        ax.text(tip[0], tip[1], tip[2], labels[i], color=colors[i], fontsize=9, fontweight="bold")


def _scene_bounds(verts: np.ndarray, origin: np.ndarray, axis_len: float) -> tuple[np.ndarray, float]:
    pts = np.vstack([verts, origin[None, :]])
    for i in range(3):
        pts = np.vstack([pts, (origin + MESH_FRAME_R[:, i] * axis_len)[None, :]])
    center = pts.mean(axis=0)
    span = float((pts.max(axis=0) - pts.min(axis=0)).max()) * 0.55 + 1e-6
    return center, span


def _render_mesh_surface(
    ax,
    verts: np.ndarray,
    faces: np.ndarray,
    *,
    max_faces: int,
) -> None:
    """Shaded triangle mesh (not a point cloud)."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if len(faces) > max_faces:
        step = max(1, len(faces) // max_faces)
        faces = faces[::step]

    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / (norms + 1e-8)

    light = np.array([0.35, -0.55, 0.85], dtype=np.float64)
    light /= np.linalg.norm(light)
    intensity = np.clip(normals @ light, 0.15, 1.0)
    base = np.array([0.75, 0.78, 0.85])
    rgba = np.hstack([np.outer(intensity, base), np.full((len(faces), 1), 0.92)])

    poly = Poly3DCollection(
        verts[faces],
        facecolors=rgba,
        edgecolors=(0.3, 0.32, 0.38, 0.18),
        linewidths=0.15,
    )
    ax.add_collection3d(poly)


def _setup_mesh_axes_view(
    ax,
    verts: np.ndarray,
    faces: np.ndarray,
    *,
    origin: np.ndarray,
    axis_len: float,
    max_faces: int,
    elev: float,
    azim: float,
    title: str,
) -> None:
    _render_mesh_surface(ax, verts, faces, max_faces=max_faces)
    draw_mesh_frame_axes(ax, origin, axis_len)
    center, span = _scene_bounds(verts, origin, axis_len)
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("X", fontsize=8, labelpad=0)
    ax.set_ylabel("Y", fontsize=8, labelpad=0)
    ax.set_zlabel("Z", fontsize=8, labelpad=0)
    ax.set_title(title, fontsize=9)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False


def rgb_mask_panel(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vis = rgb.copy()
    m = mask > 0
    vis[m] = (vis[m].astype(np.float32) * 0.55 + np.array([80, 200, 120], dtype=np.float32) * 0.45).astype(
        np.uint8
    )
    return vis


def save_sam3d_mesh_preview(
    mesh,
    rgb: np.ndarray,
    mask: np.ndarray,
    out_path: Path,
    *,
    session_id: str = "",
    max_faces: int = 24000,
    dpi: int = 150,
    frame_origin: np.ndarray | None = None,
) -> Path:
    """1×3: RGB+mask | triangle mesh + frame axes ×2 views."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    origin = frame_origin if frame_origin is not None else mesh_frame_origin(verts)
    alen = axis_length(verts)
    n_drawn = min(len(faces), max_faces)

    fig = plt.figure(figsize=(14, 4.8), facecolor="white")
    title = f"T3 SAM3D — {session_id}" if session_id else "T3 SAM3D mesh preview"
    fig.suptitle(
        f"{title} · triangle mesh ({len(faces)} faces, drawing {n_drawn}) · "
        f"RGB axes = XYZ, origin [{origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f}]",
        fontsize=11,
        y=0.98,
    )

    ax0 = fig.add_subplot(1, 3, 1)
    ax0.imshow(rgb_mask_panel(rgb, mask))
    ax0.set_title(f"RGB + mask ({(mask > 0).mean() * 100:.1f}%)")
    ax0.axis("off")

    ax1 = fig.add_subplot(1, 3, 2, projection="3d")
    _setup_mesh_axes_view(
        ax1,
        verts,
        faces,
        origin=origin,
        axis_len=alen,
        max_faces=max_faces,
        elev=22,
        azim=-58,
        title="mesh + frame",
    )

    ax2 = fig.add_subplot(1, 3, 3, projection="3d")
    _setup_mesh_axes_view(
        ax2,
        verts,
        faces,
        origin=origin,
        axis_len=alen,
        max_faces=max_faces,
        elev=16,
        azim=132,
        title="mesh + frame (alt)",
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def preview_from_glb(
    glb_path: Path,
    rgb_path: Path,
    mask_path: Path,
    out_path: Path,
    *,
    session_id: str = "",
) -> Path:
    import trimesh

    from sam3d_common import load_mask_png, load_rgb_pil

    mesh = trimesh.load(str(glb_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    rgb = load_rgb_pil(rgb_path)
    mask = load_mask_png(mask_path)
    return save_sam3d_mesh_preview(mesh, rgb, mask, out_path, session_id=session_id)
