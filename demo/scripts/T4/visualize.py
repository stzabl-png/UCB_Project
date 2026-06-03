"""T4 scale preview: depth point cloud vs metric mesh in camera / image."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scale_common import project_cam_to_image


def _subsample(pts: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    if len(pts) <= n:
        return pts
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pts), n, replace=False)
    return pts[idx]


def save_scale_scene_preview(
    rgb: np.ndarray,
    mask: np.ndarray,
    depth_m: np.ndarray,
    K: np.ndarray,
    depth_pts: np.ndarray,
    mesh_verts_cam: np.ndarray,
    out_path: Path,
    *,
    session_id: str = "",
    scale_factor: float = 1.0,
    d_real_m: float = 0.0,
) -> Path:
    """
    2×2 figure:
      RGB+mask | RGB + depth points (metric, camera frame)
      RGB + scaled mesh projected (coarse PCA align) | 3D: depth pts + mesh
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H, W = rgb.shape[:2]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # Panel 1 — mask overlay
    vis_rgb = rgb.copy()
    m = mask > 0
    vis_rgb[m] = (vis_rgb[m].astype(np.float32) * 0.55 + np.array([80, 200, 120], np.float32) * 0.45).astype(
        np.uint8
    )

    # Subsample for drawing
    pts_d = _subsample(depth_pts, 4000, seed=1)
    verts_m = _subsample(mesh_verts_cam, 6000, seed=2)

    u_d, v_d, ok_d = project_cam_to_image(pts_d, K)
    u_m, v_m, ok_m = project_cam_to_image(verts_m, K)

    fig = plt.figure(figsize=(14, 10), facecolor="white")
    sid = session_id or "session"
    fig.suptitle(
        f"T4 scale — {sid}  scale={scale_factor:.4f}  d_real={d_real_m*100:.1f}cm  "
        f"(coarse mesh overlay; T5 FP refines pose)",
        fontsize=11,
        y=0.98,
    )

    ax0 = fig.add_subplot(2, 2, 1)
    ax0.imshow(vis_rgb)
    ax0.set_title(f"RGB + mask ({100 * m.mean():.1f}%)")
    ax0.axis("off")

    ax1 = fig.add_subplot(2, 2, 2)
    ax1.imshow(rgb)
    z_d = pts_d[ok_d, 2] if ok_d.any() else pts_d[:, 2]
    sc1 = ax1.scatter(
        u_d[ok_d],
        v_d[ok_d],
        c=z_d,
        s=2,
        cmap="viridis",
        alpha=0.65,
        vmin=np.percentile(pts_d[:, 2], 5),
        vmax=np.percentile(pts_d[:, 2], 95),
    )
    ax1.set_xlim(0, W)
    ax1.set_ylim(H, 0)
    ax1.set_title(f"Depth points in image ({ok_d.sum()} shown)")
    ax1.set_aspect("equal")
    plt.colorbar(sc1, ax=ax1, fraction=0.046, label="Z (m)")

    ax2 = fig.add_subplot(2, 2, 3)
    ax2.imshow(rgb)
    ax2.scatter(
        u_m[ok_m],
        v_m[ok_m],
        c="#e74c3c",
        s=0.8,
        alpha=0.55,
        linewidths=0,
        label="scaled mesh (coarse align)",
    )
    if ok_d.any():
        ax2.scatter(
            u_d[ok_d],
            v_d[ok_d],
            c="#2ecc71",
            s=1.5,
            alpha=0.35,
            linewidths=0,
            label="depth cloud",
        )
    ax2.set_xlim(0, W)
    ax2.set_ylim(H, 0)
    ax2.set_title("Scene compare: depth (green) vs mesh (red)")
    ax2.legend(loc="lower right", fontsize=7, markerscale=3)
    ax2.set_aspect("equal")

    ax3 = fig.add_subplot(2, 2, 4, projection="3d")
    pd = _subsample(depth_pts, 2500, seed=3)
    vm = _subsample(mesh_verts_cam, 2500, seed=4)
    ax3.scatter(pd[:, 0], pd[:, 1], pd[:, 2], c="#2ecc71", s=1, alpha=0.4, label="depth")
    ax3.scatter(vm[:, 0], vm[:, 1], vm[:, 2], c="#e74c3c", s=0.6, alpha=0.5, label="mesh")
    # camera origin + view direction hint
    ax3.scatter([0], [0], [0], c="k", s=40, marker="^", label="camera")
    ax3.quiver(0, 0, 0, 0, 0, 0.15, color="gray", linewidth=1.5, arrow_length_ratio=0.2)
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Y (m)")
    ax3.set_zlabel("Z (m)")
    ax3.set_title("Camera frame (metric)")
    ax3.legend(loc="upper right", fontsize=7, markerscale=2)
    all_pts = np.vstack([pd, vm, [[0, 0, 0]]])
    c = all_pts.mean(0)
    span = max(float((all_pts.max(0) - all_pts.min(0)).max()) * 0.55, 0.05)
    ax3.set_xlim(c[0] - span, c[0] + span)
    ax3.set_ylim(c[1] - span, c[1] + span)
    ax3.set_zlim(c[2] - span, c[2] + span)
    ax3.view_init(elev=18, azim=-68)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
