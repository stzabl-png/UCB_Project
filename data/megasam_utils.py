"""
MegaSAM Intrinsics Loader
==========================
Utility to load camera intrinsics estimated by MegaSAM (DROID-SLAM + UniDepth)
for a given scene.  The intrinsics are jointly optimised over the entire video,
so they are significantly more accurate (~1-3%) than the per-frame RANSAC
estimate used by HaWoR or the diagonal heuristic used by HaPTIC.

MegaSAM output layout (produced by tools/run_megasam.py or camera_tracking_scripts/):
    mega-sam/outputs/{scene_name}_droid.npz
        intrinsic  : (3, 3) float32  — single K matrix for the whole video
        cam_c2w    : (N, 4, 4)       — per-frame camera-to-world pose
        depths     : (N, H, W)       — metric depth maps
    mega-sam/reconstructions/{scene_name}/
        intrinsics.npy  : (N, 4) = [fx, fy, cx, cy] per frame (same K replicated)
        poses.npy       : (N, 7) = [qx, qy, qz, qw, tx, ty, tz]

Usage::
    from data.megasam_utils import load_megasam_K, load_megasam_K_fx

    K = load_megasam_K('box_grab_01')   # 3x3 ndarray | None
    fx = load_megasam_K_fx('box_grab_01')  # float | None
"""

import os
import numpy as np

# ── Path to MegaSAM outputs (relative to this file's project root) ────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_THIS_DIR)
MEGASAM_DIR = os.path.join(_PROJECT_DIR, 'mega-sam')
MEGASAM_OUTPUT_DIR = os.path.join(MEGASAM_DIR, 'outputs')
MEGASAM_RECON_DIR = os.path.join(MEGASAM_DIR, 'reconstructions')


def load_megasam_K(scene_name: str) -> "np.ndarray | None":
    """Return the 3×3 intrinsic matrix K estimated by MegaSAM for *scene_name*.

    Tries the compact ``outputs/{scene_name}_droid.npz`` first (single K),
    then falls back to ``reconstructions/{scene_name}/intrinsics.npy``
    (per-frame [fx, fy, cx, cy], in which case the median is taken).

    Returns ``None`` if no MegaSAM output is found for this scene.
    """
    # Primary: _droid.npz → 'intrinsic' key (3x3)
    droid_path = os.path.join(MEGASAM_OUTPUT_DIR, f'{scene_name}_droid.npz')
    if os.path.exists(droid_path):
        try:
            d = np.load(droid_path, allow_pickle=True)
            K = d['intrinsic'].astype(np.float64)  # (3, 3)
            assert K.shape == (3, 3), f"Unexpected intrinsic shape: {K.shape}"
            return K
        except Exception as e:
            print(f"[megasam_utils] Warning: could not load {droid_path}: {e}")

    # Fallback: reconstructions/{scene}/intrinsics.npy → (N, 4) = [fx, fy, cx, cy]
    recon_path = os.path.join(MEGASAM_RECON_DIR, scene_name, 'intrinsics.npy')
    if os.path.exists(recon_path):
        try:
            intr = np.load(recon_path, allow_pickle=True)  # (N, 4)
            # Take median across frames for robustness
            fx, fy, cx, cy = np.median(intr, axis=0)
            K = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0,  0,  1]], dtype=np.float64)
            return K
        except Exception as e:
            print(f"[megasam_utils] Warning: could not load {recon_path}: {e}")

    return None


def load_megasam_K_fx(scene_name: str) -> "float | None":
    """Return only the horizontal focal length *fx* from MegaSAM.

    Convenience wrapper for HaWoR which accepts a scalar ``img_focal``
    (pixels).  Returns ``None`` when MegaSAM has not been run for this scene,
    so callers can fall back to self-estimation without extra bookkeeping.
    """
    K = load_megasam_K(scene_name)
    if K is None:
        return None
    return float(K[0, 0])


def load_megasam_poses(scene_name: str) -> "np.ndarray | None":
    """Return per-frame camera-to-world poses (N, 4, 4) from MegaSAM.

    Returns ``None`` if not available.
    """
    droid_path = os.path.join(MEGASAM_OUTPUT_DIR, f'{scene_name}_droid.npz')
    if os.path.exists(droid_path):
        try:
            d = np.load(droid_path, allow_pickle=True)
            return d['cam_c2w'].astype(np.float64)  # (N, 4, 4)
        except Exception as e:
            print(f"[megasam_utils] Warning: could not load poses from {droid_path}: {e}")
    return None


def load_megasam_depths(scene_name: str) -> "np.ndarray | None":
    """Return per-frame metric depth maps (N, H, W) from MegaSAM.

    Returns ``None`` if not available.
    """
    droid_path = os.path.join(MEGASAM_OUTPUT_DIR, f'{scene_name}_droid.npz')
    if os.path.exists(droid_path):
        try:
            d = np.load(droid_path, allow_pickle=True)
            return d['depths'].astype(np.float32)  # (N, H, W)
        except Exception as e:
            print(f"[megasam_utils] Warning: could not load depths from {droid_path}: {e}")
    return None


def K_as_haptic_intrinsics(K: np.ndarray) -> list:
    """Flatten 3×3 K to a 9-element float list compatible with HaPTIC's
    ``parse_det_seq(intrinsics=...)`` argument.

    HaPTIC does: ``np.array(intrinsics, dtype=np.float64).reshape(3, 3)``
    """
    return K.flatten().tolist()
