"""Path constants for the OakInk → baseline_3-v4 retarget pipeline.

Keep ALL OakInk-specific filesystem assumptions in one place so the rest of
Baseline1/oakink/ doesn't hardcode them.
"""
import os

# Repo root (this file lives at <root>/Baseline1/oakink/oakink_paths.py)
_THIS = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))

# ── Raw OakInk dataset (downloaded to data_hub) ───────────────────────────────
OAKINK_ROOT = os.path.join(PROJ_ROOT, "data_hub", "RawData",
                           "ThirdPersonRawData", "oakink_v1")
OAKINK_IMAGE = os.path.join(OAKINK_ROOT, "image")
OAKINK_ANNO  = os.path.join(OAKINK_IMAGE, "anno")
OAKINK_OBJ_DIR = os.path.join(OAKINK_IMAGE, "obj")          # *.obj canonical meshes

# Per-frame anno subdirs (pkl files keyed by <seq_id>__<ts>__<sbj>__<frame>__<cam>.pkl)
OAKINK_HAND_J_DIR     = os.path.join(OAKINK_ANNO, "hand_j")
OAKINK_HAND_V_DIR     = os.path.join(OAKINK_ANNO, "hand_v")
OAKINK_OBJ_TRANSF_DIR = os.path.join(OAKINK_ANNO, "obj_transf")
OAKINK_CAM_INTR_DIR   = os.path.join(OAKINK_ANNO, "cam_intr")
OAKINK_GENERAL_INFO   = os.path.join(OAKINK_ANNO, "general_info")
# general_info pkl contains: cam_extr (4x4 T_w_c), cam_intr (3x3),
#   obj_anno (4x4 T_w_o — world-frame object pose, the SAME across cameras),
#   hand_anno (dict with hand_tsl in world, hand_pose 16x4 quat, hand_shape 10x1)

# ── Our generated assets ──────────────────────────────────────────────────────
OAKINK_USD_DIR = os.path.join(PROJ_ROOT, "output", "obj_usd", "oakink")  # ycb-dex parallel

# ── Manifest + outputs ────────────────────────────────────────────────────────
CLASS_ID_MAP = os.path.join(_THIS, "class_id_map.json")
GRASPABILITY_SCAN = os.path.join(_THIS, "assets", "graspability_scan.json")

# Output dir for v4-format hdf5 — date-tagged, see Baseline1/RETRAIN_V4_FULL12.md
def episodes_dir(tag: str) -> str:
    """e.g. tag='oakink_v3_15obj_3yaw_2026-05-25' → Baseline1/data/episodes_<tag>/"""
    return os.path.join(PROJ_ROOT, "Baseline1", "data", f"episodes_{tag}")
