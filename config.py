"""
Affordance2Grasp — Global Configuration
========================================
All paths and default parameters in one place.
Uses environment variables for external tool paths (no hardcoded machine-specific paths).
"""

import os

# ============================================================
# Project Paths (auto-detected, always portable)
# ============================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
ASSETS_DIR = os.path.join(PROJECT_DIR, "assets")

# ============================================================
# data_hub — Unified Data Center
# ============================================================
DATA_HUB = os.path.join(PROJECT_DIR, "data_hub")
MESH_V1_DIR = os.path.join(DATA_HUB, "meshes", "v1")
MESH_V2_DIR = os.path.join(DATA_HUB, "meshes", "v2")
MESH_CP_DIR = os.path.join(DATA_HUB, "meshes", "contactpose")
GRASP_MESH_DIR = os.path.join(DATA_HUB, "meshes", "grasp_collection")
SEQUENCES_V1_DIR = os.path.join(DATA_HUB, "sequences", "v1")
TRAINING_M5_DIR = os.path.join(DATA_HUB, "training_m5")

HUMAN_PRIOR_DIR = os.path.join(DATA_HUB, "human_prior")  # lowercase — matches Phase 2/3 scripts
ROBOT_GT_DIR = os.path.join(DATA_HUB, "robot_gt")
TRAINING_DIR = os.path.join(DATA_HUB, "training")
REGISTRY_PATH = os.path.join(DATA_HUB, "registry.json")

# Legacy aliases (for backward compatibility)
OAKINK_OBJ_DIR = MESH_V1_DIR
OAKINK_FILTERED_DIR = SEQUENCES_V1_DIR
OAKINK2_OBJ_DIR = MESH_V2_DIR

# Output subdirectories
CONTACTS_DIR = os.path.join(OUTPUT_DIR, "contacts")
CONTACTS_V2_DIR = os.path.join(OUTPUT_DIR, "contacts_v2")
DATASET_DIR = os.path.join(OUTPUT_DIR, "dataset")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
GRASPS_DIR = os.path.join(OUTPUT_DIR, "grasps")

# ============================================================
# External Tool Paths (set via environment variables)
# Only needed for upstream data generation, NOT for training/inference.
# ============================================================
ISAAC_SIM_PATH = os.environ.get("ISAAC_SIM_PATH", "")
HAWOR_DIR  = os.environ.get("HAWOR_DIR",  os.path.join(PROJECT_DIR, "third_party", "hawor"))
HAPTIC_DIR = os.environ.get("HAPTIC_DIR", os.path.join(PROJECT_DIR, "third_party", "haptic"))
ARCTIC_ROOT = os.environ.get("ARCTIC_ROOT", "")
MANO_MODELS = os.path.join(ARCTIC_ROOT, "mano_v1_2", "models") if ARCTIC_ROOT else ""

# OakInk annotation repo (for GT camera intrinsics in batch_haptic_oakink.py)
OAKINK_ANNO_DIR = os.environ.get("OAKINK_ANNO_DIR", "")

# ContactPose (optional external dataset)
CONTACTPOSE_DIR = os.environ.get("CONTACTPOSE_DIR", "")
CONTACTPOSE_DATA_DIR = os.path.join(
    CONTACTPOSE_DIR, "ContactPose sample data", "contactpose_data"
) if CONTACTPOSE_DIR else ""

# MANO/Contact caches
HAWOR_CACHE = os.path.join(OUTPUT_DIR, "hawor_arctic_cache")
HAPTIC_CACHE = os.path.join(OUTPUT_DIR, "haptic_arctic_cache")
ONSET_JSON = os.path.join(OUTPUT_DIR, "arctic_grasp_onset.json")
CONTACT_VIS_DIR = os.path.join(OUTPUT_DIR, "contact_region_vis")

# SAM3D mesh reconstruction
SAM3D_DIR = os.environ.get("SAM3D_DIR", "")
SAM3D_CACHE = os.path.join(OUTPUT_DIR, "sam3d_obj_cache")   # triangle mesh .obj files
SAM3D_PLY_CACHE = os.path.join(OUTPUT_DIR, "sam3d_mesh_cache")  # raw Gaussian Splat .ply files

# MegaSAM camera tracking (focal + per-frame poses + depth)
MEGASAM_DIR = os.path.join(PROJECT_DIR, "mega-sam")
MEGASAM_OUTPUT = os.path.join(MEGASAM_DIR, "outputs")        # {scene}_droid.npz
MEGASAM_RECON = os.path.join(MEGASAM_DIR, "reconstructions")  # {scene}/intrinsics.npy


# ============================================================
# Default Parameters
# ============================================================

# Contact extraction
CONTACT_THRESHOLD = 0.005    # 5mm contact distance threshold
FRAME_STEP = 5               # Sample every N frames

# Video contact thresholds
GT_CONTACT_TH = 0.015        # 15mm — GT true contact
PRED_CONTACT_TH = 0.030      # 30mm — prediction threshold

# Stability filtering
MIN_FINGERS = 3              # Min fingers in simultaneous contact
MIN_STABLE_FRAMES = 10       # Min consecutive stable frames

# Dataset
NUM_POINTS = 1024            # Point cloud sample count
CONTACT_RADIUS = 0.005       # Contact label radius

# Training
TRAIN_EPOCHS = 150
TRAIN_BATCH_SIZE = 32
TRAIN_LR = 0.001

# Inference
AFFORDANCE_THRESHOLD = 0.3   # Contact probability threshold

# Simulation
OBJECT_SCALE = 1.5
TABLE_TOP_Z = 0.80
ROBOT_POSITION = [0.2, -0.05, 0.8]
ROBOT_ORIENTATION = [0.0, 0.0, 90.0]

# Robot GT
GAUSSIAN_SIGMA = 0.005       # 5mm gaussian kernel radius

# ============================================================
# Utilities
# ============================================================
def ensure_dirs():
    """Create all output directories."""
    for d in [CONTACTS_DIR, DATASET_DIR, CHECKPOINT_DIR, GRASPS_DIR,
              HUMAN_PRIOR_DIR, ROBOT_GT_DIR, TRAINING_DIR, TRAINING_M5_DIR,
              GRASP_MESH_DIR, HAWOR_CACHE, HAPTIC_CACHE, CONTACT_VIS_DIR]:
        os.makedirs(d, exist_ok=True)
