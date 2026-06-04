"""Paths and conda interpreters for Phase 2 pipeline."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

PIPELINE_VERSION = "demo.pipeline.process_razor_session 0.1.0"


def repo_root() -> Path:
    """Affordance2Grasp repository root."""
    return Path(__file__).resolve().parents[2]


def sessions_root() -> Path:
    """Default directory for Razor rsync sessions."""
    return repo_root() / "demo" / "sessions"


def _conda_python(env_name: str) -> Path | None:
    base = os.environ.get("CONDA_PREFIX")
    if base:
        root = Path(base).parents[1] if Path(base).name in ("bin", "lib") else Path(base)
    else:
        for candidate in (
            Path.home() / "miniconda3",
            Path.home() / "anaconda3",
            Path("/home/vision/miniconda3"),
        ):
            if (candidate / "envs").is_dir():
                root = candidate
                break
        else:
            return None
    py = root / "envs" / env_name / "bin" / "python"
    return py if py.is_file() else None


def bundlesdf_python() -> Path:
    py = os.environ.get("S2R_BUNDLESDF_PYTHON")
    if py and Path(py).is_file():
        return Path(py)
    found = _conda_python("bundlesdf")
    if found:
        return found
    sys_py = shutil.which("python")
    if sys_py:
        return Path(sys_py)
    raise RuntimeError(
        "bundlesdf python not found. Set S2R_BUNDLESDF_PYTHON or install env 'bundlesdf'."
    )


def sam3d_python() -> Path:
    py = os.environ.get("S2R_SAM3D_PYTHON")
    if py and Path(py).is_file():
        return Path(py)
    found = _conda_python("sam3d-objects")
    if found:
        return found
    raise RuntimeError(
        "sam3d-objects python not found. Set S2R_SAM3D_PYTHON or install env 'sam3d-objects'."
    )


def default_fp_root() -> Path:
    env = os.environ.get("FP_ROOT")
    if env:
        return Path(env).resolve()
    return repo_root() / "third_party" / "FoundationPose"


def pipeline_env(fp_root: Path | None = None) -> dict[str, str]:
    """Environment for subprocess steps (inherits os.environ)."""
    env = os.environ.copy()
    root = repo_root()
    env["UCB_ROOT"] = str(root)
    env["FP_ROOT"] = str(fp_root or default_fp_root())
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (env.get("PYTHONPATH", ""), str(root)) if p
    )
    return env
