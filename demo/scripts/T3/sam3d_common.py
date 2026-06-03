"""SAM3D mesh reconstruction helpers for Phase 2 T3 (unscaled mesh only)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))
from _session_io import repo_root  # noqa: E402


def default_sam3d_root() -> Path:
    env = os.environ.get("SAM3D_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    candidates = [
        repo_root().parent / "sam-3d-objects",
        repo_root() / "third_party" / "sam-3d-objects",
        Path("/home/vision/Project/sam-3d-objects"),
    ]
    for p in candidates:
        if (p / "checkpoints" / "hf" / "pipeline.yaml").is_file():
            return p.resolve()
    return candidates[0].resolve()


def sam3d_config_path(sam3d_root: Path | None = None) -> Path:
    root = sam3d_root or default_sam3d_root()
    return root / "checkpoints" / "hf" / "pipeline.yaml"


def check_sam3d_installed(sam3d_root: Path | None = None) -> str | None:
    root = sam3d_root or default_sam3d_root()
    cfg = sam3d_config_path(root)
    if not root.is_dir():
        return f"SAM3D repo missing: {root} (set SAM3D_ROOT)"
    if not cfg.is_file():
        return f"SAM3D config missing: {cfg}"
    return None


def prepare_sam3d_env(sam3d_root: Path) -> None:
    os.environ.setdefault("LIDRA_SKIP_INIT", "true")
    os.environ.setdefault("CUDA_HOME", os.environ.get("CONDA_PREFIX", os.environ.get("CUDA_HOME", "")))
    root = sam3d_root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def git_commit_short(path: Path) -> str | None:
    try:
        r = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if r.returncode == 0:
            return r.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def load_rgb_pil(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_mask_png(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 0).astype(np.uint8)


def mask_coverage_pct(mask_bool: np.ndarray) -> float:
    return float(mask_bool.mean() * 100.0)


def session_id_from_input(input_dir: Path, fallback: str) -> str:
    sess = input_dir / "session.json"
    if sess.is_file():
        return json.loads(sess.read_text(encoding="utf-8")).get("session_id", fallback)
    return fallback


class Sam3dInference:
    """In-process SAM3D pipeline (same flags as sam-3d-objects/scripts/image_to_mesh.py)."""

    def __init__(
        self,
        config_file: Path,
        sam3d_root: Path | None = None,
        compile_model: bool = False,
    ) -> None:
        root = sam3d_root or config_file.resolve().parent.parent.parent
        prepare_sam3d_env(root)
        import sam3d_objects  # noqa: F401
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        config = OmegaConf.load(config_file)
        config.rendering_engine = "pytorch3d"
        config.compile_model = compile_model
        config.workspace_dir = str(config_file.parent)
        self._pipeline = instantiate(config)

    def predict(self, rgb: np.ndarray, mask: np.ndarray, seed: int) -> Any:
        m = (mask > 0).astype(np.uint8) * 255
        if m.shape[:2] != rgb.shape[:2]:
            raise ValueError(f"mask shape {m.shape[:2]} != rgb {rgb.shape[:2]}")
        rgba = np.concatenate([rgb[..., :3], m[..., None]], axis=-1)
        return self._pipeline.run(
            rgba,
            None,
            seed,
            stage1_only=False,
            with_mesh_postprocess=False,
            with_texture_baking=False,
            with_layout_postprocess=False,
            use_vertex_color=True,
            stage1_inference_steps=None,
            pointmap=None,
        )


def glb_to_trimesh(glb: Any) -> Any:
    import trimesh

    if isinstance(glb, trimesh.Trimesh):
        return glb
    if hasattr(glb, "vertices") and hasattr(glb, "faces"):
        kwargs: dict[str, Any] = {
            "vertices": np.asarray(glb.vertices),
            "faces": np.asarray(glb.faces),
        }
        if hasattr(glb, "visual") and glb.visual is not None:
            kwargs["visual"] = glb.visual
        return trimesh.Trimesh(**kwargs)
    raise TypeError(f"Unexpected GLB type: {type(glb)}")


def reconstruct_raw_mesh(
    engine: Sam3dInference,
    rgb: np.ndarray,
    mask: np.ndarray,
    seed: int,
) -> tuple[Any, float, dict[str, Any]]:
    """Returns (trimesh mesh, elapsed_s, sam3d_output_summary)."""
    t0 = time.time()
    out = engine.predict(rgb, mask, seed=seed)
    glb = out.get("glb")
    if glb is None:
        raise RuntimeError("SAM3D produced no mesh (glb is None)")
    mesh = glb_to_trimesh(glb)
    gs = out.get("gs")
    summary = {
        "raw_mesh_format": "glb_trimesh",
        "has_gaussian_splat": gs is not None,
        "gaussian_format": "ply" if gs is not None else None,
    }
    return mesh, time.time() - t0, summary


def write_object_raw_glb(mesh: Any, glb_path: Path) -> None:
    """Save SAM3D mesh as GLB (native pipeline format on disk)."""
    glb_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(glb_path), file_type="glb")


def write_sam3d_meta(
    meta_path: Path,
    *,
    session_id: str,
    rgb_path: Path,
    mask_path: Path,
    mesh_path: Path,
    mesh: Any,
    frame_origin: list[float] | None = None,
    mask_coverage: float,
    time_s: float,
    seed: int,
    sam3d_root: Path,
    config_path: Path,
    pipeline_flags: dict[str, Any],
    sam3d_output_summary: dict[str, Any] | None = None,
    preview_path: Path | None = None,
) -> None:
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    extent = (verts.max(axis=0) - verts.min(axis=0)).tolist()
    payload: dict[str, Any] = {
        "tool": "sam3d",
        "session_id": session_id,
        "frame_convention": "sam3d_native",
        "scaled": False,
        "rgb_path": str(rgb_path),
        "mask_path": str(mask_path),
        "object_raw_glb": str(mesh_path.name),
        "export_format": "glb",
        "export_note": "SAM3D pipeline glb (trimesh) written directly; mesh frame = sam3d_native",
        "mesh_frame_origin": frame_origin,
        "mesh_frame_axes": "RGB = XYZ (orthonormal, identity at export time)",
        "sam3d_raw_outputs": sam3d_output_summary
        or {"raw_mesh_format": "glb_trimesh", "has_gaussian_splat": True, "gaussian_format": "ply"},
        "mask_coverage_percent": round(mask_coverage, 2),
        "n_verts": int(len(verts)),
        "n_faces": int(len(faces)),
        "bbox_extent": [round(float(x), 6) for x in extent],
        "time_s": round(time_s, 2),
        "seed": seed,
        "sam3d_root": str(sam3d_root),
        "config_path": str(config_path),
        "sam3d_commit": git_commit_short(sam3d_root),
        "pipeline_flags": pipeline_flags,
        "created_at_iso": datetime.now(timezone.utc).astimezone().isoformat(),
    }
    if preview_path is not None:
        payload["preview_png"] = str(preview_path)
    meta_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
