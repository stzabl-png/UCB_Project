"""Shared helpers for T2 segmentation (OpenCV GUI + Gradio web)."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))
from _session_io import repo_root  # noqa: E402


def sam2_paths() -> tuple[Path, Path, Path]:
    root = repo_root() / "third_party" / "sam2"
    ckpt = root / "checkpoints" / "sam2.1_hiera_tiny.pt"
    cfg_dir = root / "sam2" / "configs"
    return root, ckpt, cfg_dir


def check_sam2_installed() -> str | None:
    root, ckpt, _ = sam2_paths()
    if not root.is_dir():
        return f"SAM2 repo missing: {root}"
    if not ckpt.is_file():
        return f"SAM2 checkpoint missing: {ckpt}"
    return None


def load_rgb_pil(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def mask_coverage_pct(mask: np.ndarray) -> float:
    return float((mask > 0).mean() * 100.0)


def render_overlay(
    rgb: np.ndarray,
    mask: np.ndarray | None,
    fg: list[list[int]],
    bg: list[list[int]],
) -> np.ndarray:
    vis = rgb.copy()
    if mask is not None:
        m = mask > 0
        vis[m] = (vis[m].astype(np.float32) * 0.45 + np.array([80, 200, 120], dtype=np.float32) * 0.55).astype(
            np.uint8
        )
    for x, y in fg:
        cv2.circle(vis, (int(x), int(y)), 6, (80, 255, 120), -1)
        cv2.circle(vis, (int(x), int(y)), 6, (255, 255, 255), 1)
    for x, y in bg:
        cv2.circle(vis, (int(x), int(y)), 6, (220, 80, 80), -1)
        cv2.circle(vis, (int(x), int(y)), 6, (255, 255, 255), 1)
    return vis


def write_outputs(
    out_segment: Path,
    mask: np.ndarray,
    fg: list[list[int]],
    bg: list[list[int]],
    session_id: str,
    rgb_shape: tuple[int, int],
    source: str = "interactive",
) -> None:
    """Write SAM2 mask as mask.png only (no post-processing, no mask_raw)."""
    out_segment.mkdir(parents=True, exist_ok=True)
    if mask.shape != rgb_shape:
        raise ValueError(f"mask shape {mask.shape} != rgb {rgb_shape}")

    out = mask.astype(np.uint8)
    out[out > 0] = 255
    cv2.imwrite(str(out_segment / "mask.png"), out)

    payload: dict[str, Any] = {
        "tool": "sam2",
        "source": source,
        "session_id": session_id,
        "prompts": {
            "fg": [{"type": "point", "xy": p, "label": 1} for p in fg],
            "bg": [{"type": "point", "xy": p, "label": 0} for p in bg],
        },
        "mask_coverage_percent": round(mask_coverage_pct(out), 2),
        "created_at_iso": datetime.now(timezone.utc).astimezone().isoformat(),
    }
    (out_segment / "prompt_used.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


class Sam2Predictor:
    """In-process SAM2 image predictor (for Gradio; avoids subprocess per click)."""

    def __init__(self) -> None:
        err = check_sam2_installed()
        if err:
            raise RuntimeError(err)

        sam2_root, ckpt, cfg_dir = sam2_paths()
        if str(sam2_root) not in sys.path:
            sys.path.insert(0, str(sam2_root))

        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import OmegaConf
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(cfg_dir), version_base="1.2"):
            cfg = compose(config_name="sam2.1/sam2.1_hiera_t.yaml")
        OmegaConf.resolve(cfg)
        model = instantiate(cfg.model, _recursive_=True)
        model.eval()
        try:
            state = torch.load(ckpt, map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(ckpt, map_location=self.device)
        model.load_state_dict(state["model"] if "model" in state else state)
        model.to(self.device)
        self.predictor = SAM2ImagePredictor(model)
        self._autocast_dtype = torch.float16 if self.device == "cuda" else torch.float32

    def set_image(self, rgb: np.ndarray) -> None:
        with torch.inference_mode():
            if self.device == "cuda":
                with torch.autocast("cuda", dtype=self._autocast_dtype):
                    self.predictor.set_image(rgb)
            else:
                self.predictor.set_image(rgb)

    def predict_mask(self, fg: list[list[int]], bg: list[list[int]]) -> np.ndarray | None:
        if not fg:
            return None
        pts = np.array(fg + bg, dtype=np.float32)
        lbs = np.array([1] * len(fg) + [0] * len(bg), dtype=np.int32)
        with torch.inference_mode():
            if self.device == "cuda":
                with torch.autocast("cuda", dtype=self._autocast_dtype):
                    masks, scores, _ = self.predictor.predict(
                        point_coords=pts, point_labels=lbs, multimask_output=True
                    )
            else:
                masks, scores, _ = self.predictor.predict(
                    point_coords=pts, point_labels=lbs, multimask_output=True
                )
        best = int(np.argmax(scores))
        return (masks[best] * 255).astype(np.uint8)


def session_id_from_dirs(input_dir: Path, fallback: str) -> str:
    sess = input_dir / "session.json"
    if sess.is_file():
        return json.loads(sess.read_text()).get("session_id", fallback)
    return fallback
