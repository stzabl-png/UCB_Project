"""Dataset augmentation presets for affordance training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse


@dataclass
class AugmentConfig:
    rotation: bool = True
    scale: bool = True
    shift: bool = True
    jitter: bool = True
    dropout: bool = True


def augment_config_from_args(args: argparse.Namespace) -> AugmentConfig:
    """Map --augment-mode / --no-augment to per-transform toggles."""
    if getattr(args, "no_augment", False):
        return AugmentConfig(False, False, False, False, False)

    mode = getattr(args, "augment_mode", "full")
    if mode == "none":
        return AugmentConfig(False, False, False, False, False)
    if mode == "weak":
        return AugmentConfig(True, True, True, True, False)
    return AugmentConfig(True, True, True, True, True)
