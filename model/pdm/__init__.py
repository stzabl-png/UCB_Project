"""Pose Diffusion Model (PDM) package.

PDM predicts object-frame grasp command poses from object geometry and
affordance features. The primary supervision is converted from successful
Franka `panda_hand` executed poses at gripper close.
"""

from .model import PDM, PDMConfig
from .pose_codec import (
    TCP_OFFSET,
    command_to_executed,
    executed_to_command,
    pose9_to_command,
    command_to_pose9,
    rotation_from_6d,
    rotation_to_6d,
)

__all__ = [
    "PDM",
    "PDMConfig",
    "TCP_OFFSET",
    "command_to_executed",
    "executed_to_command",
    "pose9_to_command",
    "command_to_pose9",
    "rotation_from_6d",
    "rotation_to_6d",
]
