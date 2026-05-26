#!/usr/bin/env python3
"""Pose conversion utilities for PDM.

The simulator consumes a command pose in the sampler/candidate grasp frame:
  - position is the TCP/finger-center point in object_mesh coordinates
  - rotation columns are [finger_dir, lateral_dir, approach_dir]

The merged GT stores successful executed poses as Franka `panda_hand` wrist
poses in object_mesh coordinates. This module converts those executed wrist
poses back into simulator command poses for supervision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

TCP_OFFSET = 0.105

# Same adapter used in sim/run_grasp_sim.py:
# candidate command frame -> Franka panda_hand frame.
R_ADAPT = np.array(
    [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class CommandPose:
    """Simulator command pose in object_mesh frame."""

    position: np.ndarray  # (3,) TCP / finger-center
    rotation: np.ndarray  # (3, 3) candidate grasp frame


@dataclass(frozen=True)
class ExecutedPose:
    """Franka panda_hand wrist pose in object_mesh frame."""

    position: np.ndarray  # (3,) wrist
    rotation: np.ndarray  # (3, 3) panda_hand frame


def _as_np3(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(3)
    if not np.isfinite(arr).all():
        raise ValueError("non-finite 3-vector")
    return arr


def _as_np33(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(arr).all():
        raise ValueError("non-finite 3x3 matrix")
    return arr


def rotation_to_6d(rotation: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Convert rotation matrix to 6D representation using the first two columns."""

    if isinstance(rotation, torch.Tensor):
        return torch.cat([rotation[..., :, 0], rotation[..., :, 1]], dim=-1)
    rot = _as_np33(rotation)
    return np.concatenate([rot[:, 0], rot[:, 1]], axis=0).astype(np.float32)


def rotation_from_6d(r6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to a valid rotation matrix."""

    c1 = F.normalize(r6d[..., :3], dim=-1)
    c2_raw = r6d[..., 3:6]
    c2 = F.normalize(c2_raw - (c2_raw * c1).sum(-1, keepdim=True) * c1, dim=-1)
    c3 = torch.cross(c1, c2, dim=-1)
    return torch.stack([c1, c2, c3], dim=-1)


def rotation_from_6d_np(r6d: np.ndarray) -> np.ndarray:
    """Numpy version of 6D rotation decoding."""

    a = np.asarray(r6d, dtype=np.float64).reshape(6)
    c1 = a[:3]
    c1 = c1 / (np.linalg.norm(c1) + 1e-8)
    c2 = a[3:6]
    c2 = c2 - np.dot(c2, c1) * c1
    c2 = c2 / (np.linalg.norm(c2) + 1e-8)
    c3 = np.cross(c1, c2)
    return np.stack([c1, c2, c3], axis=1).astype(np.float32)


def is_valid_rotation(rotation: np.ndarray, atol: float = 5e-2) -> bool:
    """Loose validity check for rotation matrices from sim/HDF5."""

    try:
        rot = _as_np33(rotation)
    except ValueError:
        return False
    should_be_i = rot.T @ rot
    det = np.linalg.det(rot)
    return bool(
        np.allclose(should_be_i, np.eye(3), atol=atol)
        and np.isfinite(det)
        and abs(det - 1.0) <= atol
    )


def executed_to_command(
    wrist_position: np.ndarray,
    hand_rotation: np.ndarray,
    tcp_offset: float = TCP_OFFSET,
) -> CommandPose:
    """Convert executed Franka wrist pose to simulator command pose.

    This keeps everything in object_mesh coordinates. Only the local frame
    convention changes from panda_hand axes to candidate grasp axes.
    """

    wrist = _as_np3(wrist_position)
    r_hand = _as_np33(hand_rotation)
    approach = r_hand[:, 2]
    position = wrist + approach * float(tcp_offset)
    rotation = r_hand @ R_ADAPT.T
    return CommandPose(position=position.astype(np.float32), rotation=rotation.astype(np.float32))


def command_to_executed(
    command_position: np.ndarray,
    command_rotation: np.ndarray,
    tcp_offset: float = TCP_OFFSET,
) -> ExecutedPose:
    """Convert simulator command pose to expected panda_hand wrist pose."""

    pos = _as_np3(command_position)
    r_cmd = _as_np33(command_rotation)
    r_hand = r_cmd @ R_ADAPT
    wrist = pos - r_hand[:, 2] * float(tcp_offset)
    return ExecutedPose(position=wrist.astype(np.float32), rotation=r_hand.astype(np.float32))


def command_to_pose9(command: CommandPose) -> np.ndarray:
    """Pack command pose into PDM's 9D target vector."""

    return np.concatenate(
        [command.position.astype(np.float32), rotation_to_6d(command.rotation)],
        axis=0,
    ).astype(np.float32)


def pose9_to_command(pose9: np.ndarray) -> CommandPose:
    """Unpack a 9D PDM vector into a command pose."""

    arr = np.asarray(pose9, dtype=np.float32).reshape(9)
    return CommandPose(position=arr[:3], rotation=rotation_from_6d_np(arr[3:9]))
