"""GraspVLA (PKU-EPIC) closed-loop online adapter.

Like DP3OnlinePolicy ([[dp3_online]]) but for the GraspVLA VLA model:
  - obs: 2 RGB cameras (front + side, 256×256) + 7D proprio + language instruction
  - action: delta sequence in robot-base frame
  - server: ZMQ ROUTER (not HTTP!), default port 6666
  - pretrained on SynGrasp-1B (zero-shot, no fine-tune needed)

Source code at third_party/GraspVLA/. See memory: graspvla-baseline.md for
the complete interface spec drilled from source.

Architecture: uses partner's closed_loop_actions PolicyKind slot, with
discriminator policy_name="graspvla". Executor branches in
sim/evaluation/curobo_executor.py.

Prerequisite (before eval):
    cd third_party/GraspVLA
    conda activate graspvla_env   # python 3.9.19 + torch 2.7.1
    python -m vla_network.scripts.serve --port 6666 --path <ckpt.safetensors>
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evaluation.policies.base import EvaluationPolicy
from evaluation.specs import PolicyOutput


@dataclass
class GraspVLAOnlinePolicyConfig:
    """Configuration for a GraspVLA ZMQ inference server endpoint."""

    server_addr: str = "tcp://127.0.0.1:6666"
    instruction: str = "pick up the object"         # generic by default;
                                                     # can override per-obj for ablation
    max_chunks: int = 5
    success_dz_m: float = 0.03                       # match DP3 + collector convention
    retry_physx: int = 1
    # 2 sim cameras (LIBERO defaults — see graspvla-baseline.md memory)
    front_view_pos: tuple = (1.0, 0.0, 1.45)         # MJCF/LIBERO world → port to our IsaacSim
    front_view_quat_wxyz: tuple = (0.56, 0.43, 0.43, 0.56)
    side_view_pos: tuple = (-0.057, 1.276, 1.488)
    side_view_quat_wxyz: tuple = (0.010, 0.007, 0.591, 0.806)
    img_h: int = 256
    img_w: int = 256
    # ZMQ-specific
    request_timeout_ms: int = 60_000
    # proprio history buffer length (server uses [-4] and [-1])
    proprio_history: int = 4
    # GraspVLA model expects EE = panda_EE + REAL_EEF_TO_SIM_EEF offset.
    # Standard Franka finger → identity (no shift). Extended finger → +3cm Z.
    # Source: franka_ros_controller.py:35-46. Our setup uses standard finger.
    extended_finger: bool = False


class GraspVLAOnlinePolicy(EvaluationPolicy):
    """Adapter wrapping a remote GraspVLA inference server.

    predict() returns PolicyOutput(kind=closed_loop_actions) — same kind as
    DP3 — and the executor uses `metadata.policy_name == "graspvla"` to
    branch into the VLA-specific execution path (delta-action integration,
    2 RGB render, ZMQ client).
    """

    name = "graspvla"

    def __init__(self, config: GraspVLAOnlinePolicyConfig):
        self.config = config

    def predict(self, context: Any) -> PolicyOutput:
        return PolicyOutput(
            kind="closed_loop_actions",
            actions={
                "server_addr":       self.config.server_addr,
                "instruction":       self.config.instruction,
                "max_chunks":        self.config.max_chunks,
                "success_dz_m":      self.config.success_dz_m,
                "retry_physx":       self.config.retry_physx,
                "img_h":             self.config.img_h,
                "img_w":             self.config.img_w,
                "front_view_pos":    list(self.config.front_view_pos),
                "front_view_quat":   list(self.config.front_view_quat_wxyz),
                "side_view_pos":     list(self.config.side_view_pos),
                "side_view_quat":    list(self.config.side_view_quat_wxyz),
                "request_timeout_ms": self.config.request_timeout_ms,
                "proprio_history":   self.config.proprio_history,
                "extended_finger":   self.config.extended_finger,
            },
            metadata={
                "policy_name": self.name,                  # ← dispatcher discriminator
                "server_addr": self.config.server_addr,
                "instruction": self.config.instruction,
            },
        )
