"""DP3 (3D Diffusion Policy) closed-loop online adapter.

DP3 is a chunked receding-horizon BC policy:
  - obs = (point cloud + EE state)
  - action = horizon of EE waypoints
  - each "chunk" the policy is re-queried with fresh obs → new horizon

Unlike a2g_pdm which loads pre-computed grasp candidates and returns
ONE OpenLoopGraspCommand, DP3 needs sim state online → it returns a
``closed_loop_actions`` PolicyOutput whose ``actions`` payload carries
the HTTP server URL + chunk hyper-params for the executor to use.

The executor lives in ``sim/evaluation/curobo_executor.py``:
``execute_closed_loop_actions(scene, payload)``. It walks the chunked
rollout and stops when (a) lift_z > 3cm or (b) max_chunks reached.

The DP3 inference server itself is unchanged — see
``Baseline1/eval/dp3_inference_server.py`` in the gate3-curobo-ik
branch. Start it BEFORE eval like:

    PY_DP3=/home/accelerator/miniforge3/envs/dp3/bin/python
    $PY_DP3 Baseline1/eval/dp3_inference_server.py \
        --ckpt <path-to-ckpt> --port 8765
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evaluation.policies.base import EvaluationPolicy
from evaluation.specs import PolicyOutput


@dataclass
class DP3OnlinePolicyConfig:
    """Configuration for a single DP3 inference-server endpoint.

    A separate server is launched per ckpt; the executor then re-queries
    this server every chunk via HTTP (see Baseline1/eval/dp3_inference_server.py
    for the /predict and /info routes — request body is
    ``{"point_cloud": [T,N,3], "agent_pos": [T,D]}``, response is
    ``{"action": [n_action_steps, 8]}``).
    """

    server_url: str = "http://127.0.0.1:8765"
    max_chunks: int = 5
    success_dz_m: float = 0.03                # ★ early-stop threshold (matches collector)
    retry_physx: int = 1                      # ★ retry on PhysX NaN, DP3 is stochastic
    n_pc_points: int = 4096                   # must match shape_meta in DP3 task yaml
    request_timeout_s: int = 60               # /predict HTTP timeout


class DP3OnlinePolicy(EvaluationPolicy):
    """Adapter wrapping a remote DP3 inference server.

    ``predict()`` returns a PolicyOutput of kind ``closed_loop_actions``;
    the executor uses the ``actions`` payload to drive the rollout.
    """

    name = "dp3_online"

    def __init__(self, config: DP3OnlinePolicyConfig):
        self.config = config

    def predict(self, context: Any) -> PolicyOutput:
        return PolicyOutput(
            kind="closed_loop_actions",
            actions={
                "server_url":      self.config.server_url,
                "max_chunks":      self.config.max_chunks,
                "success_dz_m":    self.config.success_dz_m,
                "retry_physx":     self.config.retry_physx,
                "n_pc_points":     self.config.n_pc_points,
                "request_timeout": self.config.request_timeout_s,
            },
            metadata={
                "policy_name": self.name,
                "server_url":  self.config.server_url,
            },
        )
