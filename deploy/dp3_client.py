"""DP3 inference HTTP client + n_obs observation-window management.

Transport + window only. Speaks the Baseline1/eval/dp3_inference_server.py
protocol and reproduces sim/eval_dp3_titan_protocol.py's rolling-window
semantics exactly. Does NO coordinate math — frame transforms live in
dp3_frames.py; the caller packs proprio via dp3_frames.pinch_to_proprio and
passes the 8-vector here.

Server protocol (dp3_inference_server.build_app):
  GET  /info    -> {horizon, n_obs_steps, n_action_steps, action_dim,
                    point_cloud_shape, agent_pos_dim}
  POST /predict {point_cloud:[T,N,3], agent_pos:[T,D]}
                -> {action:[n_action,D], action_pred:[horizon,D], inference_ms}

Window semantics (matches the eval): the point cloud is FIXED for the whole
episode (object-centric, sampled once); only the proprio updates. Each chunk the
caller measures the current proprio and calls step(proprio); the client keeps a
sliding window of the last n_obs proprios (initial window padded with the first
proprio) and POSTs {[pc_G]*n_obs, window} to /predict. This reproduces the eval's
"overwrite latest slot + roll" exactly — there the post-execute obs equals the
next chunk's predict-time obs, so one measurement per chunk suffices (and on a
real robot, measuring fresh at predict time is strictly more correct).

Stdlib-only (urllib + json) so it runs on the razer laptop with no extra deps.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections import deque

import numpy as np


class DP3ServerError(RuntimeError):
    """Raised on any transport / HTTP / protocol error talking to the DP3 server."""


class DP3Client:
    def __init__(self, server_url: str, *, n_obs: int | None = None, timeout: float = 10.0):
        self.server_url = server_url.rstrip("/")
        self.timeout = float(timeout)
        self._info: dict | None = None
        self._n_obs: int | None = n_obs
        self._pc_G: np.ndarray | None = None
        self._window: deque | None = None  # deque[(D,) float32], maxlen = n_obs

    # ── protocol ──────────────────────────────────────────────────────────────
    def info(self, refresh: bool = False) -> dict:
        """GET /info (cached). Also resolves n_obs if not given to __init__."""
        if self._info is None or refresh:
            self._info = self._get("/info")
            if self._n_obs is None:
                self._n_obs = int(self._info["n_obs_steps"])
        return self._info

    @property
    def n_obs(self) -> int:
        if self._n_obs is None:
            self.info()
        return int(self._n_obs)

    def predict_raw(self, pc_stack, agent_stack) -> dict:
        """Stateless POST /predict. pc_stack (T,N,3), agent_stack (T,D) -> full response."""
        pc = np.asarray(pc_stack, dtype=np.float32)
        ap = np.asarray(agent_stack, dtype=np.float32)
        if pc.ndim != 3 or pc.shape[2] != 3:
            raise ValueError(f"pc_stack must be (T,N,3), got {pc.shape}")
        if ap.ndim != 2:
            raise ValueError(f"agent_stack must be (T,D), got {ap.shape}")
        if pc.shape[0] != ap.shape[0]:
            raise ValueError(f"T mismatch: pc {pc.shape[0]} vs ap {ap.shape[0]}")
        return self._post("/predict", {"point_cloud": pc.tolist(), "agent_pos": ap.tolist()})

    # ── stateful rolling window ───────────────────────────────────────────────
    def reset(self, pc_G, proprio0=None) -> None:
        """Start an episode: fix the point cloud pc_G (N,3); optionally pre-pad the
        proprio window with proprio0. If proprio0 is None the window is lazily padded
        with the first proprio passed to step()."""
        pc = np.asarray(pc_G, dtype=np.float32)
        if pc.ndim != 2 or pc.shape[1] != 3:
            raise ValueError(f"pc_G must be (N,3), got {pc.shape}")
        self._pc_G = pc
        if proprio0 is None:
            self._window = None
        else:
            p0 = np.asarray(proprio0, dtype=np.float32).reshape(-1)
            self._window = deque((p0.copy() for _ in range(self.n_obs)), maxlen=self.n_obs)

    def step(self, proprio, *, full: bool = False):
        """Roll in the current proprio (D,) and query the policy.

        Returns action (n_action, D) float32, or the full response dict if full=True.
        """
        if self._pc_G is None:
            raise RuntimeError("call reset(pc_G) before step()")
        p = np.asarray(proprio, dtype=np.float32).reshape(-1)
        n = self.n_obs
        if self._window is None:
            self._window = deque((p.copy() for _ in range(n)), maxlen=n)
        else:
            self._window.append(p.copy())
        ap = np.stack(list(self._window))                 # (n_obs, D)
        pc = np.repeat(self._pc_G[None], n, axis=0)        # (n_obs, N, 3)
        resp = self.predict_raw(pc, ap)
        if full:
            return resp
        return np.asarray(resp["action"], dtype=np.float32)

    def window(self) -> np.ndarray | None:
        """Current proprio window (n_obs, D), or None before the first step."""
        return None if self._window is None else np.stack(list(self._window))

    # ── http (stdlib) ─────────────────────────────────────────────────────────
    def _get(self, path: str) -> dict:
        try:
            with urllib.request.urlopen(self.server_url + path, timeout=self.timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:  # pragma: no cover - defensive
            raise DP3ServerError(f"GET {path} -> HTTP {e.code}: {e.read().decode()[:200]}") from e
        except (urllib.error.URLError, OSError) as e:
            raise DP3ServerError(f"GET {path} failed: {e}") from e

    def _post(self, path: str, body: dict) -> dict:
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            self.server_url + path, data=data,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            raise DP3ServerError(f"POST {path} -> HTTP {e.code}: {e.read().decode()[:200]}") from e
        except (urllib.error.URLError, OSError) as e:
            raise DP3ServerError(f"POST {path} failed: {e}") from e
