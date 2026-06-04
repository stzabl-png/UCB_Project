"""Offline tests for deploy/dp3_client.py.

Spins up a MOCK DP3 server (stdlib http.server — no torch/checkpoint) that speaks
the exact dp3_inference_server.py protocol and records every request, then drives
DP3Client against it to verify: /info parsing, fixed-PC + rolling proprio window
semantics (matches eval), request shapes, action shape, and error handling.

Run:
    python deploy/test_dp3_client.py     # self-contained
    pytest deploy/test_dp3_client.py

Optional real-server connectivity smoke test: set DP3_SERVER_URL to a live
dp3_inference_server.py and it will additionally run one info()+step() round trip.
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from dp3_client import DP3Client, DP3ServerError

N_OBS, N_ACTION, HORIZON, ACTION_DIM, N_PTS = 2, 8, 16, 8, 4096


# ── mock server (same protocol as dp3_inference_server.build_app) ──────────────
class _MockHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/info":
            self._send(200, {
                "horizon": HORIZON, "n_obs_steps": N_OBS, "n_action_steps": N_ACTION,
                "action_dim": ACTION_DIM, "point_cloud_shape": [N_PTS, 3],
                "agent_pos_dim": ACTION_DIM,
            })
        else:
            self._send(404, {"detail": "not found"})

    def do_POST(self):
        if self.path != "/predict":
            self._send(404, {"detail": "not found"})
            return
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n).decode())
        pc = np.asarray(req["point_cloud"], dtype=np.float32)
        ap = np.asarray(req["agent_pos"], dtype=np.float32)
        # validate like the real server
        if pc.ndim != 3 or pc.shape[2] != 3 or ap.ndim != 2 or pc.shape[0] != ap.shape[0]:
            self._send(400, {"detail": f"bad shapes pc={pc.shape} ap={ap.shape}"})
            return
        self.server.requests.append({"pc": pc, "ap": ap})  # record for assertions
        # deterministic action: tile the latest proprio so tests can check passthrough
        action = np.tile(ap[-1], (N_ACTION, 1))
        action_pred = np.tile(ap[-1], (HORIZON, 1))
        self._send(200, {"action": action.tolist(),
                         "action_pred": action_pred.tolist(), "inference_ms": 0.0})


class _MockServer:
    def __enter__(self):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _MockHandler)
        self.httpd.requests = []
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        return self

    def __exit__(self, *a):
        self.httpd.shutdown()
        self.httpd.server_close()


def _proprio(seed):
    rng = np.random.default_rng(seed)
    p = rng.normal(size=ACTION_DIM).astype(np.float32)
    p[3:7] /= np.linalg.norm(p[3:7])  # unit quat slot (cosmetic)
    p[7] = 0.0
    return p


# ── tests ──────────────────────────────────────────────────────────────────
def test_info():
    with _MockServer() as s:
        c = DP3Client(s.url)
        info = c.info()
        assert info["n_obs_steps"] == N_OBS
        assert info["point_cloud_shape"] == [N_PTS, 3]
        assert c.n_obs == N_OBS


def test_window_rolling_matches_eval():
    """reset(pc) + step(p0..p3): the POSTed agent_pos window must be
    [p0,p0],[p0,p1],[p1,p2],[p2,p3] — exactly the eval's sliding window."""
    with _MockServer() as s:
        c = DP3Client(s.url)
        pc_G = np.random.default_rng(0).normal(size=(N_PTS, 3)).astype(np.float32)
        c.reset(pc_G)
        ps = [_proprio(i) for i in range(4)]
        for p in ps:
            act = c.step(p)
            assert act.shape == (N_ACTION, ACTION_DIM)
        sent = s.httpd.requests
        assert len(sent) == 4
        expected = [(ps[0], ps[0]), (ps[0], ps[1]), (ps[1], ps[2]), (ps[2], ps[3])]
        for i, (a, b) in enumerate(expected):
            ap = sent[i]["ap"]
            assert ap.shape == (N_OBS, ACTION_DIM), f"chunk {i} ap shape {ap.shape}"
            np.testing.assert_allclose(ap[0], a, atol=1e-6, err_msg=f"chunk {i} slot0")
            np.testing.assert_allclose(ap[1], b, atol=1e-6, err_msg=f"chunk {i} slot1")
            # point cloud must be the SAME fixed pc_G in every slot, every chunk
            assert sent[i]["pc"].shape == (N_OBS, N_PTS, 3)
            for t in range(N_OBS):
                np.testing.assert_allclose(sent[i]["pc"][t], pc_G, atol=1e-6)


def test_action_passthrough():
    """Mock returns tiled latest proprio; verify the client surfaces it."""
    with _MockServer() as s:
        c = DP3Client(s.url)
        c.reset(np.zeros((N_PTS, 3), np.float32))
        p = _proprio(42)
        act = c.step(p)
        for k in range(N_ACTION):
            np.testing.assert_allclose(act[k], p, atol=1e-6)


def test_reset_with_proprio0():
    """reset(pc, p0) pre-pads, so the FIRST step(p1) window is [p0, p1]."""
    with _MockServer() as s:
        c = DP3Client(s.url)
        p0, p1 = _proprio(7), _proprio(8)
        c.reset(np.zeros((N_PTS, 3), np.float32), proprio0=p0)
        c.step(p1)
        ap = s.httpd.requests[-1]["ap"]
        np.testing.assert_allclose(ap[0], p0, atol=1e-6)
        np.testing.assert_allclose(ap[1], p1, atol=1e-6)


def test_predict_raw_shape_validation():
    with _MockServer() as s:
        c = DP3Client(s.url)
        for bad_pc, bad_ap in [
            (np.zeros((2, 5, 2)), np.zeros((2, 8))),   # pc last dim != 3
            (np.zeros((2, 5, 3)), np.zeros((2, 8, 1))),  # ap not 2D
            (np.zeros((3, 5, 3)), np.zeros((2, 8))),   # T mismatch
        ]:
            try:
                c.predict_raw(bad_pc, bad_ap)
                assert False, "expected ValueError"
            except ValueError:
                pass


def test_step_before_reset():
    with _MockServer() as s:
        c = DP3Client(s.url)
        try:
            c.step(_proprio(0))
            assert False, "expected RuntimeError"
        except RuntimeError:
            pass


def test_bad_pc_G_shape():
    with _MockServer() as s:
        c = DP3Client(s.url)
        try:
            c.reset(np.zeros((N_PTS, 2), np.float32))
            assert False, "expected ValueError"
        except ValueError:
            pass


def test_connection_error():
    c = DP3Client("http://127.0.0.1:1", timeout=1.0)  # nothing listening
    try:
        c.info()
        assert False, "expected DP3ServerError"
    except DP3ServerError:
        pass


def test_real_server_optional():
    url = os.environ.get("DP3_SERVER_URL")
    if not url:
        print("  [skip] DP3_SERVER_URL not set — real-server smoke test skipped")
        return
    c = DP3Client(url)
    info = c.info()
    n_pts = int(info["point_cloud_shape"][0])
    d = int(info["agent_pos_dim"])
    c.reset(np.zeros((n_pts, 3), np.float32))
    act = c.step(np.zeros(d, np.float32))
    assert act.shape == (int(info["n_action_steps"]), int(info["action_dim"]))
    print(f"  [ok] real server {url}: action {act.shape}")


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\nAll {len(tests)} test groups passed.")


if __name__ == "__main__":
    _main()
