"""Offline tests for deploy/dp3_loop.py — the closed-loop orchestrator.

Drives run_closed_loop with a scripted fake client (controls the DP3 action
sequence) + a fake "world" (records executed joints, simulates an object that
lifts after gripper close). Verifies: gripper-close trigger timing, dz early-stop
success, never-closed failure, closed-but-no-lift, IK-failure skip, proprio
gripper-bit, and last_q seeding continuity. Plus one end-to-end pass with the
REAL DP3Client against the stdlib mock server.

Run:  python deploy/test_dp3_loop.py   |   pytest deploy/test_dp3_loop.py
"""

from __future__ import annotations

import numpy as np

import dp3_frames as F
from dp3_loop import DP3Callbacks, run_closed_loop

J = 7
N_ACTION, ACTION_DIM, N_PTS = 8, 8, 16  # small N_PTS for speed
PC_G = np.zeros((N_PTS, 3), np.float32)
T_BASE_MESH = np.eye(4)
T_BASE_MESH[:3, 3] = [0.0, 0.55, 0.9]
T_EE_PINCH = np.eye(4)
T_EE_PINCH[:3, 3] = [0.08, 0.03, 0.14]


# ── scripted fake DP3 client (returns preset action chunks) ───────────────────
class FakeClient:
    def __init__(self, action_seq):
        self.action_seq = action_seq          # list of (n_action, 8) arrays
        self.i = 0
        self.reset_called = False
        self.proprios = []

    def reset(self, pc_G, proprio0=None):
        self.reset_called = True
        self.i = 0

    def step(self, proprio):
        self.proprios.append(np.asarray(proprio, dtype=np.float64).copy())
        a = self.action_seq[min(self.i, len(self.action_seq) - 1)]
        self.i += 1
        return np.asarray(a, dtype=np.float64)


# ── fake world (callbacks) ────────────────────────────────────────────────────
class FakeWorld:
    def __init__(self, *, lift_after_close=0.05, ik_fail_every=0):
        self.q = np.zeros(J)
        self.closed = False
        self.executed = []          # (q, closed)
        self.seeds = []             # seed_q seen by solve_ik
        self.lift = lift_after_close
        self._z = 1.0
        self._ik_calls = 0
        self.ik_fail_every = ik_fail_every

    def make_callbacks(self):
        return DP3Callbacks(
            measure_proprio=self.measure_proprio,
            solve_ik=self.solve_ik,
            execute=self.execute,
            close_gripper=self.close_gripper,
            get_object_z=self.get_object_z,
            get_current_q=self.get_current_q,
        )

    def get_current_q(self):
        return self.q.copy()

    def measure_proprio(self, gripper):
        p = np.zeros(ACTION_DIM)
        p[3] = 1.0          # unit quat slot (w=1)
        p[7] = gripper
        return p

    def solve_ik(self, T_base_Ree, seed_q):
        self.seeds.append(None if seed_q is None else np.asarray(seed_q).copy())
        self._ik_calls += 1
        if self.ik_fail_every and (self._ik_calls % self.ik_fail_every == 0):
            return None
        # dummy reachable: encode the target translation into joints
        return np.full(J, float(T_base_Ree[0, 3]))

    def execute(self, q, gripper_closed):
        self.q = np.asarray(q, dtype=np.float64)
        self.executed.append((self.q.copy(), gripper_closed))

    def close_gripper(self):
        self.closed = True
        self._z = 1.0 + self.lift     # object rises after close

    def get_object_z(self):
        return self._z


def _action(grip_vec):
    """(n_action, 8) with a given gripper column; pose part = identity-ish."""
    a = np.zeros((N_ACTION, ACTION_DIM))
    a[:, 3] = 1.0          # unit quat (w=1) -> R = I
    a[:, 7] = grip_vec
    return a


# ── tests ─────────────────────────────────────────────────────────────────
def test_success_lift_early_stop():
    # chunk0: no close; chunk1: gripper turns on at waypoint 3 -> close -> lift -> success
    g1 = np.zeros(N_ACTION)
    g1[3:] = 1.0
    client = FakeClient([_action(np.zeros(N_ACTION)), _action(g1)])
    w = FakeWorld(lift_after_close=0.05)
    res = run_closed_loop(client, w.make_callbacks(), PC_G, T_BASE_MESH, T_EE_PINCH, max_chunks=8)
    assert client.reset_called
    assert res.success and res.stage == "lifted"
    assert res.n_chunks == 2
    assert res.gripper_closed and w.closed
    assert res.dz > 0.03


def test_never_closed_failure():
    client = FakeClient([_action(np.zeros(N_ACTION))])  # gripper always 0
    w = FakeWorld()
    res = run_closed_loop(client, w.make_callbacks(), PC_G, T_BASE_MESH, T_EE_PINCH, max_chunks=4)
    assert not res.success and res.stage == "never_closed"
    assert res.n_chunks == 4 and not res.gripper_closed
    assert not w.closed


def test_closed_but_no_lift():
    g = np.zeros(N_ACTION)
    g[0] = 1.0
    client = FakeClient([_action(g)])
    w = FakeWorld(lift_after_close=0.0)    # closes but object never rises
    res = run_closed_loop(client, w.make_callbacks(), PC_G, T_BASE_MESH, T_EE_PINCH, max_chunks=3)
    assert w.closed and res.gripper_closed
    assert not res.success and res.stage == "max_chunks"
    assert res.dz == 0.0


def test_ik_failure_skips_but_continues():
    g = np.zeros(N_ACTION)
    g[-1] = 1.0
    client = FakeClient([_action(np.zeros(N_ACTION)), _action(g)])
    w = FakeWorld(ik_fail_every=3, lift_after_close=0.05)   # every 3rd IK call fails
    res = run_closed_loop(client, w.make_callbacks(), PC_G, T_BASE_MESH, T_EE_PINCH, max_chunks=8)
    # some IK calls failed and were skipped, but loop still progressed to success
    assert res.n_ik_fail > 0
    assert res.n_executed == len(w.executed)
    assert res.n_executed < 2 * N_ACTION  # fewer than all waypoints executed
    assert res.success


def test_proprio_gripper_bit_tracks_state():
    g = np.zeros(N_ACTION)
    g[0] = 1.0
    # chunk0 closes immediately; chunk1 proprio must carry gripper=1
    client = FakeClient([_action(g), _action(np.ones(N_ACTION))])
    w = FakeWorld(lift_after_close=0.0)   # don't early-stop, so chunk1 runs
    run_closed_loop(client, w.make_callbacks(), PC_G, T_BASE_MESH, T_EE_PINCH, max_chunks=2)
    assert client.proprios[0][7] == 0.0   # chunk0: open
    assert client.proprios[1][7] == 1.0   # chunk1: closed


def test_ik_seed_continuity():
    # seed for the very first IK = current q (zeros); later seeds = previous solved q
    client = FakeClient([_action(np.zeros(N_ACTION))])
    w = FakeWorld()
    run_closed_loop(client, w.make_callbacks(), PC_G, T_BASE_MESH, T_EE_PINCH, max_chunks=1)
    assert w.seeds[0] is not None
    np.testing.assert_allclose(w.seeds[0], np.zeros(J))         # first seed = current q
    # second IK seeded from the first solved q (all equal here since same target)
    np.testing.assert_allclose(w.seeds[1], w.executed[0][0])


def test_action_conversion_uses_frames():
    """Sanity: the loop's IK target equals dp3_frames.action_to_base_ee output."""
    a = _action(np.zeros(N_ACTION))[0]
    T_ref, grip = F.action_to_base_ee(a, T_BASE_MESH, T_EE_PINCH)
    captured = {}

    def solve_ik(T, seed):
        captured["T"] = T.copy()
        return np.zeros(J)

    w = FakeWorld()
    cb = w.make_callbacks()
    cb.solve_ik = solve_ik
    client = FakeClient([_action(np.zeros(N_ACTION))])
    run_closed_loop(client, cb, PC_G, T_BASE_MESH, T_EE_PINCH, max_chunks=1)
    np.testing.assert_allclose(captured["T"], T_ref, atol=1e-9)


def test_end_to_end_with_real_client_and_mock_server():
    """Loop + real DP3Client + stdlib mock server (action = tiled proprio, gripper=0).
    Mock never closes the gripper, so the loop must terminate cleanly as never_closed."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from dp3_client import DP3Client

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            self._send(200, {"horizon": 16, "n_obs_steps": 2, "n_action_steps": N_ACTION,
                             "action_dim": ACTION_DIM, "point_cloud_shape": [N_PTS, 3],
                             "agent_pos_dim": ACTION_DIM})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n).decode())
            ap = np.asarray(req["agent_pos"], dtype=np.float32)
            action = np.tile(ap[-1], (N_ACTION, 1)).tolist()
            self._send(200, {"action": action, "action_pred": action, "inference_ms": 0.0})

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}"
        client = DP3Client(url)
        w = FakeWorld()
        res = run_closed_loop(client, w.make_callbacks(), PC_G, T_BASE_MESH, T_EE_PINCH, max_chunks=3)
        assert res.stage == "never_closed" and not res.success
        assert res.n_chunks == 3
        assert len(w.executed) == 3 * N_ACTION  # all waypoints reachable, none closed
    finally:
        httpd.shutdown()
        httpd.server_close()


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\nAll {len(tests)} test groups passed.")


if __name__ == "__main__":
    _main()
