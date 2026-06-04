"""Smoke test for deploy/build_dp3_session.py — mechanics only (synthetic box mesh).

No real Titan session available offline, so we exercise the sampling + IO against a
trimesh box: verify pc_G shape/dtype, on-surface points, extent ~ box size, and a
well-formed meta.json. Run: python deploy/test_build_dp3_session.py
"""

from __future__ import annotations

import json
import pathlib
import tempfile

import numpy as np
import trimesh

import build_dp3_session as B


def test_build_session_box():
    box_size = np.array([0.12, 0.08, 0.05])  # 12x8x5 cm
    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        mesh_path = d / "object_base_aligned.glb"
        trimesh.creation.box(extents=box_size).export(str(mesh_path))

        T = np.eye(4)
        T[:3, 3] = [0.0, 0.55, 0.9]
        (d / "T_base_mesh.json").write_text(json.dumps({"T_base_mesh": T.tolist()}))

        out, pc_G, meta = B.build_session(mesh_path, d / "T_base_mesh.json", d / "out",
                                          num_points=4096, seed=42, n_obs=2)

        # files written
        assert (out / "pc_G.npy").exists() and (out / "meta.json").exists()
        loaded = np.load(out / "pc_G.npy")
        assert loaded.shape == (4096, 3) and loaded.dtype == np.float32

        # points lie on the box surface (within the half-extents + tiny eps)
        half = box_size / 2 + 1e-6
        assert np.all(np.abs(pc_G) <= half), "sampled points outside box"
        # extent ~ box size (area sampling covers all faces)
        extent = pc_G.max(0) - pc_G.min(0)
        np.testing.assert_allclose(extent, box_size, atol=2e-3)

        # meta well-formed
        m = json.loads((out / "meta.json").read_text())
        assert np.allclose(np.asarray(m["T_base_mesh"]), T)
        assert m["n_obs"] == 2 and m["num_points"] == 4096 and m["seed"] == 42
        assert "home_right_arm" not in m  # razer-side, intentionally absent


def test_raw_4x4_t_base_mesh():
    """read_T_base_mesh accepts a bare 4x4 (not wrapped in {'T_base_mesh':...})."""
    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        T = np.eye(4); T[0, 3] = 0.3
        p = d / "raw.json"; p.write_text(json.dumps(T.tolist()))
        np.testing.assert_allclose(B.read_T_base_mesh(p), T)


def test_seed_determinism():
    with tempfile.TemporaryDirectory() as d:
        mesh = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
        a = B.sample_pc_G(mesh, 1024, seed=7)
        b = B.sample_pc_G(mesh, 1024, seed=7)
        np.testing.assert_array_equal(a, b)


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\nAll {len(tests)} test groups passed.")


if __name__ == "__main__":
    _main()
