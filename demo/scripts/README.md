# Phase 2 Titan scripts

Per-step debug scripts live under `T1/`, `T2/`, … Model weights stay in repo defaults (`third_party/`, external SAM3D path, etc.).

| Step | Folder | Script |
|------|--------|--------|
| Shared | [_session_io.py](_session_io.py) | `resolve_session_dirs()`, `repo_root()` |
| T1 | [T1/](T1/) | `validate_input.py` |
| T2 | [T2/](T2/) | `segment_web.py` (Gradio), `segment.py` (OpenCV), `segment_common.py` |
| T3 | [T3/](T3/) | `reconstruct.py` — SAM3D → `object_raw.glb` |
| T4 | [T4/](T4/) | `scale_from_depth.py` → `object_scaled.glb` |
| T5 | [T5/](T5/) | `register_foundationpose.py` → `T_cam_mesh`, overlay |
| T6 | [T6/](T6/) | `run_pdm_grasp.py` — PDM → `candidates.json` |
| T7 | [T7/](T7/) | `write_status.py` — finalize `output/status.json` for Razor |
| T2 batch | [T2/segment_prompt.py](T2/segment_prompt.py) | SAM2 from `input/segment/prompt.json` |

**Orchestrator:** `python -m demo.pipeline` — [demo/pipeline/README.md](../pipeline/README.md)

**Titan → Razor package doc:** [TITAN_OUTPUT.md](../TITAN_OUTPUT.md) · **Automation:** [SERVER_CLIENT_PLAN.md](../SERVER_CLIENT_PLAN.md)

Spec: [demo/README.md](../README.md).
