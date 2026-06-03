# T7 — Finalize session for Razor

Writes **`output/status.json`** after T2–T6 complete. Does not run GPU steps.

## Usage

```bash
cd /home/vision/Project/Affordance2Grasp

python demo/scripts/T7/write_status.py \
  --session-dir demo/sessions/20260602_192346_chips

python demo/scripts/T7/write_status.py --session-dir ... --json
```

Exit code **0** only when `success: true` in `status.json`.

## What it checks

| Step key | Artifacts |
|----------|-----------|
| `segment` | `output/segment/mask.png`, `prompt_used.json` |
| `sam3d` | `object_raw.glb`, `sam3d_meta.json` |
| `scale` | `object_scaled.glb`, `scale.json` |
| `foundationpose` | `object_base_aligned.glb`, `T_cam_mesh.json`, `T_base_mesh.json`, … |
| `grasp_pose` | `affordance_grasp.hdf5`, `candidates.json` |

Also runs **T1** input validation (unless `--skip-input-check`), aggregates warnings from `scale.json` / `foundationpose_meta.json`, and verifies `candidates.json` ↔ `T_base_mesh.json` consistency.

## Razor documentation

See **[demo/TITAN_OUTPUT.md](../../TITAN_OUTPUT.md)** — Titan `output/` layout, frames, Razor retarget.

Spec: [demo/README.md](../../README.md) (Step T7).
