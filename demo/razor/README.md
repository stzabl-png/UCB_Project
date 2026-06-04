# Razor helpers (run on [V2AP-demo](https://github.com/jiaka1chen/V2AP-demo) laptop)

Titan perception code lives in **Affordance2Grasp**; grasp execution in **V2AP-demo** `demo/phase2/`.

## Per-session flow (with Titan daemon)

1. **Capture** — `capture_session.py` → `demo/phase2/sessions/<id>/input/`
2. **Upload** — rsync `input/` to Titan `demo/sessions/<id>/input/`
3. **Mark ready** — triggers Titan daemon (do **not** ssh-run full pipeline):

```bash
# on Titan after rsync, or:
ssh titan-demo-pipeline "cd ${UCB_ROOT} && python demo/razor/mark_upload_complete.py \
  --session-dir demo/sessions/${SESSION_ID}"
```

4. **SAM2** — operator runs `ssh -L 7860:127.0.0.1:7860 titan` → http://127.0.0.1:7860 → Save → **Done**
5. **Poll** — `output/status.json` until `success: true`
6. **Download** — rsync `output/` back
7. **Review** — `review_titan_vis.py` (T3–T6, blocking)
8. **Grasp** — `run_auto_grasp.py` (Open3D preview + Enter, blocking)

## Scripts in Affordance2Grasp

| Script | Role |
|--------|------|
| [mark_upload_complete.py](mark_upload_complete.py) | Write `input/.upload_complete` |
| [review_titan_vis.py](review_titan_vis.py) | Blocking T3–T6 PNG review on Razor |

## Grasp preview (V2AP-demo)

Default `run_auto_grasp.py`:

- Open3D selected-pose preview — **blocks until window closed**
- Enter confirm — **blocks before motion**

Skip only with `--no-visualize` / `--debug` (not for normal lab runs).
