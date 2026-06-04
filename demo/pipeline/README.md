# Phase 2 pipeline orchestrator

Razor uploads `input/` via rsync; Titan runs T1–T7 and writes `output/` + `status.json`.

## Layout

```text
demo/
├── pipeline/                 # this package (python -m demo.pipeline)
│   ├── process_razor_session.py
│   └── run_pipeline.py
├── scripts/T1 … T7           # per-step implementations
├── sessions/                 # default rsync target (gitignored)
│   └── <session_id>/
│       ├── input/
│       └── output/
├── TITAN_OUTPUT.md           # Razor consumer guide
└── SERVER_CLIENT_PLAN.md     # SSH/rsync automation
```

## Run

```bash
cd /home/vision/Project/Affordance2Grasp
export FP_ROOT="$PWD/third_party/FoundationPose"

python -m demo.pipeline.process_razor_session \
  --session-dir demo/sessions/20260602_192346_chips
```

Alternate entry (equivalent):

```bash
python -m demo.pipeline --session-dir demo/sessions/<session_id>
```

**Conda:** T3 uses `sam3d-objects`; other steps use `bundlesdf`.

## T2 (segmentation)

1. `output/segment/mask.png` present → skip (or `--skip-sam`)
2. `input/segment/prompt.json` → `demo/scripts/T2/segment_prompt.py`
3. Else **fail** (no GUI in batch mode)

## Exit code

- **0** — `output/status.json` has `"success": true`
- **1** — failed step or `success: false`
- **2** — bad CLI / missing session dir
