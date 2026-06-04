# Auto demo pipeline orchestrator

Razor uploads `input/` via rsync; Titan runs T1–T7 and writes `output/` + `status.json`.

## Recommended: Titan segment daemon (interactive T2)

**Do not** `ssh` and run `process_razor_session` for lab capture — T2 SAM2 web UI will not appear on your laptop.

Instead:

| Side | Action |
|------|--------|
| **Titan** (once per boot) | `python -m demo.pipeline.segment_daemon` |
| **Razor** | rsync `input/` → mark `input/.upload_complete` → poll `output/status.json` |
| **Operator** | SSH tunnel → browser SAM2 → Save → **Done** → daemon runs T3–T7 |
| **Razor** | rsync `output/` → `review_titan_vis.py` → `run_auto_grasp.py` |

See [SERVER_CLIENT_PLAN.md § Titan segment daemon](../SERVER_CLIENT_PLAN.md#56-titan-segment-daemon-recommended) and [demo/README.md § Daemon workflow](../README.md#titan-segment-daemon-recommended).

## Layout

```text
demo/
├── pipeline/
│   ├── segment_daemon.py       # python -m demo.pipeline.segment_daemon
│   ├── process_razor_session.py
│   └── run_pipeline.py
├── razor/
│   ├── mark_upload_complete.py
│   └── review_titan_vis.py
├── scripts/T1 … T7/
└── sessions/                   # gitignored rsync root
```

## One-shot pipeline (no interactive T2)

Use only when `input/segment/prompt.json` or `output/segment/mask.png` already exists:

```bash
cd /home/vision/Project/Affordance2Grasp
export FP_ROOT="$PWD/third_party/FoundationPose"

python -m demo.pipeline.process_razor_session \
  --session-dir demo/sessions/20260602_192346_chips
```

## Titan daemon

```bash
conda activate bundlesdf
export FP_ROOT="$PWD/third_party/FoundationPose"
cd /home/vision/Project/Affordance2Grasp

python -m demo.pipeline.segment_daemon
# optional: --port 7860 --poll-interval 5
```

**Razor after rsync:**

```bash
python demo/razor/mark_upload_complete.py \
  --session-dir demo/sessions/<session_id>
```

**Browser (laptop):**

```bash
ssh -L 7860:127.0.0.1:7860 vision@<titan-host>
# open http://127.0.0.1:7860 — Save mask, click Done
```

Poll: `output/status.json` (`state`: `waiting_segment` → `running` → `done`) and `output/daemon_state.json`.

## Conda

| Steps | Env |
|-------|-----|
| T1, T2 web, T4–T7 | `bundlesdf` |
| T3 | `sam3d-objects` (orchestrator switches automatically) |

## Exit codes

| Command | 0 | 1 |
|---------|---|---|
| `process_razor_session` | `status.json` success | failed |
| `segment_daemon` | — | session failed |
| `segment_daemon --once` | no error | — |
