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

**Only one `segment_daemon` process.** After SAM2 web saves the mask, logs show `T2 ok (mask already saved via SAM2 web UI …)` — that is expected (orchestrator does not run SAM2 twice). While T3–T7 run, `input/.upload_complete` becomes `.upload_processing` so the daemon will not start the same session again.

## Resume vs 从头开始

| 情况 | daemon 重启后 |
|------|----------------|
| `input/.upload_complete` 还在，且**没有** `input/.upload_processed` | **会自动继续**（失败中断也会重试） |
| T2 已有 `mask.png` | 跳过 SAM2 网页，从 T3 起跑（已有产物会 skip，除非 `--redo`） |
| 上次 pipeline **成功**（已 `.upload_processed`） | **不会**再处理，除非重新 `mark_upload_complete` |

**整 session 重来（含重新点 SAM2）：**

```bash
python -m demo.pipeline.reset_session \
  --session-dir demo/sessions/<session_id> \
  --requeue
# daemon 在跑会自动接单；或：
python -m demo.pipeline.segment_daemon --session-dir demo/sessions/<id> --redo
```

**只清 output、保留 mask（重做 T3–T7）：**

```bash
python -m demo.pipeline.reset_session --session-dir demo/sessions/<id> --requeue --keep-mask
```

**从队列移除（不再自动 resume）：**

```bash
python -m demo.pipeline.reset_session --session-dir demo/sessions/<id> --mark-processed
```

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
