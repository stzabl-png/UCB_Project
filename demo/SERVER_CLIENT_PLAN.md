# Titan ↔ Razor auto demo pipeline

**Audience:** Titan team (UCB_Project / GPU server) — implement processing + accept Razor uploads.  
**Razor team:** orchestrator client `run_server_client_pipeline.py`; capture + grasp already in V2AP-demo.

**Related docs**

| Doc | Location | Purpose |
|-----|----------|---------|
| Full session schema (input/output) | [README.md](README.md) | T1–T7 pipeline, JSON schemas, coordinates |
| What Razor needs from Titan | [TITAN_OUTPUT.md](TITAN_OUTPUT.md) | `status.json`, `candidates.json`, rsync minimum set |
| Razor capture / grasp | Razor `capture_session.py`, `run_auto_grasp.py` | Already implemented |

---

## 1. Goal

Automate the loop that is **manual rsync today**:

```text
[Human] place object
    → Razor capture (input/)
    → upload to Titan
    → Titan auto demo pipeline T1–T7 (output/)
    → download to Razor
    → Razor run_auto_grasp.py
```

**v1 transport:** SSH + rsync (no custom TCP server required).  
**v2 (optional):** HTTP job queue on Titan for multi-session / status API.

---

## 2. Roles and connectivity

| Machine | Role | Network |
|---------|------|---------|
| **Razor** | Robot laptop; **auto demo pipeline client**; initiates all connections | Often behind lab NAT — **does not need inbound SSH** |
| **Titan** | GPU server; **SSH server**; stores sessions; runs `python -m demo.pipeline` | Stable hostname/IP or VPN |

**Connection direction:** Razor → Titan only.

```text
Razor  ──ssh/rsync──►  Titan
       ◄──rsync──────   (output/)
```

---

## 3. SSH authentication (one-time setup)

### 3.1 On Razor (client)

Create a dedicated key (do not overwrite personal default):

```bash
ssh-keygen -t ed25519 -f ~/.ssh/demo_razor_to_titan -C "razor-auto-demo-pipeline" -N ""
```

`~/.ssh/config` on Razor:

```text
Host titan-demo-pipeline
    HostName <TITAN_HOSTNAME_OR_IP>
    User <TITAN_USER>          # e.g. vision
    IdentityFile ~/.ssh/demo_razor_to_titan
    IdentitiesOnly yes
```

Test:

```bash
ssh titan-demo-pipeline 'echo OK && hostname && nvidia-smi -L | head -1'
```

### 3.2 On Titan (server)

Append Razor **public** key to `~/.ssh/authorized_keys` for `<TITAN_USER>`:

```bash
# paste contents of demo_razor_to_titan.pub
```

Optional hardening (later): `from="<RAZOR_IP>"`, dedicated Unix user, `ForceCommand` wrapper.

### 3.3 Environment variables (both sides)

| Variable | Example | Used on |
|----------|---------|---------|
| `TITAN_SSH_HOST` | `titan-demo-pipeline` | Razor (~/.ssh/config alias) |
| `UCB_ROOT` | `/home/vision/Project/Affordance2Grasp` | Titan (repo root) |
| `DEMO_SESSIONS_ROOT` | `$UCB_ROOT/demo/sessions` | Titan |
| `RAZOR_REPO` | `/home/.../V2AP-demo` | Razor |
| `DEMO_PIPELINE_CONDA_ENV` | `bundlesdf` | Titan (FoundationPose) |

---

## 4. Directory layout (must mirror)

**Session ID:** `YYYYMMDD_HHMMSS_<object_slug>` (e.g. `20260602_192346_chips`).

| Side | Path |
|------|------|
| **Razor** | `$RAZOR_REPO/demo/phase2/sessions/<session_id>/` |
| **Titan** | `$UCB_ROOT/demo/sessions/<session_id>/` |

Both:

```text
<session_id>/
├── input/          # Razor writes → Titan reads
└── output/         # Titan writes → Razor reads
```

Create once on Titan:

```bash
mkdir -p "$UCB_ROOT/demo/sessions"
```

**Input contract:** [README.md § INPUT package](README.md#input-package-input--razor--titan) (`schema_version: "1.1"`).  
**Output contract:** [README.md § OUTPUT package](README.md#output-package-output--titan--razor) + [TITAN_OUTPUT.md](TITAN_OUTPUT.md).

---

## 5. Titan deliverables (what to implement)

### 5.1 Required CLI entry point

```bash
cd "$UCB_ROOT"
conda activate "$DEMO_PIPELINE_CONDA_ENV"

python -m demo.pipeline.process_razor_session \
  --session-dir demo/sessions/<session_id> \
  [--skip-sam] \
  [--skip-sam3d] \
  [--skip-fp] \
  [--device cuda]
```

**Pipeline order (fixed):**

| Step | ID | Output (minimum) |
|------|-----|------------------|
| Validate input | T1 | fail → `output/status.json` `success: false` |
| SAM mask | T2 | `output/segment/mask.png`; **unattended:** `input/segment/prompt.json` **or** pre-uploaded mask |
| SAM3D mesh | T3 | `output/mesh/object_raw.glb` (`sam3d-objects` env) |
| Metric scale | T4 | `output/mesh/object_scaled.glb`, `scale.json` (`bundlesdf`) |
| FoundationPose + align | T5 | `T_cam_mesh_fp.json`, `mesh_frame_align.json`, `T_cam_mesh.json`, `T_base_mesh.json`, `object_base_aligned.glb` |
| Grasp (PDM) | T6 | `output/inference/candidates.json` (`mesh_frame: base_aligned`) |
| Status | T7 | `output/status.json` (**write last**, atomic rename) |

Full step specs: [README.md § TITAN processing pipeline](README.md#titan-processing-pipeline).

### 5.2 `demo/pipeline/` module layout (implemented)

```text
Affordance2Grasp/demo/
├── pipeline/
│   ├── process_razor_session.py    # python -m demo.pipeline
│   ├── run_pipeline.py             # subprocess → demo/scripts/T1–T7
│   ├── env.py                      # bundlesdf / sam3d-objects python paths
│   └── status.py                   # running / failed status.json
├── scripts/T1 … T7/
└── sessions/                       # gitignored rsync root
```

Step scripts remain under `demo/scripts/T1` … `T7` (not duplicated in `pipeline/`).

### 5.3 `output/status.json` (critical for Razor automation)

Razor **must** read this before grasp. Write **last** (`status.json.tmp` → rename).

**Success example:**

```json
{
  "schema_version": "1.1",
  "session_id": "20260602_192346_chips",
  "success": true,
  "pipeline_version": "demo.pipeline.process_razor_session 0.1.0",
  "finished_at_iso": "2026-06-03T12:00:00+00:00",
  "steps": {
    "segment": "ok",
    "sam3d": "ok",
    "scale": "ok",
    "foundationpose": "ok",
    "grasp_pose": "ok"
  },
  "warnings": [],
  "errors": [],
  "package": {
    "required_for_grasp": [
      "output/status.json",
      "output/inference/candidates.json",
      "output/register/T_base_mesh.json",
      "output/mesh/object_base_aligned.glb"
    ]
  },
  "titan": {
    "n_candidates": 50,
    "hostname": "titan.local"
  }
}
```

**Failure:** `success: false`, non-empty `errors[]`. Partial `output/` is OK; Razor must **not** run grasp.

**`pipeline_version`:** After `python -m demo.pipeline.process_razor_session`, expect **`demo.pipeline.process_razor_session 0.1.0`**. Running T7 alone (`write_status.py`) sets `demo.scripts.T7.write_status 0.1.0` — Razor automation should require the orchestrator string.

**`steps`:** Only T2–T6 artifact keys above (T1 validation is not listed in `status.json`).

**Optional progress field (v1.1):** for Razor polling while job runs:

```json
"state": "queued | running | done | failed",
"started_at_iso": "...",
"updated_at_iso": "..."
```

Update `state`/`updated_at_iso` at each step start/end if implementing long jobs.

### 5.4 Razor operator UX (blocking review)

After rsync download and `status.json` success, Razor should **pause the pipeline** for human review:

1. **Titan vis sequence (T3→T6)** — open each PNG in order; **block until the window is closed** before the next. Files:
   - `output/vis/T3_sam3d_mesh_preview.png`
   - `output/vis/T4_scale_scene_preview.png`
   - `output/vis/T5_foundationpose_overlay.png`
   - `output/vis/T6_grasp_vis.png`  
   Reference implementation: `Affordance2Grasp/demo/razor/review_titan_vis.py` (wire into `run_server_client_pipeline.py` after download).

2. **Grasp execution** — `run_auto_grasp.py` **default** (no `--no-visualize`, no `--debug`):
   - After IK picks a candidate → **Open3D grasp preview** of the selected pose — **block until closed**.
   - **Enter / confirm** — block again before arm motion starts.

Headless/automation only: `--skip` on vis review; `--no-visualize` and/or `--debug` on grasp.

### 5.5 Razor integration (V2AP-demo client)

| Item | Value |
|------|--------|
| Titan session root | `$UCB_ROOT/demo/sessions/<session_id>/` |
| Rsync subdir | `demo/sessions` |
| Razor orchestrator | `run_server_client_pipeline.py` ([V2AP-demo](https://github.com/jiaka1chen/V2AP-demo)) |
| Upload done marker | `input/.upload_complete` (see `demo/razor/mark_upload_complete.py`) |
| Titan worker | **`segment_daemon` always on** — not one-shot SSH pipeline |

**Candidate count (Titan T6):** Before upload, Razor writes `input/session.json` → `pipeline.titan.max_candidates` (`--titan-max-candidates`, default 50). Titan T6 uses this as PDM `--n-samples`. CLI override on Titan: `process_razor_session` / `segment_daemon` / `run_pdm_grasp.py --n-samples`.

**Multi-grasp on Razor (one Titan run):** `run_server_client_pipeline.py --grasp-attempts N` runs `run_auto_grasp.py` N times with `--attempt-index 0 … N-1` (sorted candidate ranks). Set **`--titan-max-candidates` ≥ `--grasp-attempts`** so enough candidates exist after filtering.

**Suggested Razor env:**

```bash
export UCB_ROOT=/home/vision/Project/Affordance2Grasp
export PIPELINE_REMOTE_SESSIONS_SUBDIR=demo/sessions
export PIPELINE_TITAN_UPLOAD_MARKER=input/.upload_complete
export PIPELINE_TITAN_SEGMENT_TUNNEL_PORT=7860
# Do NOT ssh-run process_razor_session for normal capture (T2 web won't show on Razor).
```

**Conda on Titan:** daemon uses `bundlesdf`; full pipeline still runs T3 in `sam3d-objects` automatically.

### 5.6 Titan segment daemon (recommended)

**Problem:** `ssh titan 'python -m demo.pipeline …'` runs T2 without a display on your laptop. SAM2 must be reviewed in a **browser** via tunnel to Titan’s Flask UI (`demo/scripts/T2/segment_web.py`).

**Solution:** Start the daemon once on Titan; Razor only rsyncs + marks upload complete + polls status.

#### Titan (long-running)

```bash
cd "$UCB_ROOT"
conda activate bundlesdf
export FP_ROOT="$UCB_ROOT/third_party/FoundationPose"

python -m demo.pipeline.segment_daemon
# --port 7860  --poll-interval 5
```

#### Razor (per session)

```bash
# 1) upload input/
rsync -avz .../input/ titan-demo-pipeline:${UCB_ROOT}/demo/sessions/${SESSION}/input/

# 2) signal daemon (on Titan, or via ssh one-liner)
ssh titan-demo-pipeline "cd ${UCB_ROOT} && python demo/razor/mark_upload_complete.py \
  --session-dir demo/sessions/${SESSION}"

# 3) operator opens SAM2 (separate terminal)
ssh -L 7860:127.0.0.1:7860 titan-demo-pipeline
# browser → http://127.0.0.1:7860 — Save mask → Done

# 4) poll until output/status.json success=true, then rsync output/
```

#### Daemon behavior

| Step | Action |
|------|--------|
| Queue | `input/.upload_complete` present, no `input/.upload_processed` |
| T2 | If no `prompt.json` and no `mask.png` → start Flask on `127.0.0.1:7860`, **block until Done** |
| T3–T7 | `run_pipeline` with `--skip-sam` if mask already saved |
| Done | `mark_upload_processed`, `status.json` `success: true` |

**Artifacts for Razor polling:**

| File | `state` examples |
|------|------------------|
| `output/status.json` | `waiting_segment`, `running`, `done`, `failed` |
| `output/daemon_state.json` | same + `segment_url` |

**Non-interactive T2 (no browser):** rsync `input/segment/prompt.json` or pre-upload `output/segment/mask.png` — daemon skips web and runs batch/`--skip-sam`.

**Manual replay:** `python -m demo.pipeline.segment_daemon --session-dir demo/sessions/<id> --redo`

#### Do not use for interactive capture

```bash
# BAD for lab SAM2 labeling — UI stays on Titan, not your SSH session:
ssh titan "python -m demo.pipeline.process_razor_session --session-dir ..."
```

Use `process_razor_session` only when T2 is already satisfied (`prompt.json` or `mask.png`).

### 5.7 Rsync commands (reference)

**Razor → Titan (upload input):**

```bash
SESSION=20260602_192346_chips
rsync -avz --progress \
  "${RAZOR_REPO}/demo/phase2/sessions/${SESSION}/input/" \
  "titan-demo-pipeline:${UCB_ROOT}/demo/sessions/${SESSION}/input/"
```

**Titan → Razor (download output):**

```bash
rsync -avz --progress \
  "titan-demo-pipeline:${UCB_ROOT}/demo/sessions/${SESSION}/output/" \
  "${RAZOR_REPO}/demo/phase2/sessions/${SESSION}/output/"
```

---

## 6. End-to-end state machine (orchestrator view)

States stored in Razor `sessions/<id>/orchestrator_state.json` (future) and Titan `output/status.json`.

```text
CREATED          Razor: capture_session.py finished, input/ valid
UPLOADING        rsync input/ → Titan
UPLOAD_MARKED    input/.upload_complete written
WAITING_SEGMENT  Titan daemon: T2 Flask (operator tunnel :7860, Save, Done)
RUNNING          Titan: T3–T7 pipeline
DONE             status.json success=true
DOWNLOADING      rsync output/ ← Titan
REVIEW_VIS       Razor: blocking T3→T6 PNG popups
READY_FOR_GRASP  Razor: run_auto_grasp.py (Open3D + Enter, then motion)
GRASP_DONE       optional
FAILED           any step; preserve logs
```

**Retry policy (Razor):**

| Failed step | Retry |
|-------------|-------|
| Upload | rsync again (no re-capture) |
| Titan process | re-ssh same session (use `--skip-*` if partial outputs valid) |
| Download | rsync output again |
| Grasp | different `--rank` / `--seed` on Razor only |

---

## 7. Titan dependencies

- UCB `inference/grasp_pose.py` + checkpoints  
- **FoundationPose** (`FP_ROOT`, `bundlesdf` conda, CUDA) — `tools/batch_obj_pose_ego.py` / `run_fp()`  
- SAM2 or SAM3 (2D mask, before FP)  
- SAM3D (external install on Titan)  
- Python: `trimesh`, `fast_simplification`, `h5py`, `opencv-python`, `numpy`, `torch`

**Do not commit** `demo/sessions/` session data (gitignore).

---

## 8. What Razor does (for context — already implemented)

Titan team does **not** implement these; listed so interfaces stay aligned.

| Step | Razor script | Notes |
|------|--------------|-------|
| Capture | `capture_session.py` | `input/` pack, robot j3 spread pose |
| Validate input | auto in capture | `schema_version 1.1` |
| Grasp | `run_auto_grasp.py` | Requires `output/status.json` success |
| Retarget | open-grip IK | Uses `candidates.json` → `T_base_pinch`; not Franka `position` |

Razor reads **`grasp_point` + `rotation`** in mesh frame, **not** `position_panda_hand`.

---

## 9. Phased rollout

| Phase | Titan work | Razor work | Acceptance |
|-------|------------|------------|------------|
| **A** | `segment_daemon` + `input/.upload_complete` | rsync + mark + tunnel SAM2 + poll | One session E2E (current) |
| **B** | Stable `status.json` + daemon states | `run_server_client_pipeline.py` automates rsync/poll | No manual Titan shell |
| **C** | queue / multi-GPU policy | orchestrator_state on Razor | Multi-session lab |
| **D** (opt) | HTTP job API + queue | Client uses API + rsync for blobs | Multi-user / queue |

**Titan minimum for phase B:** sections 5.1, 5.3, 5.4, 5.5 working reliably.

---

## 10. Checklist for Titan team

- [x] Create `$UCB_ROOT/demo/sessions/`  
- [ ] Add Razor SSH public key to `authorized_keys` (Razor team)  
- [x] `python -m demo.pipeline.process_razor_session --session-dir ...` (batch T2 only)  
- [x] `python -m demo.pipeline.segment_daemon` (interactive T2 + T3–T7)  
- [x] `output/status.json` + `output/logs/process.log`  
- [x] Exit 0/1 based on `success`  
- [x] T2 batch via `input/segment/prompt.json` → `segment_prompt.py`  
- [ ] E2E: rsync chips input → full auto demo pipeline → Razor grasp (Razor client)  

---

## 11. Open questions (fill in on Titan)

| Item | Value |
|------|-------|
| `TITAN_HOST` / SSH user | |
| `UCB_ROOT` absolute path | |
| Conda env name for FP | |
| GPU sharing / queue policy | |
| SAM interactive step: always human on Titan, or use `input/segment/prompt.json`? | |

---

## 12. Changelog

| Date | Notes |
|------|-------|
| 2026-06-03 | Initial plan: SSH client on Razor, server on Titan, rsync v1 |
| 2026-06-03 | Standardize naming: auto demo pipeline; paths under `demo/` |
| 2026-06-03 | Add `segment_daemon` + `input/.upload_complete` for interactive T2 |
