# Titan ↔ Razor automation plan (S2R)

**Audience:** Titan team (UCB_Project / GPU server) — implement processing + accept Razor uploads.  
**Razor team:** orchestrator client (future `run_s2r_pipeline.py`); capture + grasp already in V2AP-demo.

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
    → Titan T2–T7 pipeline (output/)
    → download to Razor
    → Razor run_auto_grasp.py
```

**v1 transport:** SSH + rsync (no custom TCP server required).  
**v2 (optional):** HTTP job queue on Titan for multi-session / status API.

---

## 2. Roles and connectivity

| Machine | Role | Network |
|---------|------|---------|
| **Razor** | Robot laptop; **orchestrator client**; initiates all connections | Often behind lab NAT — **does not need inbound SSH** |
| **Titan** | GPU server; **SSH server**; stores sessions; runs `process_razor_session` | Stable hostname/IP or VPN |

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
ssh-keygen -t ed25519 -f ~/.ssh/s2r_razor_to_titan -C "razor-s2r" -N ""
```

`~/.ssh/config` on Razor:

```text
Host titan-s2r
    HostName <TITAN_HOSTNAME_OR_IP>
    User <TITAN_USER>          # e.g. vision or s2r
    IdentityFile ~/.ssh/s2r_razor_to_titan
    IdentitiesOnly yes
```

Test:

```bash
ssh titan-s2r 'echo OK && hostname && nvidia-smi -L | head -1'
```

### 3.2 On Titan (server)

Append Razor **public** key to `~/.ssh/authorized_keys` for `<TITAN_USER>`:

```bash
# paste contents of s2r_razor_to_titan.pub
```

Optional hardening (later): `from="<RAZOR_IP>"`, dedicated `s2r` Unix user, `ForceCommand` wrapper.

### 3.3 Environment variables (both sides)

| Variable | Example | Used on |
|----------|---------|---------|
| `TITAN_SSH_HOST` | `titan-s2r` | Razor (~/.ssh/config alias) |
| `UCB_ROOT` | `/home/vision/Project/Affordance2Grasp` | Titan (repo root) |
| `S2R_SESSIONS_ROOT` | `$UCB_ROOT/demo/sessions` | Titan |
| `RAZOR_REPO` | `/home/.../V2AP-demo` | Razor |
| `S2R_CONDA_ENV` | `bundlesdf` | Titan (FoundationPose) |

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
conda activate "$S2R_CONDA_ENV"

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
| SAM mask | T2 | `output/segment/mask.png` (+ `prompt_used.json`); batch: `input/segment/prompt.json` |
| SAM3D mesh | T3 | `output/mesh/object_raw.glb` (`sam3d-objects` env) |
| Metric scale | T4 | `output/mesh/object_scaled.glb`, `scale.json` |
| FoundationPose | T5 | `output/register/T_cam_mesh.json`, `T_base_mesh.json`, `object_base_aligned.glb` |
| Grasp (PDM) | T6 | `output/inference/candidates.json` (+ optional HDF5) |
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
    "validate_input": "ok",
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
      "output/register/T_base_mesh.json"
    ]
  },
  "titan": {
    "n_candidates": 50,
    "hostname": "titan.local"
  }
}
```

**Failure:** `success: false`, non-empty `errors[]`. Partial `output/` is OK; Razor must **not** run grasp.

**Optional progress field (v1.1):** for Razor polling while job runs:

```json
"state": "queued | running | done | failed",
"started_at_iso": "...",
"updated_at_iso": "..."
```

Update `state`/`updated_at_iso` at each step start/end if implementing long jobs.

### 5.4 Remote invocation from Razor (v1)

Razor orchestrator will run **after** rsync upload:

```bash
ssh titan-s2r "cd ${UCB_ROOT} && \
  source \$(conda info --base)/etc/profile.d/conda.sh && \
  conda activate ${S2R_CONDA_ENV} && \
  python -m demo.pipeline.process_razor_session \
    --session-dir demo/sessions/${SESSION_ID} \
    --device cuda"
```

Titan script should:

- Exit code **0** iff `success: true` in final `status.json`
- Exit code **non-zero** on fatal error (Razor treats as failed job)
- Log to `output/logs/process.log` (recommended)

### 5.5 Rsync commands (reference)

**Razor → Titan (upload input):**

```bash
SESSION=20260602_192346_chips
rsync -avz --progress \
  "${RAZOR_REPO}/demo/phase2/sessions/${SESSION}/input/" \
  "titan-s2r:${UCB_ROOT}/demo/sessions/${SESSION}/input/"
```

**Titan → Razor (download output):**

```bash
rsync -avz --progress \
  "titan-s2r:${UCB_ROOT}/demo/sessions/${SESSION}/output/" \
  "${RAZOR_REPO}/demo/phase2/sessions/${SESSION}/output/"
```

---

## 6. End-to-end state machine (orchestrator view)

States stored in Razor `sessions/<id>/orchestrator_state.json` (future) and Titan `output/status.json`.

```text
CREATED          Razor: capture_session.py finished, input/ valid
UPLOADING        rsync input/ → Titan
QUEUED/RUNNING   Titan: process_razor_session
DONE             status.json success=true
DOWNLOADING      rsync output/ ← Titan
READY_FOR_GRASP  Razor: run_auto_grasp.py
GRASP_DONE       optional terminal state
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
| **A** | Accept rsync; manual `process_razor_session` | Manual rsync (today) | One session end-to-end |
| **B** | Stable `status.json` + exit codes | Shell script: upload → ssh → download | No manual Titan login |
| **C** | `state` polling + `process.log` | Python orchestrator + `orchestrator_state.json` | Single command from Razor |
| **D** (opt) | HTTP job API + queue | Client uses API + rsync for blobs | Multi-user / queue |

**Titan minimum for phase B:** sections 5.1, 5.3, 5.4, 5.5 working reliably.

---

## 10. Checklist for Titan team

- [x] Create `$UCB_ROOT/demo/sessions/`  
- [ ] Add Razor SSH public key to `authorized_keys` (Razor team)  
- [x] `python -m demo.pipeline.process_razor_session --session-dir ...` on Titan  
- [x] `output/status.json` + `output/logs/process.log`  
- [x] Exit 0/1 based on `success`  
- [x] T2 batch via `input/segment/prompt.json` → `segment_prompt.py`  
- [ ] E2E: rsync chips input → full pipeline → Razor grasp (Razor client)  

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
