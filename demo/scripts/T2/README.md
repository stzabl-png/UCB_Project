# T2 — SAM2 segmentation

Interactive single-frame object mask for Phase 2 sessions.

## Prerequisites (one-time)

SAM2 lives in `third_party/sam2` (not under `demo/scripts`). Use **bundlesdf** for inference.

### Do **not** run plain `pip install -e .` on Titan

Upstream SAM2 now declares `torch>=2.5.1` as a **build dependency**. That makes pip download a new Torch 2.12 + CUDA 13 stack and often fails (hash mismatch on NGC mirror). **bundlesdf already has Torch 2.1.1+cu121** — keep it.

### Install for bundlesdf (recommended)

```bash
cd /home/vision/Project/Affordance2Grasp/third_party
git clone https://github.com/facebookresearch/sam2.git sam2   # if missing
cd sam2/checkpoints && bash download_ckpts.sh               # includes tiny

conda activate bundlesdf
pip install "hydra-core>=1.3.2" "iopath>=0.1.10"
SAM2_BUILD_CUDA=0 pip install -e . --no-build-isolation --no-deps

# Verify (no GUI needed):
python -c "from sam2.sam2_image_predictor import SAM2ImagePredictor; print('ok')"
```

- `SAM2_BUILD_CUDA=0` — skip optional CUDA post-processing extension (fine for clicking masks).
- `--no-build-isolation --no-deps` — do not pull Torch 2.5+; use env Torch 2.1.

Checkpoint used by `tools/sam2_server.py`:

`third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt`

Optional server smoke test (Ctrl+C after `ready`):

```bash
cd /home/vision/Project/Affordance2Grasp
python tools/sam2_server.py
```

## Usage

### A. Web UI — Flask (recommended over SSH, no X11)

Uses **Flask** (not Gradio — avoids version conflicts in `bundlesdf`).

**On Titan** (tmux):

```bash
conda activate bundlesdf
cd /home/vision/Project/Affordance2Grasp

python demo/scripts/T2/segment_web.py \
  --session-dir demo/sessions/20260602_192346_chips \
  --port 7860
```

**On your laptop**:

```bash
ssh -L 7860:127.0.0.1:7860 vision@<titan-host>
```

Browser: **http://127.0.0.1:7860** — click image (FG/BG radio), **Save mask to session**, then **Done (exit)** to stop the server (or Ctrl+C in tmux).

Options: `--redo`, `--host 127.0.0.1` (default).

`flask` is already in most envs; if missing: `pip install flask`.

### B. OpenCV window (only with local display / `ssh -X`)

```bash
python demo/scripts/T2/segment.py --session-dir demo/sessions/20260602_192346_chips
```

| Input | Action |
|-------|--------|
| Left click | Foreground point |
| Right click | Background point |
| `M` | Toggle next left-click FG vs BG |
| `C` | Clear points and mask |
| `Enter` | Save mask |
| `Q` / Esc | Quit without saving |

### T1 validate (optional)

```bash
python demo/scripts/T1/validate_input.py --session-dir ...
```

### Outputs

```text
output/segment/
├── mask.png           # SAM2 output, 0/255, same size as left_rgb.png
└── prompt_used.json   # tool=sam2, clicked points
```

## Notes

- RGB is loaded with **PIL** (correct for chips session); SAM2 server also uses PIL.
- Default SAM2 subprocess uses `bundlesdf` python (`--python` to override).
- Shared paths: `demo/scripts/_session_io.py` (`--session-dir` / `--input-dir`).

### C. Batch from `input/segment/prompt.json` (S2R / Razor)

When Razor saves click prompts in `input/segment/prompt.json` (see [demo/README.md](../../README.md)):

```bash
python demo/scripts/T2/segment_prompt.py \
  --session-dir demo/sessions/<session_id>
```

Used automatically by `python -m demo.pipeline` when mask is missing but prompt exists.
