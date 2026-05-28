# scripts/eval_combined/ — DP3 combined-model eval

Batch sim eval for the **DexYCB 162 + OakInk 207** combined DP3 model trained
on the A6000. A6000 trains with `val_ratio: 0.0` (no held-out split), so eval
is sim-based on a fixed train-set sample.

---

## Files

| file | purpose |
|---|---|
| `04_batch_eval.sh` | **The only eval entry point.** Groups eps by obj class, runs one IsaacSim subprocess per class, shares DP3 inference server. Supports `--par N` parallelism + `--resume`. |
| `README.md` | This file. |

---

## Workflow (per new ckpt)

### Step 1 — A6000 uploads new ckpt to HF

A6000 only needs to push the `.ckpt` file (config + task yaml are identical
across epochs, already uploaded once):

```bash
huggingface-cli upload UCBProject/baseline_3_v4_dexycb162_oakink207_dp3 \
    experiments/.../checkpoints/epoch=NNNN-test_mean_score=-0.XXX.ckpt \
    epoch=NNNN-test_mean_score=-0.XXX.ckpt \
    --repo-type model
```

### Step 2 — Dev box: download new ckpt

```bash
PY=/home/accelerator/miniforge3/envs/dp3/bin/python
$PY -c "from huggingface_hub import hf_hub_download; \
  hf_hub_download('UCBProject/baseline_3_v4_dexycb162_oakink207_dp3', \
    'epoch=NNNN-test_mean_score=-0.XXX.ckpt', repo_type='model', \
    local_dir='Baseline1/dp3_runs/combined_dexycb162_oakink207')"
```

### Step 3 — Run eval (DexYCB-16 + OakInk-16 train-set sample)

```bash
CKPT='Baseline1/dp3_runs/combined_dexycb162_oakink207/epoch=NNNN-*.ckpt'

# DexYCB-16 — 8 classes, ~16 eps total
bash scripts/eval_combined/04_batch_eval.sh \
    --ckpt "$CKPT" \
    --episodes-glob '/tmp/eval_combined_train_dexycb16/*.hdf5' \
    --tag eNNNN_dexycb16 --par 3

# OakInk-16 — 13 classes, ~16 eps total
bash scripts/eval_combined/04_batch_eval.sh \
    --ckpt "$CKPT" \
    --episodes-glob '/tmp/eval_combined_train_oakink16/*.hdf5' \
    --tag eNNNN_oakink16 --par 3
```

Each call takes ~5-10 min (with par=3) and writes:
- `output/dp3_eval_combined_<tag>/summary.json` — aggregate + per-class success rates
- `output/dp3_eval_combined_<tag>/per_class/<obj_label>/eval_*.json` — per-class details
- `replay_video_check/eval_combined_<tag>/<obj_label>/*.mp4` — sim videos

---

## 04_batch_eval.sh flags

| flag | default | purpose |
|---|---|---|
| `--ckpt PATH` | required | DP3 checkpoint (glob OK, picks newest) |
| `--episodes-glob PATTERN` | required | sim source eps to evaluate against |
| `--tag NAME` | required | unique tag → output dir name |
| `--par N` | 1 | parallel per-class subprocesses (RTX 5090: 3 safe, 4 OOM-risky) |
| `--resume` | off | skip classes that already have `eval_*.json` (useful for crash recovery) |
| `--n-per-class N` | all | limit eps per class for faster eval |
| `--total N` | all | cap total eps across all classes |
| `--max-chunks N` | 5 | DP3 receding-horizon chunks |
| `--port N` | 8765 | DP3 inference server port |

---

## Parallelism notes (RTX 5090, 32GB)

| par | sustained mem | OOM risk | speedup |
|-----|---------------|----------|---------|
| 1 | ~14 GB | none | baseline |
| 2 | ~21 GB | safe | 2× |
| **3** | **~28 GB** | **tight but works** | **3×** |
| 4 | ~35 GB | OOM almost certain | – |

Per-class subprocess uses ~5-7GB IsaacSim + transient cuRobo spikes ~1-2GB.
DP3 server adds ~5GB. **Recommend `--par 3` on 5090.**

---

## Pre-staged eval episode sets

Currently in `/tmp/` (re-create if cleared):

| set | sessions | classes | seed |
|---|---|---|---|
| `eval_combined_train_dexycb16` | 16 random DexYCB hdf5 | 8 obj | seed=42 |
| `eval_combined_train_oakink16` | 16 random OakInk hdf5 | 13 obj_ids | seed=42 |

Reproduce:
```python
import random, glob, shutil, os
random.seed(42)
DEX = sorted(glob.glob("Baseline1/data/episodes_b3_v4_full12_yaw/*.hdf5"))
OAK = sorted(glob.glob("Baseline1/data/episodes_b3_v4_oakink89_2026-05-26/*.hdf5"))
for tag, files in [("dexycb16", random.sample(DEX, 16)),
                   ("oakink16", random.sample(OAK, 16))]:
    d = f"/tmp/eval_combined_train_{tag}"; os.makedirs(d, exist_ok=True)
    for f in files: shutil.copy(f, d)
```

---

## Troubleshooting

- **`Failed to get rigid body velocities from backend`**: IsaacSim PhysX prim swap mid-run. Should NOT happen with `04_batch_eval.sh` (uses fresh subprocess per class). If it does, check `eval_dp3_baseline3.py` for accidental per-ep `load_object` call.
- **Server timeout (120s)**: ckpt path wrong / OOM / corrupt ckpt. Check `/tmp/dp3_server_<tag>.log`.
- **All eps fail with `dz < 0.03`**: that IS the model's true SR. Compare against prior `v4_sml` eval to spot regression.
- **`multi-class glob detected`**: you ran `sim/eval_dp3_baseline3.py` directly on a multi-class glob — use `04_batch_eval.sh` instead.

---

## Comparison against prior DexYCB-only model

Prior `v4_sml` ckpt + eval artefacts (DexYCB-only, 3000 epochs) preserved at:
- `Baseline1/dp3_runs/b3_v4_sml_3000/`
- `output/dp3_eval_b3_v4_sml_3000/` (DexYCB test-set eval, 75% SR)

Re-run prior model on same train-set sample for apples-to-apples:
```bash
bash scripts/eval_combined/04_batch_eval.sh \
    --ckpt Baseline1/dp3_runs/b3_v4_sml_3000/checkpoints/latest.ckpt \
    --episodes-glob '/tmp/eval_combined_train_dexycb16/*.hdf5' \
    --tag v4_sml_dexycb16_compare --par 3
```
