# baseline_3 v4 — DP3 Retrain on 162-ep / 10-object Dataset

Standalone retrain guide for the **baseline_3 v4** Diffusion Policy 3D (DP3)
grasp policy on the full 12-object yaw-augmented dataset (162 trajectories,
10 YCB objects after dropping `foam` / `scissors`).

Target hardware: NVIDIA A6000 (48 GB VRAM). Should also run on A100/RTX 5090.

This file is self-contained: environment setup, data download, training, and
optional simulator-side evaluation. Skim section 8 first to know what artefacts
the run produces.

---

## 0. What this is

baseline_3 = grasp-trajectory diffusion policy that takes
`(object point cloud, panda_hand pose, gripper state)` as observation and
predicts the next-N panda_hand waypoints. The v4 collector produces these
trajectories from DexYCB hand-pose sequences in IsaacSim with cuRobo motion
planning; the resulting (state, action, point cloud) tuples train the DP3
policy.

**Previous training (32 train + 8 test ep over 3 objects)** reached
**65 % train / 75 % held-out test success rate** after fixing four eval-side
scene-alignment bugs (see commit `3508d7b`). This retrain scales the data to
162 ep over 10 objects with collection-time yaw augmentation; expected
ballpark: 70-80 % across the broader object set.

---

## 1. Hardware

| Resource | Minimum     | Recommended |
|----------|-------------|-------------|
| GPU      | 24 GB VRAM  | A6000 48 GB |
| RAM      | 32 GB       | 64 GB       |
| Disk     | 20 GB       | 50 GB       |
| CUDA     | 12.1+       | 12.4        |

DP3 peak GPU usage at `batch_size=128` is ~10 GB; A6000 has ample headroom.
3000-epoch training on the 162-ep dataset is roughly **3 h** on A6000
(RTX 5090 measured at 2 h 40 min for the same workload).

---

## 2. Environment Setup (training only)

The full pipeline has two conda envs:
- **`dp3`** — for training and the inference HTTP server. *Required.*
- **`env_isaaclab`** — for sim collection and closed-loop eval. *Optional;
  only needed for section 7.*

This section sets up `dp3`. Section 7 covers `env_isaaclab` if you need to
re-run sim eval.

```bash
# 2.1 Create env
conda create -n dp3 python=3.10 -y
conda activate dp3

# 2.2 PyTorch + CUDA
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# 2.3 Clone or copy this repo
git clone -b gate3-curobo-ik git@github.com:stzabl-png/UCB_Project.git
cd UCB_Project

# 2.4 Install the DP3 package (vendored under third_party/)
cd third_party/3D-Diffusion-Policy/3D-Diffusion-Policy
pip install -e .
cd ../../..

# 2.5 Other Python deps
pip install zarr==2.16 h5py==3.10 scipy hydra-core==1.3 termcolor \
            'numpy<2' 'omegaconf==2.3.*' diffusers==0.18 wandb tqdm
```

Sanity-check the install:
```bash
python -c "import diffusion_policy_3d; import zarr; import h5py; print('OK')"
```

---

## 3. Dataset

### 3.1 Contents

162 HDF5 episodes, each a fully-recorded successful grasp + lift trajectory
in object-centric G-frame. Per-object breakdown:

| cid | object       | orig | yaw aug | total |
|-----|--------------|------|---------|-------|
| 03  | sugar        | 14   | 8       | 22    |
| 04  | tomato       | 14   | 6       | 20    |
| 05  | mustard      | 11   | 4       | 15    |
| 06  | tuna         | 11   | 6       | 17    |
| 07  | pudding      | 9    | 9       | 18    |
| 08  | gelatin      | 11   | 4       | 15    |
| 09  | potted_meat  | 11   | 7       | 18    |
| 12  | bleach       | 17   | 8       | 25    |
| 15  | drill        | 3    | 4       | 7     |
| 18  | marker       | 3    | 2       | 5     |
|     | **TOTAL**    | 104  | 58      | **162** |

Per-ep schema (HDF5):
- `state`        — float32, `(31, 8)` = `[x,y,z, qw,qx,qy,qz, gripper]` in
  object-centric G-frame, retarget-quat convention
- `action`       — float32, `(31, 8)` = `state[1:]` (shifted)
- `point_cloud`  — float32, `(31, 4096, 3)` = static CAD surface samples in
  G-frame (all 31 frames identical; object is static during collection)
- `obj_origin_G`, `obj_quat_G_wxyz`, `ycb_class_id`, etc. as HDF5 attrs

Each file is ~1.5 MB; total dataset is **238 MB**.

### 3.2 Download

The 162 files live at
`Baseline1/data/episodes_b3_v4_full12_yaw/` on the source machine. Three ways
to obtain them:

**Option A — direct rsync from the source dev box** (fastest if you have ssh):
```bash
mkdir -p Baseline1/data
rsync -av --progress \
    accelerator@<DEV_HOST>:/home/accelerator/UCB_Project/Baseline1/data/episodes_b3_v4_full12_yaw \
    Baseline1/data/
```

**Option B — tarball from a shared cloud bucket** (URL to be filled in by the
project owner before handing this README over):
```bash
mkdir -p Baseline1/data
cd Baseline1/data
wget <PASTE_DOWNLOAD_URL_HERE> -O episodes_b3_v4_full12_yaw.tar.gz
tar xzf episodes_b3_v4_full12_yaw.tar.gz   # extracts ./episodes_b3_v4_full12_yaw/
rm episodes_b3_v4_full12_yaw.tar.gz
cd -
```

**Option C — already on the machine.** If the dataset is pre-staged, just
make sure the directory layout matches
`Baseline1/data/episodes_b3_v4_full12_yaw/<162 hdf5 files>` and skip ahead.

Verify after download:
```bash
ls Baseline1/data/episodes_b3_v4_full12_yaw/ | wc -l        # expect 162
du -sh Baseline1/data/episodes_b3_v4_full12_yaw/            # expect ~238M
```

---

## 4. Build the Training Zarr

### 4.1 Stratified 80/20 train/test split

```bash
python Baseline1/scripts/split_train_test.py \
    --input-dir  Baseline1/data/episodes_b3_v4_full12_yaw \
    --output-dir Baseline1/data/dp3_full12_yaw \
    --test-ratio 0.20 \
    --seed       42
```

Expected output: `train/` with 130 files, `test/` with 32 files, stratified by
object class so each object's proportion is preserved.

### 4.2 Convert to zarr (one for train, one for test)

```bash
python Baseline1/convert_to_zarr.py \
    --input_dir   Baseline1/data/dp3_full12_yaw/train \
    --output_zarr Baseline1/data/b3_v4_full12_train.zarr

python Baseline1/convert_to_zarr.py \
    --input_dir   Baseline1/data/dp3_full12_yaw/test \
    --output_zarr Baseline1/data/b3_v4_full12_test.zarr
```

Quick zarr inspection:
```bash
python -c "
import zarr
z = zarr.open('Baseline1/data/b3_v4_full12_train.zarr', mode='r')
ee = z['meta']['episode_ends'][:]
print(f'n_episodes={len(ee)}  total_steps={int(ee[-1])}')
print(f'action shape: {z[\"data\"][\"action\"].shape}')
print(f'point_cloud shape: {z[\"data\"][\"point_cloud\"].shape}')
"
```
Expected: `n_episodes=130  total_steps=4030`, action `(4030, 8)`,
point_cloud `(4030, 4096, 3)`.

---

## 5. Create the DP3 Task Config

The training config lives inside the DP3 source tree. Create
`third_party/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/config/task/baseline1_b3_v4_full12.yaml`
with the following contents (replace `<REPO_ROOT>` with the absolute path to
your `UCB_Project/` checkout):

```yaml
# Train on the 162-ep full12_yaw dataset, 80/20 external split.
name: baseline1_b3_v4_full12
task_name: b3_v4_full12
shape_meta: &shape_meta
  obs:
    point_cloud: {shape: [4096, 3], type: point_cloud}
    agent_pos:   {shape: [8],       type: low_dim}
  action:
    shape: [8]
env_runner: null
dataset:
  _target_: diffusion_policy_3d.dataset.baseline1_dataset.Baseline1Dataset
  zarr_path: <REPO_ROOT>/Baseline1/data/b3_v4_full12_train.zarr
  lazy: false
  yaw_aug: false        # already yaw-augmented at collection time
  horizon: ${horizon}
  pad_before: ${eval:'${n_obs_steps}-1'}
  pad_after:  ${eval:'${n_action_steps}-1'}
  seed: 42
  val_ratio: 0.0        # external 80/20 split — no train/val carve
  max_train_episodes: null
```

The `zarr_path` field **must be absolute**; relative paths break because
training runs from the DP3 subdir.

---

## 6. Launch Training

```bash
cd third_party/3D-Diffusion-Policy/3D-Diffusion-Policy

RUN_DIR=$HOME/dp3_runs/b3_v4_full12_3000
mkdir -p "$(dirname $RUN_DIR)"

python train.py \
    --config-name=dp3.yaml \
    task=baseline1_b3_v4_full12 \
    hydra.run.dir=$RUN_DIR \
    training.seed=42 \
    training.device=cuda:0 \
    training.num_epochs=3000 \
    training.checkpoint_every=500 \
    training.debug=False \
    logging.mode=offline               # set to 'online' if you want W&B logging
```

Outputs:
- `$RUN_DIR/checkpoints/latest.ckpt`   — most recent checkpoint
- `$RUN_DIR/checkpoints/epoch=NNNN-test_mean_score=*.ckpt` — every 500 epochs
- `$RUN_DIR/.hydra/`                   — resolved config snapshot
- `$RUN_DIR/train.log`                 — full training log

Sanity-check that loss is decreasing in the first few epochs:
```bash
grep -E "epoch=[0-9]+ .* train_loss" $RUN_DIR/train.log | head -10
```

Each ckpt is ~3.8 GB (full diffusion model + optimizer state). At 500-epoch
cadence you'll produce 6-7 checkpoints; budget ~30 GB of disk.

**Resume after interruption:** training resumes automatically if you re-run
the same command — `training.resume: True` is the default and it picks up
from `$RUN_DIR/checkpoints/latest.ckpt`.

---

## 7. (Optional) Closed-loop Evaluation in IsaacSim

Skip this section if you only need the trained checkpoint. Evaluation requires
**IsaacSim 5.1**, **cuRobo 0.8**, and is significantly more complex to set up
than training. Brief outline only.

### 7.1 Environment

```bash
conda create -n env_isaaclab python=3.10 -y
conda activate env_isaaclab

# Install IsaacSim 5.1 via Omniverse Launcher (UI only; ~15 GB).
# Then install the python bindings:
pip install --extra-index-url=https://pypi.nvidia.com isaacsim==5.1.0

# Install cuRobo 0.8
pip install curobo==0.8.0     # or build from source per their docs

# Other deps
pip install h5py scipy requests termcolor 'numpy<2' usd-core
```

You also need the YCB CAD USD assets — `output/obj_usd_cad/ycb/`,
1.2 GB, 21 objects. These ship as a separate tarball (URL to be filled in by
the project owner) and unpack into the repo's `output/` directory.

### 7.2 Run eval

In one shell — DP3 inference server (lives in `dp3` env):
```bash
conda activate dp3
python Baseline1/eval/dp3_inference_server.py \
    --ckpt $HOME/dp3_runs/b3_v4_full12_3000/checkpoints/latest.ckpt \
    --port 8765
```

In another shell — closed-loop sim eval (lives in `env_isaaclab` env):
```bash
conda activate env_isaaclab
python sim/eval_dp3_baseline3.py \
    --episodes-glob 'Baseline1/data/dp3_full12_yaw/test/*.hdf5' \
    --n-rollouts    30 \
    --max-chunks    6 \
    --headless \
    --server-url    http://127.0.0.1:8765 \
    --result-dir    output/dp3_eval_b3_v4_full12 \
    --video         replay_video_check/eval_b3_v4_full12 \
    --video-all
```

`sim/eval_dp3_baseline3.py` was hardened in commit `3508d7b` to remove four
eval/train scene-alignment bugs that previously masked policy quality. Read
that commit message for details.

---

## 8. What "Done" Looks Like

After section 6, you should have:
1. `$HOME/dp3_runs/b3_v4_full12_3000/checkpoints/latest.ckpt` — final checkpoint
2. `$HOME/dp3_runs/b3_v4_full12_3000/checkpoints/epoch=NNNN-*.ckpt` for
   N in `{500, 1000, 1500, 2000, 2500, 3000}` — intermediate snapshots
3. `$HOME/dp3_runs/b3_v4_full12_3000/train.log` — full training log

Hand off any of these to whoever runs eval — the closed-loop sim eval
(section 7) consumes a single `.ckpt` file.

---

## 9. Reference Numbers (Prior 32-ep / 3-object Run)

For comparison, the previous (smaller) training run reached the following
sim-eval success rates after the section 6 scene-alignment fixes (commit
`3508d7b`):

| object  | train SR (10 ep) | test SR (held-out)     |
|---------|------------------|------------------------|
| sugar   | 100 %            | 67 % (2/3)             |
| tomato  |  56 %            | 100 % (3/3)            |
| mustard |  44 %            |  50 % (1/2)            |

162-ep / 10-object expectation: similar or higher per-object SR plus
generalization across the broader object set. `marker` (5 ep) and `drill`
(7 ep) have very few trajectories and may stay weak — flag those for
additional data collection in a follow-up round.

---

## 10. Troubleshooting

- **`ModuleNotFoundError: No module named 'diffusion_policy_3d'`** — `pip install -e .`
  must run from inside
  `third_party/3D-Diffusion-Policy/3D-Diffusion-Policy/` (the inner dir, not
  the outer wrapper).
- **CUDA OOM during training** — drop `dataloader.batch_size` from 128 to 64
  via `dataloader.batch_size=64` on the train command line.
- **Training loss is NaN by epoch 5** — confirm `'numpy<2'`; numpy 2 breaks
  the diffusers DDIM scheduler in this DP3 fork.
- **`zarr.errors.PathNotFoundError`** when training starts — the `zarr_path`
  in `baseline1_b3_v4_full12.yaml` must be **absolute**, not relative.
- **Resume picks up nothing** — `training.resume` is True by default; if you
  changed any config field between runs (e.g. `num_epochs`), hydra writes a
  new run dir. Use the same `hydra.run.dir` to actually resume.

---

## 11. File Index

```
Baseline1/
  RETRAIN_V4_FULL12.md           — this file
  scripts/
    split_train_test.py          — 80/20 stratified splitter
  convert_to_zarr.py             — hdf5 dir → DP3 zarr
  eval/
    dp3_inference_server.py      — HTTP server for closed-loop eval
  data/
    episodes_b3_v4_full12_yaw/   — 162 source hdf5 (download via §3)
    dp3_full12_yaw/{train,test}/ — split output (§4.1)
    b3_v4_full12_{train,test}.zarr — DP3 input (§4.2)
sim/
  eval_dp3_baseline3.py          — closed-loop sim eval (§7)
third_party/3D-Diffusion-Policy/3D-Diffusion-Policy/
  train.py                       — DP3 training entry point (§6)
  diffusion_policy_3d/
    config/task/baseline1_b3_v4_full12.yaml   — task config (§5, you create this)
    dataset/baseline1_dataset.py              — zarr loader
```
