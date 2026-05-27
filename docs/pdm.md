# Pose Diffusion Model (PDM)

Conditional diffusion over **object-frame grasp command poses**, supervised from successful merged sim grasps (`executed_panda_hand_at_close` → command frame via `model/pdm/pose_codec.py`).

Entry points live under `model/pdm/`:

| Step | Module |
|------|--------|
| Condition cache | `python3 -m model.pdm.build_condition_cache` |
| Train | `python3 -m model.pdm.train` |
| Sample candidates | `python3 -m model.pdm.sample` |
| Visualize | `python3 -m model.pdm.visualize` |

**Upstream data:** merged successful grasps ([`grasp_collect_pipeline.md`](grasp_collect_pipeline.md)) and optional affordance v6 HDF5 ([`train_affordance.md`](train_affordance.md)).

---

## Pipeline

1. **Merged GT** — `output/grasp_collect_no_rot/merged/{obj}_robot_gt_merged.hdf5` (from pool sim or legacy collect).
2. **Condition cache** (recommended) — one fixed point cloud per object: xyz + normal + affordance (`4096` points by default).
3. **Train** — DDPM noise prediction on normalized 9D pose vectors; saves `best_model.pth` + `pose_stats.pt`.
4. **Sample** — DDIM poses → per-object `{obj}_grasp.hdf5` (same candidate layout as raycast/anchored sampler).
5. **Visualize** — overlay grippers on the condition cloud; optional `overview.png` montage.

```bash
export PROJ=/home/vision/Project/Affordance2Grasp
cd "$PROJ"

# 1) Precompute object conditions (reuse affordance v6 points when available)
python3 -m model.pdm.build_condition_cache \
  --merged-dir output/grasp_collect_no_rot/merged \
  --affordance-h5 output/affordance_no_rot_executed/min20/affordance_all_soft.h5 \
  --output output/pdm/cache/conditions_4096.h5

# 2) Train
python3 -m model.pdm.train \
  --merged-dir output/grasp_collect_no_rot/merged \
  --condition-h5 output/pdm/cache/conditions_4096.h5 \
  --affordance-h5 output/affordance_no_rot_executed/min20/affordance_all_soft.h5 \
  --save-dir output/pdm/checkpoints

# 3) Sample candidates for one or many objects
python3 -m model.pdm.sample \
  --checkpoint output/pdm/checkpoints/best_model.pth \
  --condition-h5 output/pdm/cache/conditions_4096.h5 \
  --obj ycb_dex_04 \
  --n-samples 50

python3 -m model.pdm.sample \
  --checkpoint output/pdm/checkpoints/best_model.pth \
  --condition-h5 output/pdm/cache/conditions_4096.h5 \
  --all

# 4) Visualize
python3 -m model.pdm.visualize \
  --candidates-dir output/pdm/candidates \
  --condition-h5 output/pdm/cache/conditions_4096.h5 \
  --all
```

---

## Pose and condition representation

**Supervision target (9D):** `[x, y, z, rot6d]` in the **simulator command frame** (TCP / finger-center position; rotation columns = finger, lateral, approach). Labels are built from merged `executed_panda_hand_at_close` wrist poses using the same `R_ADAPT` / `TCP_OFFSET=0.105` convention as `sim/run_grasp_sim.py`.

**Object condition (per point, 7 channels):** `xyz (3) + normal (3) + affordance (1)`, `N=4096` by default. Training uses a PointNet-style global encoder (`PDMPointEncoder`) fused into the denoiser.

**Training filters** (`PDMMergedDataset`, default):

- Requires `executed_panda_hand_at_close` on each successful grasp row.
- `require_trusted_tips=True` unless `--allow-untrusted-tips`.
- Drops poses whose command TCP is farther than `--max-cmd-candidate-dist` (default `0.5` m) from stored `grasp_point`.

Pose mean/std are computed over all kept rows and stored in `pose_stats.pt` (also embedded in checkpoints).

---

## Output layout

```
output/pdm/
├── cache/
│   └── conditions_4096.h5      # data/points, normals, affordance, obj_ids
├── checkpoints/
│   ├── best_model.pth
│   ├── final_model.pth
│   ├── pose_stats.pt
│   └── training_history.json
├── candidates/
│   └── {obj_id}_grasp.hdf5     # metadata.method=pdm, candidates/*, grasp/
└── vis_overlay/
    ├── {obj_id}_pdm_overlay_top{N}.png
    └── overview.png            # montage when --all and multiple objects
```

### Condition cache HDF5

Mirrors affordance v6 layout:

| Dataset | Shape |
|---------|--------|
| `data/points` | `(M, N, 3)` |
| `data/normals` | `(M, N, 3)` |
| `data/affordance` | `(M, N)` |
| `data/obj_ids` | `(M,)` UTF-8 strings |

Objects are taken from merged files that yield at least one training row (or pass `--obj` for a subset). Rows with insane coordinates/normals are skipped (`--max-abs-coord`, default `2.0`).

### Candidate HDF5

Compatible with existing grasp sim / pool tooling:

- `metadata`: `obj_id`, `method=pdm`, `sampling_method=pdm_diffusion`, `no_rotation=True`
- `candidates/candidate_{i}`: `position`, `rotation`, `grasp_point`, attrs `name`, `score`, `gripper_width`, `approach_type=pdm`
- `grasp/`: best candidate (index 0 after sampling order)
- `mesh_prerotation/`: identity (no-rot corpus)

Run Isaac validation the same way as raycast candidates, e.g. point `run_grasp_sim.py` at `output/pdm/candidates/{obj}_grasp.hdf5` (see [`pool_grasp_sim_pipeline.md`](pool_grasp_sim_pipeline.md) for batch patterns).

### Isaac evaluation runner

Modular eval (batch, z-yaw, optional video) is documented in [`evaluation.md`](evaluation.md).

```bash
# Pool candidate → single sim episode
$ISAAC_SIM_PATH/python.sh evaluation/eval_single.py \
  --obj-id C22001 \
  --candidate-hdf5 output/grasp_collect_no_rot/candidates/pool/C22001_grasp.hdf5 \
  --selection sample --headless --save-hdf5

# GLB → PDM → sim（real machine）
$ISAAC_SIM_PATH/python.sh evaluation/eval_single.py \
  --obj-id IMG_4477 \
  --mesh data_hub/real_machine/sam3d_glb/IMG_4477.glb \
  --generate-candidate --z-yaw-deg 0 --headless

# Yaw-conditioned ckpt (checkpoints_yaw): match sim and PDM
python tools/glb_to_pdm_grasp.py --mesh ... --z-yaw-deg 90 ...
$ISAAC_SIM_PATH/python.sh evaluation/eval_single.py \
  --obj-id ... --z-yaw-deg 90 --generate-candidate --mesh ...
```

`eval_single` / `eval_batch` 调用 `glb_to_pdm_grasp` 时会加 `--random-seed`，便于多次 trial 得到不同扩散样本。训练 yaw 条件见 `model/pdm/train.py --use-yaw-condition`。

---

## CLI reference

### `build_condition_cache`

| Flag | Default | Notes |
|------|---------|--------|
| `--merged-dir` | `output/grasp_collect_no_rot/merged` | Object list source |
| `--affordance-h5` | none | Prefer aligned points/normals/labels; else mesh sample + zero affordance |
| `--mesh-root` | `data_hub/meshes/SAM3DMesh/rotated_mesh` | Fallback surface sampling |
| `--n-points` | `4096` | |
| `--output` | `output/pdm/cache/conditions_4096.h5` | |

### `train`

| Flag | Default | Notes |
|------|---------|--------|
| `--condition-h5` | none | Strongly recommended for fast IO |
| `--affordance-h5` | none | Fallback per-object conditions if not in cache |
| `--save-dir` | `output/pdm/checkpoints` | |
| `--epochs` | `300` | Cosine LR, AdamW |
| `--batch-size` | `32` | |
| `--val-ratio` | `0.2` | Best checkpoint by val MSE |
| `--allow-untrusted-tips` | off | Include non-trusted tip rows |
| `--no-cache-conditions` | off | Disable in-process condition cache |
| `--cpu` | off | |

Checkpoints: `best_model.pth` (lowest val loss), optional `checkpoint_epoch*.pth` every `--save-every` (default `100`), `final_model.pth`.

### `sample`

| Flag | Default | Notes |
|------|---------|--------|
| `--checkpoint` | required | |
| `--obj` / `--all` / `--random N` | — | Object selection |
| `--n-samples` | `50` | Poses per object |
| `--ddim-steps` | `50` | |
| `--reject-upward` | off | Drop poses with `rotation[2,2] > --max-approach-z` |
| `--output` | — | Single HDF5 path (one object only) |
| `--output-dir` | `output/pdm/candidates` | |

Condition load order: precomputed cache → affordance HDF5 → on-the-fly mesh sample.

### `visualize`

| Flag | Default | Notes |
|------|---------|--------|
| `--hdf5` | — | Explicit candidate file(s) |
| `--candidates-dir` | `output/pdm/candidates` | Used with `--all` / `--random` |
| `--top` | `20` | Overlaid poses per figure |
| `--overview` | on | Grid montage (`--no-overview` to disable) |

---

## Package map

```
model/pdm/
├── __init__.py           # public exports (PDM, pose codecs)
├── pose_codec.py         # executed ↔ command, 9D pack/unpack
├── dataset.py            # PDMMergedDataset, AffordanceStore, PDMConditionStore
├── model.py              # PDM encoder + denoiser + DDIM sample
├── build_condition_cache.py
├── train.py
├── sample.py
└── visualize.py
```

---

## Tips

- **GPU:** Training and sampling use CUDA when available (`--cpu` to force CPU).
- **Empty dataset:** Usually means no merged files, no `executed_panda_hand_at_close`, or all rows filtered (untrusted tips / outliers). Re-run train with `--allow-untrusted-tips` or inspect `skipped` counts printed at startup.
- **Affordance alignment:** For best geometry, pass the same `affordance_all_soft.h5` used for v6 training when building the condition cache.
- **Reproducibility:** Training split uses `--seed` (default `42`). Diffusion sampling uses an unseeded RNG by default. `glb_to_pdm_grasp` supports `--random-seed` (used by eval) or fixed `--seed`.
- **Sim z-yaw:** Yaw-conditioned models need the same `--z-yaw-deg` at sample time as in Isaac eval (`evaluation.md`).
