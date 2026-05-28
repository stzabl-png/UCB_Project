# Evaluation object lists

## Current: `eval_objects_merged_success_ge30.csv`

- **Criterion:** sum of `successful_grasps` in `output/grasp_collect_no_rot/robot_gt/round_R/` for **R ≥ 3**, per object, with **count ≥ 30**.
- **Regenerate:**

```bash
python tools/build_eval_object_list.py \
  --outdir output/grasp_collect_no_rot \
  --min-round 3 \
  --min-success 30 \
  --output evaluation/configs/eval_objects_merged_success_ge30.csv
```

## Sim USD bundle

Packed assets for external Isaac Sim eval:

```bash
python tools/package_eval_ge30_usd_bundle.py
# → output/eval_objects_ge30_sim_bundle/
```

## Multi-round ablation (`tools/run_round_eval.py`)

Setups **1a–3b** × **10 rounds** (default): each run is full `eval_pool` (candidate gen + sim), 5 trials/object, results under `output/round_eval/{round}_{setup}/`, summary CSV at `output/round_eval/round_eval_summary.csv`.

```bash
python tools/run_round_eval.py \
  --candidate-gpu-ids 0,1 --candidate-workers 6 \
  --sim-gpu-ids 0,1 --sim-per-gpu 4
```

Each setup runs `eval_pool` with **`--loud`** by default (`--no-loud` for quiet `--log-only`).

```bash
# Only 2a and 3b
python tools/run_round_eval.py --setups 2a,3b ...

# From 2a through 3b each round (skip 1a,1b)
python tools/run_round_eval.py --start-setup 2a ...

# Round 5 onward, only unseen ablations
python tools/run_round_eval.py --start-round 5 --setups 1b,2b,3b --resume
```

## Unseen sim bundle (`eval_unseen_all.csv`)

```bash
# 1) USD + manifest
python tools/package_eval_unseen_sim_bundle.py

# 2) 500 pool candidates per object (HDF5 only)
python evaluation/eval_pool.py \
  --obj-list evaluation/configs/eval_unseen_all.csv \
  --generate-candidate-each-trial \
  --candidates-only \
  --trials-per-obj-yaw 500 \
  --z-yaw-deg 0 \
  --no-filtering \
  --result-dir output/evaluation/unseen_yaw0_n500_poolgen \
  --candidate-gpu-ids 0,1 \
  --candidate-workers 6

# 3) HDF5 → candidates/{obj_id}.json in bundle
python tools/migrate_eval_unseen_pool_candidates_to_bundle.py
```

## Deprecated lists

- **`eval_objects_merged_success_ge20.csv`** — round≥3, success≥20 (97 objects).
- **`eval_objects_merged_success_ge40.csv`** — merged HDF5 total success≥40 (legacy rule).
