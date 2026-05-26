# Affordance training (PointNet++ v6)

Continuous soft heatmap regression on **no_rot executed** data. Entry: **`python -m model.train_v6`**.

Legacy `model/affordance/*` training has been removed; use v6 only.

## Pipeline

1. **Prepare** — `tools/prepare_affordance_executed.py` → `affordance_all.h5` + `affordance_all_soft.h5` (optional `--write-split` for train/val too)
2. **Merge** (optional) — `tools/merge_affordance_h5_splits.py` if you only have legacy train/val files
3. **Train** — `python -m model.train_v6`
4. **Infer** — `python -m model.inference_v6 --checkpoint ... --obj ID`

Soft labels are written during prepare; refresh only with `prepare_affordance_executed.py --export-soft-only`.

### Filter train/val by trusted pose count

From `affordance_all.h5` / `affordance_all_soft.h5` (unchanged), rewrite split files by merged trusted-grasp count:

```bash
python3 tools/filter_affordance_split_by_pose_count.py \
  --dataset-dir output/affordance_no_rot_executed \
  --min-trusted 10 \
  --backup
```

Default `--split-mode fixed-val` keeps original val membership (only drops objects below threshold). Use `--output-dir` to write into a separate experiment folder without touching `affordance_all*`.

## Train v6

```bash
# default GPU: cuda:0 (set CUDA_VISIBLE_DEVICES to pick another card)
python -m model.train_v6 \
  --dataset_dir output/affordance_no_rot_executed \
  --save_dir output/affordance_no_rot_executed/checkpoints_v6 \
  --epochs 300 \
  --batch_size 16
```

| Item | Detail |
|------|--------|
| Input | `xyz` (3) + `features` = concat(xyz, normals) (6ch), 4096 pts |
| Target | `data/soft_labels` in `*_soft.h5` |
| Loss | Weighted L1 (`--fg_weight 5`, foreground = soft ≥ 0.05) |
| Best ckpt | `best_v6_model.pth` by **val Pearson** (no early stop; runs full `--epochs`) |
| Vis | `vis_ep1.png`, `vis_ep20.png`, … (random 4 val objects) |

`torch.compile` is **off by default** (PyTorch 2.1 + FPS breaks). Pass `--compile` only if you know your PT version supports it.
Use `--num_workers 4` for faster IO after training runs stably with default `0`.

## Data layout

```
output/affordance_no_rot_executed/
├── affordance_all.h5
├── affordance_all_soft.h5
├── affordance_train.h5          # optional (--write-split)
├── affordance_val.h5
├── affordance_train_soft.h5     # v6 training
├── affordance_val_soft.h5
├── objects_train_val_split.json
└── checkpoints_v6/
```

## Inference (v6)

```bash
python -m model.inference_v6 \
  --checkpoint output/affordance_no_rot_executed/checkpoints_v6/best_v6_model.pth \
  --obj ycb_dex_04 \
  --save-dir output/affordance_no_rot_executed/inf
# → inf/npz/{obj}.npz  and  inf/png/{obj}.png

python -m model.inference_v6 --checkpoint ... --random 4 --save-dir output/inf_v6
python -m model.inference_v6 --checkpoint ... --split val --save-dir output/inf_v6
# → inf_v6/png/{obj}.png  and  inf_v6/all_objects_grid.png (montage, default)

python -m model.inference_v6 --compose-grid-only --save-dir output/inf_v6   # grid from existing png/
python -m model.inference_v6 --checkpoint ... --h5 output/.../affordance_all_soft.h5 --all --save-dir output/inf_v6
python -m model.inference_v6 --checkpoint ... --split val   # default batch-size 64 objects per GPU forward
```

See also: [`prepare_affordance_executed.md`](prepare_affordance_executed.md).
