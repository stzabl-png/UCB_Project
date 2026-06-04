# DP3 Baseline — A/B Diversity Eval (±5cm XY jitter + 4 yaw), 116 objects

Raw **per-episode** results for the **DP3 (3D Diffusion Policy)** baseline under the A/B
placement-diversity protocol. Intended for computing **per-category success rates** (group the
116 objects into semantic categories, then SR per category).

## Experiment
- **Policy**: DP3 baseline `combined_dexycb162_oakink207` (epoch 2800), titan-protocol sim eval (gate3 IsaacSim).
- **Condition**: A/B diversity, seed 42 (byte-parity with the partner `eval_pool` enumeration — same seed ⇒ same placement as the main method):
  - **A** — object XY jitter: **±5 cm random** (base XY = (0, 0.55), no y-bias).
  - **B** — z-yaw grid: **{0, 90, 180, 270}°**.
- **116 objects × 4 yaw × 10 trials = 4640 episodes** (exactly 40 per object).
- **Overall SR = 25.09% (1164 / 4640).**

## Files
| file | what |
|---|---|
| `per_ep_results.tar.gz` | 4640 per-episode JSONs. Extract: `tar xzf per_ep_results.tar.gz`. Filename = `{obj_id}_dp3_titan_protocol_t{trial}_yaw{yaw:03d}.json`. |
| `per_object_summary.csv` | `obj_id, coarse_category, n_success, n_total, success_rate` (one row per object). |
| `compute_sr.py` | reads the tar.gz directly → prints per-object + per-category SR; accepts an optional `obj_id→category` CSV. |

## Per-episode JSON schema (key fields)
| field | meaning |
|---|---|
| `obj_id` | object id, e.g. `A01001`, `unseen_000`, `ycb_dex_01`, `Y03006` |
| `success` | bool — grasp success (object lift `dz > 3 cm`) |
| `z_delta_m` | object lift (m) |
| `z_yaw_deg` | B: yaw applied this episode (0 / 90 / 180 / 270) |
| `obj_xy_offset` | A: `[dx, dy]` jitter applied (m) |
| `trial` | trial index (0–9) within a (obj, yaw) |
| `failure_stage` | stage label if failed, else `null` |
| `execution` / `scene` / `policy_output` | extra metadata (n_chunks, ycb_class_id, placement, …) |

## Object categories
Coarse split is by `obj_id` prefix (already in the CSV):
- `A` / `C` / `O` / `S` → **OakInk** classes (68 objects)
- `unseen_XXX` → **unseen** objects (29)
- `Y` / `ycb_dex_XX` → **DexYCB** objects (19)

**Finer semantic category classification is yours** (per the OakInk action-class / DexYCB object
taxonomy). To use it: make a CSV `obj_id,category` and pass it to `compute_sr.py --categories`.

## Compute per-category SR
```bash
python compute_sr.py                          # per-object + coarse-category SR
python compute_sr.py --categories your_map.csv   # SR per your semantic category
```
SR = `mean(success)` over the episodes in each group. Each object has exactly 40 episodes
(4 yaw × 10 trials), so per-object SR is over 40, per-category SR is over (40 × #objects-in-category).
