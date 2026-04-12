# UCB_Project — Training & Evaluation Results (gm_zt_results)

This branch contains the results of two 200-epoch training runs of the Affordance2Grasp pipeline, documenting a critical bug discovery and fix.

## Overview

| Version | Folder | Best val_F1 | val_fc_mm | Status |
|---------|--------|-------------|-----------|--------|
| **V1** (before fix) | `before_touch_src_200/` | 0.000 | 94.38mm | Segmentation head collapsed |
| **V2** (after fix) | `after_touch_src_200/` | **0.139** | **89.90mm** | Working segmentation + grasp generation |

## Bugs Found & Fixed (V1 → V2)

Three critical bugs caused V1 to produce zero F1 across all 200 epochs:

1. **Sigmoid–Loss Mismatch** (`model/pointnet2.py`): Model applied `torch.sigmoid()` on output, but losses (`FocalLoss` → `F.cross_entropy`, `TverskyLoss` → `F.softmax`) expect raw logits. Double-activation destroyed gradients.

2. **Label Binarization** (`model/train.py`): Labels are continuous floats [0, 0.9999]. Code did `.long()` directly → ALL labels became 0 (floor to int). The model trained with 100% negative labels.

3. **Class Imbalance** (`model/losses.py`): Only ~0.4% of points are positive. Added class weight `[1.0, 50.0]` to `FocalLoss` to handle extreme imbalance.

---

## V1: before_touch_src_200

Training with **unmodified upstream source code** — 200 epochs, val_F1 = 0.0 throughout.

```
before_touch_src_200/
├── EXPERIMENT_REPORT.md                    # Detailed experiment report
├── checkpoints_pretrained/
│   ├── best_model.pth                      # Pretrained checkpoint (from HuggingFace)
│   └── best_m5_model.pth                   # Pretrained M5 model
├── checkpoints_v5/
│   ├── checkpoint_epoch200.pth             # 200-epoch checkpoint (F1=0.0)
│   ├── final_model.pth                     # Same as epoch 200
│   ├── training_history.json               # Full training metrics
│   └── split_info.json                     # Train/val split info
├── compare/
│   ├── pretrained_vs_200epoch_summary.json # Pretrained vs 200-epoch comparison
│   └── pretrained_vs_200epoch_summary.md   # Comparison report (markdown)
├── fresh_clone_results/
│   ├── training_history_fresh.json         # Training from clean clone (same result)
│   └── split_info_fresh.json
├── analysis/                               # Visualization and comparison scripts
│   └── compare_pretrained_vs_200epoch.py
└── batch_inference.py                      # Batch evaluation script
```

### Key V1 Results
- val_F1 = 0.000 for all 200 epochs
- val_accuracy = 0.997 (trivially predicting all-negative due to 0.4% positive rate)
- Force-center error: 94.38mm (still learns spatial structure)
- Reproduced identically in both local and fresh-clone training runs

---

## V2: after_touch_src_200

Training with **3 bug fixes applied** — 200 epochs, val_F1 = 0.139 (best at epoch 149).

```
after_touch_src_200/
├── REPORT_v2_200epoch.md                   # Detailed V2 report with diffs
├── checkpoints/
│   ├── best_model.pth                      # Best checkpoint (epoch 149, F1=13.88%)
│   ├── final_model.pth                     # Final checkpoint (epoch 200)
│   └── training_history.json               # Full 200-epoch training metrics
├── grasps/
│   └── A16013_grasp.hdf5                   # Inference output (11 grasp candidates)
└── source_diffs/
    ├── v2_bugfix.patch                     # Git patch of all changes
    ├── pointnet2.py                        # Fixed model (removed sigmoid)
    ├── losses.py                           # Fixed loss (class weight 50×)
    ├── train.py                            # Fixed label binarization
    └── predictor.py                        # Fixed inference auto-detection
```

### Key V2 Results
- Best val_F1 = 0.139 (epoch 149), vs 0.000 in V1
- val_precision = 0.108, val_recall = 0.249
- Force-center error: 89.90mm (best), 93.27mm at best-F1 epoch
- F1 > 0 for all 200 epochs
- Training time: 3.3 hours on 8× RTX PRO 6000

### A16013 Inference
- 11 valid grasp candidates generated
- Best grasp: `dynamic_back_y30` (score=95.1, horizontal approach)
- Gripper widths: 4.3–7.7cm across candidates

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | PointNet2Seg (976,645 params) |
| Input | 7 channels (xyz + normals + human_prior) |
| Output | 2-class segmentation + 3D force center |
| Dataset | 11,636 train / 2,910 val samples (1,024 pts each) |
| Optimizer | AdamW (lr=0.001, weight_decay=1e-4) |
| Scheduler | CosineAnnealingWarmRestarts (T_0=50) |
| Batch size | 32/GPU × 8 GPUs = 256 effective |
| Loss | Focal (α=0.75, γ=2) + Tversky (α=0.3, β=0.7) |
| GPUs | 8× NVIDIA RTX PRO 6000 Blackwell |

---

## How to Apply V2 Fixes

```bash
cd UCB_Project
git apply after_touch_src_200/source_diffs/v2_bugfix.patch
```

Or manually copy the fixed source files:
```bash
cp after_touch_src_200/source_diffs/pointnet2.py model/
cp after_touch_src_200/source_diffs/losses.py model/
cp after_touch_src_200/source_diffs/train.py model/
cp after_touch_src_200/source_diffs/predictor.py inference/
```
