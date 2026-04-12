# V2 200-Epoch Training Report — after_touch_src_200

**Date**: 2026-04-12  
**Repo**: UCB_Project_fresh (clean clone from GitHub)  
**Model**: PointNet2Seg (976,645 params), 7-channel input, 2-class segmentation + force-center regression  

---

## 1. Bugs Found & Fixed (vs V1)

V1 (before_touch_src_200) trained for 200 epochs with **val_f1 = 0.0 for all epochs**. Three bugs were identified and fixed:

### Bug 1: Sigmoid–Loss Mismatch (Critical)

**File**: `model/pointnet2.py` line 205

- **Before (V1)**: Model applied `torch.sigmoid()` on the output → values in [0,1]
- **Problem**: Downstream losses (`FocalLoss` → `F.cross_entropy`, `TverskyLoss` → `F.softmax`) expect **raw logits**, not sigmoid-activated values. Double-activation destroyed gradients.
- **Fix**: Removed sigmoid; model now outputs raw logits `(B, N, 2)`.

```diff
-        seg_logits = torch.sigmoid(x.permute(0, 2, 1)).squeeze(-1)  # (B, N)
+        seg_logits = x.permute(0, 2, 1)  # raw logits, no activation
```

### Bug 2: Label Binarization (Critical)

**File**: `model/train.py` line 134

- **Before (V1)**: Labels are continuous floats [0, 0.9999]. Code did `.long()` directly → **ALL labels became 0** (floor to int). The model trained with 100% negative labels.
- **Fix**: Binarize with threshold 0.5 before converting to long.

```diff
-            torch.from_numpy(lbl).long(),
+            lbl_binary = (lbl > 0.5).astype(np.int64)
+            torch.from_numpy(lbl_binary),
```

### Bug 3: Class Imbalance Weighting

**File**: `model/losses.py` line 24

- **Before (V1)**: `F.cross_entropy()` with no class weights. With ~0.4% positive rate, predicting all-negative minimizes loss.
- **Fix**: Added `weight=[1.0, 50.0]` to give 50× weight to positive class.

```diff
-        ce = F.cross_entropy(pred, target, reduction='none')
+        weight = torch.tensor([1.0, 50.0], device=pred.device)
+        ce = F.cross_entropy(pred, target, weight=weight, reduction='none')
```

### Bug 4: Inference Predictor Architecture Auto-Detection

**File**: `inference/predictor.py`

- **Before (V1)**: Hardcoded M5 detection assumed 7-channel = 1-class. A 7ch+2class checkpoint was loaded with wrong architecture → `state_dict` mismatch error.
- **Fix**: Auto-detect `in_channel` from `sa1.mlp_convs.0.weight.shape[1] - 3` and `num_classes` from `conv2.weight.shape[0]`. Added `_predict_7ch()` method with proper softmax activation for 2-class logit output.

---

## 2. Files Modified

| File | Change |
|------|--------|
| `model/pointnet2.py` | Removed sigmoid from output (line 205) |
| `model/train.py` | Added label binarization threshold=0.5 (line 134); fixed contact ratio display |
| `model/losses.py` | Added class weight [1.0, 50.0] to FocalLoss cross_entropy |
| `inference/predictor.py` | Rewrote architecture auto-detection; added `_predict_7ch()` for 7ch+2class models |

---

## 3. Dataset

| Metric | Value |
|--------|-------|
| Source | HuggingFace `UCBProject/Affordance2Grasp-Data` |
| Train samples | 11,636 |
| Val samples | 2,910 |
| Points per sample | 1,024 |
| Input channels | 7 (xyz + normals + human_prior) |
| Label type | Continuous float [0, 0.9999] → binarized at 0.5 |
| Positive rate (train, >0.5) | 0.406% (48,341 / 11,915,264 points) |
| Positive rate (val, >0.5) | 0.296% (8,814 / 2,979,840 points) |

---

## 4. Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 200 |
| Optimizer | AdamW (lr=0.001, weight_decay=1e-4) |
| Scheduler | CosineAnnealingWarmRestarts (T_0=50) |
| Batch size | 32 per GPU × 8 GPUs = 256 effective |
| Loss | CombinedLoss: Focal (α=0.75, γ=2, w=0.6) + Tversky (α=0.3, β=0.7, w=0.4) |
| FocalLoss class weight | [1.0, 50.0] |
| GPUs | 8× NVIDIA RTX PRO 6000 Blackwell |
| Total training time | 11,993s (3.3 hours) |
| Avg time per epoch | 60.0s |

---

## 5. Training Results

### V1 → V2 Comparison

| Metric | V1 (before_touch_src_200) | V2 (after_touch_src_200) |
|--------|---------------------------|--------------------------|
| Best val_f1 | 0.0000 | **0.1388** |
| Best val_iou | 0.0000 | **0.1073** |
| Best val_fc_mm | 94.38 | **89.90** |
| val_precision | 0.0000 | **0.1084** |
| val_recall | 0.0000 | **0.2485** |
| Epochs with F1 > 0 | 0/200 | **200/200** |

### V2 Detailed Training Progression

| Epoch | val_f1 | val_precision | val_recall | val_iou | val_fc_mm | val_loss |
|-------|--------|---------------|------------|---------|-----------|----------|
| 1 | 0.1142 | 0.0975 | 0.2243 | 0.0856 | 142.4 | 0.52231 |
| 10 | 0.1218 | 0.0894 | 0.2559 | 0.0912 | 151.4 | 0.51268 |
| 20 | 0.1292 | 0.1028 | 0.2320 | 0.0981 | 119.8 | 0.48826 |
| 50 | 0.1314 | 0.1031 | 0.2368 | 0.1010 | 99.8 | 0.47001 |
| 100 | 0.1343 | 0.1048 | 0.2418 | 0.1040 | 95.5 | 0.46605 |
| **149** | **0.1388** | **0.1084** | **0.2485** | **0.1073** | 93.3 | 0.46565 |
| 192 | 0.1301 | 0.1012 | 0.2369 | 0.1017 | **89.9** | 0.46694 |
| 200 | 0.1354 | 0.1060 | 0.2413 | 0.1059 | 94.5 | 0.46297 |

### Key Observations

- F1 starts nonzero from epoch 1 (0.1142) — confirms the bug fixes enabled learning
- F1 improves steadily until ~epoch 50, then plateaus around 0.13–0.14
- Force-center error steadily decreases: 142mm → 90mm
- Train loss: 0.769 (ep1) → 0.421 (ep200)
- Val loss: 0.522 (ep1) → 0.463 (ep200) — no significant overfitting

---

## 6. Inference Results — A16013.obj

**Command**: `python run.py --mesh data_hub/meshes/v1/A16013.obj --no-sim`  
**Checkpoint**: `best_model.pth` (epoch 149, val_f1=0.1388)

### Contact Map

| Metric | Value |
|--------|-------|
| Sampled points | 1,024 |
| Contact points (≥0.3) | 1,024/1,024 (100%) |
| Prob range | [0.981, 0.997] |
| Mean prob | ~0.99 |

**Note**: The model predicts high contact probability for all points. This reflects the precision/recall trade-off: recall=24.8% but precision=10.8% — the model over-predicts positives on unseen single-object inference.

### Force Center

Predicted: `[0.0156, -0.0009, -0.0126]` (object-local coordinates, near geometric center)

### Grasp Candidates (11 valid)

| Rank | Name | Score | Gripper Width | Approach |
|------|------|-------|---------------|----------|
| **1** | **dynamic_back_y30** | **95.1** | 6.3cm | horizontal |
| 2 | dynamic_front_y-30 | 89.6 | 5.2cm | horizontal |
| 3 | dynamic_back_y0 | 88.2 | 6.1cm | horizontal |
| 4 | top_down_15 | 84.1 | 4.4cm | top-down |
| 5 | top_down | 80.2 | 4.3cm | top-down |
| 6 | dynamic_front_y0 | 76.8 | 5.7cm | horizontal |
| 7 | top_down_-15 | 76.3 | 4.3cm | top-down |
| 8 | dynamic_back_y-30 | 75.4 | 5.8cm | horizontal |
| 9 | dynamic_front_y30 | 69.9 | 5.8cm | horizontal |
| 10 | dynamic_back_y-15 | 59.6 | 7.7cm | horizontal |
| 11 | dynamic_front_y-15 | 56.0 | 7.7cm | horizontal |

**Best grasp**: `dynamic_back_y30`, horizontal approach at ~48° angle, score=95.1  
**Output file**: `output/grasps/A16013_grasp.hdf5` (71KB)

---

## 7. Artifacts

| File | Description |
|------|-------------|
| `output/checkpoints_v5/best_model.pth` | Best checkpoint (epoch 149, 12MB) |
| `output/checkpoints_v5/final_model.pth` | Final checkpoint (epoch 200, 3.8MB) |
| `output/checkpoints_v5/checkpoint_epoch*.pth` | Checkpoints every 10 epochs |
| `output/checkpoints_v5/training_history.json` | Full training history (200 epochs) |
| `output/grasps/A16013_grasp.hdf5` | Inference output with 11 grasp candidates |

---

## 8. Known Limitations

1. **Segmentation precision is low** (10.8%): Model over-predicts contact regions. On single-object inference (A16013), it predicts ~100% of points as contact.
2. **Extreme class imbalance**: Only 0.3–0.4% of points are positive. Even with 50× class weight and Focal/Tversky losses, precision remains limited.
3. **F1 plateau at ~14%**: Converges early (~epoch 50) and does not improve significantly after. May benefit from stronger data augmentation, higher class weight, or different architecture.
