# Pretrained vs 200-Epoch Checkpoint Comparison

## Validation Summary

| Metric | Pretrained | 200-epoch | Delta (200 - pre) |
|---|---:|---:|---:|
| accuracy | 97.78% | 0.42% | -97.36% |
| precision | 8.26% | 0.42% | -7.84% |
| recall | 42.97% | 100.00% | 57.03% |
| f1 | 13.85% | 0.83% | -13.02% |
| iou | 7.44% | 0.42% | -7.03% |

| Metric | Pretrained | 200-epoch | Delta (200 - pre) |
|---|---:|---:|---:|
| macro_f1 | 0.31% | 0.47% | 0.15% |
| macro_iou | 0.23% | 0.42% | 0.18% |

## Threshold and Force Center

- Label positive threshold for binarization: 0.10
- Validation positive rate after binarization: 0.42%
- Pretrained best threshold: 0.50, F1: 13.85%
- 200-epoch best threshold: 0.05, F1: 0.83%
- Force-center mean error (mm): pretrained 355.17, 200-epoch 90.40

## Continuous-Label Fit

| Metric | Pretrained | 200-epoch | Delta (200 - pre) |
|---|---:|---:|---:|
| mae | 0.283315 | 0.269459 | -0.013856 |
| mse | 0.082694 | 0.072896 | -0.009798 |
| pearson_r | 0.214521 | 0.049801 | -0.164720 |

## Top-K Retrieval (Binary Labels)

| Metric | Pretrained | 200-epoch | Delta (200 - pre) |
|---|---:|---:|---:|
| top_1_percent_precision | 7.27% | 4.42% | -2.85% |
| top_1_percent_recall | 17.45% | 10.61% | -6.84% |
| top_2_percent_precision | 8.23% | 3.63% | -4.59% |
| top_2_percent_recall | 39.53% | 17.46% | -22.07% |
| top_5_percent_precision | 7.01% | 1.45% | -5.56% |
| top_5_percent_recall | 84.22% | 17.46% | -66.76% |

## Mesh Inference Benchmark

| Metric | Pretrained | 200-epoch | Delta (200 - pre) |
|---|---:|---:|---:|
| time_s_mean | 0.0273 | 0.0249 | -0.0024 |
| time_s_median | 0.0289 | 0.0235 | -0.0053 |
| contact_0.3_mean | 1001.8333 | 0.0000 | -1001.8333 |
| contact_0.5_mean | 627.5833 | 0.0000 | -627.5833 |

## 200-Epoch Training Snapshot

- Epoch 200: val_f1=0.0, val_iou=0.0, val_fc_mm=91.52
- Best epoch in history: epoch=1, val_f1=0.0, val_iou=0.0, val_fc_mm=179.17
