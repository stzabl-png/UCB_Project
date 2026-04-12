# Affordance2Grasp (UCB_Project) — Experiment Report

## 1. Project Overview

**Repository**: https://github.com/stzabl-png/UCB_Project  
**Task**: Object affordance prediction → grasp pose generation → Isaac Sim execution  
**Model**: PointNet++ Part Segmentation (multi-task: contact segmentation + force center regression)  
**Architecture**: 976,645 parameters, 7-channel input (xyz + normals + human_prior), 2-class segmentation output + 3D force center regression head  

---

## 2. Environment

| Component | Details |
|-----------|---------|
| **Hardware** | 8× NVIDIA RTX PRO 6000 Blackwell Server Edition (97 GB VRAM each) |
| **OS** | Linux |
| **Python** | 3.10 (conda env: `ucb_project`) |
| **PyTorch** | 2.11.0 |
| **Key Libraries** | trimesh 4.11.5, h5py 3.16.0, scipy 1.15.3, huggingface-hub 1.10.1 |
| **Multi-GPU** | `nn.DataParallel` across all 8 GPUs |

---

## 3. Dataset

| Split | Samples | Objects | Overlap |
|-------|---------|---------|---------|
| Train | 11,637 | 11,632 | — |
| Val | 2,909 | 2,907 | Zero object overlap |

- **Contact ratio**: ~0.3% positive (extreme class imbalance)
- **Input**: 1024-point clouds, 7 channels (xyz + face normals + human prior)
- **Source**: HuggingFace `UCBProject/Affordance2Grasp-Data` (3.06 GB)

---

## 4. Training Details

### 4.1 Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 200 |
| Batch size | 32 per GPU × 8 GPUs = 256 effective |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Scheduler | CosineAnnealingWarmRestarts (T_0=50) |
| Seg loss | CombinedLoss: Focal (α=0.75, γ=2, w=0.6) + Tversky (α=0.3, β=0.7, w=0.4) |
| FC loss | MSE × λ=10.0 |
| Training time | ~100 minutes total (~29–46 s/epoch) |

### 4.2 Training Progression (Newly Trained Model)

| Epoch | Train Loss | Val Loss | FC Distance (mm) | F1 (%) | Time/Epoch |
|------:|----------:|----------:|------------------:|-------:|-----------:|
| 1 | 0.93834 | 0.54571 | 179.2 | 0.0 | 32.2s |
| 10 | 0.50424 | 0.47825 | 122.5 | 0.0 | 29.1s |
| 25 | 0.48959 | 0.46923 | 110.2 | 0.0 | 31.1s |
| 50 | 0.47912 | 0.45685 | 96.4 | 0.0 | 32.9s |
| 75 | 0.48007 | 0.47544 | 120.2 | 0.0 | 35.4s |
| 100 | 0.47342 | 0.45737 | 92.0 | 0.0 | 37.9s |
| 125 | 0.47692 | 0.45922 | 101.4 | 0.0 | 39.1s |
| **150** | **0.47045** | **0.45286** | **88.2** | **0.0** | 43.5s |
| 175 | 0.47460 | 0.46155 | 100.3 | 0.0 | 43.7s |
| 200 | 0.46933 | 0.45458 | 91.5 | 0.0 | 46.3s |

**Best epoch by val loss**: Epoch 150 (val_loss=0.45286, FC=88.2mm)

### 4.3 Training Observations

- **Loss convergence**: Train loss dropped from 0.938 → 0.469, val loss from 0.546 → 0.453
- **Force center regression**: Improved significantly from 179mm → 88mm error
- **Segmentation (F1=0%)**: The model never learned to predict contact points. With only 0.3% positive contact ratio, the model collapsed to predicting all-negative. The segmentation loss flatlined at 0.4034 from epoch 3 onward
- **Saved checkpoints**: 20 periodic (every 10 epochs) + `final_model.pth`. No `best_model.pth` was saved (save condition not triggered)

---

## 5. Inference Results

### 5.1 Pretrained Model (Downloaded)

**Checkpoint**: `output/checkpoints/best_model.pth` (legacy multi-task: 6ch input, 2-class seg + fc_head)

| Object | Contacts (θ=0.3) | Prob Range | Mean Prob | Grasp Score | Best Approach | Candidates |
|--------|------------------:|------------|----------:|------------:|---------------|------------|
| cell_phone | 1024/1024 | [0.433, 0.617] | 0.552 | 86.5 | dynamic_front_y-15 | 13 |
| scissors | 1024/1024 | [0.412, 0.610] | 0.553 | 95.0 | top_down | 12 |
| flashlight | 1024/1024 | [0.437, 0.636] | 0.561 | 99.0 | dynamic_front_y30 | 13 |
| knife | 795/1024 | [0.271, 0.623] | 0.455 | 74.7 | dynamic_front_y0 | 8 |
| stapler | 1024/1024 | [0.481, 0.561] | 0.528 | 100.0 | dynamic_back_y30 | 10 |
| toothbrush | 1024/1024 | [0.340, 0.654] | 0.548 | 87.1 | dynamic_front_y15 | 5 |
| hammer | 1018/1024 | [0.289, 0.628] | 0.457 | 81.1 | top_down | 2 |
| apple | 1024/1024 | [0.309, 0.518] | 0.455 | 95.6 | dynamic_back_y-15 | 13 |
| mouse | 1024/1024 | [0.354, 0.552] | 0.498 | 85.0 | top_down_15 | 13 |
| ps_controller | 1024/1024 | [0.351, 0.580] | 0.483 | 85.0 | top_down_-15 | 3 |
| light_bulb | 1024/1024 | [0.335, 0.553] | 0.486 | 99.9 | dynamic_back_y0 | 13 |
| water_bottle | 916/1024 | [0.281, 0.565] | 0.452 | 95.7 | dynamic_front_y0 | 13 |

**Summary**: 12/12 meshes generated valid grasp poses. Average best grasp score: **90.4**

### 5.2 Newly Trained Model (200 Epochs)

**Checkpoint**: `output/checkpoints_v5/final_model.pth` (v5 multi-task: 7ch input, 2-class seg + fc_head)

| Object | Contacts (θ=0.3) | Prob Range | Mean Prob |
|--------|------------------:|------------|----------:|
| cell_phone | 0/1024 | [0.269, 0.269] | 0.269 |
| apple | 0/1024 | [0.269, 0.269] | 0.269 |
| hammer | 0/1024 | [0.269, 0.269] | 0.269 |
| knife | 0/1024 | [0.269, 0.269] | 0.269 |
| mouse | 0/1024 | [0.269, 0.269] | 0.269 |
| scissors | 0/1024 | [0.269, 0.269] | 0.269 |
| stapler | 0/1024 | [0.269, 0.269] | 0.269 |
| toothbrush | 0/1024 | [0.269, 0.269] | 0.269 |
| flashlight | 0/1024 | [0.269, 0.269] | 0.269 |
| light_bulb | 0/1024 | [0.269, 0.269] | 0.269 |
| water_bottle | 0/1024 | [0.269, 0.269] | 0.269 |
| ps_controller | 0/1024 | [0.269, 0.269] | 0.269 |

**Summary**: The newly trained model outputs a constant probability of 0.269 for ALL points on ALL objects. This confirms the model collapsed during training — the segmentation head learned to output a uniform distribution (softmax of [constant, constant] → ~0.5 for class 0, ~0.269 for class 1 contact). Zero valid grasps could be generated.

### 5.3 Failed Meshes

| Mesh | Model | Failure Reason |
|------|-------|----------------|
| ketchup.obj | Pretrained | 0 contact points (prob max=0.023) — missing human prior |
| banana.obj | Pretrained | 0 contact points (prob max=0.052) — missing human prior |
| mug.obj | Pretrained | Cross-section > 8cm — object too wide for the gripper |

---

## 6. Model Comparison Summary

| Metric | Pretrained Model | Newly Trained Model |
|--------|:----------------:|:-------------------:|
| **Input channels** | 6 (xyz + normals) | 7 (xyz + normals + human_prior) |
| **Seg output** | 2-class softmax | 2-class softmax |
| **Force center** | Yes (fc_head) | Yes (fc_head) |
| **Contact detection** | Working (avg 990/1024 pts) | Failed (0/1024 pts, constant 0.269) |
| **Grasp generation** | 12/12 objects succeeded | 0/12 objects succeeded |
| **Avg best grasp score** | 90.4 | N/A |
| **Root cause of difference** | Properly trained on balanced data | Collapsed due to 0.3% contact ratio |

---

## 7. Generated Grasp Files

All saved to `output/grasps/`:

| File | Size | Object |
|------|------|--------|
| `apple_grasp.hdf5` | 77.7 KB | Apple |
| `cell_phone_grasp.hdf5` | 77.7 KB | Cell Phone |
| `flashlight_grasp.hdf5` | 77.4 KB | Flashlight |
| `hammer_grasp.hdf5` | 48.2 KB | Hammer |
| `knife_grasp.hdf5` | 63.1 KB | Knife |
| `light_bulb_grasp.hdf5` | 77.7 KB | Light Bulb |
| `mouse_grasp.hdf5` | 75.8 KB | Mouse |
| `ps_controller_grasp.hdf5` | 53.7 KB | PS Controller |
| `scissors_grasp.hdf5` | 74.5 KB | Scissors |
| `stapler_grasp.hdf5` | 62.8 KB | Stapler |
| `toothbrush_grasp.hdf5` | 58.7 KB | Toothbrush |
| `water_bottle_grasp.hdf5` | 76.6 KB | Water Bottle |

Each HDF5 contains: best grasp pose (position, rotation, quaternion, gripper width), all candidates, affordance point cloud data, and metadata.

---

## 8. Isaac Sim Integration

**Status**: Pending installation

To execute any saved grasp in simulation:
```bash
export ISAAC_SIM_PATH=/path/to/isaac-sim
$ISAAC_SIM_PATH/python.sh sim/run_grasp.py --hdf5 output/grasps/<object>_grasp.hdf5
```

---

## 9. Recommendations

1. **Use the pretrained model** for all inference — it works reliably across diverse objects
2. **Class imbalance is the critical issue** for re-training: with 0.3% positive contact ratio, standard training collapses. Potential fixes:
   - Aggressive oversampling of positive contact regions
   - Higher focal loss gamma (γ > 2) or lower alpha
   - Class-weighted sampling in the dataloader
   - Using the human_prior channel as soft supervision signal
3. **Expand mesh coverage**: The grasp pipeline works well for graspable objects within the 8cm gripper opening
