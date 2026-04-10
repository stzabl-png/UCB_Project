# Affordance2Grasp

Learning where to grasp objects by combining human contact priors with robot simulation.

**Pipeline**: Video → Hand Reconstruction → Contact Map → Affordance Model → Grasp Pose → Sim Execution

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Upstream: Human Prior Generation                           │
│  Video → HaWoR/HaPTIC → MANO hand vertices                 │
│  Object mesh + GT pose → Spatial alignment                  │
│  KDTree proximity → ContactMap (human_prior)                │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Downstream: Robot GT via Simulation                        │
│  Human prior → Sample grasp candidates                      │
│  Isaac Sim → Physics-based grasp evaluation                 │
│  Successful grasps → Contact region → robot_gt labels       │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Affordance Model (PointNet++ Multi-Task)                   │
│  Input:  point_cloud (N,3) + normals (N,3) + human_prior    │
│  Output: contact segmentation (N,) + force center (3,)      │
│  Loss:   Focal+Tversky + λ·MSE                             │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install

```bash
git clone https://github.com/<your-org>/Affordance2Grasp.git
cd Affordance2Grasp
pip install -r requirements.txt
```

### 2. Download Data & Checkpoints

```bash
pip install huggingface-hub
python download_data.py
```

This downloads from [UCBProject/Affordance2Grasp-Data](https://huggingface.co/datasets/UCBProject/Affordance2Grasp-Data):
- `data_hub/` — meshes, human priors, training data
- `output/dataset/` — preprocessed HDF5 train/val splits
- `output/checkpoints/` — trained model weights

### 3. Train

```bash
# Full training (200 epochs)
python run.py --train --epochs 200

# Or directly:
python -m model.train --epochs 200 --batch_size 16
```

### 4. Inference

```bash
# Generate grasp pose for an object
python run.py --mesh data_hub/meshes/grasp_collection/A16013.obj --no-sim
```

### 5. Simulation (requires Isaac Sim)

```bash
# Set Isaac Sim path
export ISAAC_SIM_PATH=/path/to/isaac-sim

# Full pipeline: inference + sim
python run.py --mesh path/to/object.obj
```

## Project Structure

```
Affordance2Grasp/
├── run.py                  # Unified CLI entry point
├── config.py               # Global configuration (paths + params)
├── download_data.py        # HuggingFace data downloader
│
├── model/                  # PointNet++ model
│   ├── pointnet2.py        #   Architecture (multi-task seg + force center)
│   ├── train.py            #   Training script
│   ├── losses.py           #   Focal + Tversky loss
│   └── metrics.py          #   Evaluation metrics + visualization
│
├── inference/              # Inference + grasp pose generation
│   ├── predictor.py        #   AffordancePredictor API
│   └── grasp_pose.py       #   Mesh → grasp pose HDF5
│
├── data/                   # Data processing pipeline
│   ├── build_dataset.py    #   Build HDF5 training dataset
│   ├── aggregate_prior.py  #   Multi-sequence contact aggregation
│   ├── aggregate_robot_gt.py  # Sim results → robot GT labels
│   └── video_to_training_data.py  # Video → training data bridge
│
├── sim/                    # Isaac Sim execution
│   ├── run_grasp.py        #   Franka + cuRobo grasping
│   └── env_config/         #   Simulation environment setup
│
├── tools/                  # Auxiliary tools
│   ├── gen_m5_training_data.py  # Generate M5 training data
│   └── random_grasp_sampler.py  # Random grasp candidate sampling
│
├── data_hub/               # Data center (from HuggingFace)
│   ├── meshes/             #   Object meshes (v1, v2, grasp_collection)
│   ├── human_prior/        #   Human contact priors (HDF5)
│   ├── training/           #   Training data (human_prior + robot_gt)
│   └── registry.json       #   Object registry
│
└── output/                 # Generated outputs
    ├── dataset/            #   HDF5 train/val splits
    ├── checkpoints/        #   Model weights
    └── grasps/             #   Generated grasp poses
```

## Training Data Format

Each training sample (HDF5):

| Field | Shape | Description |
|-------|-------|-------------|
| `point_cloud` | (N, 3) | Sampled surface points |
| `normals` | (N, 3) | Surface normals |
| `human_prior` | (N,) | Human contact probability [0,1] — model input |
| `robot_gt` | (N,) | Robot grasp success label [0,1] — supervision |
| `force_center` | (3,) | Grasp force center coordinate |

## Configuration

All paths auto-detect from the project root. External tools (only needed for data generation, not training) use environment variables:

```bash
# Optional: copy and edit
cp .env.example .env
```

See [.env.example](.env.example) for available options.

## License

MIT
