# NUNO Training Pipeline for Darcy Flow

This directory contains a complete training pipeline for the Non-Uniform Neural Operator (NUNO) on the 2D Darcy flow benchmark with non-uniform point cloud data.

## Overview

The pipeline consists of four main components:

1. **Data Generation** (`generate_nuno_darcy.py`) - Creates non-uniform point cloud data
2. **Dataset** (`nuno_darcy_dataset.py`) - PyTorch Dataset for loading the data
3. **Training** (`train_nuno_darcy.py`) - Main training script
4. **Validation** (`nuno_validator.py`) - Validation with heatmap visualization
5. **Configuration** (`nuno_config.yaml`) - Hyperparameters and settings

## Critical Design Decision: Fixed Geometry

**IMPORTANT**: NUNO requires the same point locations (coordinates) for all samples because:

1. The K-D tree domain decomposition is computed once during `model.setup_domain()`
2. The grid sizes for each subdomain are determined by the point distribution
3. Changing coordinates between samples would require rebuilding the entire decomposition

The original `generate_local_darcy.py` generated different random coordinates for each sample, which is incompatible with NUNO. The fixed version generates coordinates once and reuses them for all samples.

## File Descriptions

### `generate_nuno_darcy.py`
Generates non-uniform Darcy flow data with **fixed geometry**.

**Key Features:**
- Uses `FixedGeometryDarcySampler` class to ensure consistent coordinates
- Samples from PhysicsNeMo's `Darcy2D` generator at random (but fixed) locations
- Stores coordinates once per shard, not per sample
- Supports train/val splits with configurable sizes

**Usage:**
```bash
# Generate default dataset (10k train, 1k val, 1024 points)
python generate_nuno_darcy.py

# Custom generation
python generate_nuno_darcy.py \
    --output data/nuno_darcy \
    --train-samples 10000 \
    --val-samples 1000 \
    --num-points 1024 \
    --seed 42

# Visualize generated data
python generate_nuno_darcy.py --visualize data/nuno_darcy/train/darcy_nuno_train_000.h5
```

**HDF5 Data Format:**
```
shard.h5
├── coordinates    (N, 2)           # Fixed point locations [-1, 1]
├── permeability   (num_samples, N, 1)  # Input field
└── pressure       (num_samples, N, 1)  # Target field
```

### `nuno_darcy_dataset.py`
PyTorch Dataset class for loading the generated data.

**Key Features:**
- Loads HDF5 shards with fixed geometry
- Supports normalization of input/output fields
- Memory-efficient: loads samples on-demand
- Provides `get_coordinates()` method for domain decomposition setup

**Usage:**
```python
from nuno_darcy_dataset import create_dataloader

normalizer = {
    'permeability': (1.25, 0.75),
    'darcy': (4.52e-2, 2.79e-2)
}

train_loader, train_dataset = create_dataloader(
    data_dir="data/nuno_darcy",
    split="train",
    batch_size=16,
    normalizer=normalizer
)

# Get coordinates for NUNO setup
coordinates = train_dataset.get_coordinates()  # (N, 2)
```

### `train_nuno_darcy.py`
Main training script following PhysicsNeMo patterns.

**Key Features:**
- Hydra configuration management
- PhysicsNeMo logging (PythonLogger, LaunchLogger)
- Checkpoint save/load support
- StaticCaptureTraining for optimized training
- Validation with visualization

**Usage:**
```bash
# Run with default config
python train_nuno_darcy.py

# Override config values
python train_nuno_darcy.py training.batch_size=32 training.max_pseudo_epochs=50

# Quick test run
python train_nuno_darcy.py training.max_pseudo_epochs=2 training.pseudo_epoch_sample_size=64
```

**Training Flow:**
1. Load train/val data from HDF5 shards
2. Extract fixed coordinates from dataset
3. Create NUNO model and call `setup_domain(coordinates)`
4. Train with MSE loss, Adam optimizer, LR scheduling
5. Validate periodically with heatmap visualization

### `nuno_validator.py`
Validator for point cloud predictions with heatmap visualization.

**Key Features:**
- Creates 4-panel figure: Input, Ground Truth, Prediction, Relative Error
- Uses Delaunay triangulation for smooth heatmap rendering
- Falls back to scatter plots for sparse data
- Compatible with PhysicsNeMo's LaunchLogger

**Visualization Output:**
```
[Input (Permeability)] [Ground Truth] [Prediction] [Relative Error]
```

### `nuno_config.yaml`
Configuration file with all hyperparameters.

**Sections:**
- `data`: Data directory and loader settings
- `arch`: Model architecture (NUNO + decoder)
- `normalizer`: Data normalization parameters
- `scheduler`: Learning rate schedule
- `training`: Batch size, epochs, checkpoint frequency
- `validation`: Validation frequency and visualization settings

## Quick Start Guide

### Step 1: Generate Data
```bash
# Generate a small test dataset (faster)
python generate_nuno_darcy.py \
    --train-samples 2000 \
    --val-samples 200 \
    --num-points 1024
```

### Step 2: Verify Data
```bash
python generate_nuno_darcy.py --visualize data/nuno_darcy/train/darcy_nuno_train_000.h5
```

### Step 3: Run Training
```bash
# Quick test (5-10 minutes)
python train_nuno_darcy.py

# Full training (adjust config as needed)
python train_nuno_darcy.py \
    training.max_pseudo_epochs=100 \
    training.pseudo_epoch_sample_size=4096 \
    training.batch_size=32
```

### Step 4: Check Results
- Checkpoints: `./checkpoints/`
- Validation plots: Logged via LaunchLogger
- Console output: Loss values per epoch

## Configuration for Quick Testing

The default `nuno_config.yaml` is configured for quick testing:

```yaml
training:
  batch_size: 16
  max_pseudo_epochs: 8
  pseudo_epoch_sample_size: 512

validation:
  sample_size: 64
  validation_pseudo_epochs: 2
```

This should complete in 5-10 minutes on a GPU and demonstrate:
- Decreasing training loss
- Validation visualization output
- Checkpoint saving

## Configuration for Full Training

For proper training, modify the config:

```yaml
training:
  batch_size: 32
  max_pseudo_epochs: 100
  pseudo_epoch_sample_size: 4096

validation:
  sample_size: 256
  validation_pseudo_epochs: 10
```

## Directory Structure

After running the pipeline:
```
project/
├── data/
│   └── nuno_darcy/
│       ├── train/
│       │   ├── darcy_nuno_train_000.h5
│       │   └── ...
│       └── val/
│           └── darcy_nuno_val_000.h5
├── checkpoints/
│   └── model_epoch_*.pt
├── outputs/                    # Hydra outputs
│   └── YYYY-MM-DD/
│       └── HH-MM-SS/
│           ├── .hydra/
│           └── nuno_validation_step_*.png
├── generate_nuno_darcy.py
├── nuno_darcy_dataset.py
├── nuno_validator.py
├── train_nuno_darcy.py
└── nuno_config.yaml
```

## Troubleshooting

### CUDA Out of Memory
Reduce batch size:
```bash
python train_nuno_darcy.py training.batch_size=8
```

### No Data Found
Ensure data was generated first:
```bash
python generate_nuno_darcy.py --output data/nuno_darcy
```

### Visualization Errors
If triangulation fails (e.g., collinear points), disable it:
```yaml
validation:
  use_triangulation: false
```

### Windows-specific Issues
- Set `num_workers: 0` in config (already default)
- Ensure HDF5 files are closed properly before re-running

## Expected Results

After training for ~8 pseudo-epochs with the test config:
- **Training loss**: Should decrease from ~1.0 to ~0.3-0.5
- **Validation loss**: Should decrease similarly
- **Visualization**: Prediction should roughly match ground truth

For full training (~100 epochs), expect:
- **Training loss**: ~0.01-0.05
- **Validation loss**: ~0.02-0.10
- **Visualization**: Close match with small relative errors

## Key Differences from FNO Training

| Aspect | FNO | NUNO |
|--------|-----|------|
| Input format | Grid (B, C, H, W) | Point cloud (B, N, C) |
| Data generation | On-the-fly | Pre-generated HDF5 |
| Geometry | Fixed grid resolution | Fixed point locations |
| Domain setup | None | `model.setup_domain(coords)` |
| Validation viz | imshow | tricontourf/scatter |

## References

- NUNO Paper: Liu et al., "NUNO: A General Framework for Learning Parametric PDEs with Non-Uniform Data" (ICML 2023)
- PhysicsNeMo: https://github.com/NVIDIA/PhysicsNeMo
