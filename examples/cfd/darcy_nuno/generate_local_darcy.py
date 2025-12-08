# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
NUNO Data Generation Pipeline for Darcy Flow
---------------------------------------------
This script generates non-uniform point cloud data from the PhysicsNeMo Darcy generator.

CRITICAL: NUNO requires FIXED geometry (same coordinates for all samples) because
the K-D tree domain decomposition is computed once and reused. This script ensures
all samples share the same coordinate locations.

Process:
1. Generate a SINGLE set of non-uniform coordinates (fixed for all samples)
2. Generate high-res uniform grid data on GPU
3. Interpolate values at the fixed coordinates using GPU bilinear sampling
4. Store coordinates once, then store field values per sample

Data Format (HDF5):
- coordinates: (N, 2) - FIXED across all samples, stored once per shard
- permeability: (num_samples, N, 1) - input field
- pressure: (num_samples, N, 1) - target field

Configuration is handled via command-line arguments for flexibility.
"""

import os
import argparse
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Optional, Tuple

# Import the existing Darcy generator from PhysicsNeMo
from physicsnemo.datapipes.benchmarks.darcy import Darcy2D


class FixedGeometryDarcySampler:
    """
    Non-uniform sampler with FIXED coordinates for NUNO compatibility.
    
    NUNO's K-D tree domain decomposition assumes the same point locations
    for all samples. This class ensures coordinates are generated once
    and reused for all field interpolations.
    
    Parameters
    ----------
    num_points : int
        Number of non-uniform sample points
    seed : int, optional
        Random seed for reproducible coordinate generation
    device : torch.device, optional
        Device for computations
    """
    
    def __init__(
        self, 
        num_points: int, 
        seed: int = 42,
        device: Optional[torch.device] = None
    ):
        self.num_points = num_points
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Generate FIXED coordinates once
        # Using a fixed seed ensures reproducibility
        torch.manual_seed(seed)
        
        # Coordinates in [-1, 1] range for grid_sample compatibility
        # Shape: (1, 1, N, 2) for grid_sample
        self.coords_for_sample = torch.rand(
            (1, 1, num_points, 2), 
            device=self.device, 
            dtype=torch.float32
        ) * 2 - 1
        
        # Also store as (N, 2) for saving
        self.coordinates = self.coords_for_sample.squeeze(0).squeeze(0)  # (N, 2)
        
        print(f"Generated fixed geometry with {num_points} points")
        print(f"Coordinate range: [{self.coordinates.min():.3f}, {self.coordinates.max():.3f}]")
    
    def sample_batch(self, batch: dict) -> dict:
        """
        Sample field values at fixed coordinates.
        
        Parameters
        ----------
        batch : dict
            Dictionary with 'permeability' and 'darcy' tensors of shape (B, 1, H, W)
            
        Returns
        -------
        dict
            Dictionary with:
            - 'coordinates': (N, 2) fixed coordinates
            - 'permeability': (B, N, 1) sampled input field
            - 'pressure': (B, N, 1) sampled target field
        """
        k_grid = batch['permeability'].to(self.device)
        p_grid = batch['darcy'].to(self.device)
        
        batch_size = k_grid.shape[0]
        
        # Expand fixed coordinates for batch
        # (1, 1, N, 2) -> (B, 1, N, 2)
        coords_batch = self.coords_for_sample.expand(batch_size, -1, -1, -1)
        
        # Interpolate using grid_sample (GPU-accelerated)
        # align_corners=False is standard for neural operators
        k_sampled = F.grid_sample(k_grid, coords_batch, align_corners=False, mode='bilinear')
        p_sampled = F.grid_sample(p_grid, coords_batch, align_corners=False, mode='bilinear')
        
        # Reshape: (B, C, 1, N) -> (B, N, C)
        k_out = k_sampled.squeeze(2).permute(0, 2, 1)  # (B, N, 1)
        p_out = p_sampled.squeeze(2).permute(0, 2, 1)  # (B, N, 1)
        
        return {
            'coordinates': self.coordinates.cpu().numpy(),  # (N, 2) - same for all
            'permeability': k_out.cpu().numpy(),            # (B, N, 1)
            'pressure': p_out.cpu().numpy()                 # (B, N, 1)
        }


def create_shard(
    root_dir: str, 
    split: str, 
    shard_idx: int, 
    n_points: int,
    coordinates: np.ndarray
) -> h5py.File:
    """
    Create a new HDF5 shard file with fixed coordinates.
    
    Parameters
    ----------
    root_dir : str
        Root directory for data
    split : str
        Data split ('train', 'val', 'test')
    shard_idx : int
        Shard index
    n_points : int
        Number of points per sample
    coordinates : np.ndarray
        Fixed coordinates (N, 2)
        
    Returns
    -------
    h5py.File
        Open HDF5 file handle
    """
    os.makedirs(os.path.join(root_dir, split), exist_ok=True)
    fname = os.path.join(root_dir, split, f"darcy_nuno_{split}_{shard_idx:03d}.h5")
    
    f = h5py.File(fname, "w")
    
    # Store FIXED coordinates once (not per sample)
    f.create_dataset("coordinates", data=coordinates.astype('float32'))
    
    # Create resizable datasets for field values
    # Chunking: (1, n_points, 1) allows efficient single-sample reading
    f.create_dataset(
        "permeability", 
        shape=(0, n_points, 1), 
        maxshape=(None, n_points, 1),
        chunks=(1, n_points, 1), 
        dtype='float32'
    )
    f.create_dataset(
        "pressure", 
        shape=(0, n_points, 1), 
        maxshape=(None, n_points, 1),
        chunks=(1, n_points, 1), 
        dtype='float32'
    )
    
    return f


def append_to_shard(h5f: h5py.File, data_dict: dict) -> None:
    """
    Append sampled data to HDF5 shard.
    
    Parameters
    ----------
    h5f : h5py.File
        Open HDF5 file
    data_dict : dict
        Dictionary with 'permeability' and 'pressure' arrays
    """
    n_new = data_dict['permeability'].shape[0]
    
    for key in ['permeability', 'pressure']:
        dset = h5f[key]
        old_shape = dset.shape[0]
        dset.resize(old_shape + n_new, axis=0)
        dset[old_shape:] = data_dict[key]


def visualize_sample(shard_path: str, sample_idx: int = 0) -> None:
    """
    Visualize a single sample from the generated HDF5 shard.
    
    Creates a side-by-side scatter plot of Permeability and Pressure
    on the non-uniform grid, plus a density visualization.
    
    Parameters
    ----------
    shard_path : str
        Path to the .h5 shard file
    sample_idx : int
        Index of the sample to plot
    """
    if not os.path.exists(shard_path):
        print(f"Error: File {shard_path} not found.")
        return

    print(f"Loading sample {sample_idx} from {shard_path}...")
    
    with h5py.File(shard_path, 'r') as f:
        n_samples = f['permeability'].shape[0]
        if sample_idx >= n_samples:
            print(f"Error: Index {sample_idx} out of range (0-{n_samples-1})")
            return

        # Coordinates are stored once (fixed geometry)
        coords = f['coordinates'][:]          # (N, 2)
        perm = f['permeability'][sample_idx]  # (N, 1)
        press = f['pressure'][sample_idx]     # (N, 1)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot Permeability
    sc1 = axes[0].scatter(
        coords[:, 0], coords[:, 1], 
        c=perm[:, 0], s=5, cmap='viridis', alpha=0.8
    )
    axes[0].set_title("Permeability (Input)")
    axes[0].set_xlim(-1, 1)
    axes[0].set_ylim(-1, 1)
    axes[0].set_aspect('equal')
    plt.colorbar(sc1, ax=axes[0])

    # Plot Pressure
    sc2 = axes[1].scatter(
        coords[:, 0], coords[:, 1], 
        c=press[:, 0], s=5, cmap='magma', alpha=0.8
    )
    axes[1].set_title("Pressure (Target)")
    axes[1].set_xlim(-1, 1)
    axes[1].set_ylim(-1, 1)
    axes[1].set_aspect('equal')
    plt.colorbar(sc2, ax=axes[1])

    # Plot point density (2D histogram)
    axes[2].hist2d(coords[:, 0], coords[:, 1], bins=32, cmap='Blues')
    axes[2].set_title("Point Density")
    axes[2].set_xlim(-1, 1)
    axes[2].set_ylim(-1, 1)
    axes[2].set_aspect('equal')

    plt.suptitle(f"Sample {sample_idx} from {os.path.basename(shard_path)}")
    plt.tight_layout()
    plt.savefig(f"sample_visualization_{sample_idx}.png", dpi=150)
    plt.show()


def generate_data(
    output_root: str = "data/nuno_darcy",
    total_train_samples: int = 10000,
    total_val_samples: int = 1000,
    samples_per_shard: int = 2000,
    num_points: int = 1024,
    resolution: int = 256,
    batch_size: int = 32,
    seed: int = 42
) -> None:
    """
    Main data generation function.
    
    Parameters
    ----------
    output_root : str
        Output directory
    total_train_samples : int
        Number of training samples
    total_val_samples : int
        Number of validation samples
    samples_per_shard : int
        Samples per HDF5 file
    num_points : int
        Non-uniform points per sample
    resolution : int
        Resolution of uniform grid for generation
    batch_size : int
        Batch size for generation
    seed : int
        Random seed for coordinate generation
    """
    print("=" * 60)
    print("NUNO Darcy Flow Data Generator")
    print("=" * 60)
    print(f"Output: {output_root}")
    print(f"Train samples: {total_train_samples}")
    print(f"Val samples: {total_val_samples}")
    print(f"Points per sample: {num_points}")
    print(f"Resolution: {resolution}")
    print(f"Seed: {seed}")
    print("=" * 60)
    
    # Initialize uniform generator (PhysicsNeMo)
    uniform_loader = Darcy2D(
        resolution=resolution,
        batch_size=batch_size,
        normaliser=None  # Store raw values, normalize during training
    )
    
    # Initialize fixed-geometry sampler
    sampler = FixedGeometryDarcySampler(num_points, seed=seed)
    
    # Get fixed coordinates for all shards
    fixed_coords = sampler.coordinates.cpu().numpy()
    
    # Generate training data
    print("\n--- Generating Training Data ---")
    _generate_split(
        uniform_loader, sampler, fixed_coords,
        output_root, "train", total_train_samples, samples_per_shard, num_points
    )
    
    # Generate validation data
    print("\n--- Generating Validation Data ---")
    _generate_split(
        uniform_loader, sampler, fixed_coords,
        output_root, "val", total_val_samples, samples_per_shard, num_points
    )
    
    print("\n" + "=" * 60)
    print("Data generation complete!")
    print("=" * 60)


def _generate_split(
    uniform_loader,
    sampler: FixedGeometryDarcySampler,
    fixed_coords: np.ndarray,
    output_root: str,
    split: str,
    total_samples: int,
    samples_per_shard: int,
    num_points: int
) -> None:
    """Generate data for a single split (train/val/test)."""
    shard_idx = 0
    samples_in_shard = 0
    total_written = 0
    
    current_h5 = create_shard(output_root, split, shard_idx, num_points, fixed_coords)
    
    uniform_iter = iter(uniform_loader)
    
    try:
        while total_written < total_samples:
            # Get uniform batch
            batch = next(uniform_iter)
            
            # Sample at fixed coordinates
            sampled_data = sampler.sample_batch(batch)
            batch_len = sampled_data['permeability'].shape[0]
            
            # Don't exceed total samples
            if total_written + batch_len > total_samples:
                remaining = total_samples - total_written
                sampled_data['permeability'] = sampled_data['permeability'][:remaining]
                sampled_data['pressure'] = sampled_data['pressure'][:remaining]
                batch_len = remaining
            
            # Sharding logic
            if samples_in_shard + batch_len > samples_per_shard:
                current_h5.close()
                shard_idx += 1
                samples_in_shard = 0
                current_h5 = create_shard(output_root, split, shard_idx, num_points, fixed_coords)
                print(f"  Opened shard {shard_idx}")
            
            # Write to disk
            append_to_shard(current_h5, sampled_data)
            
            samples_in_shard += batch_len
            total_written += batch_len
            
            if total_written % 500 == 0 or total_written == total_samples:
                print(f"  Progress: {total_written}/{total_samples} samples")
                
    finally:
        current_h5.close()


def main():
    parser = argparse.ArgumentParser(description="Generate NUNO Darcy Flow Data")
    parser.add_argument("--output", type=str, default="data/nuno_darcy", help="Output directory")
    parser.add_argument("--train-samples", type=int, default=1000, help="Training samples")
    parser.add_argument("--val-samples", type=int, default=100, help="Validation samples")
    parser.add_argument("--samples-per-shard", type=int, default=2000, help="Samples per HDF5 file")
    parser.add_argument("--num-points", type=int, default=1024, help="Non-uniform points")
    parser.add_argument("--resolution", type=int, default=256, help="Source grid resolution")
    parser.add_argument("--batch-size", type=int, default=32, help="Generation batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for coordinates")
    parser.add_argument("--visualize", type=str, default=None, help="Path to shard to visualize")
    parser.add_argument("--viz-idx", type=int, default=0, help="Sample index to visualize")
    
    args = parser.parse_args()
    
    if args.visualize:
        visualize_sample(args.visualize, args.viz_idx)
    else:
        generate_data(
            output_root=args.output,
            total_train_samples=args.train_samples,
            total_val_samples=args.val_samples,
            samples_per_shard=args.samples_per_shard,
            num_points=args.num_points,
            resolution=args.resolution,
            batch_size=args.batch_size,
            seed=args.seed
        )


if __name__ == "__main__":
    main()