import os
import argparse
import h5py
import numpy as np
from tqdm import tqdm
from datetime import datetime
from typing import Tuple, Optional, Dict

# PhysicsNeMo import
from physicsnemo.datapipes.benchmarks.darcy import Darcy2D

class DarcyMasterGridGenerator:
    """
    Generates high-resolution, normalized, uniform grid data for Darcy Flow 
    and saves it to HDF5 shards.
    
    This creates the 'Master Dataset' containing the full physics resolution.
    Non-uniform sampling will be handled on-the-fly during training.
    
    Parameters
    ----------
    output_dir : str
        Directory to save HDF5 shards.
    resolution : int
        Grid resolution (e.g., 512 for 512x512).
    normaliser : Dict[str, Tuple[float, float]]
        Mean and std dev for normalization, passed to the simulator.
    batch_size : int
        Internal batch size for the PhysicsNeMo generator.
    samples_per_shard : int
        Number of samples to store per HDF5 file.
    """
    
    def __init__(
        self, 
        output_dir: str, 
        resolution: int,
        normaliser: Dict[str, Tuple[float, float]],
        batch_size: int = 32,
        samples_per_shard: int = 2000
    ):
        self.output_dir = output_dir
        self.resolution = resolution
        self.batch_size = batch_size
        self.samples_per_shard = samples_per_shard
        self.date_str = datetime.now().strftime('%Y%m%d')
        
        # Initialize the PhysicsNeMo generator with normalization
        self.simulator = Darcy2D(
            resolution=resolution,
            batch_size=batch_size,
            normaliser=normaliser
        )

    def generate(self, split: str, num_samples: int, seed: int = 42) -> None:
        """
        Run the generation loop for a specific split (train/val/test).
        """
        print(f"Generating {num_samples} {split} samples @ {self.resolution}x{self.resolution}...")
        
        # Ensure reproducibility for the physics generation
        np.random.seed(seed)
        
        shard_idx = 0
        samples_in_shard = 0
        total_written = 0
        
        # Open first shard
        current_h5 = self._create_shard(split, shard_idx)
        
        # Simulator is an infinite iterator, so we control the loop manually
        sim_iter = iter(self.simulator)
        
        pbar = tqdm(total=num_samples, desc=f"  {split.capitalize()}", unit="samples")
        
        try:
            while total_written < num_samples:
                # Fetch batch from simulator (already normalized)
                # batch keys: 'permeability', 'darcy' (pressure)
                # shapes: (B, 1, H, W)
                #
                # IMPORTANT: Darcy2D generates ONE unique random field per
                # iteration, replicated across the batch dimension.
                # We take only the first sample to avoid storing duplicates.
                batch = next(sim_iter)
                
                # Move to CPU numpy — take only first (unique) sample
                k_batch = batch['permeability'][0:1].cpu().numpy()
                p_batch = batch['darcy'][0:1].cpu().numpy()
                
                # Permute to HDF5 friendly format: (B, H, W, C)
                # (B, 1, H, W) -> (B, H, W, 1)
                k_batch = np.transpose(k_batch, (0, 2, 3, 1))
                p_batch = np.transpose(p_batch, (0, 2, 3, 1))
                
                batch_len = k_batch.shape[0]
                
                # Trim if we exceed requested total
                if total_written + batch_len > num_samples:
                    remaining = num_samples - total_written
                    k_batch = k_batch[:remaining]
                    p_batch = p_batch[:remaining]
                    batch_len = remaining
                
                # Check shard capacity and create new shard if full
                if samples_in_shard + batch_len > self.samples_per_shard:
                    current_h5.close()
                    shard_idx += 1
                    samples_in_shard = 0
                    current_h5 = self._create_shard(split, shard_idx)
                    tqdm.write(f"  [Info] Created new shard: {split}/shard-{shard_idx:03d}")
                
                # Write to disk
                self._append_to_shard(current_h5, k_batch, p_batch)
                
                samples_in_shard += batch_len
                total_written += batch_len
                pbar.update(batch_len)
                
        finally:
            pbar.close()
            current_h5.close()

    def _create_shard(self, split: str, idx: int) -> h5py.File:
        """Initialize a new HDF5 file with expandable datasets."""
        folder = os.path.join(self.output_dir, split)
        os.makedirs(folder, exist_ok=True)
        
        fname = os.path.join(
            folder, 
            f"{self.date_str}-darcy-{self.resolution}x{self.resolution}-shard-{idx:03d}.h5"
        )
        f = h5py.File(fname, "w")
        
        # Shape: (N, H, W, 1)
        # We use chunks=(1, H, W, 1) for efficient single-sample reading
        for name in ["permeability", "pressure"]:
            f.create_dataset(
                name,
                shape=(0, self.resolution, self.resolution, 1),
                maxshape=(None, self.resolution, self.resolution, 1),
                chunks=(1, self.resolution, self.resolution, 1),
                dtype='float32',
                compression="gzip"
            )
        # Store normalization info in the HDF5 file's attributes
        if self.simulator.normaliser:
            f.attrs['permeability_mean'] = self.simulator.normaliser['permeability'][0]
            f.attrs['permeability_std'] = self.simulator.normaliser['permeability'][1]
            f.attrs['darcy_mean'] = self.simulator.normaliser['darcy'][0]
            f.attrs['darcy_std'] = self.simulator.normaliser['darcy'][1]

        return f

    def _append_to_shard(self, h5f: h5py.File, k_data: np.ndarray, p_data: np.ndarray):
        """Resize datasets and append new batch."""
        n_new = k_data.shape[0]
        
        dset_k = h5f["permeability"]
        dset_p = h5f["pressure"]
        
        # Resize
        dset_k.resize(dset_k.shape[0] + n_new, axis=0)
        dset_p.resize(dset_p.shape[0] + n_new, axis=0)
        
        # Write
        dset_k[-n_new:] = k_data
        dset_p[-n_new:] = p_data

# ==================================================================================
# CLI Entry Point
# ==================================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Uniform Darcy Master Dataset")
    parser.add_argument("--output", type=str, default="data/darcy_master_normalized", help="Output path for HDF5 shards")
    parser.add_argument("--resolution", type=int, default=512, help="Grid resolution of the simulation")
    parser.add_argument("--train-samples", type=int, default=1000, help="Number of training samples to generate")
    parser.add_argument("--val-samples", type=int, default=50, help="Number of validation samples to generate")
    parser.add_argument("--shard-size", type=int, default=50, help="Number of samples per HDF5 file")
    parser.add_argument("--batch-size", type=int, default=32, help="Internal batch size for the generator")
    
    args = parser.parse_args()

    # Normalization constants from the original config
    # This ensures data is generated in the [-1, 1] range expected by the training script
    normalizer = {
        'permeability': (1.25, 0.75),
        'darcy': (4.52e-2, 2.79e-2)
    }

    generator = DarcyMasterGridGenerator(
        output_dir=args.output,
        resolution=args.resolution,
        normaliser=normalizer,
        batch_size=args.batch_size,
        samples_per_shard=args.shard_size
    )
    
    # 1. Generate Training Data
    generator.generate("train", args.train_samples, seed=42)
    
    # 2. Generate Validation Data (using a different seed for statistical independence)
    generator.generate("val", args.val_samples, seed=12345)
    
    print("\nGeneration Complete. Master data saved to:", args.output)