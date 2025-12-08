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
NUNO Darcy Flow Dataset
-----------------------
PyTorch Dataset for loading non-uniform Darcy flow data for NUNO training.

Key Features:
- Loads HDF5 shards with fixed geometry (same coordinates for all samples)
- Supports normalization of input/output fields
- Memory-efficient: loads samples on-demand from disk
- Provides coordinates separately for domain decomposition setup
"""

import os
import glob
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


class NUNODarcyDataset(Dataset):
    """
    Dataset for non-uniform Darcy flow data.
    
    This dataset loads point cloud data where all samples share the same
    coordinate locations (fixed geometry), which is required for NUNO's
    K-D tree domain decomposition.
    
    Parameters
    ----------
    data_dir : str
        Directory containing HDF5 shards
    split : str
        Data split ('train', 'val', 'test')
    normalizer : dict, optional
        Normalization parameters with keys 'permeability' and 'darcy',
        each containing (mean, std) tuple
    
    Attributes
    ----------
    coordinates : Tensor
        Fixed coordinates for all samples, shape (N, 2)
    num_points : int
        Number of points per sample
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        normalizer: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        super().__init__()
        
        self.data_dir = data_dir
        self.split = split
        self.normalizer = normalizer or {
            'permeability': (0.0, 1.0),
            'darcy': (0.0, 1.0)
        }
        
        # Find all shard files
        split_dir = os.path.join(data_dir, split)
        self.shard_files = sorted(glob.glob(os.path.join(split_dir, "*.h5")))
        
        if len(self.shard_files) == 0:
            raise ValueError(f"No HDF5 files found in {split_dir}")
        
        print(f"Found {len(self.shard_files)} shard(s) for split '{split}'")
        
        # Load coordinates from first shard (they're the same across all shards)
        with h5py.File(self.shard_files[0], 'r') as f:
            self.coordinates = torch.from_numpy(f['coordinates'][:]).float()
            self.num_points = self.coordinates.shape[0]
        
        print(f"Loaded fixed geometry: {self.num_points} points")
        
        # Build index mapping: global_idx -> (shard_idx, local_idx)
        self._build_index()
    
    def _build_index(self) -> None:
        """Build mapping from global index to (shard_idx, local_idx)."""
        self.index_map: List[Tuple[int, int]] = []
        self.shard_lengths: List[int] = []
        
        for shard_idx, shard_path in enumerate(self.shard_files):
            with h5py.File(shard_path, 'r') as f:
                n_samples = f['permeability'].shape[0]
                self.shard_lengths.append(n_samples)
                
                for local_idx in range(n_samples):
                    self.index_map.append((shard_idx, local_idx))
        
        self.total_samples = len(self.index_map)
        print(f"Total samples: {self.total_samples}")
    
    def __len__(self) -> int:
        return self.total_samples
    
    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """
        Get a single sample.
        
        Parameters
        ----------
        idx : int
            Global sample index
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'coordinates': (N, 2) point coordinates
            - 'permeability': (N, 1) normalized input field
            - 'pressure': (N, 1) normalized target field
        """
        shard_idx, local_idx = self.index_map[idx]
        shard_path = self.shard_files[shard_idx]
        
        with h5py.File(shard_path, 'r') as f:
            perm = torch.from_numpy(f['permeability'][local_idx]).float()  # (N, 1)
            press = torch.from_numpy(f['pressure'][local_idx]).float()     # (N, 1)
        
        # Normalize
        perm_mean, perm_std = self.normalizer['permeability']
        press_mean, press_std = self.normalizer['darcy']
        
        perm = (perm - perm_mean) / perm_std
        press = (press - press_mean) / press_std
        
        return {
            'coordinates': self.coordinates,  # (N, 2)
            'permeability': perm,              # (N, 1)
            'pressure': press                  # (N, 1)
        }
    
    def get_coordinates(self) -> Tensor:
        """
        Get the fixed coordinates for domain decomposition setup.
        
        Returns
        -------
        Tensor
            Coordinates of shape (N, 2)
        """
        return self.coordinates.clone()
    
    def get_normalizer(self) -> Dict[str, Tuple[float, float]]:
        """Get normalization parameters."""
        return self.normalizer


def create_dataloader(
    data_dir: str,
    split: str,
    batch_size: int,
    normalizer: Optional[Dict[str, Tuple[float, float]]] = None,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = True
) -> Tuple[DataLoader, NUNODarcyDataset]:
    """
    Create DataLoader for NUNO Darcy training.
    
    Parameters
    ----------
    data_dir : str
        Directory containing data
    split : str
        Data split ('train', 'val', 'test')
    batch_size : int
        Batch size
    normalizer : dict, optional
        Normalization parameters
    shuffle : bool
        Whether to shuffle data
    num_workers : int
        Number of data loading workers
    pin_memory : bool
        Whether to pin memory for GPU transfer
    drop_last : bool
        Whether to drop the last incomplete batch
        
    Returns
    -------
    tuple
        (DataLoader, Dataset) tuple
    """
    dataset = NUNODarcyDataset(data_dir, split, normalizer)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )
    
    return loader, dataset


def collate_fn_nuno(batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    """
    Custom collate function for NUNO data.
    
    Since coordinates are the same for all samples, we only need to
    return them once (not stacked).
    
    Parameters
    ----------
    batch : list
        List of sample dictionaries
        
    Returns
    -------
    dict
        Batched data with:
        - 'coordinates': (N, 2) - single set of coordinates
        - 'permeability': (B, N, 1) - batched input
        - 'pressure': (B, N, 1) - batched target
    """
    return {
        'coordinates': batch[0]['coordinates'],  # Same for all, just take first
        'permeability': torch.stack([b['permeability'] for b in batch], dim=0),
        'pressure': torch.stack([b['pressure'] for b in batch], dim=0)
    }


if __name__ == "__main__":
    # Test the dataset
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/nuno_darcy")
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()
    
    # Create dataset with default normalization
    normalizer = {
        'permeability': (1.25, 0.75),
        'darcy': (4.52e-2, 2.79e-2)
    }
    
    dataset = NUNODarcyDataset(args.data_dir, args.split, normalizer)
    
    print(f"\nDataset info:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Points per sample: {dataset.num_points}")
    print(f"  Coordinates shape: {dataset.coordinates.shape}")
    
    # Test loading a sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Coordinates: {sample['coordinates'].shape}")
    print(f"  Permeability: {sample['permeability'].shape}")
    print(f"  Pressure: {sample['pressure'].shape}")
    
    # Test dataloader
    loader, _ = create_dataloader(args.data_dir, args.split, batch_size=4, normalizer=normalizer)
    
    batch = next(iter(loader))
    print(f"\nBatch:")
    print(f"  Coordinates: {batch['coordinates'].shape}")
    print(f"  Permeability: {batch['permeability'].shape}")
    print(f"  Pressure: {batch['pressure'].shape}")