"""
Non-Uniform Neural Operator (NUNO) for PhysicsNeMo

This module implements NUNO for 2D problems, enabling neural operator learning
on non-uniform point clouds by decomposing the domain into subdomains via KD-tree,
interpolating to uniform grids, processing with FNO, and interpolating back.

Reference:
    Liu et al., "Domain Agnostic Fourier Neural Operators", NeurIPS 2023

Adapted for NVIDIA PhysicsNeMo environment.
"""

from typing import List, Tuple, Optional, Union
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from physicsnemo.models import Module
from physicsnemo.models.meta import ModelMetaData
from physicsnemo.models.fno.fno import FNO
from scipy.interpolate import LinearNDInterpolator, RBFInterpolator

# KDTree imports - assumes kdtree package is in the same directory or PYTHONPATH
from physicsnemo.models.fno.tree import KDTree
from physicsnemo.models.layers import SpectralConv2d



class UnitGaussianNormalizer:
    """Unit Gaussian Normalizer."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def encode(self, x):
        return (x - self.mean) / self.std

    def decode(self, x):
        return x * self.std + self.mean


@dataclass
class MetaData(ModelMetaData):
    """Metadata for NUNO model."""
    name: str = "NonUniformNeuralOperator"
    # Optimization
    jit: bool = True
    cuda_graphs: bool = True
    amp: bool = False


# Idea: Prep Data -> Forward on prepped data 

class NUNO(Module):
    def __init__(
            self,
            n_subdomains: int,
            output_dim: int,
            spatial_dim: int,
            oversamp_ratio: float = 1.5,
            fno_width: int = 20,
            ) -> None:
        super().__init__(meta=MetaData())

        self.model = FNO(
            in_channels=output_dim * n_subdomains,
            out_channels=output_dim * n_subdomains,
            latent_channels=fno_width,
        ).cuda()

        self.spatial_dim = spatial_dim
        self.output_dim = output_dim
        self.n_subdomains = n_subdomains
        self.oversamp_ratio = oversamp_ratio

        
    def prep2D(self, values, coordinates, n_train, n_test):

        # Gather subdomain info
        assert self.spatial_dim == coordinates.shape[1] == 2\
        
        # == TREE == 
        self.tree = KDTree(
            points=coordinates.tolist(),
            dim=self.spatial_dim, 
            n_subdomains=self.n_subdomains, 
            n_blocks=8,
            return_indices=True,
        )
        self.tree.solve()
        self.bbox_sd = self.tree.get_subdomain_bounding_boxes()
        self.indices_sd = self.tree.get_subdomain_indices()
        assert self.indices_sd is not None, "Subdomain indices not found."
        self.input_xy_sd = np.zeros((np.max([len(self.indices_sd[i]) 
            for i in range(self.n_subdomains)]), self.n_subdomains, 2))
        for i in range(self.n_subdomains):
            # Normalize to [-1, 1]
            xy = coordinates[self.indices_sd[i], :]
            _min, _max = np.min(xy, axis=0, keepdims=True), \
                np.max(xy, axis=0, keepdims=True)
            xy = (xy - _min) / (_max - _min) * 2 - 1
            # Long side alignment
            bbox = self.bbox_sd[i]
            if bbox[0][1] - bbox[0][0] < bbox[1][1] - bbox[1][0]:
                xy = np.flip(xy, axis=1)
            self.input_xy_sd[:len(self.indices_sd[i]), i, :] = xy
        print("Finish KD-Tree splitting.")

        # == INTERPOLATION == 
        print("Start Interpolation")
        n_total = values.shape[0]
        # Padded mesh grid size
        max_grid_size_x, max_grid_size_y = -1, -1
        grid_sizes = []
        is_transposed = [False] * self.n_subdomains
        for i in range(self.n_subdomains):
            n_points = len(self.indices_sd[i])
            bbox = self.bbox_sd[i]
            # Calculate the grid size, where the aspect ratio of the discrete grid 
            # remains the same as the that of the original subdomain (bbox)
            grid_size_x = np.sqrt(n_points * self.oversamp_ratio * \
                (bbox[0][1] - bbox[0][0]) / (bbox[1][1] - bbox[1][0]))
            grid_size_y = grid_size_x * (bbox[1][1] - bbox[1][0]) / (bbox[0][1] - bbox[0][0])
            grid_size_x, grid_size_y = max(int(np.round(grid_size_x)), 2), \
                max(int(np.round(grid_size_y)), 2)
            grid_sizes.append((grid_size_x, grid_size_y))
            # Long side alignment to reduce paddings
            if bbox[0][1] - bbox[0][0] < bbox[1][1] - bbox[1][0]:
                grid_size_x, grid_size_y = grid_size_y, grid_size_x
                is_transposed[i] = True
            max_grid_size_x, max_grid_size_y = max(max_grid_size_x, 
                grid_size_x), max(max_grid_size_y, grid_size_y)
        for i in range(self.n_subdomains):
            grid_size_x, grid_size_y = grid_sizes[i]
            print (f" Subdomain {i}: grid size = ({grid_size_x}, {grid_size_y}), ")
        input_u_sd_grid = []
        point_cloud = coordinates
        # values shape: (B, N, T, C)
        # shape: (1200, 3809, 31, 3) (Batch, Num-point, Time steps, Output Channels) 
        # In our case: No time steps, C = 1 -> shape: (B,N,1)        

        # input_u = Tensor.numpy(values)  # (B, N, T, C)
        if isinstance(values, torch.Tensor):
            input_u = values.detach().cpu().numpy() # Safe move to CPU
        else:
            input_u = values

        if isinstance(coordinates, torch.Tensor):
            coordinates = coordinates.detach().cpu().numpy()

        point_cloud_val = np.transpose(input_u, (1, 2, 0)) 
        interp_linear = LinearNDInterpolator(point_cloud, point_cloud_val)
        interp_rbf = RBFInterpolator(point_cloud, point_cloud_val, neighbors=6)

        for i in range(self.n_subdomains):
            grid_size_x, grid_size_y = grid_sizes[i]
            bbox = self.bbox_sd[i]
            # Linear interpolation
            grid_x = np.linspace(bbox[0][0], bbox[0][1], num=grid_size_x)
            grid_y = np.linspace(bbox[1][0], bbox[1][1], num=grid_size_y)
            grid_x, grid_y = np.meshgrid(grid_x, grid_y)
            grid_val = interp_linear(grid_x, grid_y)
            # Fill nan values
            # Was: [..., 0, 0, 0] -> [..., 0, 0] for no time steps (not sure on this)
            nan_indices = np.isnan(grid_val)[..., 0, 0]
            fill_vals = interp_rbf(np.stack((grid_x[nan_indices], grid_y[nan_indices]), axis=1))
            grid_val[nan_indices] = fill_vals
            # Resize to the same size via FFT-IFFT
            freq = np.fft.rfft2(grid_val, axes=(0, 1))
            s1_padded, s2_padded = max_grid_size_y, max_grid_size_x 
            if is_transposed[i]:
                s1_padded, s2_padded = s2_padded, s1_padded
            square_freq = np.zeros((s1_padded, 
                s2_padded // 2, self.output_dim, n_total)) + 0j # n_total - batch size
            square_freq[:min(s1_padded//2, freq.shape[0]//2), 
                    :min(s2_padded//2+1, freq.shape[1]//2+1), ...] = \
                freq[:min(s1_padded//2, freq.shape[0]//2), 
                    :min(s2_padded//2+1, freq.shape[1]//2+1), ...]
            square_freq[-min(s1_padded//2, freq.shape[0]//2):, 
                    :min(s2_padded//2+1, freq.shape[1]//2+1), ...] = \
                freq[-min(s1_padded//2, freq.shape[0]//2):,
                    :min(s2_padded//2+1, freq.shape[1]//2+1), ...]
            grid_val = np.fft.irfft2(square_freq, 
                s=(s1_padded, s2_padded), axes=(0, 1))
            if is_transposed[i]:
                # Old: (1, 0, 2, 3, 4) -> New: (1, 0, 2, 3)
                grid_val = np.transpose(grid_val, (1, 0, 2, 3))
            # Old: (4, 0, 1, 2, 3) -> New: (3, 0, 1, 2)
            # Resulting shape: (Batch, Grid_Y, Grid_X, Channels)
            input_u_sd_grid.append(np.transpose(grid_val, (3, 0, 1, 2)))
        # Final shape: (Batch, Height, Width, Channels, N_subdomains) 
        input_u_sd_grid = np.transpose(np.array(input_u_sd_grid), (1, 2, 3, 4, 0))
        
        print(input_u_sd_grid.shape)

        input_u_sd = np.zeros((n_total, 
            np.max([len(self.indices_sd[i]) for i in range(self.n_subdomains)]), self.output_dim, self.n_subdomains))
        input_u_sd_mask = np.zeros((1, 
            np.max([len(self.indices_sd[i]) for i in range(self.n_subdomains)]), 1, self.n_subdomains))
        for i in range(self.n_subdomains):
            input_u_sd[:, :len(self.indices_sd[i]), ..., i] = input_u[:, self.indices_sd[i], ...]
            input_u_sd_mask[:, :len(self.indices_sd[i]), ..., i] = 1.

        print("Finish interpolation.")
        input_xy_sd = torch.from_numpy(self.input_xy_sd).cuda().float()
        input_xy_sd = input_xy_sd.unsqueeze(0).repeat([n_total, 1, 1, 1])\
            .permute(0, 2, 1, 3)\
            .reshape(n_total * self.n_subdomains, -1, 1, 2)
            # shape: (batch * n_subdomains, n_points_sd_padded, 1, 2)

        s1_padded, s2_padded = input_u_sd_grid.shape[1:3]
        input_u_sd_grid = torch.from_numpy(
            input_u_sd_grid.reshape(n_total, s1_padded, s2_padded, -1)).float()
        input_u_sd = torch.from_numpy(input_u_sd).float()
        input_u_sd_mask = torch.from_numpy(input_u_sd_mask).cuda().float()

        train_a_sd = input_u_sd_grid[:n_train, ..., :].cuda()
        test_a_sd = input_u_sd_grid[-n_test:, ..., :].cuda()

        train_u_sd = input_u_sd[:n_train, ..., :, :].cuda()
        test_u_sd = input_u_sd[-n_test:, ..., :, :].cuda()

        eps = 1e-5

        u_mean = torch.mean(train_u_sd, 0)
        u_std = torch.std(train_u_sd, 0) + eps

        a_mean = torch.mean(train_a_sd, 0)
        a_std = torch.std(train_a_sd, 0) + eps

        a_normalizer = UnitGaussianNormalizer(a_mean, a_std)
        train_a_sd = a_normalizer.encode(train_a_sd)
        test_a_sd = a_normalizer.encode(test_a_sd)

        y_normalizer = UnitGaussianNormalizer(u_mean, 
                                              u_std)
        self.input_xy_sd = input_xy_sd
        self.input_u_sd_mask = input_u_sd_mask
        self.y_normalizer = y_normalizer

        return train_a_sd, train_u_sd, test_a_sd, test_u_sd, y_normalizer, a_normalizer

    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for 2D NUNO.
        Input x: (Batch, Height, Width, Channels_in * N_subdomains)
        Output:  (Batch, Points, Channels_out, N_subdomains) - Decoded & Masked
        """
       # 1. FIX: Permute Input to Channel-First (B, C, H, W)
        # Current: (Batch, H, W, C) -> Target: (Batch, C, H, W)
        x = x.permute(0, 3, 1, 2)

        batch_size = x.shape[0]
        # Note: Now H and W are at indices 2 and 3
        h, w = x.shape[2], x.shape[3]
        
        # 2. Backbone FNO Pass
        out = self.model(x)

        # 3. Reshape for Grid Sampling
        # ... (rest of the function follows the previous logic) ...
        
        # Split channels and subdomains
        # (Batch, C, H, W) -> (Batch, output_dim, n_subdomains, H, W)
        out = out.reshape(batch_size, self.output_dim, self.n_subdomains, h, w)

        # Merge Batch and N_sd for grid_sample
        # Target: (Batch * N_sd, Channels, H, W)
        out = out.permute(0, 2, 1, 3, 4).reshape(-1, self.output_dim, h, w)

    
        current_batch_flat = out.shape[0]
        
        grid_slice = self.input_xy_sd[:current_batch_flat, ...]
        
        u = F.grid_sample(input=out, grid=grid_slice, 
                          padding_mode='border', align_corners=False)


        # 4. Interpolation
        # u = F.grid_sample(input=out, grid=self.input_xy_sd, 
        #                  padding_mode='border', align_corners=False)

        # 5. Post-Process
        u = u.squeeze(-1) # (Batch*N_sd, Channels, Points)
        
        # Reshape to (Batch, N_sd, Channels, Points)
        u = u.reshape(batch_size, self.n_subdomains, self.output_dim, -1)
        
        # Permute to (Batch, Points, Channels, N_sd)
        u = u.permute(0, 3, 2, 1)


        # 6. Decode
        u = self.y_normalizer.decode(u)
        u = u * self.input_u_sd_mask

        return u

# ============================================================================
# Main function for testing basic functionality
# ============================================================================
if __name__ == "__main__":

    n_points = 4096
    in_channels = 1
    out_channels = 1
    batch_size = 256
    n_subdomains = 8

    # Create clustered points (non-uniform distribution)
    cluster1 = np.random.randn(n_points // 3, 2) * 0.15 + np.array([0.25, 0.25])
    cluster2 = np.random.randn(n_points // 3, 2) * 0.1 + np.array([0.75, 0.75])
    cluster3 = np.random.randn(n_points - 2 * (n_points // 3), 2) * 0.2 + np.array([0.5, 0.5])
    coordinates = np.vstack([cluster1, cluster2, cluster3])
    coordinates = np.clip(coordinates, 0.01, 0.99)

    def sample_function(coords):
        return np.sin(2 * np.pi * coords[:, 0]) * np.cos(2 * np.pi * coords[:, 1])

    def generate_random_waves(coords, batch_size):
        """
        Generates random sine waves: A * sin(2*pi*f1*x + p1) * cos(2*pi*f2*y + p2)
        """
        # coords: (N_points, 2)
        x = coords[:, 0:1] # (N, 1)
        y = coords[:, 1:2] # (N, 1)
        
        # Random Parameters for each batch item
        # Shape: (Batch, 1) to broadcast against (N, 1)
        freq_x = np.random.uniform(1, 3, (batch_size, 1))
        freq_y = np.random.uniform(1, 3, (batch_size, 1))
        phase_x = np.random.uniform(0, 2*np.pi, (batch_size, 1))
        phase_y = np.random.uniform(0, 2*np.pi, (batch_size, 1))
        amp = np.random.uniform(0.5, 1.5, (batch_size, 1))

        # Calculate values using broadcasting
        # (B, 1) * sin((B, 1) * (N, 1) + (B, 1)) ...
        val = amp * np.sin(2 * np.pi * freq_x * x.T + phase_x) * \
                    np.cos(2 * np.pi * freq_y * y.T + phase_y)
        
        return val # Shape: (Batch, N_points)

    values_np = generate_random_waves(coordinates, batch_size)
    values = torch.from_numpy(values_np).float().unsqueeze(-1)
    # values = values.expand(batch_size, -1, -1).clone()
    
    print(f"   - Values shape: {values.shape}")
    print(f"   - Values range: [{values.min().item():.3f}, {values.max().item():.3f}]")

    model = NUNO(
        n_subdomains=n_subdomains,
        output_dim=1,
        spatial_dim=2
    )

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    n_train = batch_size
    n_test = 0

    train_a, train_u, _, _, y_norm, a_norm = model.prep2D(values, coordinates, n_train, n_test)

    print(f"Train Grid Shape (train_a): {train_a.shape}") # (Batch, H, W, C*N_sd)
    print(f"Train Points Shape (train_u): {train_u.shape}") # (Batch, Points, C, N_sd)

    train_a_decoded = a_norm.decode(train_a).cpu().numpy()
    train_u_cpu = train_u.cpu().numpy()
    coords_tensor = model.input_xy_sd.cpu()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    batch_idx = 0
    
    H, W = train_a_decoded.shape[1], train_a_decoded.shape[2]
    grid_reshaped = train_a_decoded[batch_idx].reshape(H, W, out_channels, n_subdomains)

    for i in range(n_subdomains):
        grid_data = grid_reshaped[:, :, 0, i]
        
        bbox = model.bbox_sd[i]
        x_min, x_max = bbox[0][0], bbox[0][1]
        y_min, y_max = bbox[1][0], bbox[1][1]
        
        im = ax.imshow(grid_data, 
                       extent=[x_min, x_max, y_min, y_max], 
                       origin='lower', 
                       cmap='viridis', 
                       alpha=0.6, # Make transparent to see points
                       interpolation='nearest')
        
        flat_idx = batch_idx * n_subdomains + i
        local_coords = coords_tensor[flat_idx, :, 0, :].numpy()

        # reverse rotation
        width = x_max - x_min
        height = y_max - y_min
        
        if width < height:
            local_coords = np.flip(local_coords, axis=1)
        
        # Remove padding (points at 0,0 usually indicate padding if using simplified padding)
        phys_x = (local_coords[:, 0] + 1) / 2 * (x_max - x_min) + x_min
        phys_y = (local_coords[:, 1] + 1) / 2 * (y_max - y_min) + y_min
        
        point_vals = train_u_cpu[batch_idx, :, 0, i]
        
        mask = np.abs(point_vals) > 1e-6 
        
        ax.scatter(phys_x[mask], phys_y[mask], c='orange', s=5, label='Points' if i==0 else "")

        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                 linewidth=1, edgecolor='red', facecolor='none', linestyle='--')
        ax.add_patch(rect)
        
        # Label
        cx, cy = (x_min + x_max)/2, (y_min + y_max)/2
        ax.text(cx, cy, str(i), color='white', fontweight='bold', ha='center', va='center')

    ax.set_title("Training Data Check: Grids (Background) vs Points (Red)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label="Grid Value")
    plt.legend(loc='upper right')
    plt.show()

    print("\n--- Testing Single Forward Pass ---")
    
    import torch.optim as optim

    # 1. Define an optimizer (Essential for updating weights)
    # using a standard learning rate of 1e-3
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    max_epoch = 200
    
    print(f"Starting {max_epoch}-epoch sanity check...")
    print("-" * 50)

    for epoch in range(max_epoch):
        # --- A. Setup ---
        model.train()
        optimizer.zero_grad()  # Reset gradients from previous epoch
        
        x = train_a
        
        # --- B. Forward Pass ---
        prediction = model(x)
        
        # --- C. Debug Prints (Run only on first epoch) ---
        if epoch == 0:
            print(f"[Debug] Forward pass successful!")
            print(f"[Debug] Input Shape (x):      {x.shape}") 
            print(f"[Debug] Output Shape (pred):  {prediction.shape}") 
            print(f"[Debug] Target Shape (y):     {train_u.shape}")
            print(f"[Debug] Prediction Range:     [{prediction.min().item():.3f}, {prediction.max().item():.3f}]")
            
            # Check Mask
            mask = model.input_u_sd_mask
            print(f"[Debug] Mask Mean value:      {mask.float().mean().item():.3f}")
            
            # Check Sample Values
            valid_idx = torch.nonzero(mask.flatten())[0]
            print(f"[Debug] Sample Valid Pred:    {prediction.flatten()[valid_idx].item():.5f}")
            print(f"[Debug] Sample Valid Targ:    {train_u.flatten()[valid_idx].item():.5f}")
            print("-" * 50)

        loss = torch.nn.functional.mse_loss(prediction, train_u) # UPDATE TO MULTILP LOSS

        loss.backward()   
        optimizer.step()  

        # --- F. Report ---
        print(f"Epoch {epoch+1}/{max_epoch} | Loss: {loss.item():.6f}")