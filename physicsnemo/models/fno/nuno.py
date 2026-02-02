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
from physicsnemo.models import Module
from physicsnemo.models.meta import ModelMetaData
import torch.nn.functional as F
from scipy.interpolate import LinearNDInterpolator, RBFInterpolator

# KDTree imports - assumes kdtree package is in the same directory or PYTHONPATH
from physicsnemo.models.fno.tree import KDTree
from physicsnemo.models.layers import SpectralConv2d

@dataclass
class MetaData(ModelMetaData):
    """Metadata for the NUNO model following PhysicsNeMo conventions."""
    name: str = "NUNO"
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    var_dim: int = 1  # Variable dimension index for input/output



class FNO2DBlock(nn.Module):
    """
    Single FNO block combining spectral convolution with local linear transform.
    """
    
    def __init__(self, width: int, modes1: int, modes2: int):
        """
        Initialize FNO block.
        
        Args:
            width: Hidden channel width
            modes1: Fourier modes in first dimension
            modes2: Fourier modes in second dimension
        """
        super().__init__()
        self.conv = SpectralConv2d(width, width, modes1, modes2)
        self.w = nn.Conv2d(width, width, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spectral conv + linear conv with GELU activation."""
        x1 = self.conv(x)
        x2 = self.w(x)
        return F.gelu(x1 + x2)


class FNO2DCore(nn.Module):
    """
    Core FNO2D network for processing uniform grid data.
    
    This is the neural network backbone used within NUNO for each subdomain.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 32,
        num_layers: int = 4,
    ):
        """
        Initialize FNO2D core network.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            modes1: Number of Fourier modes in first spatial dimension
            modes2: Number of Fourier modes in second spatial dimension
            width: Hidden channel width
            num_layers: Number of FNO blocks
        """
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.num_layers = num_layers
        
        # Lifting layer: input channels -> hidden width
        self.fc0 = nn.Linear(in_channels, width)
        
        # FNO blocks
        self.blocks = nn.ModuleList([
            FNO2DBlock(width, modes1, modes2) for _ in range(num_layers)
        ])
        
        # Projection layers: hidden width -> output channels
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through FNO2D.
        
        Args:
            x: Input tensor of shape (batch, height, width, in_channels)
            
        Returns:
            Output tensor of shape (batch, height, width, out_channels)
        """
        # Lift to higher dimension
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # (batch, width, height, width_spatial)
        
        # Apply FNO blocks
        for block in self.blocks:
            x = block(x)
        
        # Project back
        x = x.permute(0, 2, 3, 1)  # (batch, height, width_spatial, width)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x


class DomainDecomposition:
    """
    Handles KD-tree based domain decomposition for NUNO.
    
    This class encapsulates all preprocessing logic:
    - Splitting domain into subdomains via KD-tree
    - Computing normalized coordinates for interpolation
    - Managing grid sizes and padding
    """
    
    def __init__(
        self,
        coordinates: np.ndarray,
        n_subdomains: int = 8,
        n_blocks: int = 8,
        smallest_points: int = 8,
        max_depth: Optional[int] = None,
        oversamp_ratio: float = 1.5,
    ):
        """
        Initialize domain decomposition.
        
        Args:
            coordinates: Point cloud coordinates of shape (N, 2)
            n_subdomains: Number of subdomains to create
            n_blocks: Number of blocks for KL divergence approximation
            smallest_points: Minimum points per subdomain
            max_depth: Maximum tree depth (None for unlimited)
            oversamp_ratio: Oversampling ratio for grid resolution
        """
        self.coordinates = coordinates
        self.n_subdomains = n_subdomains
        self.n_blocks = n_blocks
        self.smallest_points = smallest_points
        self.max_depth = max_depth
        self.oversamp_ratio = oversamp_ratio
        
        # Build KD-tree
        self._build_tree()
        
        # Compute grid information
        self._compute_grid_info()
        
        # Prepare interpolation coordinates
        self._prepare_interp_coords()
    
    def _build_tree(self):
        """Build KD-tree for domain decomposition."""
        point_cloud = self.coordinates.tolist()
        self.tree = KDTree(
            point_cloud,
            dim=2,
            n_subdomains=self.n_subdomains,
            n_blocks=self.n_blocks,
            smallest_points=self.smallest_points,
            max_depth=self.max_depth,
            return_indices=True,
        )
        self.tree.solve()
        
        # Get subdomain information
        self.bbox_sd = self.tree.get_subdomain_bounding_boxes()
        self.indices_sd = self.tree.get_subdomain_indices()
        self.n_subdomains = len(self.indices_sd)  # Actual number after tree solve
    
    def _compute_grid_info(self):
        """Compute grid sizes and padding information."""
        max_grid_size_x, max_grid_size_y = -1, -1
        self.grid_sizes = []
        self.is_transposed = [False] * self.n_subdomains
        
        for i in range(self.n_subdomains):
            n_points = len(self.indices_sd[i])
            bbox = self.bbox_sd[i]
            
            # Calculate grid size preserving aspect ratio
            bbox_width = bbox[0][1] - bbox[0][0]
            bbox_height = bbox[1][1] - bbox[1][0]
            
            # Avoid division by zero
            if bbox_height < 1e-10:
                bbox_height = 1e-10
            if bbox_width < 1e-10:
                bbox_width = 1e-10
            
            grid_size_x = np.sqrt(n_points * self.oversamp_ratio * bbox_width / bbox_height)
            grid_size_y = grid_size_x * bbox_height / bbox_width
            grid_size_x = max(int(np.round(grid_size_x)), 2)
            grid_size_y = max(int(np.round(grid_size_y)), 2)
            self.grid_sizes.append((grid_size_x, grid_size_y))
            
            # Long side alignment to reduce padding
            if bbox_width < bbox_height:
                grid_size_x, grid_size_y = grid_size_y, grid_size_x
                self.is_transposed[i] = True
            
            max_grid_size_x = max(max_grid_size_x, grid_size_x)
            max_grid_size_y = max(max_grid_size_y, grid_size_y)
        
        self.max_grid_size = (max_grid_size_x, max_grid_size_y)
        self.s1_padded = max_grid_size_y
        self.s2_padded = max_grid_size_x
        
        # Maximum points in any subdomain (for padding)
        self.max_points_sd = max(len(self.indices_sd[i]) for i in range(self.n_subdomains))
    
    def _prepare_interp_coords(self):
        """Prepare normalized interpolation coordinates for each subdomain."""
        # Coordinates normalized to [-1, 1] for grid_sample
        self.coords_normalized = np.zeros((self.max_points_sd, self.n_subdomains, 2))
        
        for i in range(self.n_subdomains):
            xy = self.coordinates[self.indices_sd[i], :]
            _min = np.min(xy, axis=0, keepdims=True)
            _max = np.max(xy, axis=0, keepdims=True)
            
            # Avoid division by zero
            scale = _max - _min
            scale[scale < 1e-10] = 1e-10
            
            xy_norm = (xy - _min) / scale * 2 - 1
            
            # Long side alignment
            bbox = self.bbox_sd[i]
            if bbox[0][1] - bbox[0][0] < bbox[1][1] - bbox[1][0]:
                xy_norm = np.flip(xy_norm, axis=1)
            
            self.coords_normalized[:len(self.indices_sd[i]), i, :] = xy_norm
    
    def get_interpolation_grids(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get meshgrid coordinates for each subdomain."""
        grids = []
        for i in range(self.n_subdomains):
            bbox = self.bbox_sd[i]
            grid_size_x, grid_size_y = self.grid_sizes[i]
            
            grid_x = np.linspace(bbox[0][0], bbox[0][1], num=grid_size_x)
            grid_y = np.linspace(bbox[1][0], bbox[1][1], num=grid_size_y)
            grid_x, grid_y = np.meshgrid(grid_x, grid_y)
            grids.append((grid_x, grid_y))
        
        return grids


class NUNO(Module):
    """
    Non-Uniform Neural Operator (NUNO) for 2D problems.
    
    NUNO enables neural operator learning on non-uniform point clouds by:
    1. Decomposing the domain into subdomains via KD-tree
    2. Interpolating point cloud data to uniform grids per subdomain
    3. Processing grids through FNO
    4. Interpolating back to the original point cloud
    
    This implementation follows PhysicsNeMo conventions while maintaining
    the original NUNO algorithm logic.
    
    Args:
        coordinates: Point cloud coordinates of shape (N, 2)
        in_channels: Number of input channels/features per point
        out_channels: Number of output channels/features per point
        n_subdomains: Number of subdomains for domain decomposition
        modes: Number of Fourier modes in each dimension
        width: Hidden channel width for FNO
        num_layers: Number of FNO blocks
        n_blocks: KD-tree parameter for KL divergence approximation
        smallest_points: Minimum points per subdomain
        max_depth: Maximum KD-tree depth
        oversamp_ratio: Oversampling ratio for grid resolution
        rbf_neighbors: Number of neighbors for RBF interpolation fallback
        
    Example:
        >>> coords = np.random.rand(100, 2)  # 100 points in 2D
        >>> model = NUNO(coords, in_channels=1, out_channels=1, n_subdomains=4)
        >>> model.setup_interpolation(coords, sample_values)  # Setup with training data
        >>> x = torch.randn(32, 100, 1)  # batch of 32, 100 points, 1 channel
        >>> y = model(x)  # Output: (32, 100, 1)
    """
    
    # PhysicsNeMo metadata
    meta = MetaData()
    
    def __init__(
        self,
        coordinates: np.ndarray,
        in_channels: int,
        out_channels: int,
        n_subdomains: int = 8,
        modes: int = 12,
        width: int = 32,
        num_layers: int = 4,
        n_blocks: int = 8,
        smallest_points: int = 8,
        max_depth: Optional[int] = None,
        oversamp_ratio: float = 1.5,
        rbf_neighbors: int = 6,
    ):
        super().__init__(
            meta=ModelMetaData(
                name="NUNO",
                cuda_graphs=False,
                amp_cpu=False,
                amp_gpu=True,
                torch_fx=False,
            )
        )
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_subdomains = n_subdomains
        self.modes = modes
        self.width = width
        self.num_layers = num_layers
        self.rbf_neighbors = rbf_neighbors
        
        # Domain decomposition (computed once, coordinates are fixed)
        self.decomposition = DomainDecomposition(
            coordinates=coordinates,
            n_subdomains=n_subdomains,
            n_blocks=n_blocks,
            smallest_points=smallest_points,
            max_depth=max_depth,
            oversamp_ratio=oversamp_ratio,
        )
        
        # Update actual number of subdomains after tree solve
        self.n_subdomains = self.decomposition.n_subdomains
        
        # FNO backbone - processes all subdomains together
        # Input: in_channels * n_subdomains, Output: out_channels * n_subdomains
        self.fno = FNO2DCore(
            in_channels=in_channels * self.n_subdomains,
            out_channels=out_channels * self.n_subdomains,
            modes1=modes,
            modes2=modes,
            width=width,
            num_layers=num_layers,
        )
        
        # Store grid sizes
        self.s1_padded = self.decomposition.s1_padded
        self.s2_padded = self.decomposition.s2_padded
        
        # Register coordinates for grid_sample as buffer
        coords_tensor = torch.from_numpy(
            self.decomposition.coords_normalized
        ).float()
        self.register_buffer('coords_normalized', coords_tensor)
        
        # Interpolators will be set during setup_interpolation
        self._interp_linear = None
        self._interp_rbf = None
        self._interpolators_ready = False
    
    def setup_interpolation(
        self,
        coordinates: np.ndarray,
        sample_values: Optional[np.ndarray] = None,
    ):
        """
        Setup interpolators for point cloud to grid conversion.
        
        This method should be called once before training/inference with
        representative data to establish the interpolation scheme.
        
        Args:
            coordinates: Point cloud coordinates (N, 2)
            sample_values: Optional sample values for establishing interpolators.
                          Shape: (N, C) or (batch, N, C). If provided, uses the
                          mean across batch/samples for interpolator setup.
        """
        if sample_values is not None:
            if sample_values.ndim == 3:
                # Average across batch
                sample_values = sample_values.mean(axis=0)
            
            self._interp_linear = LinearNDInterpolator(coordinates, sample_values)
            self._interp_rbf = RBFInterpolator(
                coordinates, sample_values, neighbors=self.rbf_neighbors
            )
            self._interpolators_ready = True
        else:
            # Create placeholder - will use per-sample interpolation
            self._interpolators_ready = False
    
    def _interpolate_to_grid(
        self,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Interpolate point cloud values to uniform grids for each subdomain.
        
        This follows the reference implementation's preprocessing logic.
        
        Args:
            values: Point values of shape (batch, N, in_channels)
            
        Returns:
            Grid values of shape (batch, s1_padded, s2_padded, in_channels * n_subdomains)
        """
        batch_size = values.shape[0]
        device = values.device
        dtype = values.dtype
        
        values_np = values.detach().cpu().numpy()
        coordinates = self._coordinates
        
        # Output: list of subdomain grids, each (batch, s1_padded, s2_padded, in_channels)
        all_subdomain_grids = []
        
        for i in range(self.n_subdomains):
            bbox = self.decomposition.bbox_sd[i]
            grid_size_x, grid_size_y = self.decomposition.grid_sizes[i]
            
            # Create grid coordinates for this subdomain
            grid_x = np.linspace(bbox[0][0], bbox[0][1], num=grid_size_x)
            grid_y = np.linspace(bbox[1][0], bbox[1][1], num=grid_size_y)
            grid_x, grid_y = np.meshgrid(grid_x, grid_y)
            # grid_x, grid_y shapes: (grid_size_y, grid_size_x)
            
            subdomain_grid_batch = []
            
            for b in range(batch_size):
                sample_values = values_np[b]  # (N, in_channels)
                
                # Create interpolator for this sample
                interp_linear = LinearNDInterpolator(coordinates, sample_values)
                
                # Interpolate to grid
                grid_val = interp_linear(grid_x, grid_y)
                # grid_val shape: (grid_size_y, grid_size_x, in_channels)
                
                # Fill NaN values with RBF interpolation
                nan_mask = np.isnan(grid_val)
                if nan_mask.any():
                    nan_indices = nan_mask[..., 0]
                    if nan_indices.any():
                        interp_rbf = RBFInterpolator(
                            coordinates, sample_values, neighbors=self.rbf_neighbors
                        )
                        fill_coords = np.stack(
                            (grid_x[nan_indices], grid_y[nan_indices]), axis=1
                        )
                        fill_vals = interp_rbf(fill_coords)
                        grid_val[nan_indices] = fill_vals
                
                # Resize via FFT-IFFT to padded size (following reference lines 253-271)
                freq = np.fft.rfft2(grid_val, axes=(0, 1))
                
                # Target size depends on transposition
                s1, s2 = self.s1_padded, self.s2_padded
                if self.decomposition.is_transposed[i]:
                    s1, s2 = s2, s1
                
                # Zero-pad in frequency domain
                square_freq = np.zeros((s1, s2 // 2 + 1, self.in_channels), dtype=np.complex128)
                
                # Copy low frequencies (both positive and negative)
                min_h1 = min(s1 // 2, freq.shape[0] // 2)
                min_w1 = min(s2 // 2 + 1, freq.shape[1])
                
                square_freq[:min_h1, :min_w1, :] = freq[:min_h1, :min_w1, :]
                square_freq[-min_h1:, :min_w1, :] = freq[-min_h1:, :min_w1, :]
                
                # Transform back
                grid_val = np.fft.irfft2(square_freq, s=(s1, s2), axes=(0, 1))
                
                # Transpose if needed to align long side
                if self.decomposition.is_transposed[i]:
                    grid_val = np.transpose(grid_val, (1, 0, 2))
                
                # Now grid_val has shape (s1_padded, s2_padded, in_channels)
                subdomain_grid_batch.append(grid_val)
            
            # Stack batch: (batch, s1_padded, s2_padded, in_channels)
            subdomain_grid_batch = np.stack(subdomain_grid_batch, axis=0)
            all_subdomain_grids.append(subdomain_grid_batch)
        
        # Stack subdomains - need to match reference ordering
        # Reference: (n_samples, s1, s2, out_dim, n_sd) then reshape to (n_samples, s1, s2, out_dim * n_sd)
        # This gives ordering: [ch0_sd0, ch0_sd1, ..., ch0_sdn, ch1_sd0, ch1_sd1, ...]
        # i.e., channels first, then subdomains
        
        # Stack along a new axis: (batch, s1, s2, in_channels, n_subdomains)
        grid_stacked = np.stack(all_subdomain_grids, axis=-1)
        # Reshape to (batch, s1, s2, in_channels * n_subdomains) - this gives correct ordering
        grid_combined = grid_stacked.reshape(
            batch_size, self.s1_padded, self.s2_padded, -1
        )
        
        return torch.from_numpy(grid_combined).to(device=device, dtype=dtype)
    
    def _interpolate_from_grid(
        self,
        grid_values: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Interpolate from uniform grids back to point cloud.
        
        Uses F.grid_sample for efficient GPU-based interpolation.
        
        Args:
            grid_values: Grid values of shape (batch, s1, s2, C * n_subdomains)
            batch_size: Original batch size
            
        Returns:
            Point values of shape (batch, N, C)
        """
        # Reshape for grid_sample: need (batch * n_subdomains, C, s1, s2)
        grid = grid_values.reshape(
            batch_size,
            self.s1_padded,
            self.s2_padded,
            self.out_channels,
            self.n_subdomains,
        )
        grid = grid.permute(0, 4, 3, 1, 2)  # (batch, n_sd, C, s1, s2)
        grid = grid.reshape(
            batch_size * self.n_subdomains,
            self.out_channels,
            self.s1_padded,
            self.s2_padded,
        )
        
        # Prepare sampling coordinates
        # coords_normalized: (max_points, n_subdomains, 2)
        coords = self.coords_normalized.unsqueeze(0).expand(batch_size, -1, -1, -1)
        coords = coords.permute(0, 2, 1, 3)  # (batch, n_sd, max_points, 2)
        coords = coords.reshape(batch_size * self.n_subdomains, -1, 1, 2)
        
        # grid_sample expects (N, C, H, W) input and (N, H_out, W_out, 2) grid
        sampled = F.grid_sample(
            input=grid,
            grid=coords,
            mode='bilinear',
            padding_mode='border',
            align_corners=False,
        )
        # sampled: (batch * n_sd, C, max_points, 1)
        
        sampled = sampled.squeeze(-1).permute(0, 2, 1)  # (batch * n_sd, max_points, C)
        sampled = sampled.reshape(batch_size, self.n_subdomains, -1, self.out_channels)
        sampled = sampled.permute(0, 2, 3, 1)  # (batch, max_points, C, n_sd)
        
        # Aggregate subdomains - reconstruct full point cloud
        # Each point belongs to exactly one subdomain
        n_total_points = len(self.decomposition.coordinates)
        output = torch.zeros(
            batch_size, n_total_points, self.out_channels,
            device=grid_values.device, dtype=grid_values.dtype
        )
        
        for i in range(self.n_subdomains):
            indices = self.decomposition.indices_sd[i]
            n_pts = len(indices)
            output[:, indices, :] = sampled[:, :n_pts, :, i]
        
        return output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through NUNO.
        
        Args:
            x: Input point cloud values of shape (batch, N, in_channels)
               where N is the number of points matching the coordinates
               used during initialization.
               
        Returns:
            Output point cloud values of shape (batch, N, out_channels)
        """
        batch_size = x.shape[0]
        coordinates = self.decomposition.coordinates
        
        # Step 1: Interpolate point cloud to subdomain grids
        grid_input = self._interpolate_to_grid(x)
        # grid_input: (batch, s1, s2, in_channels * n_subdomains)
        
        # Step 2: Process through FNO
        grid_output = self.fno(grid_input)
        # grid_output: (batch, s1, s2, out_channels * n_subdomains)
        
        # Step 3: Interpolate back to point cloud
        output = self._interpolate_from_grid(grid_output, batch_size)
        # output: (batch, N, out_channels)
        
        return output
    
    def get_subdomain_info(self) -> dict:
        """Return information about the domain decomposition."""
        return {
            'n_subdomains': self.n_subdomains,
            'grid_sizes': self.decomposition.grid_sizes,
            'padded_size': (self.s1_padded, self.s2_padded),
            'points_per_subdomain': [
                len(self.decomposition.indices_sd[i])
                for i in range(self.n_subdomains)
            ],
            'bounding_boxes': self.decomposition.bbox_sd,
        }


def count_params(model: nn.Module) -> int:
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# Main function for testing basic functionality
# ============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("=" * 60)
    print("NUNO (Non-Uniform Neural Operator) - Basic Functionality Test")
    print("=" * 60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Test parameters
    n_points = 100
    in_channels = 1
    out_channels = 1
    batch_size = 4
    n_subdomains = 4
    
    # Generate random non-uniform point cloud
    # Using a non-uniform distribution to demonstrate NUNO's capability
    print("\n1. Generating non-uniform point cloud...")
    
    # Create clustered points (non-uniform distribution)
    cluster1 = np.random.randn(n_points // 3, 2) * 0.15 + np.array([0.25, 0.25])
    cluster2 = np.random.randn(n_points // 3, 2) * 0.1 + np.array([0.75, 0.75])
    cluster3 = np.random.randn(n_points - 2 * (n_points // 3), 2) * 0.2 + np.array([0.5, 0.5])
    coordinates = np.vstack([cluster1, cluster2, cluster3])
    
    # Clip to [0, 1] domain
    coordinates = np.clip(coordinates, 0.01, 0.99)
    
    print(f"   - Number of points: {n_points}")
    print(f"   - Coordinate range: [{coordinates.min():.3f}, {coordinates.max():.3f}]")
    
    # Generate sample function values (e.g., a simple function)
    print("\n2. Generating sample function values...")
    
    def sample_function(coords):
        """Sample function: f(x, y) = sin(2*pi*x) * cos(2*pi*y)"""
        return np.sin(2 * np.pi * coords[:, 0]) * np.cos(2 * np.pi * coords[:, 1])
    
    values_np = sample_function(coordinates)
    values = torch.from_numpy(values_np).float().unsqueeze(0).unsqueeze(-1)
    values = values.expand(batch_size, -1, -1)  # (batch, N, 1)
    
    print(f"   - Values shape: {values.shape}")
    print(f"   - Values range: [{values.min().item():.3f}, {values.max().item():.3f}]")
    
    # Initialize NUNO model
    print("\n3. Initializing NUNO model...")
    
    model = NUNO(
        coordinates=coordinates,
        in_channels=in_channels,
        out_channels=out_channels,
        n_subdomains=n_subdomains,
        modes=4,
        width=16,
        num_layers=4,
    )
    
    print(f"   - Model parameters: {count_params(model):,}")
    print(f"   - Actual subdomains: {model.n_subdomains}")
    print(f"   - Padded grid size: {model.s1_padded} x {model.s2_padded}")
    
    # Print subdomain information
    info = model.get_subdomain_info()
    print(f"\n   Subdomain information:")
    for i in range(model.n_subdomains):
        print(f"     - Subdomain {i}: {info['points_per_subdomain'][i]} points, "
              f"grid {info['grid_sizes'][i]}")
    
    # Test forward pass
    print("\n4. Testing forward pass...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   - Device: {device}")
    
    model = model.to(device)
    values = values.to(device)
    
    try:
        with torch.no_grad():
            output = model(values)
        
        print(f"   - Input shape: {values.shape}")
        print(f"   - Output shape: {output.shape}")
        print(f"   - Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
        print("   ✓ Forward pass successful!")
    except Exception as e:
        print(f"   ✗ Forward pass failed: {e}")
        raise
    
    # Test backward pass
    print("\n5. Testing backward pass...")
    
    model.train()
    values_train = values.clone().requires_grad_(False)
    target = torch.randn_like(output)
    
    try:
        output = model(values_train)
        loss = F.mse_loss(output, target)
        loss.backward()
        
        # Check gradients
        has_grad = all(
            p.grad is not None for p in model.parameters() if p.requires_grad
        )
        print(f"   - Loss: {loss.item():.6f}")
        print(f"   - All parameters have gradients: {has_grad}")
        print("   ✓ Backward pass successful!")
    except Exception as e:
        print(f"   ✗ Backward pass failed: {e}")
        raise
    
    # Quick training test
    print("\n6. Testing mini training loop...")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    initial_loss = None
    final_loss = None
    
    for epoch in range(20):
        optimizer.zero_grad()
        output = model(values_train)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        
        if epoch == 0:
            initial_loss = loss.item()
        if epoch == 4:
            final_loss = loss.item()
    
    print(f"   - Initial loss: {initial_loss:.6f}")
    print(f"   - Final loss: {final_loss:.6f}")
    print(f"   - Loss decreased: {final_loss < initial_loss}")
    print("   ✓ Training loop test successful!")
    
    print("\n" + "=" * 60)
    print("All tests passed! NUNO is ready for use.")
    print("=" * 60)
    
    # Visualization (optional - only if matplotlib is available)
    try:
        print("\n7. Creating visualization...")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Point cloud with subdomain coloring
        ax = axes[0]
        colors = []
        for i in range(model.n_subdomains):
            indices = model.decomposition.indices_sd[i]
            colors.extend([i] * len(indices))
        
        # Reorder colors to match original point order
        color_arr = np.zeros(n_points)
        for i in range(model.n_subdomains):
            indices = model.decomposition.indices_sd[i]
            color_arr[indices] = i
        
        scatter = ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=color_arr,
            cmap='tab10',
            s=30,
            alpha=0.7,
        )
        ax.set_title(f'Domain Decomposition ({model.n_subdomains} subdomains)')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        plt.colorbar(scatter, ax=ax, label='Subdomain')
        
        # Plot 2: Input values
        ax = axes[1]
        scatter = ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=values[0, :, 0].cpu().numpy(),
            cmap='viridis',
            s=30,
        )
        ax.set_title('Input Values')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        plt.colorbar(scatter, ax=ax, label='Value')
        
        # Plot 3: Output values (after training)
        ax = axes[2]
        with torch.no_grad():
            output_vis = model(values_train)
        scatter = ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=output_vis[0, :, 0].cpu().numpy(),
            cmap='viridis',
            s=30,
        )
        ax.set_title('Output Values (after training)')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        plt.colorbar(scatter, ax=ax, label='Value')
        
        plt.tight_layout()
        plt.savefig('images/nuno_test_visualization.png', dpi=150)
        print("   - Visualization saved to nuno_test_visualization.png")
        plt.close()
        
    except Exception as e:
        print(f"   - Visualization skipped: {e}")