# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Comprehensive test suite for NUNO (Non-Uniform Neural Operator) implementation.

Run with: pytest test_nuno.py -v
Or simply: python test_nuno.py
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

# Import the module to test
from physicsnemo.models.fno.nuno import (
    KDTree,
    KDTreeNode,
    DomainDecomposition,
    NUNO,
    inverse_distance_weighting,
    points_to_grid_2d,
    points_to_grid_3d,
    grid_to_points_2d,
    grid_to_points_3d,
    resize_grid_fft,
    count_parameters,
    create_nuno_from_points,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def random_2d_points():
    """Generate random 2D points in unit square."""
    np.random.seed(42)
    torch.manual_seed(42)
    n_points = 500
    points = np.random.rand(n_points, 2)
    return points


@pytest.fixture
def random_3d_points():
    """Generate random 3D points in unit cube."""
    np.random.seed(42)
    torch.manual_seed(42)
    n_points = 500
    points = np.random.rand(n_points, 3)
    return points


@pytest.fixture
def non_uniform_2d_points():
    """Generate non-uniform 2D points (clustered)."""
    np.random.seed(42)
    # Create clusters
    cluster1 = np.random.randn(200, 2) * 0.1 + [0.2, 0.2]
    cluster2 = np.random.randn(200, 2) * 0.1 + [0.8, 0.8]
    cluster3 = np.random.randn(100, 2) * 0.05 + [0.5, 0.5]
    points = np.vstack([cluster1, cluster2, cluster3])
    # Clip to unit square
    points = np.clip(points, 0.01, 0.99)
    return points


# =============================================================================
# KDTree Tests
# =============================================================================

class TestKDTree:
    """Tests for K-D tree domain decomposition."""
    
    def test_kdtree_creation_2d(self, random_2d_points):
        """Test basic K-D tree creation in 2D."""
        points_list = random_2d_points.tolist()
        tree = KDTree(points_list, dim=2, n_subdomains=8, return_indices=True)
        tree.solve()
        
        # Check we have the right number of subdomains
        assert len(tree.nodes) == 8, f"Expected 8 subdomains, got {len(tree.nodes)}"
        
        # Check all points are accounted for
        indices = tree.get_subdomain_indices()
        all_indices = np.concatenate(indices)
        assert len(all_indices) == len(random_2d_points), "Not all points accounted for"
        assert len(np.unique(all_indices)) == len(random_2d_points), "Duplicate indices found"
    
    def test_kdtree_creation_3d(self, random_3d_points):
        """Test K-D tree creation in 3D."""
        points_list = random_3d_points.tolist()
        tree = KDTree(points_list, dim=3, n_subdomains=8, return_indices=True)
        tree.solve()
        
        assert len(tree.nodes) == 8
        indices = tree.get_subdomain_indices()
        all_indices = np.concatenate(indices)
        assert len(np.unique(all_indices)) == len(random_3d_points)
    
    def test_kdtree_bounding_boxes(self, random_2d_points):
        """Test that bounding boxes are valid."""
        points_list = random_2d_points.tolist()
        tree = KDTree(points_list, dim=2, n_subdomains=4, return_indices=True)
        tree.solve()
        
        bboxes = tree.get_subdomain_bounding_boxes()
        subdomain_points = tree.get_subdomain_points()
        
        for i, (bbox, pts) in enumerate(zip(bboxes, subdomain_points)):
            # Check bbox contains all points
            for d in range(2):
                assert pts[:, d].min() >= bbox[d][0] - 1e-10, \
                    f"Points outside bbox in subdomain {i}, dim {d}"
                assert pts[:, d].max() <= bbox[d][1] + 1e-10, \
                    f"Points outside bbox in subdomain {i}, dim {d}"
    
    def test_kdtree_non_uniform_points(self, non_uniform_2d_points):
        """Test K-D tree with non-uniform (clustered) points."""
        points_list = non_uniform_2d_points.tolist()
        tree = KDTree(points_list, dim=2, n_subdomains=8, return_indices=True)
        tree.solve()
        
        # Should still produce 8 subdomains
        assert len(tree.nodes) == 8
        
        # All points should be assigned
        indices = tree.get_subdomain_indices()
        all_indices = np.concatenate(indices)
        assert len(np.unique(all_indices)) == len(non_uniform_2d_points)
    
    def test_kdtree_small_n_subdomains(self, random_2d_points):
        """Test K-D tree with just 2 subdomains."""
        points_list = random_2d_points.tolist()
        tree = KDTree(points_list, dim=2, n_subdomains=2, return_indices=True)
        tree.solve()
        
        assert len(tree.nodes) == 2
    
    def test_kdtree_large_n_subdomains(self, random_2d_points):
        """Test K-D tree with many subdomains."""
        points_list = random_2d_points.tolist()
        tree = KDTree(points_list, dim=2, n_subdomains=16, return_indices=True)
        tree.solve()
        
        # May have fewer if points don't allow further splitting
        assert len(tree.nodes) <= 16
        assert len(tree.nodes) >= 2


# =============================================================================
# Interpolation Tests
# =============================================================================

class TestInterpolation:
    """Tests for interpolation utilities."""
    
    def test_idw_basic(self):
        """Test basic inverse distance weighting."""
        # Simple test: interpolate at source points should return source values
        source_points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        source_values = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        
        # Query at same points
        result = inverse_distance_weighting(source_points, source_values, source_points, k=4)
        
        # Should be close to original values
        assert torch.allclose(result, source_values, atol=1e-5), \
            f"IDW at source points should return source values"
    
    def test_idw_midpoint(self):
        """Test IDW at midpoint between sources."""
        source_points = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
        source_values = torch.tensor([[0.0], [2.0]])
        query_points = torch.tensor([[1.0, 0.0]])  # Midpoint
        
        result = inverse_distance_weighting(source_points, source_values, query_points, k=2)
        
        # At midpoint, should be average (1.0)
        assert torch.allclose(result, torch.tensor([[1.0]]), atol=1e-5), \
            f"IDW at midpoint should be average, got {result}"
    
    def test_idw_multichannel(self):
        """Test IDW with multiple channels."""
        source_points = torch.rand(100, 2)
        source_values = torch.rand(100, 5)  # 5 channels
        query_points = torch.rand(50, 2)
        
        result = inverse_distance_weighting(source_points, source_values, query_points, k=6)
        
        assert result.shape == (50, 5), f"Wrong output shape: {result.shape}"
    
    def test_points_to_grid_2d_shape(self):
        """Test points_to_grid_2d output shape."""
        points = torch.rand(100, 2)
        values = torch.rand(100, 3)  # 3 channels
        grid_size = (16, 16)
        bbox = ((0.0, 1.0), (0.0, 1.0))
        
        grid = points_to_grid_2d(points, values, grid_size, bbox)
        
        assert grid.shape == (3, 16, 16), f"Wrong shape: {grid.shape}"
    
    def test_points_to_grid_3d_shape(self):
        """Test points_to_grid_3d output shape."""
        points = torch.rand(100, 3)
        values = torch.rand(100, 2)  # 2 channels
        grid_size = (8, 8, 8)
        bbox = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
        
        grid = points_to_grid_3d(points, values, grid_size, bbox)
        
        assert grid.shape == (2, 8, 8, 8), f"Wrong shape: {grid.shape}"
    
    def test_grid_to_points_2d_roundtrip(self):
        """Test grid_to_points_2d consistency."""
        # Create a simple linear field on a grid
        H, W = 16, 16
        C = 2
        grid = torch.zeros(C, H, W)
        
        # Create x and y coordinate grids
        for h in range(H):
            for w in range(W):
                grid[0, h, w] = w / (W - 1)  # x-like: increases with width
                grid[1, h, w] = h / (H - 1)  # y-like: increases with height
        
        bbox = ((0.0, 1.0), (0.0, 1.0))
        
        # Query at grid center
        query_points = torch.tensor([[0.5, 0.5]])
        result = grid_to_points_2d(grid, query_points, bbox)
        
        # At center, both channels should be ~0.5
        expected = torch.tensor([[0.5, 0.5]])
        assert torch.allclose(result.squeeze(0).T, expected, atol=0.1), \
            f"Center value wrong: {result}"
    
    def test_grid_to_points_3d_shape(self):
        """Test grid_to_points_3d output shape."""
        grid = torch.rand(2, 8, 8, 8)  # (C, D, H, W)
        query_points = torch.rand(50, 3)
        bbox = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
        
        result = grid_to_points_3d(grid, query_points, bbox)
        
        assert result.shape == (2, 50), f"Wrong shape: {result.shape}"
    
    def test_points_grid_roundtrip_2d(self):
        """Test round-trip: points -> grid -> points."""
        torch.manual_seed(42)
        n_points = 100
        points = torch.rand(n_points, 2)
        values = torch.rand(n_points, 1)
        
        grid_size = (32, 32)
        bbox = ((0.0, 1.0), (0.0, 1.0))
        
        # Points to grid
        grid = points_to_grid_2d(points, values, grid_size, bbox, k=6)
        
        # Grid back to points
        reconstructed = grid_to_points_2d(grid, points, bbox)
        reconstructed = reconstructed.T  # (C, N) -> (N, C)
        
        # Should be reasonably close (not exact due to interpolation)
        error = (reconstructed - values).abs().mean()
        assert error < 0.5, f"Round-trip error too large: {error}"
    
    def test_resize_grid_fft_2d(self):
        """Test FFT-based grid resizing for 2D."""
        grid = torch.rand(2, 3, 16, 16)  # (B, C, H, W)
        target_size = (32, 32)
        
        resized = resize_grid_fft(grid, target_size)
        
        assert resized.shape == (2, 3, 32, 32), f"Wrong shape: {resized.shape}"
    
    def test_resize_grid_fft_3d(self):
        """Test FFT-based grid resizing for 3D."""
        grid = torch.rand(2, 3, 8, 8, 8)  # (B, C, D, H, W)
        target_size = (16, 16, 16)
        
        resized = resize_grid_fft(grid, target_size)
        
        assert resized.shape == (2, 3, 16, 16, 16), f"Wrong shape: {resized.shape}"
    
    def test_resize_grid_preserves_mean(self):
        """Test that FFT resize preserves mean value approximately."""
        grid = torch.rand(1, 1, 16, 16)
        original_mean = grid.mean()
        
        resized = resize_grid_fft(grid, (32, 32))
        resized_mean = resized.mean()
        
        assert torch.allclose(original_mean, resized_mean, rtol=0.1), \
            f"Mean not preserved: {original_mean} vs {resized_mean}"


# =============================================================================
# Domain Decomposition Tests
# =============================================================================

class TestDomainDecomposition:
    """Tests for DomainDecomposition class."""
    
    def test_decomposition_creation_2d(self, random_2d_points):
        """Test basic domain decomposition creation."""
        points_tensor = torch.from_numpy(random_2d_points).float()
        decomp = DomainDecomposition(points_tensor, n_subdomains=4)
        
        info = decomp.get_subdomain_info()
        assert info['n_subdomains'] == 4
        assert info['n_points'] == len(random_2d_points)
        assert info['dim'] == 2
    
    def test_decomposition_creation_3d(self, random_3d_points):
        """Test domain decomposition in 3D."""
        points_tensor = torch.from_numpy(random_3d_points).float()
        decomp = DomainDecomposition(points_tensor, n_subdomains=4)
        
        info = decomp.get_subdomain_info()
        assert info['dim'] == 3
    
    def test_interpolate_to_grids_2d(self, random_2d_points):
        """Test interpolation from points to grids."""
        points_tensor = torch.from_numpy(random_2d_points).float()
        n_points = len(random_2d_points)
        
        decomp = DomainDecomposition(points_tensor, n_subdomains=4)
        
        # Create random values
        values = torch.rand(n_points, 3)  # 3 channels
        
        grids = decomp.interpolate_to_grids(values)
        
        # Check shape: (n_subdomains, C, H, W)
        assert grids.dim() == 4, f"Wrong number of dimensions: {grids.dim()}"
        assert grids.shape[0] == 4, f"Wrong n_subdomains: {grids.shape[0]}"
        assert grids.shape[1] == 3, f"Wrong n_channels: {grids.shape[1]}"
    
    def test_interpolate_to_grids_batched(self, random_2d_points):
        """Test batched interpolation to grids."""
        points_tensor = torch.from_numpy(random_2d_points).float()
        n_points = len(random_2d_points)
        
        decomp = DomainDecomposition(points_tensor, n_subdomains=4)
        
        # Batched values
        batch_size = 8
        values = torch.rand(batch_size, n_points, 2)
        
        grids = decomp.interpolate_to_grids(values)
        
        # Check shape: (B, n_subdomains, C, H, W)
        assert grids.dim() == 5
        assert grids.shape[0] == batch_size
        assert grids.shape[1] == 4  # n_subdomains
        assert grids.shape[2] == 2  # channels
    
    def test_interpolate_to_points_2d(self, random_2d_points):
        """Test interpolation from grids back to points."""
        points_tensor = torch.from_numpy(random_2d_points).float()
        n_points = len(random_2d_points)
        
        decomp = DomainDecomposition(points_tensor, n_subdomains=4)
        
        values = torch.rand(n_points, 3)
        grids = decomp.interpolate_to_grids(values)
        
        # Interpolate back
        reconstructed, mask = decomp.interpolate_to_points(grids)
        
        assert reconstructed.shape == values.shape, \
            f"Shape mismatch: {reconstructed.shape} vs {values.shape}"
        assert mask.shape == (n_points,)
    
    def test_align_long_side_mode(self, random_2d_points):
        """Test that align_long_side=True works without crashing."""
        points_tensor = torch.from_numpy(random_2d_points).float()
        n_points = len(random_2d_points)
        
        # Enable align_long_side explicitly
        decomp = DomainDecomposition(
            points_tensor, 
            n_subdomains=4,
            align_long_side=True
        )
        
        info = decomp.get_subdomain_info()
        
        # Check that final_grid_sizes may differ from grid_sizes when transposed
        assert 'final_grid_sizes' in info
        assert len(info['final_grid_sizes']) == 4
        
        # Test interpolation still works
        values = torch.rand(n_points, 2)
        grids = decomp.interpolate_to_grids(values)
        
        # All grids should have same shape (resized to max)
        assert grids.dim() == 4  # (n_subdomains, C, H, W)
        
        # Interpolate back
        reconstructed, mask = decomp.interpolate_to_points(grids)
        assert reconstructed.shape == values.shape
    
    def test_roundtrip_error_2d(self, random_2d_points):
        """Test round-trip interpolation error is reasonable."""
        points_tensor = torch.from_numpy(random_2d_points).float()
        n_points = len(random_2d_points)
        
        # Use more subdomains and higher oversampling for better reconstruction
        decomp = DomainDecomposition(
            points_tensor, 
            n_subdomains=8, 
            oversamp_ratio=2.0
        )
        
        values = torch.rand(n_points, 1)
        grids = decomp.interpolate_to_grids(values, k=8)
        reconstructed, _ = decomp.interpolate_to_points(grids)
        
        # Compute relative error
        error = (reconstructed - values).abs().mean() / values.abs().mean()
        
        # Error should be reasonable (< 60% for random data with interpolation)
        # Note: Some error is expected due to:
        # 1. IDW interpolation smoothing
        # 2. Grid discretization
        # 3. FFT resize for mismatched grid sizes
        assert error < 0.6, f"Round-trip error too large: {error}"
    
    def test_grid_sizes_vary(self, non_uniform_2d_points):
        """Test that grid sizes adapt to subdomain point density."""
        points_tensor = torch.from_numpy(non_uniform_2d_points).float()
        
        decomp = DomainDecomposition(points_tensor, n_subdomains=4)
        
        info = decomp.get_subdomain_info()
        grid_sizes = info['grid_sizes']
        final_grid_sizes = info['final_grid_sizes']
        
        # Grid sizes should exist and be valid
        assert len(grid_sizes) == 4
        assert len(final_grid_sizes) == 4
        for gs in grid_sizes:
            assert len(gs) == 2  # 2D
            assert all(s >= 2 for s in gs)  # Minimum grid size
        
        # With align_long_side=False (default), grid_sizes == final_grid_sizes
        for gs, fgs in zip(grid_sizes, final_grid_sizes):
            assert gs == fgs, "With align_long_side=False, sizes should match"


# =============================================================================
# NUNO Model Tests
# =============================================================================

class TestNUNOModel:
    """Tests for the NUNO model."""
    
    def test_nuno_creation_2d(self):
        """Test NUNO model creation for 2D."""
        model = NUNO(
            in_channels=1,
            out_channels=1,
            n_subdomains=4,
            dimension=2,
            num_fno_layers=2,
            latent_channels=16,
            num_fno_modes=8,
            padding=4
        )
        
        assert model.in_channels == 1
        assert model.out_channels == 1
        assert model.n_subdomains == 4
        assert model.dimension == 2
    
    def test_nuno_creation_3d(self):
        """Test NUNO model creation for 3D."""
        model = NUNO(
            in_channels=2,
            out_channels=1,
            n_subdomains=4,
            dimension=3,
            num_fno_layers=2,
            latent_channels=16,
            num_fno_modes=4,
            padding=2
        )
        
        assert model.dimension == 3
    
    def test_nuno_setup_domain(self, random_2d_points):
        """Test domain setup."""
        model = NUNO(
            in_channels=1,
            out_channels=1,
            n_subdomains=4,
            dimension=2,
            num_fno_layers=2,
            latent_channels=16
        )
        
        points_tensor = torch.from_numpy(random_2d_points).float()
        model.setup_domain(points_tensor)
        
        assert model.domain_decomp is not None
    
    def test_nuno_forward_without_setup_raises(self):
        """Test that forward without setup_domain raises error."""
        model = NUNO(
            in_channels=1,
            out_channels=1,
            n_subdomains=4,
            dimension=2,
            num_fno_layers=2,
            latent_channels=16
        )
        
        x = torch.rand(10, 100, 1)
        
        with pytest.raises(RuntimeError, match="Domain decomposition not set up"):
            model(x)
    
    def test_nuno_forward_2d(self, random_2d_points):
        """Test NUNO forward pass in 2D."""
        torch.manual_seed(42)
        
        model = NUNO(
            in_channels=1,
            out_channels=1,
            n_subdomains=4,
            dimension=2,
            num_fno_layers=2,
            latent_channels=16,
            num_fno_modes=8,
            padding=4,
            coord_features=False  # Disable for simplicity
        )
        
        points_tensor = torch.from_numpy(random_2d_points).float()
        model.setup_domain(points_tensor)
        
        n_points = len(random_2d_points)
        batch_size = 4
        x = torch.rand(batch_size, n_points, 1)
        
        y = model(x)
        
        assert y.shape == (batch_size, n_points, 1), f"Wrong output shape: {y.shape}"
    
    def test_nuno_forward_unbatched(self, random_2d_points):
        """Test NUNO forward pass without batch dimension."""
        torch.manual_seed(42)
        
        model = NUNO(
            in_channels=1,
            out_channels=1,
            n_subdomains=4,
            dimension=2,
            num_fno_layers=2,
            latent_channels=16,
            coord_features=False
        )
        
        points_tensor = torch.from_numpy(random_2d_points).float()
        model.setup_domain(points_tensor)
        
        n_points = len(random_2d_points)
        x = torch.rand(n_points, 1)
        
        y = model(x)
        
        assert y.shape == (n_points, 1), f"Wrong output shape: {y.shape}"
    
    def test_nuno_forward_multichannel(self, random_2d_points):
        """Test NUNO with multiple input/output channels."""
        torch.manual_seed(42)
        
        model = NUNO(
            in_channels=3,
            out_channels=2,
            n_subdomains=4,
            dimension=2,
            num_fno_layers=2,
            latent_channels=16,
            coord_features=False
        )
        
        points_tensor = torch.from_numpy(random_2d_points).float()
        model.setup_domain(points_tensor)
        
        n_points = len(random_2d_points)
        batch_size = 4
        x = torch.rand(batch_size, n_points, 3)
        
        y = model(x)
        
        assert y.shape == (batch_size, n_points, 2), f"Wrong output shape: {y.shape}"
    
    def test_nuno_forward_with_coord_features(self, random_2d_points):
        """Test NUNO with coordinate features enabled."""
        torch.manual_seed(42)
        
        model = NUNO(
            in_channels=1,
            out_channels=1,
            n_subdomains=4,
            dimension=2,
            num_fno_layers=2,
            latent_channels=16,
            coord_features=True  # Enable coord features
        )
        
        points_tensor = torch.from_numpy(random_2d_points).float()
        model.setup_domain(points_tensor)
        
        n_points = len(random_2d_points)
        x = torch.rand(2, n_points, 1)
        
        y = model(x)
        
        assert y.shape == (2, n_points, 1)
    
    def test_nuno_forward_return_grid(self, random_2d_points):
        """Test NUNO forward with return_grid=True."""
        torch.manual_seed(42)
        
        model = NUNO(
            in_channels=1,
            out_channels=1,
            n_subdomains=4,
            dimension=2,
            num_fno_layers=2,
            latent_channels=16,
            coord_features=False
        )
        
        points_tensor = torch.from_numpy(random_2d_points).float()
        model.setup_domain(points_tensor)
        
        n_points = len(random_2d_points)
        x = torch.rand(2, n_points, 1)
        
        point_out, grid_out = model(x, return_grid=True)
        
        assert point_out.shape == (2, n_points, 1)
        assert grid_out.dim() == 5  # (B, n_subdomains, C, H, W)
        assert grid_out.shape[0] == 2
        assert grid_out.shape[1] == 4  # n_subdomains
        assert grid_out.shape[2] == 1  # out_channels
    
    def test_nuno_gradient_flow(self, random_2d_points):
        """Test that gradients flow through the model."""
        torch.manual_seed(42)
        
        model = NUNO(
            in_channels=1,
            out_channels=1,
            n_subdomains=4,
            dimension=2,
            num_fno_layers=2,
            latent_channels=16,
            coord_features=False
        )
        
        points_tensor = torch.from_numpy(random_2d_points).float()
        model.setup_domain(points_tensor)
        
        n_points = len(random_2d_points)
        x = torch.rand(2, n_points, 1, requires_grad=True)
        
        y = model(x)
        loss = y.sum()
        loss.backward()
        
        # Check gradients exist for model parameters
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        
        assert has_grad, "No gradients in model parameters"
    
    def test_nuno_forward_on_grid(self, random_2d_points):
        """Test forward_on_grid method."""
        torch.manual_seed(42)
        
        model = NUNO(
            in_channels=1,
            out_channels=1,
            n_subdomains=4,
            dimension=2,
            num_fno_layers=2,
            latent_channels=16,
            coord_features=False
        )
        
        # Create grid input directly (B, C*n_subdomains, H, W)
        x = torch.rand(2, 4, 16, 16)  # 4 = in_channels * n_subdomains
        
        y = model.forward_on_grid(x)
        
        assert y.shape == (2, 4, 16, 16)  # 4 = out_channels * n_subdomains
    
    def test_nuno_forward_on_grid_small(self):
        """Test forward_on_grid with small grids (dynamic padding)."""
        torch.manual_seed(42)
        
        model = NUNO(
            in_channels=1,
            out_channels=1,
            n_subdomains=4,
            dimension=2,
            num_fno_layers=2,
            latent_channels=16,
            num_fno_modes=16,  # Large modes
            coord_features=False
        )
        
        # Create small grid that would normally fail with modes=16
        # The dynamic padding should handle this
        x = torch.rand(2, 4, 8, 10)  # Small grid: 8x10
        
        y = model.forward_on_grid(x)
        
        # Output should be same spatial size as input
        assert y.shape == (2, 4, 8, 10), f"Wrong output shape: {y.shape}"
    
    def test_nuno_3d_forward(self, random_3d_points):
        """Test NUNO forward pass in 3D."""
        torch.manual_seed(42)
        
        model = NUNO(
            in_channels=1,
            out_channels=1,
            n_subdomains=4,
            dimension=3,
            num_fno_layers=2,
            latent_channels=16,
            num_fno_modes=4,
            padding=2,
            coord_features=False
        )
        
        points_tensor = torch.from_numpy(random_3d_points).float()
        model.setup_domain(points_tensor)
        
        n_points = len(random_3d_points)
        batch_size = 2
        x = torch.rand(batch_size, n_points, 1)
        
        y = model(x)
        
        assert y.shape == (batch_size, n_points, 1), f"Wrong output shape: {y.shape}"


# =============================================================================
# Utility Function Tests
# =============================================================================

class TestUtilities:
    """Tests for utility functions."""
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = nn.Linear(10, 5)
        n_params = count_parameters(model)
        
        expected = 10 * 5 + 5  # weights + bias
        assert n_params == expected, f"Wrong param count: {n_params} vs {expected}"
    
    def test_create_nuno_from_points_2d(self, random_2d_points):
        """Test convenience constructor for 2D."""
        points_tensor = torch.from_numpy(random_2d_points).float()
        
        model = create_nuno_from_points(
            points_tensor,
            in_channels=1,
            out_channels=1,
            n_subdomains=4,
            num_fno_layers=2,
            latent_channels=16
        )
        
        assert model.domain_decomp is not None
        assert model.dimension == 2
    
    def test_create_nuno_from_points_3d(self, random_3d_points):
        """Test convenience constructor for 3D."""
        points_tensor = torch.from_numpy(random_3d_points).float()
        
        model = create_nuno_from_points(
            points_tensor,
            in_channels=1,
            out_channels=1,
            n_subdomains=4,
            num_fno_layers=2,
            latent_channels=16
        )
        
        assert model.domain_decomp is not None
        assert model.dimension == 3


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_single_point_subdomain(self):
        """Test handling when subdomain has very few points."""
        # Create points that force small subdomains
        np.random.seed(42)
        points = np.random.rand(32, 2)  # Only 32 points
        points_list = points.tolist()
        
        # Request 8 subdomains - may not get all 8
        tree = KDTree(points_list, dim=2, n_subdomains=8, 
                     smallest_points=2, return_indices=True)
        tree.solve()
        
        # Should have created some subdomains
        assert len(tree.nodes) >= 2
    
    def test_collinear_points(self):
        """Test handling of collinear points."""
        # Points along a line
        t = np.linspace(0, 1, 100)
        points = np.stack([t, t], axis=1)  # y = x
        points_list = points.tolist()
        
        tree = KDTree(points_list, dim=2, n_subdomains=4, return_indices=True)
        tree.solve()
        
        # Should still work, though splits may be limited
        assert len(tree.nodes) >= 1
    
    def test_duplicate_points(self):
        """Test handling of duplicate points."""
        points = np.array([
            [0.5, 0.5],
            [0.5, 0.5],  # Duplicate
            [0.1, 0.1],
            [0.9, 0.9],
        ] * 25)  # 100 points with duplicates
        
        points_list = points.tolist()
        tree = KDTree(points_list, dim=2, n_subdomains=4, return_indices=True)
        tree.solve()
        
        assert len(tree.nodes) >= 1
    
    def test_very_small_bbox(self):
        """Test handling of very small bounding boxes."""
        # Points in a tiny region
        points = np.random.rand(100, 2) * 1e-6 + 0.5
        points_tensor = torch.from_numpy(points).float()
        
        decomp = DomainDecomposition(points_tensor, n_subdomains=4)
        
        # Should still work
        info = decomp.get_subdomain_info()
        assert info['n_subdomains'] == 4
    
    def test_large_batch_size(self, random_2d_points):
        """Test with large batch size."""
        torch.manual_seed(42)
        
        model = NUNO(
            in_channels=1,
            out_channels=1,
            n_subdomains=4,
            dimension=2,
            num_fno_layers=2,
            latent_channels=16,
            coord_features=False
        )
        
        points_tensor = torch.from_numpy(random_2d_points).float()
        model.setup_domain(points_tensor)
        
        n_points = len(random_2d_points)
        batch_size = 32
        x = torch.rand(batch_size, n_points, 1)
        
        y = model(x)
        
        assert y.shape == (batch_size, n_points, 1)


# =============================================================================
# Numerical Accuracy Tests
# =============================================================================

class TestNumericalAccuracy:
    """Tests for numerical accuracy and stability."""
    
    def test_idw_no_nan(self):
        """Test that IDW doesn't produce NaN values."""
        source_points = torch.rand(100, 2)
        source_values = torch.rand(100, 3)
        query_points = torch.rand(50, 2)
        
        result = inverse_distance_weighting(source_points, source_values, query_points)
        
        assert not torch.isnan(result).any(), "NaN values in IDW output"
        assert not torch.isinf(result).any(), "Inf values in IDW output"
    
    def test_grid_sample_boundary(self):
        """Test grid sampling at boundaries."""
        grid = torch.rand(3, 16, 16)
        bbox = ((0.0, 1.0), (0.0, 1.0))
        
        # Query at corners
        corners = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])
        
        result = grid_to_points_2d(grid, corners, bbox)
        
        assert not torch.isnan(result).any(), "NaN at corners"
    
    def test_fft_resize_stability(self):
        """Test FFT resize numerical stability."""
        # Test with various sizes
        for size in [(8, 8), (16, 16), (32, 32)]:
            grid = torch.rand(1, 1, *size)
            
            # Upscale then downscale
            up = resize_grid_fft(grid, (size[0] * 2, size[1] * 2))
            down = resize_grid_fft(up, size)
            
            assert not torch.isnan(down).any(), f"NaN after resize at size {size}"
    
    def test_model_output_range(self, random_2d_points):
        """Test that model outputs are in reasonable range."""
        torch.manual_seed(42)
        
        model = NUNO(
            in_channels=1,
            out_channels=1,
            n_subdomains=4,
            dimension=2,
            num_fno_layers=2,
            latent_channels=16,
            coord_features=False
        )
        
        points_tensor = torch.from_numpy(random_2d_points).float()
        model.setup_domain(points_tensor)
        
        # Input in [0, 1]
        x = torch.rand(2, len(random_2d_points), 1)
        y = model(x)
        
        # Output should not explode
        assert y.abs().max() < 1000, f"Output too large: {y.abs().max()}"
        assert not torch.isnan(y).any(), "NaN in model output"


# =============================================================================
# 3D Specific Tests
# =============================================================================

class Test3DSpecific:
    """Tests specific to 3D functionality."""
    
    def test_3d_grid_ordering(self):
        """Test that 3D grid ordering is correct.
        
        This is a critical test for the bug found in points_to_grid_3d.
        The meshgrid with indexing='xy' and reshape must be consistent.
        """
        # Create a simple 3D field where we know the expected values
        # f(x, y, z) = x + 10*y + 100*z
        
        # Create grid points
        n = 8
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 1, n)
        z = torch.linspace(0, 1, n)
        
        # Create all points
        points_list = []
        values_list = []
        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                for k, zk in enumerate(z):
                    points_list.append([xi.item(), yj.item(), zk.item()])
                    values_list.append([xi.item() + 10*yj.item() + 100*zk.item()])
        
        points = torch.tensor(points_list)
        values = torch.tensor(values_list)
        
        # Interpolate to grid (should be exact since points are on grid)
        grid_size = (n, n, n)
        bbox = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
        
        grid = points_to_grid_3d(points, values, grid_size, bbox, k=1)
        
        # Check grid shape
        assert grid.shape == (1, n, n, n), f"Wrong shape: {grid.shape}"
        
        # Verify some specific values
        # grid[0, d, h, w] should be x[w] + 10*y[h] + 100*z[d]
        # where x, y, z are linspace(0, 1, n)
        for d in range(n):
            for h in range(n):
                for w in range(n):
                    expected = w/(n-1) + 10*h/(n-1) + 100*d/(n-1)
                    actual = grid[0, d, h, w].item()
                    # Allow some tolerance due to interpolation
                    assert abs(actual - expected) < 0.5, \
                        f"Grid value wrong at ({d}, {h}, {w}): {actual} vs {expected}"
    
    def test_3d_roundtrip(self, random_3d_points):
        """Test 3D points -> grid -> points roundtrip."""
        points_tensor = torch.from_numpy(random_3d_points).float()
        n_points = len(random_3d_points)
        
        decomp = DomainDecomposition(points_tensor, n_subdomains=4, oversamp_ratio=2.0)
        
        values = torch.rand(n_points, 2)
        grids = decomp.interpolate_to_grids(values, k=8)
        reconstructed, _ = decomp.interpolate_to_points(grids)
        
        error = (reconstructed - values).abs().mean()
        
        # Error should be reasonable
        assert error < 1.0, f"3D round-trip error too large: {error}"


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def test_training_step_simulation(self, random_2d_points):
        """Simulate a training step."""
        torch.manual_seed(42)
        
        model = NUNO(
            in_channels=1,
            out_channels=1,
            n_subdomains=4,
            dimension=2,
            num_fno_layers=2,
            latent_channels=16,
            coord_features=False
        )
        
        points_tensor = torch.from_numpy(random_2d_points).float()
        model.setup_domain(points_tensor)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        n_points = len(random_2d_points)
        batch_size = 4
        
        # Simulate a few training steps
        for _ in range(3):
            x = torch.rand(batch_size, n_points, 1)
            y_target = torch.rand(batch_size, n_points, 1)
            
            optimizer.zero_grad()
            y_pred = model(x)
            loss = ((y_pred - y_target) ** 2).mean()
            loss.backward()
            optimizer.step()
        
        # If we get here without error, the test passes
        assert True
    
    def test_inference_mode(self, random_2d_points):
        """Test model in inference mode."""
        torch.manual_seed(42)
        
        model = NUNO(
            in_channels=1,
            out_channels=1,
            n_subdomains=4,
            dimension=2,
            num_fno_layers=2,
            latent_channels=16,
            coord_features=False
        )
        
        points_tensor = torch.from_numpy(random_2d_points).float()
        model.setup_domain(points_tensor)
        
        model.eval()
        
        with torch.no_grad():
            x = torch.rand(4, len(random_2d_points), 1)
            y = model(x)
        
        assert y.shape == (4, len(random_2d_points), 1)
    
    def test_model_save_load(self, random_2d_points, tmp_path):
        """Test saving and loading model weights."""
        torch.manual_seed(42)
        
        model = NUNO(
            in_channels=1,
            out_channels=1,
            n_subdomains=4,
            dimension=2,
            num_fno_layers=2,
            latent_channels=16,
            coord_features=False
        )
        
        points_tensor = torch.from_numpy(random_2d_points).float()
        model.setup_domain(points_tensor)
        
        # Get output before save
        x = torch.rand(2, len(random_2d_points), 1)
        with torch.no_grad():
            y_before = model(x)
        
        # Save
        save_path = tmp_path / "nuno_model.pt"
        torch.save(model.state_dict(), save_path)
        
        # Create new model and load
        model2 = NUNO(
            in_channels=1,
            out_channels=1,
            n_subdomains=4,
            dimension=2,
            num_fno_layers=2,
            latent_channels=16,
            coord_features=False
        )
        model2.setup_domain(points_tensor)
        model2.load_state_dict(torch.load(save_path))
        
        # Get output after load
        with torch.no_grad():
            y_after = model2(x)
        
        assert torch.allclose(y_before, y_after), "Outputs differ after load"


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Run with pytest for full output
    # Or run directly for quick check
    print("Running NUNO tests...")
    print("=" * 60)
    
    # Quick smoke tests
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate test data
    n_points = 200
    points_2d = np.random.rand(n_points, 2)
    points_3d = np.random.rand(n_points, 3)
    
    print("\n1. Testing K-D Tree...")
    tree = KDTree(points_2d.tolist(), dim=2, n_subdomains=4, return_indices=True)
    tree.solve()
    print(f"   Created {len(tree.nodes)} subdomains")
    
    print("\n2. Testing Domain Decomposition...")
    decomp = DomainDecomposition(torch.from_numpy(points_2d).float(), n_subdomains=4)
    info = decomp.get_subdomain_info()
    print(f"   Grid sizes: {info['grid_sizes']}")
    print(f"   Final grid sizes: {info['final_grid_sizes']}")
    print(f"   Max grid size: {info['max_grid_size']}")
    
    print("\n3. Testing Interpolation...")
    values = torch.rand(n_points, 2)
    grids = decomp.interpolate_to_grids(values)
    print(f"   Grid shape: {grids.shape}")
    reconstructed, _ = decomp.interpolate_to_points(grids)
    print(f"   Reconstructed shape: {reconstructed.shape}")
    error = (reconstructed - values).abs().mean()
    print(f"   Round-trip error: {error:.4f}")
    
    print("\n4. Testing NUNO Model (2D)...")
    model = NUNO(
        in_channels=1,
        out_channels=1,
        n_subdomains=4,
        dimension=2,
        num_fno_layers=2,
        latent_channels=16,
        num_fno_modes=8,  # Smaller modes for faster testing
        coord_features=False
    )
    model.setup_domain(torch.from_numpy(points_2d).float())
    print(f"   Max grid size: {model.domain_decomp.max_grid_size}")
    print(f"   FNO modes: {model.num_fno_modes}")
    x = torch.rand(2, n_points, 1)
    y = model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {y.shape}")
    print(f"   Parameters: {count_parameters(model)}")
    
    print("\n5. Testing NUNO Model (3D)...")
    model_3d = NUNO(
        in_channels=1,
        out_channels=1,
        n_subdomains=4,
        dimension=3,
        num_fno_layers=2,
        latent_channels=16,
        num_fno_modes=4,  # Smaller modes for 3D
        coord_features=False
    )
    model_3d.setup_domain(torch.from_numpy(points_3d).float())
    print(f"   Max grid size: {model_3d.domain_decomp.max_grid_size}")
    x_3d = torch.rand(2, n_points, 1)
    y_3d = model_3d(x_3d)
    print(f"   Input shape: {x_3d.shape}")
    print(f"   Output shape: {y_3d.shape}")
    
    print("\n6. Testing Gradient Flow...")
    x = torch.rand(2, n_points, 1, requires_grad=True)
    y = model(x)
    loss = y.sum()
    loss.backward()
    grad_exists = any(p.grad is not None and p.grad.abs().sum() > 0 
                      for p in model.parameters())
    print(f"   Gradients flow correctly: {grad_exists}")
    
    print("\n" + "=" * 60)
    print("All smoke tests passed!")
    print("\nFor comprehensive testing, run: pytest test_nuno.py -v")