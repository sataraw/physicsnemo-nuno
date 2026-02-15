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
Point Samplers for non-uniform data generation from uniform grids.
Coordinates are generated in the normalized domain [-1, 1] x [-1, 1].
"""

import torch
from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def inverse_distance_weighting(
    source_points: Tensor,
    source_values: Tensor,
    query_points: Tensor,
    k: int = 6,
    power: float = 2.0,
    epsilon: float = 1e-10
) -> Tensor:
    """
    Inverse Distance Weighting (IDW) interpolation.
    
    Parameters
    ----------
    source_points : Tensor
        Source point coordinates, shape (N, D)
    source_values : Tensor
        Values at source points, shape (N, C) or (N,)
    query_points : Tensor
        Query point coordinates, shape (M, D)
    k : int, optional
        Number of nearest neighbors to use, by default 6
    power : float, optional
        Power parameter for distance weighting, by default 2.0
    epsilon : float, optional
        Small constant to avoid division by zero, by default 1e-10
        
    Returns
    -------
    Tensor
        Interpolated values at query points, shape (M, C) or (M,)
    """
    # Ensure proper shapes
    if source_values.dim() == 1:
        source_values = source_values.unsqueeze(-1)
        squeeze_output = True
    else:
        squeeze_output = False
    
    N, D = source_points.shape
    M = query_points.shape[0]
    C = source_values.shape[1]
    
    # Compute pairwise distances: (M, N)
    # query_points: (M, D) -> (M, 1, D)
    # source_points: (N, D) -> (1, N, D)
    diff = query_points.unsqueeze(1) - source_points.unsqueeze(0)
    distances = torch.sqrt((diff ** 2).sum(dim=-1) + epsilon)  # (M, N)
    
    # Find k nearest neighbors
    k = min(k, N)
    knn_distances, knn_indices = torch.topk(distances, k, dim=1, largest=False)
    
    # Compute weights: w_i = 1 / d_i^power
    weights = 1.0 / (knn_distances ** power + epsilon)  # (M, k)
    weights = weights / weights.sum(dim=1, keepdim=True)  # Normalize
    
    # Gather values for k nearest neighbors
    knn_values = source_values[knn_indices]  # (M, k, C)
    
    # Weighted sum
    result = (weights.unsqueeze(-1) * knn_values).sum(dim=1)  # (M, C)
    
    if squeeze_output:
        result = result.squeeze(-1)
    
    return result


def grid_to_points_2d(
    grid: Tensor,
    query_points: Tensor,
    bbox: Tuple[Tuple[float, float], Tuple[float, float]],
    mode: str = 'bilinear',
    padding_mode: str = 'border',
    align_corners: bool = False
) -> Tensor:
    """
    Interpolate from a 2D grid to scattered points using grid_sample.
    
    Parameters
    ----------
    grid : Tensor
        Grid values, shape (B, C, H, W) or (C, H, W)
    query_points : Tensor
        Query point coordinates, shape (N, 2) or (B, N, 2)
    bbox : Tuple[Tuple[float, float], Tuple[float, float]]
        Bounding box ((x_min, x_max), (y_min, y_max))
    mode : str, optional
        Interpolation mode, by default 'bilinear'
    padding_mode : str, optional
        Padding mode for out-of-bounds, by default 'border'
    align_corners : bool, optional
        Whether to align corners, by default False
        
    Returns
    -------
    Tensor
        Values at query points, shape (B, C, N) or (C, N)
    """
    # Handle batch dimension
    had_batch = grid.dim() == 4
    if grid.dim() == 3:
        grid = grid.unsqueeze(0)
    if query_points.dim() == 2:
        query_points = query_points.unsqueeze(0)
    
    B, C, H, W = grid.shape
    N = query_points.shape[1]
    
    # Normalize coordinates to [-1, 1] for grid_sample
    normalized_points = query_points.clone()
    normalized_points[..., 0] = 2 * (query_points[..., 0] - bbox[0][0]) / (bbox[0][1] - bbox[0][0]) - 1
    normalized_points[..., 1] = 2 * (query_points[..., 1] - bbox[1][0]) / (bbox[1][1] - bbox[1][0]) - 1
    
    # Reshape for grid_sample: (B, N, 1, 2)
    grid_coords = normalized_points.unsqueeze(2)
    
    # Apply grid_sample
    # grid_sample expects (B, C, H, W) and (B, H_out, W_out, 2)
    
    # Match batch size for grid_coords
    if grid.shape[0] != grid_coords.shape[0]:
        if grid_coords.shape[0] == 1:
            grid_coords = grid_coords.expand(grid.shape[0], -1, -1, -1)
        else:
            raise ValueError(
                f"grid_sampler(): expected grid and input to have same batch size, "
                f"but got input with sizes {grid.shape} and grid with sizes {grid_coords.shape}"
            )

    values = F.grid_sample(
        grid, grid_coords, 
        mode=mode, padding_mode=padding_mode, align_corners=align_corners
    )  # (B, C, N, 1)
    
    values = values.squeeze(-1)  # (B, C, N)
    
    if not had_batch:
        values = values.squeeze(0)
    
    return values


class PointSampler:
    """
    Base class for point cloud samplers.

    Generates a fixed set of coordinates in the [-1, 1] domain and samples
    values from uniform grids at these locations using interpolation.

    Parameters
    ----------
    num_points : int
        The number of points to sample in the point cloud.
    device : torch.device
        The device to store coordinates on.
    """
    def __init__(self, num_points: int, device: torch.device):
        self.num_points = num_points
        self.device = device
        self.coordinates = self._generate_coordinates().to(device)

    def _generate_coordinates(self) -> torch.Tensor:
        """
        Abstract method to generate the point cloud coordinates.
        Must be implemented by subclasses.

        Returns
        -------
        torch.Tensor
            A tensor of shape (N, 2) with coordinates in the range [-1, 1].
        """
        raise NotImplementedError("Subclasses must implement _generate_coordinates")

    def sample(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Samples values from a uniform grid at the pre-defined coordinates.

        Parameters
        ----------
        grid : torch.Tensor
            A uniform grid tensor of shape (B, C, H, W) defined over [-1, 1]^2.

        Returns
        -------
        torch.Tensor
            A tensor of point values of shape (B, N, C).
        """
        # The grid and coordinates are both in the [-1, 1] x [-1, 1] domain.
        bbox = ((-1.0, 1.0), (-1.0, 1.0))
        
        # grid_to_points_2d returns (B, C, N), we want (B, N, C)
        # align_corners=False is standard for grid_sample when points are not pixels.
        sampled_values = grid_to_points_2d(
            grid,
            self.coordinates,
            bbox,
            align_corners=False
        )
        return sampled_values.permute(0, 2, 1)

class RandomSampler(PointSampler):
    """
    Generates randomly distributed points in the [-1, 1] x [-1, 1] domain.
    """
    def _generate_coordinates(self) -> torch.Tensor:
        print(f"Generating {self.num_points} random points in [-1, 1] x [-1, 1].")
        # torch.rand gives [0, 1), so scale to [-1, 1)
        return torch.rand(self.num_points, 2, device=self.device) * 2 - 1

class UniformSampler(PointSampler):
    """
    Generates points on a uniform grid covering the [-1, 1] x [-1, 1] domain.
    The number of points will be close to `num_points`, but adjusted to form a
    perfect grid.
    """
    def _generate_coordinates(self) -> torch.Tensor:
        # Calculate grid dimensions that best approximate num_points
        pts_per_axis = int(self.num_points**0.5)
        self.num_points = pts_per_axis * pts_per_axis
        print(f"Generating {self.num_points} uniform points in a {pts_per_axis}x{pts_per_axis} grid over [-1, 1]^2.")
        
        x = torch.linspace(-1, 1, pts_per_axis, device=self.device)
        y = torch.linspace(-1, 1, pts_per_axis, device=self.device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        return coords

class LocallyDenseSampler(PointSampler):
    """
    Generates points with a locally dense region (a box) inside the [-1, 1]^2 domain.

    Parameters
    ----------
    num_points : int
        Total number of points to sample.
    device : torch.device
        The device to store coordinates on.
    box_size : Tuple[float, float]
        The (width, height) of the dense box, relative to the domain size (2x2).
    center : Tuple[float, float]
        The (x, y) center of the dense box in [-1, 1]^2.
    inner_ratio : float
        The fraction of points to be sampled inside the dense box.
    """
    def __init__(
        self,
        num_points: int,
        device: torch.device,
        box_size: Tuple[float, float] = (0.4, 0.4), # e.g., 0.4 corresponds to 20% of the 2-unit width
        center: Tuple[float, float] = (0.0, 0.0),
        inner_ratio: float = 0.6
    ):
        self.box_w, self.box_h = box_size
        self.center_x, self.center_y = center
        self.inner_ratio = inner_ratio
        
        self.box_x_min = self.center_x - self.box_w / 2
        self.box_x_max = self.center_x + self.box_w / 2
        self.box_y_min = self.center_y - self.box_h / 2
        self.box_y_max = self.center_y + self.box_h / 2

        super().__init__(num_points, device)

    def _generate_coordinates(self) -> torch.Tensor:
        num_inner = int(self.num_points * self.inner_ratio)
        num_outer = self.num_points - num_inner
        
        print(f"Generating {self.num_points} locally dense points in [-1, 1]^2:")
        print(f"  - {num_inner} points inside box centered at ({self.center_x:.2f}, {self.center_y:.2f}) with size ({self.box_w:.2f}, {self.box_h:.2f})")
        print(f"  - {num_outer} points outside.")

        # Generate points inside the box
        inner_pts_x = self.box_x_min + torch.rand(num_inner, 1, device=self.device) * self.box_w
        inner_pts_y = self.box_y_min + torch.rand(num_inner, 1, device=self.device) * self.box_h
        inner_pts = torch.cat([inner_pts_x, inner_pts_y], dim=1)

        # Generate points outside the box
        outer_pts = torch.empty(num_outer, 2, device=self.device)
        i = 0
        while i < num_outer:
            # Generate points in the full [-1, 1] domain
            rand_pts = torch.rand(num_outer - i, 2, device=self.device) * 2 - 1
            # Find points that are actually outside the box
            is_outside = ~((rand_pts[:, 0] >= self.box_x_min) & (rand_pts[:, 0] <= self.box_x_max) & \
                           (rand_pts[:, 1] >= self.box_y_min) & (rand_pts[:, 1] <= self.box_y_max))
            
            num_found = is_outside.sum()
            if num_found > 0:
                found_pts = rand_pts[is_outside]
                take_n = min(num_found, num_outer - i)
                outer_pts[i : i + take_n] = found_pts[:take_n]
                i += take_n
        
        return torch.cat([inner_pts, outer_pts], dim=0)

def get_sampler(name, num_points, box_size, center, inner_ratio, device: torch.device) -> PointSampler:
    """
    Factory function to create a sampler from a Hydra config.
    """
    if name == "random":
        return RandomSampler(num_points, device)
    elif name == "uniform":
        return UniformSampler(num_points, device)
    elif name == "locally_dense":
        return LocallyDenseSampler(
            num_points=num_points,
            device=device,
            box_size=box_size,
            center=center,
            inner_ratio=inner_ratio
        )