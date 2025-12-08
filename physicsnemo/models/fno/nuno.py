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
Non-Uniform Neural Operator (NUNO) for PhysicsNeMo

This module implements the NUNO framework for learning parametric PDEs with non-uniform data.
NUNO extends mesh-based neural operators (like FNO) to handle non-uniform point cloud data
through K-D tree-based domain decomposition and adaptive grid interpolation.

Reference:
    Liu, S., Hao, Z., Ying, C., Su, H., Cheng, Z., & Zhu, J. (2023).
    NUNO: A General Framework for Learning Parametric PDEs with Non-Uniform Data.
    Proceedings of the 40th International Conference on Machine Learning.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import physicsnemo  # noqa: F401 for docs
import physicsnemo.models.layers as layers
from physicsnemo.models.meta import ModelMetaData
from physicsnemo.models.mlp import FullyConnected
from physicsnemo.models.module import Module


# ===================================================================
# K-D Tree Implementation for Domain Decomposition
# ===================================================================

BIG_CONSTANT = 1e100


class KDTreeNode:
    """
    A Tree Node used by KDTree for domain decomposition.
    
    This node represents a subdomain in the K-D tree structure,
    containing points and implementing splitting logic based on
    KL divergence optimization.
    """
    
    def __init__(
        self,
        depth: int,
        points: List[List[float]],
        dim: int,
        n_blocks: int,
        smallest_points: int,
        max_depth: Optional[int],
        borders: List[Tuple[float, float]]
    ) -> None:
        """
        Initialize a K-D tree node.
        
        Parameters
        ----------
        depth : int
            Current depth in the tree
        points : List[List[float]]
            List of points in this node's subdomain
        dim : int
            Dimensionality of the points
        n_blocks : int
            Number of blocks for KL divergence calculation
        smallest_points : int
            Minimum number of points allowed in a subdomain
        max_depth : Optional[int]
            Maximum tree depth (None for no limit)
        borders : List[Tuple[float, float]]
            Boundaries of this subdomain
        """
        self.depth = depth
        self.points = points
        self.points_np = np.array(points)
        self.dim = dim
        self.n_blocks = n_blocks
        self.smallest_points = smallest_points
        self.max_depth = max_depth
        self.borders = borders
        self.alive = True
        
        # Calculate the bounding box
        self.bbox = [
            (np.min(self.points_np[:, i]), np.max(self.points_np[:, i])) 
            for i in range(self.dim)
        ]
        self.choose_split()
        self.ready = True

    def cal_kl_divergence(
        self, 
        point_set: np.ndarray, 
        bbox: List[Tuple[float, float]]
    ) -> Tuple[List[int], float]:
        """
        Calculate KL divergence of point distribution vs uniform distribution.
        
        Parameters
        ----------
        point_set : np.ndarray
            Set of points to analyze
        bbox : List[Tuple[float, float]]
            Bounding box for the point set
            
        Returns
        -------
        Tuple[List[int], float]
            Shape of discrete blocks and KL divergence value
        """
        # Determine shape of discrete blocks approximating distribution
        # Shape: K1 x K2 x ... x Kd ≈ n_blocks
        # such that Ki / Kj ≈ bbox_len_i / bbox_len_j
        shape = [None] * self.dim
        
        # Handle degenerate bounding boxes
        bbox_lengths = []
        for j in range(self.dim):
            length = bbox[j][1] - bbox[j][0]
            bbox_lengths.append(max(length, 1e-10))
        
        shape[0] = self.n_blocks * np.prod([
            bbox_lengths[0] / bbox_lengths[j] 
            for j in range(1, self.dim)
        ])
        shape[0] = shape[0] ** (1 / self.dim)
        
        for j in range(1, self.dim):
            shape[j] = bbox_lengths[j] / bbox_lengths[0] * shape[0]
        
        shape = [max(int(np.round(l)), 1) for l in shape]
        
        # Count discrete distribution of points
        block_proportions = np.zeros(shape)
        for point in point_set:
            ind = []
            for j in range(self.dim):
                idx = int((point[j] - bbox[j][0]) / bbox_lengths[j] * shape[j])
                ind.append(min(idx, shape[j] - 1))
            ind = tuple(ind)
            block_proportions[ind] += 1
        
        block_proportions = block_proportions / len(point_set)
        
        # Calculate KL divergence: KL(counted || uniform)
        kl_result = block_proportions * np.log(
            block_proportions * np.prod(shape),
            out=np.zeros_like(block_proportions),
            where=(block_proportions != 0)
        )
        kl_result = np.sum(kl_result)
        
        return shape, kl_result

    def try_split(self, i: int) -> Tuple[bool, int]:
        """
        Try to split in the i-th dimension and find optimal split point.
        
        Parameters
        ----------
        i : int
            Dimension index to split along
            
        Returns
        -------
        Tuple[bool, int]
            Success flag and optimal split point index
        """
        cond_kls = []
        split_points = []
        ok = False
        
        i_shape = self.K_shape[i]
        for j in range(i_shape - 1):
            # Find split point
            split_val = (j + 1) / i_shape * (self.bbox[i][1] - self.bbox[i][0]) + self.bbox[i][0]
            split_point = min(
                self.points_np[:, i].searchsorted(split_val + 1e-8) + 1,
                len(self.points)
            )
            split_points.append(split_point)
            
            # Check left partition
            left_points = self.points_np[:split_point, :]
            if split_point < max(self.smallest_points, self.n_blocks):
                cond_kls.append(BIG_CONSTANT)
                continue
                
            _, left_kl = self.cal_kl_divergence(
                left_points,
                [(np.min(left_points[:, k]), np.max(left_points[:, k])) 
                 for k in range(self.dim)]
            )
            left_proportion = split_point / len(self.points)
            
            # Check right partition
            right_points = self.points_np[split_point:, :]
            if len(self.points) - split_point < max(self.smallest_points, self.n_blocks):
                cond_kls.append(BIG_CONSTANT)
                continue
                
            _, right_kl = self.cal_kl_divergence(
                right_points,
                [(np.min(right_points[:, k]), np.max(right_points[:, k])) 
                 for k in range(self.dim)]
            )
            right_proportion = (len(self.points) - split_point) / len(self.points)
            
            # Conditional KL
            cond_kl = left_proportion * left_kl + right_proportion * right_kl
            cond_kls.append(cond_kl)
            ok = True
        
        return ok, split_points[np.argmin(cond_kls)]

    def choose_split(self) -> None:
        """Choose optimal split dimension and point."""
        # Check corner cases
        if (self.max_depth is not None and self.depth >= self.max_depth) or \
           len(self.points) < 2 * max(self.smallest_points, self.n_blocks):
            self.set_dead()
            return
        
        # Split along dimension with largest scale
        self.dim_chosen = np.argmax([e[1] - e[0] for e in self.bbox])
        self.points.sort(key=lambda x: x[self.dim_chosen])
        self.points_np = np.array(self.points)
        
        # Calculate overall KL divergence
        self.K_shape, self.overall_kl = self.cal_kl_divergence(self.points, self.bbox)
        
        # Choose optimal split point
        ok, self.split_chosen = self.try_split(self.dim_chosen)
        if not ok:
            self.set_dead()

    def set_dead(self) -> None:
        """Mark this node as inactive (cannot be split further)."""
        self.overall_kl = None
        self.alive = False

    def split(self) -> Tuple["KDTreeNode", "KDTreeNode"]:
        """
        Split node according to optimal KL gain.
        
        Returns
        -------
        Tuple[KDTreeNode, KDTreeNode]
            Left and right child nodes
        """
        borders_l = list(self.borders)
        borders_r = list(self.borders)
        split_val = self.points[self.split_chosen][self.dim_chosen]
        
        borders_l[self.dim_chosen] = (borders_l[self.dim_chosen][0], split_val)
        borders_r[self.dim_chosen] = (split_val, borders_r[self.dim_chosen][1])
        
        return (
            KDTreeNode(
                self.depth + 1, self.points[:self.split_chosen],
                self.dim, self.n_blocks, self.smallest_points,
                self.max_depth, borders_l
            ),
            KDTreeNode(
                self.depth + 1, self.points[self.split_chosen:],
                self.dim, self.n_blocks, self.smallest_points,
                self.max_depth, borders_r
            )
        )

    def simple_split(self) -> Tuple["KDTreeNode", "KDTreeNode"]:
        """
        Split node at the median.
        
        Returns
        -------
        Tuple[KDTreeNode, KDTreeNode]
            Left and right child nodes
        """
        m = len(self.points) >> 1
        borders_l = list(self.borders)
        borders_r = list(self.borders)
        split_val = self.points[m][self.dim_chosen]
        
        borders_l[self.dim_chosen] = (borders_l[self.dim_chosen][0], split_val)
        borders_r[self.dim_chosen] = (split_val, borders_r[self.dim_chosen][1])
        
        return (
            KDTreeNode(
                self.depth + 1, self.points[:m],
                self.dim, self.n_blocks, self.smallest_points,
                self.max_depth, borders_l
            ),
            KDTreeNode(
                self.depth + 1, self.points[m:],
                self.dim, self.n_blocks, self.smallest_points,
                self.max_depth, borders_r
            )
        )

    def get_bounding_box(self) -> List[Tuple[float, float]]:
        """Get the bounding box of points in this node."""
        return self.bbox

    def get_borders(self) -> List[Tuple[float, float]]:
        """Get the subdomain borders."""
        return self.borders


class KDTree:
    """
    K-D Tree for splitting scattered data into more uniformly distributed subdomains.
    
    This implementation optimizes subdomain decomposition based on KL divergence,
    resulting in subdomains that are more uniformly distributed internally.
    
    Parameters
    ----------
    points : List[List[float]]
        List of points, e.g., [[1, 2], [1, 2.5], ...] for 2D
    dim : int
        Dimensionality of the points
    n_subdomains : int, optional
        Number of subdomains to generate, by default 32
    n_blocks : int, optional
        Number of blocks for KL divergence approximation, by default 8
    smallest_points : int, optional
        Minimum points per subdomain, by default 8
    max_depth : Optional[int], optional
        Maximum tree depth, by default None
    group : int, optional
        Number of groups for separate splitting, by default 1
    return_indices : bool, optional
        Whether to track original point indices, by default False
    overall_borders : Optional[List[Tuple[float, float]]], optional
        Domain boundaries, by default None (uses bounding box)
        
    Example
    -------
    >>> points = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], ...]
    >>> tree = KDTree(points, dim=2, n_subdomains=8, return_indices=True)
    >>> tree.solve()
    >>> subdomain_indices = tree.get_subdomain_indices()
    >>> bounding_boxes = tree.get_subdomain_bounding_boxes()
    """
    
    def __init__(
        self,
        points: List[List[float]],
        dim: int,
        n_subdomains: int = 32,
        n_blocks: int = 8,
        smallest_points: int = 8,
        max_depth: Optional[int] = None,
        group: int = 1,
        return_indices: bool = False,
        overall_borders: Optional[List[Tuple[float, float]]] = None
    ) -> None:
        self.return_indices = return_indices
        self.dim = dim
        self.smallest_points = smallest_points
        self.n_subdomains = n_subdomains
        self.group = group
        
        if overall_borders is None:
            points_np = np.array(points)
            overall_borders = [
                (np.min(points_np[:, i]), np.max(points_np[:, i])) 
                for i in range(self.dim)
            ]
        
        if return_indices:
            points = [list(points[i]) + [i] for i in range(len(points))]
        
        self.nodes = [KDTreeNode(
            0, points, dim, n_blocks, smallest_points, max_depth, overall_borders
        )]
        self.overall_bbox = self.nodes[0].bbox

    def solve(self, prepro: bool = False, postpro: bool = False) -> None:
        """
        Execute K-D tree splitting until target number of subdomains is reached.
        
        Parameters
        ----------
        prepro : bool, optional
            Whether to call preprocessing, by default False
        postpro : bool, optional
            Whether to call postprocessing, by default False
        """
        if prepro:
            self.preprocess()
        
        # Generate groups if requested
        if self.group > 1:
            self.nodes = self._split(self.nodes, self.group, simple=True)
        
        # Split each group
        tot_nodes = []
        for node in self.nodes:
            tot_nodes = tot_nodes + self._split(
                [node], self.n_subdomains // max(self.group, 1), simple=False
            )
        self.nodes = tot_nodes
        
        if postpro:
            self.postprocess()

    def _split(
        self, 
        nodes: List[KDTreeNode], 
        n_subdomains: int, 
        simple: bool = False
    ) -> List[KDTreeNode]:
        """Internal splitting method."""
        while len(nodes) < n_subdomains:
            ind = [i for i in range(len(nodes)) if nodes[i].alive and nodes[i].ready]
            
            if len(ind) == 0:
                print("Warning: early termination - no nodes available for splitting.")
                break
            
            if simple:
                # Split node with most points
                node_utilities = [len(nodes[i].points) for i in ind]
                node_chosen = np.argmax(node_utilities)
                son_a, son_b = nodes[ind[node_chosen]].simple_split()
            else:
                # Split node with highest KL divergence × point count
                node_utilities = [
                    nodes[i].overall_kl * len(nodes[i].points) for i in ind
                ]
                node_chosen = np.argmax(node_utilities)
                son_a, son_b = nodes[ind[node_chosen]].split()
            
            nodes[ind[node_chosen]] = son_a
            nodes.append(son_b)
        
        return nodes

    def preprocess(self) -> None:
        """Override for custom preprocessing."""
        pass

    def postprocess(self) -> None:
        """Override for custom postprocessing."""
        pass

    def get_subdomain_points(self) -> List[np.ndarray]:
        """
        Get points in each subdomain.
        
        Returns
        -------
        List[np.ndarray]
            List of point arrays for each subdomain
        """
        if self.return_indices:
            return [node.points_np[:, :-1] for node in self.nodes]
        else:
            return [node.points_np for node in self.nodes]

    def get_subdomain_bounding_boxes(self) -> List[List[Tuple[float, float]]]:
        """
        Get bounding boxes for each subdomain.
        
        Returns
        -------
        List[List[Tuple[float, float]]]
            Bounding boxes as [(min, max) for each dimension] for each subdomain
        """
        return [node.get_bounding_box() for node in self.nodes]

    def get_subdomain_borders(
        self,
        shrink: bool = False,
        shrink_tol: float = 0.25,
        shrink_proportion: float = 0.25
    ) -> List[List[Tuple[float, float]]]:
        """
        Get subdomain borders with optional shrinking.
        
        Parameters
        ----------
        shrink : bool, optional
            Whether to shrink borders, by default False
        shrink_tol : float, optional
            Shrink tolerance threshold, by default 0.25
        shrink_proportion : float, optional
            Shrink proportion, by default 0.25
            
        Returns
        -------
        List[List[Tuple[float, float]]]
            Border coordinates for each subdomain
        """
        borders = [node.get_borders() for node in self.nodes]
        
        if not shrink:
            return borders
        
        bboxes = [node.get_bounding_box() for node in self.nodes]
        for j in range(len(borders)):
            border = list(borders[j])
            bbox = bboxes[j]
            for i in range(self.dim):
                new_l, new_r = border[i]
                bbox_range = bbox[i][1] - bbox[i][0]
                if bbox[i][0] - border[i][0] > bbox_range * shrink_tol:
                    new_l = bbox[i][0] - (bbox[i][0] - border[i][0]) * shrink_proportion
                if border[i][1] - bbox[i][1] > bbox_range * shrink_tol:
                    new_r = bbox[i][1] + (border[i][1] - bbox[i][1]) * shrink_proportion
                border[i] = (new_l, new_r)
            borders[j] = border
        
        return borders

    def get_subdomain_indices(self) -> Optional[List[np.ndarray]]:
        """
        Get original point indices for each subdomain.
        
        Returns
        -------
        Optional[List[np.ndarray]]
            Index arrays if return_indices=True, else None
        """
        if self.return_indices:
            return [node.points_np[:, -1].astype(int) for node in self.nodes]
        else:
            return None

    def sort_nodes_by_n_points(self) -> None:
        """Sort subdomains by number of points (ascending)."""
        self.nodes.sort(key=lambda node: len(node.points))


# ===================================================================
# Interpolation Utilities (PyTorch-based, no SciPy dependency)
# ===================================================================

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


def points_to_grid_2d(
    points: Tensor,
    values: Tensor,
    grid_size: Tuple[int, int],
    bbox: Tuple[Tuple[float, float], Tuple[float, float]],
    k: int = 6,
    power: float = 2.0
) -> Tensor:
    """
    Interpolate scattered points to a uniform 2D grid.
    
    Parameters
    ----------
    points : Tensor
        Point coordinates, shape (N, 2)
    values : Tensor
        Values at points, shape (N, C)
    grid_size : Tuple[int, int]
        Output grid size (H, W)
    bbox : Tuple[Tuple[float, float], Tuple[float, float]]
        Bounding box ((x_min, x_max), (y_min, y_max))
    k : int, optional
        Number of neighbors for IDW, by default 6
    power : float, optional
        IDW power parameter, by default 2.0
        
    Returns
    -------
    Tensor
        Grid values, shape (C, H, W)
    """
    device = points.device
    dtype = points.dtype
    H, W = grid_size
    
    # Create grid coordinates
    x = torch.linspace(bbox[0][0], bbox[0][1], W, device=device, dtype=dtype)
    y = torch.linspace(bbox[1][0], bbox[1][1], H, device=device, dtype=dtype)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
    query_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # (H*W, 2)
    
    # Interpolate using IDW
    grid_values = inverse_distance_weighting(points, values, query_points, k=k, power=power)
    
    # Reshape to grid
    C = values.shape[1] if values.dim() > 1 else 1
    grid_values = grid_values.reshape(H, W, C).permute(2, 0, 1)  # (C, H, W)
    
    return grid_values


def points_to_grid_3d(
    points: Tensor,
    values: Tensor,
    grid_size: Tuple[int, int, int],
    bbox: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    k: int = 8,
    power: float = 2.0
) -> Tensor:
    """
    Interpolate scattered points to a uniform 3D grid.
    
    Parameters
    ----------
    points : Tensor
        Point coordinates, shape (N, 3)
    values : Tensor
        Values at points, shape (N, C)
    grid_size : Tuple[int, int, int]
        Output grid size (D, H, W)
    bbox : Tuple[...]
        Bounding box ((x_min, x_max), (y_min, y_max), (z_min, z_max))
    k : int, optional
        Number of neighbors for IDW, by default 8
    power : float, optional
        IDW power parameter, by default 2.0
        
    Returns
    -------
    Tensor
        Grid values, shape (C, D, H, W)
    """
    device = points.device
    dtype = points.dtype
    D, H, W = grid_size
    
    # Create grid coordinates
    x = torch.linspace(bbox[0][0], bbox[0][1], W, device=device, dtype=dtype)
    y = torch.linspace(bbox[1][0], bbox[1][1], H, device=device, dtype=dtype)
    z = torch.linspace(bbox[2][0], bbox[2][1], D, device=device, dtype=dtype)
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='xy')
    query_points = torch.stack([
        grid_x.flatten(), grid_y.flatten(), grid_z.flatten()
    ], dim=-1)  # (H*W*D, 3)
    
    # Interpolate using IDW
    grid_values = inverse_distance_weighting(points, values, query_points, k=k, power=power)
    
    # Reshape to grid
    # meshgrid with indexing='xy' produces shape (H, W, D)
    C = values.shape[1] if values.dim() > 1 else 1
    grid_values = grid_values.reshape(H, W, D, C).permute(3, 2, 0, 1)  # (C, D, H, W)
    
    return grid_values


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
    values = F.grid_sample(
        grid, grid_coords, 
        mode=mode, padding_mode=padding_mode, align_corners=align_corners
    )  # (B, C, N, 1)
    
    values = values.squeeze(-1)  # (B, C, N)
    
    if not had_batch:
        values = values.squeeze(0)
    
    return values


def grid_to_points_3d(
    grid: Tensor,
    query_points: Tensor,
    bbox: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    mode: str = 'bilinear',
    padding_mode: str = 'border',
    align_corners: bool = False
) -> Tensor:
    """
    Interpolate from a 3D grid to scattered points using grid_sample.
    
    Parameters
    ----------
    grid : Tensor
        Grid values, shape (B, C, D, H, W) or (C, D, H, W)
    query_points : Tensor
        Query point coordinates, shape (N, 3) or (B, N, 3)
    bbox : Tuple[...]
        Bounding box ((x_min, x_max), (y_min, y_max), (z_min, z_max))
    mode : str, optional
        Interpolation mode ('bilinear' uses trilinear for 5D), by default 'bilinear'
    padding_mode : str, optional
        Padding mode, by default 'border'
    align_corners : bool, optional
        Whether to align corners, by default False
        
    Returns
    -------
    Tensor
        Values at query points, shape (B, C, N) or (C, N)
    """
    # Handle batch dimension
    had_batch = grid.dim() == 5
    if grid.dim() == 4:
        grid = grid.unsqueeze(0)
    if query_points.dim() == 2:
        query_points = query_points.unsqueeze(0)
    
    B, C, D, H, W = grid.shape
    N = query_points.shape[1]
    
    # Normalize coordinates to [-1, 1]
    normalized_points = query_points.clone()
    normalized_points[..., 0] = 2 * (query_points[..., 0] - bbox[0][0]) / (bbox[0][1] - bbox[0][0]) - 1
    normalized_points[..., 1] = 2 * (query_points[..., 1] - bbox[1][0]) / (bbox[1][1] - bbox[1][0]) - 1
    normalized_points[..., 2] = 2 * (query_points[..., 2] - bbox[2][0]) / (bbox[2][1] - bbox[2][0]) - 1
    
    # Reshape for grid_sample: (B, N, 1, 1, 3)
    grid_coords = normalized_points.unsqueeze(2).unsqueeze(3)
    
    # Apply grid_sample
    # Note: F.grid_sample automatically uses trilinear interpolation for 5D input
    # when mode='bilinear'
    values = F.grid_sample(
        grid, grid_coords,
        mode=mode, padding_mode=padding_mode, align_corners=align_corners
    )  # (B, C, N, 1, 1)
    
    values = values.squeeze(-1).squeeze(-1)  # (B, C, N)
    
    if not had_batch:
        values = values.squeeze(0)
    
    return values


def resize_grid_fft(
    grid: Tensor,
    target_size: Union[Tuple[int, ...], List[int]]
) -> Tensor:
    """
    Resize a grid using FFT-based interpolation.
    
    This method preserves spectral content better than standard interpolation.
    The output is scaled to preserve the mean value of the input.
    
    Parameters
    ----------
    grid : Tensor
        Input grid, shape (B, C, *spatial_dims) or (C, *spatial_dims)
    target_size : Union[Tuple[int, ...], List[int]]
        Target spatial dimensions
        
    Returns
    -------
    Tensor
        Resized grid
    """
    had_batch = grid.dim() > len(target_size) + 1
    if not had_batch:
        grid = grid.unsqueeze(0)
    
    B, C = grid.shape[:2]
    spatial_dims = grid.shape[2:]
    n_dims = len(spatial_dims)
    
    # Calculate scale factor to preserve mean
    # FFT normalization: irfft divides by output size, so we need to compensate
    source_numel = 1
    target_numel = 1
    for s in spatial_dims:
        source_numel *= s
    for t in target_size:
        target_numel *= t
    scale_factor = target_numel / source_numel
    
    if n_dims == 2:
        # 2D FFT resize
        freq = torch.fft.rfft2(grid, dim=(-2, -1))
        target_h, target_w = target_size
        
        # Create output frequency tensor
        out_freq = torch.zeros(
            (B, C, target_h, target_w // 2 + 1),
            dtype=freq.dtype, device=freq.device
        )
        
        # Copy frequency components
        min_h = min(spatial_dims[0] // 2, target_h // 2)
        min_w = min(spatial_dims[1] // 2 + 1, target_w // 2 + 1)
        
        out_freq[:, :, :min_h, :min_w] = freq[:, :, :min_h, :min_w]
        out_freq[:, :, -min_h:, :min_w] = freq[:, :, -min_h:, :min_w]
        
        # Inverse FFT with scaling to preserve mean
        result = torch.fft.irfft2(out_freq, s=target_size, dim=(-2, -1)) * scale_factor
        
    elif n_dims == 3:
        # 3D FFT resize
        freq = torch.fft.rfftn(grid, dim=(-3, -2, -1))
        target_d, target_h, target_w = target_size
        
        out_freq = torch.zeros(
            (B, C, target_d, target_h, target_w // 2 + 1),
            dtype=freq.dtype, device=freq.device
        )
        
        min_d = min(spatial_dims[0] // 2, target_d // 2)
        min_h = min(spatial_dims[1] // 2, target_h // 2)
        min_w = min(spatial_dims[2] // 2 + 1, target_w // 2 + 1)
        
        out_freq[:, :, :min_d, :min_h, :min_w] = freq[:, :, :min_d, :min_h, :min_w]
        out_freq[:, :, :min_d, -min_h:, :min_w] = freq[:, :, :min_d, -min_h:, :min_w]
        out_freq[:, :, -min_d:, :min_h, :min_w] = freq[:, :, -min_d:, :min_h, :min_w]
        out_freq[:, :, -min_d:, -min_h:, :min_w] = freq[:, :, -min_d:, -min_h:, :min_w]
        
        result = torch.fft.irfftn(out_freq, s=target_size, dim=(-3, -2, -1)) * scale_factor
    else:
        raise ValueError(f"Unsupported number of dimensions: {n_dims}")
    
    if not had_batch:
        result = result.squeeze(0)
    
    return result


# ===================================================================
# Domain Decomposition Manager
# ===================================================================

class DomainDecomposition:
    """
    Manages domain decomposition for non-uniform data.
    
    This class handles the decomposition of non-uniform point cloud data
    into subdomains and provides methods for interpolation between
    point clouds and uniform grids.
    
    Parameters
    ----------
    points : Union[np.ndarray, Tensor]
        Point cloud coordinates, shape (N, D)
    n_subdomains : int, optional
        Number of subdomains, by default 8
    n_blocks : int, optional
        Blocks for KL divergence, by default 8
    smallest_points : int, optional
        Minimum points per subdomain, by default 8
    oversamp_ratio : float, optional
        Oversampling ratio for grid size, by default 1.5
    align_long_side : bool, optional
        Whether to transpose grids to align long sides (reduces padding).
        Disabled by default for stability. Use with caution as it may
        affect interpolation accuracy after resize operations.
        
    Example
    -------
    >>> points = torch.randn(1000, 2)
    >>> decomp = DomainDecomposition(points, n_subdomains=8)
    >>> values = torch.randn(1000, 3)  # 3 channels
    >>> grids = decomp.interpolate_to_grids(values)
    >>> reconstructed = decomp.interpolate_to_points(grids)
    """
    
    def __init__(
        self,
        points: Union[np.ndarray, Tensor],
        n_subdomains: int = 8,
        n_blocks: int = 8,
        smallest_points: int = 8,
        oversamp_ratio: float = 1.5,
        align_long_side: bool = False
    ) -> None:
        # Convert to numpy for K-D tree
        if isinstance(points, Tensor):
            self.points_tensor = points
            points_np = points.detach().cpu().numpy()
        else:
            self.points_tensor = torch.from_numpy(points).float()
            points_np = points
        
        self.n_points, self.dim = points_np.shape
        self.n_subdomains = n_subdomains
        self.oversamp_ratio = oversamp_ratio
        self.align_long_side = align_long_side
        
        # Build K-D tree
        points_list = points_np.tolist()
        self.tree = KDTree(
            points_list, 
            dim=self.dim, 
            n_subdomains=n_subdomains,
            n_blocks=n_blocks,
            smallest_points=smallest_points,
            return_indices=True
        )
        self.tree.solve()
        
        # Get subdomain information
        self.subdomain_bboxes = self.tree.get_subdomain_bounding_boxes()
        self.subdomain_indices = self.tree.get_subdomain_indices()
        
        # Calculate grid sizes for each subdomain
        self._calculate_grid_info()
    
    def _calculate_grid_info(self) -> None:
        """Calculate grid sizes and metadata for each subdomain."""
        self.grid_sizes = []  # Sizes BEFORE any transposition
        self.final_grid_sizes = []  # Sizes AFTER transposition (actual output shapes)
        self.is_transposed = []
        
        for i, bbox in enumerate(self.subdomain_bboxes):
            n_pts = len(self.subdomain_indices[i])
            
            # Calculate grid size maintaining aspect ratio
            if self.dim == 2:
                bbox_size_x = bbox[0][1] - bbox[0][0] + 1e-10
                bbox_size_y = bbox[1][1] - bbox[1][0] + 1e-10
                
                grid_size_x = np.sqrt(n_pts * self.oversamp_ratio * bbox_size_x / bbox_size_y)
                grid_size_y = grid_size_x * bbox_size_y / bbox_size_x
                
                grid_size_x = max(int(np.round(grid_size_x)), 2)
                grid_size_y = max(int(np.round(grid_size_y)), 2)
                
                # Store the natural grid size (matches bbox aspect ratio)
                self.grid_sizes.append((grid_size_x, grid_size_y))
                
                # Long side alignment: transpose if height > width to make width >= height
                transposed = False
                final_size_x, final_size_y = grid_size_x, grid_size_y
                if self.align_long_side and grid_size_y > grid_size_x:
                    # After transpose, dimensions will be swapped
                    final_size_x, final_size_y = grid_size_y, grid_size_x
                    transposed = True
                
                self.final_grid_sizes.append((final_size_x, final_size_y))
                self.is_transposed.append(transposed)
                
            elif self.dim == 3:
                bbox_sizes = [bbox[j][1] - bbox[j][0] + 1e-10 for j in range(3)]
                
                # Calculate proportional grid sizes
                vol_factor = (n_pts * self.oversamp_ratio) ** (1/3)
                scale_factor = vol_factor / (bbox_sizes[0] * bbox_sizes[1] * bbox_sizes[2]) ** (1/3)
                
                grid_size = tuple(max(int(np.round(scale_factor * s)), 2) for s in bbox_sizes)
                
                self.grid_sizes.append(grid_size)
                self.final_grid_sizes.append(grid_size)  # No transposition for 3D
                self.is_transposed.append(False)
        
        # Calculate max grid size from FINAL (post-transpose) shapes
        self.max_grid_size = tuple(
            max(fgs[j] for fgs in self.final_grid_sizes) 
            for j in range(self.dim)
        )
        self.n_points_per_subdomain = [len(idx) for idx in self.subdomain_indices]
        self.max_points_per_subdomain = max(self.n_points_per_subdomain)
    
    def get_normalized_coords(self, subdomain_idx: int) -> Tensor:
        """
        Get normalized coordinates for a subdomain (in [-1, 1]).
        
        Parameters
        ----------
        subdomain_idx : int
            Subdomain index
            
        Returns
        -------
        Tensor
            Normalized coordinates, shape (n_points_in_subdomain, D)
        """
        indices = self.subdomain_indices[subdomain_idx]
        points = self.points_tensor[indices]
        bbox = self.subdomain_bboxes[subdomain_idx]
        
        normalized = points.clone()
        for d in range(self.dim):
            _min, _max = bbox[d]
            normalized[:, d] = 2 * (points[:, d] - _min) / (_max - _min + 1e-10) - 1
        
        # Handle transposition for 2D
        if self.dim == 2 and self.is_transposed[subdomain_idx]:
            normalized = normalized.flip(dims=[1])
        
        return normalized
    
    def interpolate_to_grids(
        self,
        values: Tensor,
        k: int = 6,
        power: float = 2.0,
        resize_to_max: bool = True
    ) -> Tensor:
        """
        Interpolate point cloud values to uniform grids per subdomain.
        
        Parameters
        ----------
        values : Tensor
            Values at points, shape (N, C) or (B, N, C)
        k : int, optional
            Neighbors for IDW, by default 6
        power : float, optional
            IDW power, by default 2.0
        resize_to_max : bool, optional
            Resize all grids to same size, by default True
            
        Returns
        -------
        Tensor
            Grid values, shape (n_subdomains, C, *max_grid_size) if resize_to_max
            or list of grids with different sizes
        """
        has_batch = values.dim() == 3
        if not has_batch:
            values = values.unsqueeze(0)
        
        B, N, C = values.shape
        device = values.device
        dtype = values.dtype
        
        all_grids = []
        
        for i in range(self.n_subdomains):
            indices = self.subdomain_indices[i]
            subdomain_values = values[:, indices, :]  # (B, n_pts, C)
            subdomain_points = self.points_tensor[indices].to(device)  # (n_pts, D)
            bbox = self.subdomain_bboxes[i]
            grid_size = self.grid_sizes[i]  # Natural size (matches bbox)
            final_size = self.final_grid_sizes[i]  # Size after transpose
            
            batch_grids = []
            for b in range(B):
                if self.dim == 2:
                    grid = points_to_grid_2d(
                        subdomain_points, subdomain_values[b],
                        grid_size, tuple(bbox), k=k, power=power
                    )
                    # Handle transposition to align long sides
                    if self.is_transposed[i]:
                        grid = grid.transpose(-2, -1)
                else:  # dim == 3
                    grid = points_to_grid_3d(
                        subdomain_points, subdomain_values[b],
                        grid_size, tuple(bbox), k=k, power=power
                    )
                
                batch_grids.append(grid)
            
            subdomain_grid = torch.stack(batch_grids, dim=0)  # (B, C, *final_size)
            
            # Resize to max grid size if requested
            # Use final_size (post-transpose) for comparison
            if resize_to_max and final_size != self.max_grid_size:
                subdomain_grid = resize_grid_fft(subdomain_grid, self.max_grid_size)
            
            all_grids.append(subdomain_grid)
        
        # Stack all subdomains
        result = torch.stack(all_grids, dim=1)  # (B, n_subdomains, C, *max_grid_size)
        
        if not has_batch:
            result = result.squeeze(0)
        
        return result
    
    def interpolate_to_points(
        self,
        grids: Tensor,
        mode: str = 'bilinear',
        padding_mode: str = 'border'
    ) -> Tuple[Tensor, Tensor]:
        """
        Interpolate from subdomain grids back to point cloud.
        
        Parameters
        ----------
        grids : Tensor
            Grid values, shape (n_subdomains, C, *grid_size) or (B, n_subdomains, C, *grid_size)
        mode : str, optional
            Interpolation mode, by default 'bilinear'
        padding_mode : str, optional
            Padding mode, by default 'border'
            
        Returns
        -------
        Tuple[Tensor, Tensor]
            - Values at original point locations, shape (N, C) or (B, N, C)
            - Mask indicating valid (non-padded) values, shape (N,) or (B, N)
        """
        has_batch = grids.dim() == 5 if self.dim == 2 else grids.dim() == 6
        if not has_batch:
            grids = grids.unsqueeze(0)
        
        B = grids.shape[0]
        C = grids.shape[2]
        device = grids.device
        dtype = grids.dtype
        
        # Output tensor
        output = torch.zeros((B, self.n_points, C), device=device, dtype=dtype)
        
        for i in range(self.n_subdomains):
            indices = self.subdomain_indices[i]
            subdomain_grid = grids[:, i]  # (B, C, *grid_size)
            bbox = self.subdomain_bboxes[i]
            
            # Get normalized coordinates
            norm_coords = self.get_normalized_coords(i).to(device)  # (n_pts, D)
            norm_coords = norm_coords.unsqueeze(0).expand(B, -1, -1)  # (B, n_pts, D)
            
            # Interpolate
            if self.dim == 2:
                # Handle transposition
                if self.is_transposed[i]:
                    subdomain_grid = subdomain_grid.transpose(-2, -1)
                
                # grid_sample expects coords in [-1, 1], already normalized
                grid_coords = norm_coords.unsqueeze(2)  # (B, n_pts, 1, 2)
                values = F.grid_sample(
                    subdomain_grid, grid_coords,
                    mode=mode, padding_mode=padding_mode, align_corners=False
                )  # (B, C, n_pts, 1)
                values = values.squeeze(-1).permute(0, 2, 1)  # (B, n_pts, C)
            else:  # dim == 3
                grid_coords = norm_coords.unsqueeze(2).unsqueeze(3)  # (B, n_pts, 1, 1, 3)
                # Note: F.grid_sample uses 'bilinear' for both 4D and 5D input
                # For 5D input, it automatically uses trilinear interpolation internally
                values = F.grid_sample(
                    subdomain_grid, grid_coords,
                    mode=mode, padding_mode=padding_mode, align_corners=False
                )  # (B, C, n_pts, 1, 1)
                values = values.squeeze(-1).squeeze(-1).permute(0, 2, 1)  # (B, n_pts, C)
            
            # Place values at original indices
            output[:, indices, :] = values
        
        # Create mask (all points are valid in this implementation)
        mask = torch.ones((B, self.n_points), device=device, dtype=torch.bool)
        
        if not has_batch:
            output = output.squeeze(0)
            mask = mask.squeeze(0)
        
        return output, mask
    
    def get_subdomain_info(self) -> Dict:
        """Get information about the domain decomposition."""
        return {
            'n_subdomains': self.n_subdomains,
            'n_points': self.n_points,
            'dim': self.dim,
            'grid_sizes': self.grid_sizes,
            'final_grid_sizes': self.final_grid_sizes,
            'max_grid_size': self.max_grid_size,
            'subdomain_bboxes': self.subdomain_bboxes,
            'points_per_subdomain': self.n_points_per_subdomain,
            'indices': self.subdomain_indices
        }


# ===================================================================
# NUNO Model
# ===================================================================

@dataclass
class NUNOMetaData(ModelMetaData):
    name: str = "NonUniformNeuralOperator"
    # Optimization
    jit: bool = False  # Domain decomposition uses numpy
    cuda_graphs: bool = False
    amp: bool = False
    # Inference
    onnx_cpu: bool = False
    onnx_gpu: bool = False
    onnx_runtime: bool = False
    # Physics informed
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False


class NUNO(Module):
    """
    Non-Uniform Neural Operator (NUNO).
    
    NUNO extends mesh-based neural operators (like FNO) to handle non-uniform
    point cloud data through K-D tree-based domain decomposition and adaptive
    grid interpolation.
    
    Note
    ----
    The NUNO architecture:
    1. Decomposes non-uniform point cloud into subdomains using K-D tree
    2. Interpolates each subdomain to uniform grids
    3. Processes grids through underlying FNO
    4. Interpolates back to point cloud
    
    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    n_subdomains : int, optional
        Number of subdomains for domain decomposition, by default 8
    decoder_layers : int, optional
        Number of decoder layers, by default 1
    decoder_layer_size : int, optional
        Size of decoder layers, by default 32
    decoder_activation_fn : str, optional
        Decoder activation function, by default "silu"
    dimension : int, optional
        Spatial dimension (2 or 3), by default 2
    latent_channels : int, optional
        FNO latent channels, by default 32
    num_fno_layers : int, optional
        Number of FNO layers, by default 4
    num_fno_modes : Union[int, List[int]], optional
        FNO modes, by default 16
    padding : int, optional
        FNO padding, by default 8
    padding_type : str, optional
        FNO padding type, by default "constant"
    activation_fn : str, optional
        Activation function, by default "gelu"
    coord_features : bool, optional
        Use coordinate features, by default True
    n_blocks : int, optional
        K-D tree blocks for KL divergence, by default 8
    smallest_points : int, optional
        Minimum points per subdomain, by default 8
    oversamp_ratio : float, optional
        Grid oversampling ratio, by default 1.5
    idw_neighbors : int, optional
        IDW interpolation neighbors, by default 6
    idw_power : float, optional
        IDW power parameter, by default 2.0
        
    Example
    -------
    >>> # Create NUNO model
    >>> model = NUNO(
    ...     in_channels=1,
    ...     out_channels=1,
    ...     n_subdomains=8,
    ...     dimension=2,
    ...     num_fno_layers=4,
    ... )
    >>> # Set up domain decomposition (done once)
    >>> points = torch.randn(1000, 2)  # Point cloud coordinates
    >>> model.setup_domain(points)
    >>> # Forward pass
    >>> x = torch.randn(32, 1000, 1)  # (batch, n_points, channels)
    >>> y = model(x)  # (batch, n_points, out_channels)
    
    Note
    ----
    Reference: Liu, S., et al. "NUNO: A General Framework for Learning 
    Parametric PDEs with Non-Uniform Data." ICML 2023.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_subdomains: int = 8,
        decoder_layers: int = 1,
        decoder_layer_size: int = 32,
        decoder_activation_fn: str = "silu",
        dimension: int = 2,
        latent_channels: int = 32,
        num_fno_layers: int = 4,
        num_fno_modes: Union[int, List[int]] = 16,
        padding: int = 8,
        padding_type: str = "constant",
        activation_fn: str = "gelu",
        coord_features: bool = True,
        n_blocks: int = 8,
        smallest_points: int = 8,
        oversamp_ratio: float = 1.5,
        idw_neighbors: int = 6,
        idw_power: float = 2.0
    ) -> None:
        super().__init__(meta=NUNOMetaData())
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_subdomains = n_subdomains
        self.dimension = dimension
        self.latent_channels = latent_channels
        self.n_blocks = n_blocks
        self.smallest_points = smallest_points
        self.oversamp_ratio = oversamp_ratio
        self.idw_neighbors = idw_neighbors
        self.idw_power = idw_power
        
        # Domain decomposition (initialized later via setup_domain)
        self.domain_decomp: Optional[DomainDecomposition] = None
        
        # Build FNO components
        # Input projection: maps subdomain grids to hidden dimension
        # Input channels = in_channels * n_subdomains (all subdomains concatenated)
        self._build_network(
            in_channels, out_channels, n_subdomains,
            decoder_layers, decoder_layer_size, decoder_activation_fn,
            latent_channels, num_fno_layers, num_fno_modes,
            padding, padding_type, activation_fn, coord_features
        )
    
    def _build_network(
        self,
        in_channels: int,
        out_channels: int,
        n_subdomains: int,
        decoder_layers: int,
        decoder_layer_size: int,
        decoder_activation_fn: str,
        latent_channels: int,
        num_fno_layers: int,
        num_fno_modes: Union[int, List[int]],
        padding: int,
        padding_type: str,
        activation_fn: str,
        coord_features: bool
    ) -> None:
        """
        Build the neural network components.
        
        Uses PhysicsNeMo's FNO encoders for the spectral convolution backbone,
        ensuring consistency with the rest of PhysicsNeMo and leveraging
        well-tested implementations.
        """
        from physicsnemo.models.fno.fno import FNO2DEncoder, FNO3DEncoder
        
        self.activation_fn = layers.get_activation(activation_fn)
        
        # We process all subdomains together by stacking in channel dimension
        # Input shape: (B, n_subdomains, H, W, in_channels) -> (B, in_channels * n_subdomains, H, W)
        total_in_channels = in_channels * n_subdomains
        total_out_channels = out_channels * n_subdomains
        
        # Store for forward pass
        self.coord_features = coord_features
        self.latent_channels = latent_channels
        
        # Store modes for dynamic padding calculation
        if isinstance(num_fno_modes, int):
            if self.dimension == 2:
                self.num_fno_modes = [num_fno_modes, num_fno_modes]
            else:
                self.num_fno_modes = [num_fno_modes, num_fno_modes, num_fno_modes]
        else:
            self.num_fno_modes = list(num_fno_modes)
        
        # Use PhysicsNeMo's FNO encoder - handles lift network, spectral convs, padding
        if self.dimension == 2:
            self.spec_encoder = FNO2DEncoder(
                in_channels=total_in_channels,
                num_fno_layers=num_fno_layers,
                fno_layer_size=latent_channels,
                num_fno_modes=num_fno_modes,
                padding=padding,
                padding_type=padding_type,
                activation_fn=self.activation_fn,
                coord_features=coord_features,
            )
        elif self.dimension == 3:
            self.spec_encoder = FNO3DEncoder(
                in_channels=total_in_channels,
                num_fno_layers=num_fno_layers,
                fno_layer_size=latent_channels,
                num_fno_modes=num_fno_modes,
                padding=padding,
                padding_type=padding_type,
                activation_fn=self.activation_fn,
                coord_features=coord_features,
            )
        else:
            raise ValueError(f"Unsupported dimension: {self.dimension}. Only 2D and 3D supported.")
        
        # Decoder network: projects from latent space to output channels
        self.decoder_net = FullyConnected(
            in_features=latent_channels,
            layer_size=decoder_layer_size,
            out_features=total_out_channels,
            num_layers=decoder_layers,
            activation_fn=decoder_activation_fn,
        )
    
    def setup_domain(
        self,
        points: Union[np.ndarray, Tensor],
        force_rebuild: bool = False
    ) -> None:
        """
        Set up domain decomposition for the given point cloud.
        
        This should be called once before training when the point cloud
        geometry is known. The decomposition can be reused for all samples
        with the same geometry.
        
        Parameters
        ----------
        points : Union[np.ndarray, Tensor]
            Point cloud coordinates, shape (N, D)
        force_rebuild : bool, optional
            Force rebuilding even if already set up, by default False
        """
        if self.domain_decomp is not None and not force_rebuild:
            return
        
        self.domain_decomp = DomainDecomposition(
            points,
            n_subdomains=self.n_subdomains,
            n_blocks=self.n_blocks,
            smallest_points=self.smallest_points,
            oversamp_ratio=self.oversamp_ratio
        )
    
    def forward_on_grid(self, x: Tensor) -> Tensor:
        """
        Forward pass on pre-interpolated grid data.
        
        Uses the spec_encoder's components (lift_network, spectral convolutions)
        but with dynamic padding to handle small subdomain grids that may be
        smaller than the FNO modes require.
        
        Parameters
        ----------
        x : Tensor
            Grid input, shape (B, C * n_subdomains, *spatial_dims)
            
        Returns
        -------
        Tensor
            Grid output, shape (B, out_C * n_subdomains, *spatial_dims)
        """
        # Store original spatial size for output
        original_spatial_shape = x.shape[2:]
        
        # Add coordinate features (using encoder's method)
        if self.coord_features:
            coord_feat = self.spec_encoder.meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)
        
        # Lift to latent space (using encoder's lift network)
        x = self.spec_encoder.lift_network(x)
        
        # Dynamic padding to ensure grid is large enough for spectral modes
        # This is necessary for NUNO because subdomain grids can be quite small
        # For rfft, output size is N//2+1, so we need N >= 2*modes - 1
        if self.dimension == 2:
            h, w = x.shape[-2:]
            min_h = self.num_fno_modes[0]
            min_w = 2 * self.num_fno_modes[1] - 1  # For rfft dimension
            
            pad_h = max(0, min_h - h) + self.spec_encoder.pad[0]
            pad_w = max(0, min_w - w) + self.spec_encoder.pad[1]
            
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=self.spec_encoder.padding_type)
        else:
            d, h, w = x.shape[-3:]
            min_d = self.num_fno_modes[0]
            min_h = self.num_fno_modes[1]
            min_w = 2 * self.num_fno_modes[2] - 1  # For rfft dimension
            
            pad_d = max(0, min_d - d) + self.spec_encoder.pad[0]
            pad_h = max(0, min_h - h) + self.spec_encoder.pad[1]
            pad_w = max(0, min_w - w) + self.spec_encoder.pad[2]
            
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d), mode=self.spec_encoder.padding_type)
        
        # Spectral convolution layers (using encoder's layers)
        for k, (conv, spconv) in enumerate(zip(self.spec_encoder.conv_layers, self.spec_encoder.spconv_layers)):
            if k < len(self.spec_encoder.conv_layers) - 1:
                x = self.spec_encoder.activation_fn(conv(x) + spconv(x))
            else:
                x = conv(x) + spconv(x)
        
        # Crop back to original spatial size (not using encoder's fixed ipad)
        if self.dimension == 2:
            x = x[..., :original_spatial_shape[0], :original_spatial_shape[1]]
        else:
            x = x[..., :original_spatial_shape[0], :original_spatial_shape[1], :original_spatial_shape[2]]
        
        # Reshape for decoder using encoder's helper
        y_shape = x.shape
        x_flat, _ = self.spec_encoder.grid_to_points(x)
        
        # Decode to output channels
        y = self.decoder_net(x_flat)
        
        # Reshape back to grid using encoder's helper
        y = self.spec_encoder.points_to_grid(y, y_shape)
        
        return y
    
    def forward(
        self,
        x: Tensor,
        return_grid: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass for point cloud input.
        
        Parameters
        ----------
        x : Tensor
            Point cloud values, shape (B, N, C) or (N, C)
        return_grid : bool, optional
            Also return intermediate grid representation, by default False
            
        Returns
        ------- 
        Union[Tensor, Tuple[Tensor, Tensor]]
            Point cloud output, shape (B, N, out_C) or (N, out_C)
            If return_grid=True, also returns grid output
            
        Raises
        ------
        RuntimeError
            If setup_domain() has not been called
        """
        if self.domain_decomp is None:
            raise RuntimeError(
                "Domain decomposition not set up. Call setup_domain() first."
            )
        
        has_batch = x.dim() == 3
        if not has_batch:
            x = x.unsqueeze(0)
        
        B, N, C = x.shape
        
        # Step 1: Interpolate to subdomain grids
        # Output: (B, n_subdomains, C, *grid_size)
        grids = self.domain_decomp.interpolate_to_grids(
            x, k=self.idw_neighbors, power=self.idw_power
        )
        
        # Reshape for FNO: (B, C * n_subdomains, *grid_size)
        spatial_dims = grids.shape[3:]
        grids = grids.permute(0, 2, 1, *range(3, 3 + len(spatial_dims)))
        grids = grids.reshape(B, -1, *spatial_dims)
        
        # Step 2: Process through FNO
        grid_out = self.forward_on_grid(grids)
        
        # Reshape back: (B, n_subdomains, out_C, *grid_size)
        grid_out = grid_out.reshape(
            B, self.out_channels, self.n_subdomains, *spatial_dims
        )
        grid_out = grid_out.permute(0, 2, 1, *range(3, 3 + len(spatial_dims)))
        
        # Step 3: Interpolate back to points
        point_out, _ = self.domain_decomp.interpolate_to_points(grid_out)
        
        if not has_batch:
            point_out = point_out.squeeze(0)
            grid_out = grid_out.squeeze(0)
        
        if return_grid:
            return point_out, grid_out
        return point_out
    
    def forward_from_grid(
        self,
        grids: Tensor,
        output_to_points: bool = True
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass starting from pre-computed subdomain grids.
        
        Useful when grids are pre-computed and cached for efficiency.
        
        Parameters
        ----------
        grids : Tensor
            Subdomain grids, shape (B, n_subdomains, C, *grid_size)
        output_to_points : bool, optional
            Interpolate output back to points, by default True
            
        Returns
        -------
        Union[Tensor, Tuple[Tensor, Tensor]]
            If output_to_points: point values (B, N, out_C)
            Else: grid values (B, n_subdomains, out_C, *grid_size)
        """
        B = grids.shape[0]
        spatial_dims = grids.shape[3:]
        
        # Reshape for FNO
        grids_flat = grids.permute(0, 2, 1, *range(3, 3 + len(spatial_dims)))
        grids_flat = grids_flat.reshape(B, -1, *spatial_dims)
        
        # Process through FNO
        grid_out = self.forward_on_grid(grids_flat)
        
        # Reshape back
        grid_out = grid_out.reshape(
            B, self.out_channels, self.n_subdomains, *spatial_dims
        )
        grid_out = grid_out.permute(0, 2, 1, *range(3, 3 + len(spatial_dims)))
        
        if output_to_points and self.domain_decomp is not None:
            point_out, _ = self.domain_decomp.interpolate_to_points(grid_out)
            return point_out
        
        return grid_out


# ===================================================================
# Utility Functions
# ===================================================================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_nuno_from_points(
    points: Union[np.ndarray, Tensor],
    in_channels: int,
    out_channels: int,
    **kwargs
) -> NUNO:
    """
    Convenience function to create NUNO with domain decomposition.
    
    Parameters
    ----------
    points : Union[np.ndarray, Tensor]
        Point cloud coordinates
    in_channels : int
        Input channels
    out_channels : int
        Output channels
    **kwargs
        Additional arguments for NUNO
        
    Returns
    -------
    NUNO
        NUNO model with domain set up
    """
    # Determine dimension from points
    if isinstance(points, Tensor):
        dim = points.shape[-1]
    else:
        dim = points.shape[-1]
    
    model = NUNO(
        in_channels=in_channels,
        out_channels=out_channels,
        dimension=dim,
        **kwargs
    )
    model.setup_domain(points)
    
    return model