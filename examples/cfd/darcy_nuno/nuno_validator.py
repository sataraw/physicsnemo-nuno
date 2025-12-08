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
NUNO Validator for Point Cloud Data
------------------------------------
Validates NUNO model predictions and creates visualizations for non-uniform data.

Unlike grid-based validators that use imshow, this validator uses scatter plots
with interpolated colormaps to visualize point cloud data, creating heatmap-like
visualizations from scattered points.
"""

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from typing import Dict, Optional, Tuple

import torch
from torch import FloatTensor, Tensor
from physicsnemo.launch.logging import LaunchLogger


class NUNOValidator:
    """
    Validator for NUNO point cloud predictions.
    
    Compares model output against ground truth, inverts normalization,
    and creates heatmap-style visualizations from scattered point data
    using triangulation-based interpolation.
    
    Parameters
    ----------
    loss_fun : callable
        Loss function for computing validation error
    norm : dict
        Normalization parameters with keys 'permeability' and 'darcy',
        each containing (mean, std) tuple
    font_size : float, optional
        Font size for plot labels
    use_triangulation : bool, optional
        If True, use triangulation for smooth heatmaps. If False, use scatter plots.
    """
    
    def __init__(
        self,
        loss_fun,
        norm: Dict[str, Tuple[float, float]] = None,
        font_size: float = 14.0,
        use_triangulation: bool = True
    ):
        self.norm = norm or {
            'permeability': (0.0, 1.0),
            'darcy': (0.0, 1.0)
        }
        self.criterion = loss_fun
        self.font_size = font_size
        self.use_triangulation = use_triangulation
        self.headers = ("Input (Permeability)", "Ground Truth", "Prediction", "Relative Error")
    
    def compare(
        self,
        coordinates: FloatTensor,
        invar: FloatTensor,
        target: FloatTensor,
        prediction: FloatTensor,
        step: int,
        logger: LaunchLogger
    ) -> float:
        """
        Compare model output against target and generate visualization.
        
        Parameters
        ----------
        coordinates : FloatTensor
            Point coordinates, shape (N, 2) or (B, N, 2)
        invar : FloatTensor
            Normalized input to model, shape (B, N, 1)
        target : FloatTensor
            Normalized ground truth, shape (B, N, 1)
        prediction : FloatTensor
            Normalized model output, shape (B, N, 1)
        step : int
            Current iteration/epoch for logging
        logger : LaunchLogger
            Logger for saving figures
            
        Returns
        -------
        float
            Validation loss (MSE on normalized values)
        """
        # Compute loss on normalized values
        loss = self.criterion(prediction, target)
        
        # Get normalization parameters
        perm_mean, perm_std = self.norm['permeability']
        darcy_mean, darcy_std = self.norm['darcy']
        
        # Denormalize for visualization
        invar_denorm = invar * perm_std + perm_mean
        target_denorm = target * darcy_std + darcy_mean
        pred_denorm = prediction * darcy_std + darcy_mean
        
        # Pick first sample from batch for visualization
        if coordinates.dim() == 3:
            coords = coordinates[0].cpu().numpy()  # (N, 2)
        else:
            coords = coordinates.cpu().numpy()      # (N, 2)
            
        invar_np = invar_denorm[0, :, 0].cpu().numpy()           # (N,)
        target_np = target_denorm[0, :, 0].cpu().numpy()         # (N,)
        pred_np = pred_denorm[0, :, 0].detach().cpu().numpy()    # (N,)
        
        # Compute relative error
        # Avoid division by zero
        rel_error = (pred_np - target_np) / (np.abs(target_np).max() + 1e-10)
        
        # Create visualization
        fig = self._create_figure(coords, invar_np, target_np, pred_np, rel_error)
        
        # Log figure
        logger.log_figure(figure=fig, artifact_file=f"nuno_validation_step_{step:03d}.png")
        plt.close(fig)
        
        return loss.item()
    
    def _create_figure(
        self,
        coords: np.ndarray,
        invar: np.ndarray,
        target: np.ndarray,
        prediction: np.ndarray,
        rel_error: np.ndarray
    ) -> plt.Figure:
        """
        Create a 4-panel visualization figure.
        
        Parameters
        ----------
        coords : ndarray
            Point coordinates (N, 2)
        invar : ndarray
            Input field values (N,)
        target : ndarray
            Ground truth values (N,)
        prediction : ndarray
            Predicted values (N,)
        rel_error : ndarray
            Relative error values (N,)
            
        Returns
        -------
        Figure
            Matplotlib figure with 4 subplots
        """
        plt.rcParams.update({'font.size': self.font_size})
        
        x, y = coords[:, 0], coords[:, 1]
        data = [invar, target, prediction, rel_error]
        cmaps = ['viridis', 'magma', 'magma', 'RdBu_r']
        
        if self.use_triangulation:
            fig = self._create_triangulation_figure(x, y, data, cmaps)
        else:
            fig = self._create_scatter_figure(x, y, data, cmaps)
        
        return fig
    
    def _create_triangulation_figure(
        self,
        x: np.ndarray,
        y: np.ndarray,
        data: list,
        cmaps: list
    ) -> plt.Figure:
        """Create figure using Delaunay triangulation for smooth heatmaps."""
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Create triangulation once
        triang = tri.Triangulation(x, y)
        
        for idx, (ax, values, cmap, title) in enumerate(zip(axes, data, cmaps, self.headers)):
            # For error plot, center colormap at zero
            if idx == 3:
                vmax = max(abs(values.min()), abs(values.max()))
                vmin = -vmax
            else:
                vmin, vmax = values.min(), values.max()
            
            # Create filled contour plot from triangulation
            tcf = ax.tricontourf(triang, values, levels=50, cmap=cmap, vmin=vmin, vmax=vmax)
            
            # Add colorbar
            cbar = fig.colorbar(tcf, ax=ax, fraction=0.046, pad=0.04)
            
            ax.set_title(title)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_aspect('equal')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        
        plt.tight_layout()
        return fig
    
    def _create_scatter_figure(
        self,
        x: np.ndarray,
        y: np.ndarray,
        data: list,
        cmaps: list
    ) -> plt.Figure:
        """Create figure using scatter plots (fallback for sparse data)."""
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        for idx, (ax, values, cmap, title) in enumerate(zip(axes, data, cmaps, self.headers)):
            # For error plot, center colormap at zero
            if idx == 3:
                vmax = max(abs(values.min()), abs(values.max()))
                vmin = -vmax
            else:
                vmin, vmax = values.min(), values.max()
            
            sc = ax.scatter(x, y, c=values, s=3, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.8)
            
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            
            ax.set_title(title)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_aspect('equal')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        
        plt.tight_layout()
        return fig


class NUNOValidatorSimple:
    """
    Simplified validator that only computes loss without visualization.
    
    Useful for frequent validation during training when full visualization
    is too expensive.
    
    Parameters
    ----------
    loss_fun : callable
        Loss function for computing validation error
    """
    
    def __init__(self, loss_fun):
        self.criterion = loss_fun
    
    def compare(
        self,
        prediction: FloatTensor,
        target: FloatTensor
    ) -> float:
        """
        Compute validation loss.
        
        Parameters
        ----------
        prediction : FloatTensor
            Model output
        target : FloatTensor
            Ground truth
            
        Returns
        -------
        float
            Validation loss
        """
        return self.criterion(prediction, target).item()


def create_standalone_visualization(
    coordinates: np.ndarray,
    invar: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    save_path: str = "nuno_visualization.png",
    use_triangulation: bool = True
) -> None:
    """
    Create standalone visualization without LaunchLogger.
    
    Parameters
    ----------
    coordinates : ndarray
        Point coordinates (N, 2)
    invar : ndarray
        Input field (N,)
    target : ndarray
        Ground truth (N,)
    prediction : ndarray
        Model prediction (N,)
    save_path : str
        Output file path
    use_triangulation : bool
        Whether to use triangulation for smooth plots
    """
    x, y = coordinates[:, 0], coordinates[:, 1]
    rel_error = (prediction - target) / (np.abs(target).max() + 1e-10)
    
    headers = ("Input (Permeability)", "Ground Truth", "Prediction", "Relative Error")
    data = [invar, target, prediction, rel_error]
    cmaps = ['viridis', 'magma', 'magma', 'RdBu_r']
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    if use_triangulation:
        triang = tri.Triangulation(x, y)
        
        for idx, (ax, values, cmap, title) in enumerate(zip(axes, data, cmaps, headers)):
            if idx == 3:
                vmax = max(abs(values.min()), abs(values.max()))
                vmin = -vmax
            else:
                vmin, vmax = values.min(), values.max()
            
            tcf = ax.tricontourf(triang, values, levels=50, cmap=cmap, vmin=vmin, vmax=vmax)
            fig.colorbar(tcf, ax=ax, fraction=0.046, pad=0.04)
            
            ax.set_title(title)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_aspect('equal')
    else:
        for idx, (ax, values, cmap, title) in enumerate(zip(axes, data, cmaps, headers)):
            if idx == 3:
                vmax = max(abs(values.min()), abs(values.max()))
                vmin = -vmax
            else:
                vmin, vmax = values.min(), values.max()
            
            sc = ax.scatter(x, y, c=values, s=3, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.8)
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            
            ax.set_title(title)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved visualization to {save_path}")


if __name__ == "__main__":
    # Test visualization with random data
    import torch
    
    print("Testing NUNOValidator with random data...")
    
    # Create random point cloud
    np.random.seed(42)
    n_points = 1024
    coords = np.random.rand(n_points, 2) * 2 - 1  # [-1, 1]
    
    # Create smooth test functions
    x, y = coords[:, 0], coords[:, 1]
    invar = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
    target = np.exp(-5 * (x**2 + y**2))
    prediction = target + 0.1 * np.random.randn(n_points)  # Add noise
    
    # Test standalone visualization
    create_standalone_visualization(
        coords, invar, target, prediction,
        save_path="test_nuno_validation.png",
        use_triangulation=True
    )
    
    print("Test complete!")
