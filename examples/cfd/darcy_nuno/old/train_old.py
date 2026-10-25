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
NUNO Training Script for 2D Darcy Flow
--------------------------------------
Training script for the Non-Uniform Neural Operator (NUNO) on the 
2D Darcy flow benchmark problem with non-uniform point cloud data.

This script demonstrates:
1. Loading non-uniform point cloud data with fixed geometry
2. Setting up NUNO's K-D tree domain decomposition
3. Training with PhysicsNeMo utilities (logging, checkpointing)
4. Validation with point cloud visualization

Usage:
    python train_nuno_darcy.py
    
    # Override config values:
    python train_nuno_darcy.py training.batch_size=32 training.max_pseudo_epochs=10
"""

import os
from math import ceil

import hydra
from omegaconf import DictConfig
import torch
from torch.nn import MSELoss
from torch.optim import Adam, lr_scheduler

# PhysicsNeMo imports
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint
from physicsnemo.launch.logging import PythonLogger, LaunchLogger

# Local imports
from physicsnemo.models.fno.nuno import NUNO  # Your NUNO implementation
from nuno_darcy_dataset import NUNODarcyDataset, create_dataloader
from nuno_validator import NUNOValidator

@hydra.main(version_base="1.3", config_path=".", config_name="nuno_config")
def nuno_trainer(cfg: DictConfig) -> None:
    """
    Training for NUNO on 2D Darcy flow with non-uniform data.
    
    This training script demonstrates how to:
    1. Load non-uniform point cloud data from HDF5 files
    2. Set up NUNO's domain decomposition using fixed coordinates
    3. Train the model with proper logging and checkpointing
    4. Validate with heatmap-style visualizations
    
    The data uses fixed geometry (same coordinates for all samples),
    which is required for NUNO's K-D tree domain decomposition.
    """
    # Initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()
    
    # Initialize logging
    log = PythonLogger(name="nuno_darcy")
    log.file_logging()
    LaunchLogger.initialize()
    
    log.info("=" * 60)
    log.info("NUNO Training for 2D Darcy Flow")
    log.info("=" * 60)
    
    # Setup normalization
    norm_vars = cfg.normalizer
    normalizer = {
        'permeability': (norm_vars.permeability.mean, norm_vars.permeability.std_dev),
        'darcy': (norm_vars.darcy.mean, norm_vars.darcy.std_dev)
    }
    
    # Create dataloaders
    log.info("Loading training data...")
    train_loader, train_dataset = create_dataloader(
        data_dir=cfg.data.data_dir,
        split="train",
        batch_size=cfg.training.batch_size,
        normalizer=normalizer,
        shuffle=True,
        num_workers=cfg.data.num_workers
    )
    
    log.info("Loading validation data...")
    # Note: Don't drop_last for validation to use all available samples
    val_loader, val_dataset = create_dataloader(
        data_dir=cfg.data.data_dir,
        split="val",
        batch_size=cfg.training.batch_size,
        normalizer=normalizer,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        drop_last=False  # Important: use all validation samples
    )
    
    # Warn if validation dataset is small
    if len(val_dataset) < cfg.training.batch_size:
        log.warning(f"Validation dataset ({len(val_dataset)} samples) is smaller than batch size ({cfg.training.batch_size}). Consider generating more validation data.")
    
    # Get fixed coordinates for domain decomposition
    coordinates = train_dataset.get_coordinates()
    num_points = train_dataset.num_points
    log.info(f"Point cloud geometry: {num_points} points")
    
    # Create NUNO model
    log.info("Creating NUNO model...")
    model = NUNO(
        in_channels=cfg.arch.nuno.in_channels,
        out_channels=cfg.arch.decoder.out_features,
        n_subdomains=cfg.arch.nuno.n_subdomains,
        decoder_layers=cfg.arch.decoder.layers,
        decoder_layer_size=cfg.arch.decoder.layer_size,
        dimension=cfg.arch.nuno.dimension,
        latent_channels=cfg.arch.nuno.latent_channels,
        num_fno_layers=cfg.arch.nuno.fno_layers,
        num_fno_modes=cfg.arch.nuno.fno_modes,
        padding=cfg.arch.nuno.padding,
        coord_features=cfg.arch.nuno.coord_features,
        n_blocks=cfg.arch.nuno.n_blocks,
        smallest_points=cfg.arch.nuno.smallest_points,
        oversamp_ratio=cfg.arch.nuno.oversamp_ratio
    ).to(dist.device)
    
    # Setup domain decomposition with the fixed coordinates
    # This is a critical step for NUNO - the K-D tree is built once
    log.info("Setting up domain decomposition...")
    model.setup_domain(coordinates)
    
    # Log model info
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model parameters: {n_params:,}")
    log.info(f"Subdomains: {model.n_subdomains}")
    log.info(f"Max grid size: {model.domain_decomp.max_grid_size}")
    
    # Setup training components
    loss_fun = MSELoss(reduction="mean")
    optimizer = Adam(model.parameters(), lr=cfg.scheduler.initial_lr)
    scheduler = lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda=lambda step: cfg.scheduler.decay_rate ** step
    )
    
    # Setup validator
    validator = NUNOValidator(
        loss_fun=MSELoss(reduction="mean"),
        norm=normalizer,
        use_triangulation=cfg.validation.use_triangulation
    )
    
    # Checkpoint setup
    ckpt_args = {
        "path": "./checkpoints",
        "optimizer": optimizer,
        "scheduler": scheduler,
        "models": model,
    }
    loaded_pseudo_epoch = load_checkpoint(device=dist.device, **ckpt_args)
    
    # Calculate iterations per pseudo-epoch
    steps_per_pseudo_epoch = ceil(
        cfg.training.pseudo_epoch_sample_size / cfg.training.batch_size
    )
    
    log_args = {
        "name_space": "train",
        "num_mini_batch": steps_per_pseudo_epoch,
        "epoch_alert_freq": 1,
    }
    
    # Define forward passes
    @StaticCaptureTraining(
        model=model, optim=optimizer, logger=log, use_amp=False, use_graphs=False
    )
    def forward_train(invars, target):
        """Training forward pass."""
        pred = model(invars)
        loss = loss_fun(pred, target)
        return loss
    
    @StaticCaptureEvaluateNoGrad(
        model=model, logger=log, use_amp=False, use_graphs=False
    )
    def forward_eval(invars):
        """Evaluation forward pass."""
        return model(invars)
    
    # Training loop
    if loaded_pseudo_epoch == 0:
        log.success("Training started...")
    else:
        log.warning(f"Resuming training from pseudo epoch {loaded_pseudo_epoch + 1}.")
    
    for pseudo_epoch in range(
        max(1, loaded_pseudo_epoch + 1), cfg.training.max_pseudo_epochs + 1
    ):
        # Training phase
        model.train()
        with LaunchLogger(**log_args, epoch=pseudo_epoch) as logger:
            train_iter = iter(train_loader)
            for step in range(steps_per_pseudo_epoch):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)
                
                # Move to device
                invar = batch['permeability'].to(dist.device)  # (B, N, 1)
                target = batch['pressure'].to(dist.device)      # (B, N, 1)
                
                # Forward pass
                loss = forward_train(invar, target)
                logger.log_minibatch({"loss": loss.detach()})
            
            logger.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})
        
        # Save checkpoint
        if pseudo_epoch % cfg.training.rec_results_freq == 0:
            save_checkpoint(**ckpt_args, epoch=pseudo_epoch)
            log.info(f"Saved checkpoint at epoch {pseudo_epoch}")
        
        # Validation phase
        if pseudo_epoch % cfg.validation.validation_pseudo_epochs == 0:
            model.eval()
            with LaunchLogger("valid", epoch=pseudo_epoch) as logger:
                total_loss = 0.0
                num_val_batches = 0
                
                for val_step, batch in enumerate(val_loader):
                    invar = batch['permeability'].to(dist.device)
                    target = batch['pressure'].to(dist.device)
                    coords = batch['coordinates'].to(dist.device)
                    
                    # Forward pass
                    prediction = forward_eval(invar)

                    # Validate (with visualization on first batch)
                    if val_step == 0:
                        val_loss = validator.compare(
                            coords, invar, target, prediction,
                            pseudo_epoch, logger
                        )
                    else:
                        val_loss = loss_fun(prediction, target).item()
                    
                    total_loss += val_loss
                    num_val_batches += 1
                
                avg_val_loss = total_loss / max(num_val_batches, 1)
                logger.log_epoch({"Validation error": avg_val_loss})
                log.info(f"Epoch {pseudo_epoch}: Validation loss = {avg_val_loss:.6f}")
                    
        # Update learning rate
        if pseudo_epoch % cfg.scheduler.decay_pseudo_epochs == 0:
            scheduler.step()
            log.info(f"Learning rate decayed to {optimizer.param_groups[0]['lr']:.2e}")
    
    # Final checkpoint
    save_checkpoint(**ckpt_args, epoch=cfg.training.max_pseudo_epochs)
    log.success("Training completed!")


def visualize_subdomains(model, coordinates):
    """Visualize K-D tree subdomain assignments."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    coords = coordinates.cpu().numpy()
    
    # Get subdomain info from domain decomposition
    info = model.domain_decomp.get_subdomain_info()
    indices_list = info['indices']
    
    # Color each point by its subdomain
    colors = np.zeros(len(coords))
    for subdomain_idx, point_indices in enumerate(indices_list):
        colors[point_indices] = subdomain_idx
    
    plt.figure(figsize=(8, 8))
    plt.scatter(coords[:, 0], coords[:, 1], c=colors, cmap='tab10', s=5)
    plt.colorbar(label='Subdomain ID')
    plt.title(f'K-D Tree Decomposition ({len(indices_list)} subdomains)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.savefig('subdomain_visualization.png', dpi=150)

if __name__ == "__main__":
    nuno_trainer()