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
NUNO Training Script for 2D Darcy Flow with Live Sampling
----------------------------------------------------------
This script trains a Non-Uniform Neural Operator (NUNO) on the 2D Darcy
flow benchmark problem. It differs from a standard FNO script in that
it loads high-resolution uniform grid data and then samples it to a
non-uniform point cloud on-the-fly during training.

This script demonstrates:
1. Loading uniform grid data from HDF5 shards.
2. Using a modular 'PointSampler' to create a fixed non-uniform geometry.
3. Setting up NUNO's K-D tree domain decomposition with this geometry.
4. Training the model by interpolating grid data to the point cloud each step.
5. Accurate GPU time measurement and logging.
6. Post-training evaluation on a different sampling distribution.

Usage:
    python train_nuno_darcy.py
    
    # Override config values (e.g., change sampler)
    python train_nuno_darcy.py sampler=locally_dense training.batch_size=32
"""

import os
import glob
import h5py
from math import ceil

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.nn import MSELoss
from torch.optim import Adam, lr_scheduler
from torch.utils.data import Dataset, DataLoader

# PhysicsNeMo imports
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint
from physicsnemo.launch.logging import PythonLogger, LaunchLogger

# Local imports
from physicsnemo.models.fno.nuno import NUNO
from point_sampler import get_sampler
from nuno_validator import NUNOValidator


class GridDarcyDataset(Dataset):
    """
    Dataset to load uniform grid Darcy data from HDF5 shards.
    
    Parameters
    ----------
    data_dir : str
        Directory containing the HDF5 shards.
    split : str
        "train" or "val".
    """
    def __init__(self, data_dir: str, split: str):
        self.data_dir = data_dir
        self.split = split
        
        self.file_paths = sorted(glob.glob(os.path.join(self.data_dir, self.split, "*.h5")))
        if not self.file_paths:
            raise FileNotFoundError(f"No HDF5 files found in {os.path.join(self.data_dir, self.split)}")

        self.num_samples_per_file = []
        self.file_cumulative_samples = [0]
        
        # Pre-scan files to get sample counts
        for file_path in self.file_paths:
            with h5py.File(file_path, 'r') as f:
                num_in_file = f['permeability'].shape[0]
                self.num_samples_per_file.append(num_in_file)
                self.file_cumulative_samples.append(self.file_cumulative_samples[-1] + num_in_file)

        self.total_samples = self.file_cumulative_samples[-1]

    def __len__(self):
        return self.total_samples

    def __getitem__(self, index):
        # Find which file the index belongs to
        file_idx = next(i for i, total in enumerate(self.file_cumulative_samples) if total > index) - 1
        local_idx = index - self.file_cumulative_samples[file_idx]
        
        with h5py.File(self.file_paths[file_idx], 'r') as f:
            # Data is stored as (H, W, C), convert to (C, H, W) for PyTorch
            permeability = torch.from_numpy(f['permeability'][local_idx, ...]).permute(2, 0, 1)
            pressure = torch.from_numpy(f['pressure'][local_idx, ...]).permute(2, 0, 1)

        return {'permeability': permeability, 'pressure': pressure}


@hydra.main(version_base="1.3", config_path=".", config_name="nuno_config")
def nuno_trainer(cfg: DictConfig) -> None:
    """
    Main training function for NUNO on 2D Darcy flow with live sampling.
    """
    # Initialize distributed manager and logging
    DistributedManager.initialize()
    dist = DistributedManager()
    log = PythonLogger(name="nuno_darcy_live_sampling")
    log.file_logging()
    LaunchLogger.initialize(use_wandb=False) # WandB can be enabled here if desired
    
    log.info("=" * 80)
    log.info("NUNO Training for 2D Darcy Flow with Live On-the-Fly Sampling")
    log.info("=" * 80)
    log.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg)}")

    # SAMPLER: Create the fixed point cloud geometry for this training run
    log.info(f"Initializing training sampler: {cfg.sampler.name}")
    train_sampler = get_sampler(cfg, dist.device)
    coordinates = train_sampler.coordinates
    log.info(f"Generated fixed point cloud geometry: {train_sampler.num_points} points")

    # DATA LOADING: Load uniform grid data
    log.info("Loading uniform grid training data...")
    train_dataset = GridDarcyDataset(data_dir=cfg.data.data_dir, split="train")
    train_loader = DataLoader(train_dataset, 
                              batch_size=cfg.training.batch_size, 
                              shuffle=True, 
                              num_workers=cfg.data.num_workers)
    
    log.info("Loading uniform grid validation data...")
    val_dataset = GridDarcyDataset(data_dir=cfg.data.data_dir, split="val")
    val_loader = DataLoader(val_dataset, 
                            batch_size=cfg.training.batch_size, 
                            shuffle=False, 
                            num_workers=cfg.data.num_workers)

    if not train_dataset or not val_dataset:
        log.error("Data could not be loaded. Please generate master grid data first.")
        return

    # MODEL
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
    
    # Critical step for NUNO: set up domain decomposition with the fixed coordinates
    log.info("Setting up NUNO domain decomposition...")
    model.setup_domain(coordinates)
    visualize_subdomains(model, coordinates) # Optional visualization
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model parameters: {n_params:,}")
    log.info(f"Training on point clouds of size {train_sampler.num_points}")
    log.info(f"Max grid size in subdomains: {model.domain_decomp.max_grid_size}")
    
    # TRAINING COMPONENTS
    loss_fun = MSELoss(reduction="mean")
    optimizer = Adam(model.parameters(), lr=cfg.scheduler.initial_lr)
    scheduler = lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: cfg.scheduler.decay_rate ** step
    )
    
    # VALIDATOR (needs normalization constants)
    normalizer = {
        'permeability': (cfg.normalizer.permeability.mean, cfg.normalizer.permeability.std_dev),
        'darcy': (cfg.normalizer.darcy.mean, cfg.normalizer.darcy.std_dev)
    }
    validator = NUNOValidator(
        loss_fun=MSELoss(reduction="none"), # Use 'none' for per-sample loss, then mean
        norm=normalizer,
        use_triangulation=cfg.validation.use_triangulation
    )
    
    # CHECKPOINTING
    ckpt_path = f"./checkpoints/darcy_nuno_{cfg.sampler.name}"
    os.makedirs(ckpt_path, exist_ok=True)
    ckpt_args = { "path": ckpt_path, "optimizer": optimizer, "scheduler": scheduler, "models": model }
    loaded_pseudo_epoch = load_checkpoint(device=dist.device, **ckpt_args)
    
    # Define forward passes for training and evaluation
    @StaticCaptureTraining(model=model, optim=optimizer, logger=log, use_amp=False)
    def forward_train(invars, target):
        pred = model(invars)
        loss = loss_fun(pred, target)
        return loss
    
    @StaticCaptureEvaluateNoGrad(model=model, logger=log, use_amp=False)
    def forward_eval(invars):
        return model(invars)
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    if loaded_pseudo_epoch == 0:
        log.success("Training started...")
    else:
        log.warning(f"Resuming training from pseudo epoch {loaded_pseudo_epoch + 1}.")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    steps_per_pseudo_epoch = ceil(cfg.training.pseudo_epoch_sample_size / cfg.training.batch_size)
    log_args = { "name_space": "train", "num_mini_batch": steps_per_pseudo_epoch, "epoch_alert_freq": 1 }

    for pseudo_epoch in range(max(1, loaded_pseudo_epoch + 1), cfg.training.max_pseudo_epochs + 1):
        model.train()
        epoch_gpu_time_ms = 0.0
        
        with LaunchLogger(**log_args, epoch=pseudo_epoch) as logger:
            train_iter = iter(train_loader)
            for step in range(steps_per_pseudo_epoch):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)
                
                # Move uniform grids to device
                grid_k = batch['permeability'].to(dist.device)
                grid_p = batch['pressure'].to(dist.device)

                k_shift = grid_k.mean()
                p_shift = grid_p.mean()

                grid_k = grid_k - k_shift
                grid_p = grid_p - p_shift

                # LIVE SAMPLING: Interpolate grid data to the fixed point cloud
                invar = train_sampler.sample(grid_k)
                target = train_sampler.sample(grid_p)

                # Timed forward/backward pass
                start_event.record()
                loss = forward_train(invar, target)
                end_event.record()
                torch.cuda.synchronize()
                
                epoch_gpu_time_ms += start_event.elapsed_time(end_event)
                logger.log_minibatch({"loss": loss.detach()})
            
            logger.log_epoch({
                "Learning Rate": optimizer.param_groups[0]["lr"],
                "Epoch GPU Time (ms)": epoch_gpu_time_ms,
            })
        
        # Save checkpoint
        if pseudo_epoch % cfg.training.rec_results_freq == 0:
            save_checkpoint(**ckpt_args, epoch=pseudo_epoch)
            log.info(f"Saved checkpoint to {ckpt_path} at epoch {pseudo_epoch}")
        
        # Update learning rate
        if pseudo_epoch % cfg.scheduler.decay_pseudo_epochs == 0:
            scheduler.step()
            log.info(f"Learning rate decayed to {optimizer.param_groups[0]['lr']:.2e}")

        # ====================================================================
        # VALIDATION PHASE
        # ====================================================================
        if pseudo_epoch % cfg.validation.validation_pseudo_epochs == 0:
            model.eval()
            with LaunchLogger("valid", epoch=pseudo_epoch) as logger:
                total_loss = 0.0
                for val_step, batch in enumerate(val_loader):
                    grid_k = batch['permeability'].to(dist.device)
                    grid_p = batch['pressure'].to(dist.device)
                    
                    k_shift = grid_k.mean()
                    p_shift = grid_p.mean() 

                    grid_k = grid_k - k_shift
                    grid_p = grid_p - p_shift

                    # Sample validation data to the same point cloud geometry
                    invar = train_sampler.sample(grid_k)
                    target = train_sampler.sample(grid_p)
                    
                    prediction = forward_eval(invar)

                    # For the first batch, run detailed validation with plots
                    if val_step == 0:
                        val_loss = validator.compare(
                            coordinates, invar, target, prediction, pseudo_epoch, logger
                        )
                    else: # For subsequent batches, just compute loss
                        val_loss = loss_fun(prediction, target).item()
                    
                    total_loss += val_loss
                
                avg_val_loss = total_loss / len(val_loader)
                logger.log_epoch({"Validation Loss": avg_val_loss})
                log.info(f"Epoch {pseudo_epoch}: Validation loss = {avg_val_loss:.6f}")

    log.success("Training completed!")
    
    # ========================================================================
    # POST-TRAINING EVALUATION
    # ========================================================================
    if cfg.evaluation.run_after_training:
        post_training_evaluation(
            cfg=cfg,
            model=model,
            val_loader=val_loader,
            dist=dist,
            log=log,
            ckpt_path=ckpt_path,
        )


def visualize_subdomains(model, coordinates):
    """Visualize K-D tree subdomain assignments."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        log = PythonLogger(name="visualize")
        coords_np = coordinates.cpu().numpy()
        
        if model.domain_decomp is None:
            log.warning("Domain decomposition not set up, skipping visualization.")
            return

        info = model.domain_decomp.get_subdomain_info()
        indices_list = info['indices']
        
        colors = np.zeros(len(coords_np))
        for subdomain_idx, point_indices in enumerate(indices_list):
            colors[point_indices] = subdomain_idx
        
        plt.figure(figsize=(8, 8))
        plt.scatter(coords_np[:, 0], coords_np[:, 1], c=colors, cmap='viridis', s=5, alpha=0.8)
        cbar = plt.colorbar(label='Subdomain ID')
        cbar.set_ticks(np.arange(len(indices_list)))
        plt.title(f'K-D Tree Decomposition ({len(indices_list)} subdomains)')
        plt.xlabel('x-coordinate')
        plt.ylabel('y-coordinate')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig('subdomain_visualization.png', dpi=200)
        log.info("Saved subdomain visualization to 'subdomain_visualization.png'")
        plt.close()

    except ImportError:
        log.warning("Matplotlib not found. Skipping subdomain visualization.")

def post_training_evaluation(
    cfg: DictConfig,
    model: NUNO,
    val_loader: DataLoader,
    dist: DistributedManager,
    log: PythonLogger,
    ckpt_path: str,
) -> None:
    """
    Performs post-training evaluation on a different geometry.

    Loads the final model, sets up a new sampler, re-initializes the model's
    domain decomposition for the new geometry, and runs a forward pass.
    """
    log.info("=" * 80)
    log.info("Starting Post-Training Evaluation")
    log.info("=" * 80)

    # Load the final trained model from the specified checkpoint path
    log.info(f"Loading final model from {ckpt_path}")
    eval_ckpt_args = {"path": ckpt_path, "models": model}
    load_checkpoint(device=dist.device, **eval_ckpt_args)
    model.eval()

    # Set up a new sampler for evaluation
    log.info(f"Initializing evaluation sampler: {cfg.evaluation.sampler.name}")
    eval_sampler = get_sampler(
        OmegaConf.create({"sampler": cfg.evaluation.sampler}), dist.device
    )
    eval_coords = eval_sampler.coordinates
    log.info(f"Evaluation geometry: {eval_sampler.num_points} points")

    # The NUNO model's domain decomposition is fixed. For a fair evaluation
    # on a NEW geometry, we must re-initialize the decomposition.
    log.warning("Re-initializing model's domain decomposition for evaluation geometry.")
    model.setup_domain(eval_coords, force_rebuild=True)

    # Get one sample from the validation set for testing
    test_batch = next(iter(val_loader))
    grid_k = test_batch["permeability"].to(dist.device)
    
    print(f"DEBUG: Min={grid_k.min():.3f}, Max={grid_k.max():.3f}, Mean={grid_k.mean():.3f}")


    grid_p = test_batch["pressure"].to(dist.device)

    # Sample the grid using the evaluation sampler
    test_invar = eval_sampler.sample(grid_k)
    test_target = eval_sampler.sample(grid_p)

    # Define forward eval pass
    @StaticCaptureEvaluateNoGrad(model=model, logger=log, use_amp=False)
    def forward_eval(invars):
        return model(invars)

    # Run forward pass and compute loss
    test_pred = forward_eval(test_invar)
    loss_fun = MSELoss(reduction="mean")
    eval_loss = loss_fun(test_pred, test_target).item()

    log.success(
        f"Evaluation on new geometry ('{cfg.evaluation.sampler.name}') complete."
    )
    log.info(f"Final evaluation loss: {eval_loss:.6f}")



if __name__ == "__main__":
    nuno_trainer()