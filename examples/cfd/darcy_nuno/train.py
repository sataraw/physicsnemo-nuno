"""
Train NUNO on 2D Darcy Flow data.

Darcy2D generates uniform grids (B, 1, H, W).  We sample these onto a fixed
point cloud using grid_sample, then pass the point-cloud data through NUNO's
prep2D to partition into subdomains and normalise.

No Hydra / omegaconf — just constants at the top.
"""

import torch
import torch.nn.functional as F
import numpy as np
import copy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from torch.utils.data import DataLoader, TensorDataset

from physicsnemo.datapipes.benchmarks.darcy import Darcy2D
from physicsnemo.models.fno.nuno import NUNO
from point_sampler import get_sampler

# ──────────────────── hyperparameters ────────────────────
NUM_POINTS = 3809  # points in the fixed point cloud
NUM_SAMPLES = 200
BATCH_SIZE = 20  # for both Darcy generator & training
DARCY_RESOLUTION = 512  # resolution of the Darcy generator grid
N_SUBDOMAINS = 8  # NUNO subdomain count
MAX_EPOCHS = 100
LR = 1e-3
VAL_SPLIT = 0.1
VAL_EVERY = 10  # validate & visualise every N epochs
N_VIS = 4  # samples to visualise spatially
SEED = 42

# Known Darcy normalization constants (for pre-normalization)
PERM_MEAN, PERM_STD = 1.25, 0.75
DARCY_MEAN, DARCY_STD = 4.52e-2, 2.79e-2
# ─────────────────────────────────────────────────────────


# =====================================================================
#  Data generation — replaces generate_random_waves(coords, batch_size)
# =====================================================================
def generate_darcy_pointcloud(
    coords_gs: torch.Tensor,
    num_samples: int,
    batch_size: int,
    resolution: int,
    device: torch.device,
):
    """
    Sample Darcy flow fields onto a fixed point cloud, then pre-normalise.

    We pre-normalise with known Darcy statistics so that prep2D sees
    ~N(0,1) data.  This is important because we call prep2D twice
    (once for inputs, once for targets) and the second call overwrites
    the internal normalisation state.  With pre-normalised data the
    stored stats are ~(mean=0, std=1), making the normaliser effectively
    an identity — so it doesn't matter which call's stats survive.

    Parameters
    ----------
    coords_gs : Tensor, shape (1, 1, N, 2)
        Fixed sampling coordinates in [-1, 1]^2 (grid_sample convention).
    num_samples : int
        How many (input, target) pairs to produce.
    batch_size : int
        Batch size handed to Darcy2D.
    resolution : int
        Spatial resolution of the generated grids.
    device : torch.device

    Returns
    -------
    permeability : Tensor (num_samples, N, 1)   — input field  (pre-normalised)
    pressure     : Tensor (num_samples, N, 1)   — target field (pre-normalised)
    """
    generator = Darcy2D(
        resolution=resolution,
        batch_size=batch_size,
        normaliser=None,
    )

    perm_chunks, press_chunks = [], []
    count = 0

    for batch in generator:
        k_grid = batch["permeability"].to(device)  # (B, 1, H, W)
        p_grid = batch["darcy"].to(device)  # (B, 1, H, W)

        # Darcy2D produces one unique random field per iteration,
        # replicated across the batch dimension.  Take only the first
        # sample from each batch to avoid duplicates.
        k_grid = k_grid[0:1]  # (1, 1, H, W)
        p_grid = p_grid[0:1]  # (1, 1, H, W)

        coords_1 = coords_gs[:1]  # (1, 1, N, 2)

        with torch.no_grad():
            k_pts = F.grid_sample(
                k_grid, coords_1, align_corners=False, mode="bilinear"
            )  # (1, 1, 1, N)
            p_pts = F.grid_sample(
                p_grid, coords_1, align_corners=False, mode="bilinear"
            )  # (1, 1, 1, N)

        # reshape -> (1, N, 1)
        k_pts = k_pts.squeeze(2).permute(0, 2, 1)
        p_pts = p_pts.squeeze(2).permute(0, 2, 1)

        # Pre-normalise so prep2D's internal normaliser becomes ~identity
        k_pts = (k_pts - PERM_MEAN) / PERM_STD
        p_pts = (p_pts - DARCY_MEAN) / DARCY_STD

        perm_chunks.append(k_pts)
        press_chunks.append(p_pts)

        count += 1
        if count % 200 == 0:
            print(f"  generated {count}/{num_samples} ...")
        if count >= num_samples:
            break

    perm = torch.cat(perm_chunks, dim=0)[:num_samples]  # (S, N, 1)
    press = torch.cat(press_chunks, dim=0)[:num_samples]

    print(f"  permeability  range: [{perm.min():.4f}, {perm.max():.4f}]  (pre-normalised)")
    print(f"  pressure      range: [{press.min():.4f}, {press.max():.4f}]  (pre-normalised)")
    return perm, press





# =====================================================================
#  Visualisation helpers
# =====================================================================
def plot_losses(train_hist, val_hist, save_path="darcy_losses.png"):
    """Training & validation loss curves."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_hist, label="train")
    if val_hist:
        ve, vl = zip(*val_hist)
        ax.plot(ve, vl, "o-", label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("NUNO - Darcy Flow Training")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  saved loss curves -> {save_path}")


def plot_predictions(model, val_loader, device, epoch, n_samples=4, save_path=None):
    """Scatter: predicted vs true values, plus per-sample error histograms."""
    if save_path is None:
        save_path = f"darcy_pred_epoch{epoch:03d}.png"

    model.eval()
    x_batch, y_batch = next(iter(val_loader))
    with torch.no_grad():
        pred_batch = model(x_batch.to(device))

    n_samples = min(n_samples, x_batch.shape[0])

    fig, axes = plt.subplots(2, n_samples, figsize=(4 * n_samples, 7))
    if n_samples == 1:
        axes = axes[:, None]

    for i in range(n_samples):
        true = y_batch[i].cpu().flatten().numpy()
        pred = pred_batch[i].cpu().flatten().numpy()

        ax = axes[0, i]
        ax.scatter(true, pred, s=2, alpha=0.4)
        lo, hi = min(true.min(), pred.min()), max(true.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_xlabel("true")
        ax.set_ylabel("pred")
        ax.set_title(f"sample {i}")
        ax.set_aspect("equal", "box")

        ax = axes[1, i]
        err = pred - true
        ax.hist(err, bins=40, edgecolor="k", linewidth=0.3)
        ax.set_xlabel("pred - true")
        ax.set_title(f"MAE={np.mean(np.abs(err)):.4f}")

    fig.suptitle(f"Epoch {epoch}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  saved scatter plot -> {save_path}")


def plot_spatial_fields(
    model, train_a, train_u,
    raw_perm_vis, raw_press_vis, coords_np,
    device, epoch,
    n_samples=3, save_path=None,
):
    """
    Side-by-side spatial comparison using tricontourf on the original
    point-cloud coordinates.

    Columns per sample:
        [Input Permeability]  [GT Pressure]  [Predicted Pressure]  [Abs Error]

    The model output has shape (B, max_pts_per_sd, output_dim, n_subdomains).
    For subdomain i, the first len(indices_sd[i]) entries correspond to
    coords_np[indices_sd[i]].  We reconstruct the full field by placing
    each subdomain's values at the correct original coordinates.
    """
    if save_path is None:
        save_path = f"darcy_spatial_epoch{epoch:03d}.png"

    model.eval()
    n_samples = min(n_samples, raw_perm_vis.shape[0], train_a.shape[0])

    with torch.no_grad():
        preds = model(train_a[:n_samples].to(device))
    targets = train_u[:n_samples]

    # Reconstruct full fields at original coordinates using indices_sd
    N = coords_np.shape[0]
    n_sd = model.n_subdomains
    indices_sd = model.indices_sd  # list of arrays, one per subdomain

    def _reconstruct_field(tensor):
        """
        Map (n_samples, max_pts_sd, output_dim, n_subdomains) back to
        (n_samples, N) at original coordinate positions.
        """
        v = tensor.cpu().float().numpy()
        n = v.shape[0]
        field = np.full((n, N), np.nan)
        for sd in range(n_sd):
            idx = indices_sd[sd]      # original point indices for this subdomain
            n_pts = len(idx)
            # v[:, :n_pts, 0, sd] → values for this subdomain's points
            # (squeeze output_dim which is 1)
            if v.ndim == 4:
                vals = v[:, :n_pts, 0, sd]  # (n, n_pts)
            elif v.ndim == 3:
                vals = v[:, :n_pts, sd]     # (n, n_pts)
            else:
                vals = v[:, :n_pts]
            field[:, idx] = vals
        return field

    pred_full = _reconstruct_field(preds)    # (n_samples, N)
    true_full = _reconstruct_field(targets)  # (n_samples, N)

    # Verify reconstruction: compare reconstructed GT vs raw GT
    for i in range(min(n_samples, 3)):
        raw_gt = raw_press_vis[i].cpu().squeeze().numpy()  # (N,)
        recon_gt = true_full[i]                              # (N,)
        recon_diff = np.abs(raw_gt - recon_gt).mean()
        if i == 0:
            print(f"  [viz] Reconstruction verification:")
        print(f"    sample {i}: |raw_GT - reconstructed_GT| = {recon_diff:.6f}"
              f"  (should be ~0 if reconstruction is correct)")

    # Check for NaN (points not covered by any subdomain — shouldn't happen)
    n_nan = np.isnan(pred_full[0]).sum()
    if n_nan > 0:
        print(f"  [viz] WARNING: {n_nan}/{N} points have NaN in reconstructed field")
        pred_full = np.nan_to_num(pred_full, nan=0.0)
        true_full = np.nan_to_num(true_full, nan=0.0)

    # Build triangulation on the original N input points
    tri_input = Triangulation(coords_np[:, 0], coords_np[:, 1])

    n_cols = 4
    fig, axes = plt.subplots(
        n_samples, n_cols,
        figsize=(5 * n_cols, 4.5 * n_samples),
        squeeze=False,
    )

    for i in range(n_samples):
        perm_vals = raw_perm_vis[i].cpu().squeeze().numpy()
        press_vals = raw_press_vis[i].cpu().squeeze().numpy()
        pred_i = pred_full[i]
        true_i = true_full[i]

        # Col 0 — Input permeability
        ax = axes[i, 0]
        tc = ax.tricontourf(tri_input, perm_vals, levels=32, cmap="viridis")
        fig.colorbar(tc, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Input (perm)" if i == 0 else "")
        ax.set_ylabel(f"sample {i}")
        ax.set_aspect("equal")

        # Col 1 — GT pressure (from raw pre-prep2D data)
        ax = axes[i, 1]
        tc = ax.tricontourf(tri_input, press_vals, levels=32, cmap="RdBu_r")
        fig.colorbar(tc, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("GT (pressure)" if i == 0 else "")
        ax.set_aspect("equal")

        # Col 2 — Predicted pressure (reconstructed from subdomains)
        ax = axes[i, 2]
        tc = ax.tricontourf(tri_input, pred_i, levels=32, cmap="RdBu_r")
        fig.colorbar(tc, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Prediction" if i == 0 else "")
        ax.set_aspect("equal")

        # Col 3 — Absolute error vs reconstructed GT
        ax = axes[i, 3]
        abs_err = np.abs(pred_i - true_i)
        tc = ax.tricontourf(tri_input, abs_err, levels=32, cmap="hot_r")
        fig.colorbar(tc, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"|Error| (MAE={abs_err.mean():.4f})" if i == 0
                      else f"MAE={abs_err.mean():.4f}")
        ax.set_aspect("equal")

    fig.suptitle(f"Spatial Fields — Epoch {epoch}", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved spatial fields -> {save_path}")


# =====================================================================
#  Main
# =====================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # -- 1. Fixed point-cloud coordinates --
    torch.manual_seed(SEED)
    # grid_sample expects coords in [-1, 1]
    #coords_gs = torch.rand(1, 1, NUM_POINTS, 2, device=device) * 2 - 1

    from point_sampler import LocallyDenseSampler

    sampler = LocallyDenseSampler(num_points=NUM_POINTS, device=device)

    coords_gs = sampler._generate_coordinates().unsqueeze(0).unsqueeze(0)

    # prep2D expects coords in [0, 1]  (NUNO convention)
    coords_np = (coords_gs.squeeze().cpu().numpy() + 1.0) / 2.0  # (N, 2)

    # -- 2. Generate Darcy data as point clouds --
    print("Generating Darcy flow point-cloud data ...")
    perm, press = generate_darcy_pointcloud(
        coords_gs, NUM_SAMPLES, BATCH_SIZE, DARCY_RESOLUTION, device
    )
    # perm:  (S, N, 1) — permeability (input,  pre-normalised)
    # press: (S, N, 1) — pressure     (target, pre-normalised)

    # Save a few raw samples for spatial visualisation (before prep2D mangles them)
    raw_perm_vis = perm[:N_VIS].clone()
    raw_press_vis = press[:N_VIS].clone()

    # -- 3. Initialise NUNO --
    model = NUNO(
        n_subdomains=N_SUBDOMAINS,
        output_dim=1,
        spatial_dim=2,
    ).to(device)

    # -- 4. prep2D — partitions points into subdomains & normalises --
    #
    # CRITICAL: prep2D overwrites internal model state.  We deepcopy for
    # the second call (targets) so the original model is untouched.
    #
    # Raw data check
    print(f"\n  Raw data check:")
    print(f"    perm shape: {perm.shape}")
    print(f"    perm[0] stats: mean={perm[0].mean():.4f}, std={perm[0].std():.4f}")
    print(f"    perm[1] stats: mean={perm[1].mean():.4f}, std={perm[1].std():.4f}")
    print(f"    perm[0] vs perm[1] diff: {(perm[0] - perm[1]).abs().mean().item():.6f}")
    print(f"    coords_np shape: {coords_np.shape}")

    print("\nRunning prep2D on original model (inputs) ...")
    train_a, _, _, _, y_norm, a_norm = model.prep2D(
        perm, coords_np, n_train=NUM_SAMPLES, n_test=0
    )
    print(
        f"  train_a shape: {train_a.shape}  "
        f"range: [{train_a.min():.4f}, {train_a.max():.4f}]"
    )

    # === KEY DIAGNOSTIC: where does the collapse happen? ===
    print(f"\n  === Collapse diagnostic ===")

    # 1. Check raw data — do samples actually differ before prep2D?
    print(f"    Raw perm checks:")
    for i in range(min(5, len(perm))):
        for j in range(i+1, min(5, len(perm))):
            d = (perm[i] - perm[j]).abs().mean().item()
            print(f"      perm[{i}] vs [{j}]: {d:.8f}")

    # 2. Check train_a — multiple pairs, not just 0 vs 1
    print(f"    train_a pairwise checks:")
    any_differ = False
    for i in range(min(5, len(train_a))):
        for j in range(i+1, min(5, len(train_a))):
            d = (train_a[i] - train_a[j]).abs().mean().item()
            tag = " ← DIFFER!" if d > 1e-8 else ""
            if d > 1e-8:
                any_differ = True
            print(f"      train_a[{i}] vs [{j}]: {d:.8f}{tag}")

    # 3. Batch variance — is it truly all identical or just a few?
    batch_var = train_a.var(dim=0).mean().item()
    print(f"    train_a batch variance: {batch_var:.8f}")
    if batch_var < 1e-10:
        print(f"    -> ALL samples in train_a are identical!")
    else:
        print(f"    -> Samples vary (batch variance > 0)")

    # 4. Check unique samples
    ta_flat = train_a.reshape(len(train_a), -1)
    # Quick uniqueness: hash each sample
    unique_hashes = set()
    for i in range(len(ta_flat)):
        h = ta_flat[i].sum().item()  # crude hash
        unique_hashes.add(round(h, 6))
    print(f"    Approximate unique samples (by sum): {len(unique_hashes)}/{len(train_a)}")

    # 5. Check normalizer stats
    a_mean_t = a_norm.mean.cpu() if isinstance(a_norm.mean, torch.Tensor) else a_norm.mean
    a_std_t = a_norm.std.cpu() if isinstance(a_norm.std, torch.Tensor) else a_norm.std
    print(f"    a_norm.std range: [{a_std_t.min():.6f}, {a_std_t.max():.6f}]")
    print(f"  === End diagnostic ===\n")

    # Deepcopy the model, call prep2D on the copy for targets, discard copy
    print("Running prep2D on model copy (targets) ...")
    model_copy = copy.deepcopy(model)
    _, train_u, _, _, y_norm_press, _ = model_copy.prep2D(
        press, coords_np, n_train=NUM_SAMPLES, n_test=0
    )

    # CRITICAL FIX: The original model's y_normalizer was set by prep2D(perm),
    # so it has PERMEABILITY statistics.  But the forward pass uses
    # y_normalizer.decode() on predictions that should match PRESSURE targets.
    # Copy the pressure-based y_normalizer from the copy to the original model.
    model.y_normalizer = model_copy.y_normalizer
    print(f"  Replaced model.y_normalizer with pressure-based normalizer")
    print(f"    y_norm mean range: [{model.y_normalizer.mean.min():.4f}, "
          f"{model.y_normalizer.mean.max():.4f}]")
    print(f"    y_norm std range:  [{model.y_normalizer.std.min():.4f}, "
          f"{model.y_normalizer.std.max():.4f}]")

    del model_copy
    print(
        f"  train_u shape: {train_u.shape}  "
        f"range: [{train_u.min():.4f}, {train_u.max():.4f}]"
    )

    # Input difference check
    _input_diff = (train_a[0] - train_a[1]).abs().mean().item()
    if _input_diff < 1e-8:
        print("\n  WARNING: Inputs are identical after prep2D!")
    else:
        print(f"\n  OK — inputs differ (diff={_input_diff:.6f})")

    # Model output check (untrained models may produce constants — that's OK)
    model.eval()
    with torch.no_grad():
        _p0 = model(train_a[0:1])
        _p1 = model(train_a[1:2])
    _pred_diff = (_p0 - _p1).abs().mean().item()
    print(f"  Pre-training pred diff: {_pred_diff:.6f}")
    if _pred_diff < 1e-8 and _input_diff > 1e-8:
        print("  Note: inputs differ but predictions don't — normal for untrained model.")

    # -- 4b. Shape diagnostic & alignment --
    model.eval()
    with torch.no_grad():
        test_pred = model(train_a[:2])
    print(f"\n  ** Shape check **")
    print(f"     model(train_a[:2]) -> {test_pred.shape}")
    print(f"     train_u[:2]        -> {train_u[:2].shape}")

    if test_pred.shape[1:] != train_u[:2].shape[1:]:
        print(f"  !!  SHAPE MISMATCH - aligning target to model output")
        mask = model.input_u_sd_mask
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        mask = mask.bool()
        print(f"     mask shape: {mask.shape}, non-zero: {mask.sum()} / {mask.numel()}")

        S = train_u.shape[0]
        C = train_u.shape[-1]
        train_u_flat = train_u.reshape(S, -1, C)
        mask_flat = mask.reshape(-1)
        train_u = train_u_flat[:, mask_flat, :]

        if test_pred.dim() == 4 and train_u.dim() == 3:
            train_u = train_u.unsqueeze(2)

        print(f"     train_u after alignment: {train_u.shape}")
        assert test_pred.shape[1:] == train_u[:2].shape[1:], (
            f"Still mismatched: pred {test_pred.shape} vs target {train_u[:2].shape}"
        )
        print(f"  OK  Shapes match!\n")
    else:
        print(f"  OK  Shapes already match!\n")

    # Free raw data
    del perm, press
    torch.cuda.empty_cache()

    # -- 5. DataLoaders --
    dataset = TensorDataset(train_a, train_u)
    val_size = int(VAL_SPLIT * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # -- 6. Optimiser --
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()

    # -- 7. Training loop --
    train_loss_hist = []
    val_loss_hist = []

    print(
        f"Training for {MAX_EPOCHS} epochs "
        f"({train_size} train / {val_size} val samples)\n"
    )

    for epoch in range(1, MAX_EPOCHS + 1):
        # -- train --
        model.train()
        running = 0.0
        n_batches = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()

            # First-batch gradient diagnostic (epoch 1 only)
            if epoch == 1 and n_batches == 0:
                grad_norms = {}
                for name, p in model.named_parameters():
                    if p.grad is not None:
                        grad_norms[name] = p.grad.norm().item()
                total_grad = sum(grad_norms.values())
                print(f"\n  Gradient diagnostic (first batch):")
                print(f"    Total grad norm: {total_grad:.6f}")
                if total_grad < 1e-10:
                    print("    WARNING: Gradients are essentially zero!")
                # Show top 5 layers by gradient magnitude
                sorted_grads = sorted(grad_norms.items(), key=lambda x: x[1], reverse=True)
                for name, norm in sorted_grads[:5]:
                    print(f"    {name}: {norm:.6f}")
                print()

            optimizer.step()
            running += loss.item()
            n_batches += 1

        avg_train = running / n_batches
        train_loss_hist.append(avg_train)

        # -- validate & visualise --
        if epoch % VAL_EVERY == 0 or epoch == 1 or epoch == MAX_EPOCHS:
            model.eval()
            val_running = 0.0
            vn = 0
            with torch.no_grad():
                for x, y in val_loader:
                    p = model(x)
                    val_running += loss_fn(p, y).item()
                    vn += 1
            avg_val = val_running / max(vn, 1)
            val_loss_hist.append((epoch, avg_val))
            print(
                f"  epoch {epoch:3d}/{MAX_EPOCHS}  "
                f"train {avg_train:.6f}  val {avg_val:.6f}"
            )

            plot_predictions(model, val_loader, device, epoch)
            plot_spatial_fields(
                model, train_a, train_u,
                raw_perm_vis, raw_press_vis, coords_np,
                device, epoch,
                n_samples=N_VIS,
            )
        else:
            print(f"  epoch {epoch:3d}/{MAX_EPOCHS}  train {avg_train:.6f}")

    # -- 8. Final plots & sanity check --
    model.eval()
    with torch.no_grad():
        _p0 = model(train_a[0:1])
        _p1 = model(train_a[1:2])
    print(f"\n  Post-training sanity check:")
    print(f"    pred sample 0: mean={_p0.mean():.4f}, std={_p0.std():.4f}")
    print(f"    pred sample 1: mean={_p1.mean():.4f}, std={_p1.std():.4f}")
    print(f"    pred diff: {(_p0 - _p1).abs().mean():.6f}")

    plot_losses(train_loss_hist, val_loss_hist, save_path="darcy_losses.png")
    plot_spatial_fields(
        model, train_a, train_u,
        raw_perm_vis, raw_press_vis, coords_np,
        device, MAX_EPOCHS,
        n_samples=N_VIS,
        save_path="darcy_spatial_final.png",
    )
    print("\nDone.")


if __name__ == "__main__":
    main()