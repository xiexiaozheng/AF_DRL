"""
train_phase1.py – Phase 1: Ordinal Regression Pretraining
===========================================================

Pre-trains the Actor network using ordinal regression loss on the
dual-pixel autofocus dataset.  The target distribution for each sample
is a soft label centred on the ground-truth *relative movement*
(gt_focus_index − focus_index), implemented as:

    y_k = exp(-(r_f - r_k)^2 / T) / Σ_i exp(-(r_f - r_i)^2 / T)

where r_f is the rank of the ground-truth relative movement, r_k is
the rank of position k, and T is a temperature hyper-parameter.

The model outputs a distribution over the ``NUM_FOCUS_POSITIONS``
classes (absolute positions) during Phase 1.  After pre-training, the
backbone weights are transferred to the RL actor (see ``train_phase2.py``).

Usage
-----
    python train_phase1.py \\
        --txt_path data/train.txt \\
        --data_root data/raw/ \\
        --epochs 50 \\
        --batch_size 64 \\
        --lr 1e-3 \\
        --output_dir checkpoints/phase1
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import AutofocusDataset, NUM_FOCUS_POSITIONS
from models import AutofocusActorCritic


# ---------------------------------------------------------------------------
# Ordinal Regression Target Distribution
# ---------------------------------------------------------------------------

def ordinal_target(
    gt_focus_index: torch.Tensor,
    focus_index: torch.Tensor,
    num_positions: int = NUM_FOCUS_POSITIONS,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Build the soft ordinal-regression target distribution.

    During pretraining, the target rank is the *absolute* GT focus position
    (``gt_focus_index``), so the model learns to predict absolute positions.
    When transitioning to RL (Phase 2), the output head is changed to
    predict relative movements instead.

    Parameters
    ----------
    gt_focus_index : (B,) int tensor
    focus_index    : (B,) int tensor
    num_positions  : int
    temperature    : float  (T in the formula)

    Returns
    -------
    target : (B, num_positions) float tensor, sums to 1 per row.
    """
    B = gt_focus_index.shape[0]
    device = gt_focus_index.device

    # Ranks: 0..num_positions-1
    ranks = torch.arange(num_positions, device=device, dtype=torch.float32)  # (P,)

    # Ground-truth rank — here we directly use gt_focus_index as the rank
    # (absolute position label for pretrain phase)
    rf = gt_focus_index.float().unsqueeze(1)  # (B, 1)

    # Numerator: exp( -(rf - rk)^2 / T )
    diff_sq = (rf - ranks.unsqueeze(0)) ** 2  # (B, P)
    logits = -diff_sq / temperature            # (B, P)

    target = F.softmax(logits, dim=-1)
    return target


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def ordinal_regression_loss(
    pred_logits: torch.Tensor,
    gt_focus_index: torch.Tensor,
    focus_index: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Cross-entropy between predicted logits and soft ordinal target.

    Parameters
    ----------
    pred_logits    : (B, NUM_FOCUS_POSITIONS) raw logits from the model.
    gt_focus_index : (B,) int tensor.
    focus_index    : (B,) int tensor (current lens position).
    temperature    : float

    Returns
    -------
    Scalar loss (mean over batch).
    """
    target = ordinal_target(gt_focus_index, focus_index, temperature=temperature)
    log_probs = F.log_softmax(pred_logits, dim=-1)
    loss = -(target * log_probs).sum(dim=-1).mean()
    return loss


# ---------------------------------------------------------------------------
# Collate helper
# ---------------------------------------------------------------------------

def collate_fn(batch):
    """Custom collate for ``AutofocusDataset`` samples."""
    states, fis, gts, metas = zip(*batch)
    images = torch.stack([s["image"] for s in states])
    lens_pe = torch.stack([s["lens_pe"] for s in states])
    roi_pe = torch.stack([s["roi_pe"] for s in states])
    temperature = torch.stack([s["temperature"] for s in states])
    fis = torch.tensor(fis, dtype=torch.long)
    gts = torch.tensor(gts, dtype=torch.long)
    state_batch = {
        "image": images,
        "lens_pe": lens_pe,
        "roi_pe": roi_pe,
        "temperature": temperature,
    }
    return state_batch, fis, gts, metas


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: AutofocusActorCritic,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    temperature: float = 2.0,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for state_batch, focus_index, gt_focus_index, _meta in loader:
        # Move to device
        for k in state_batch:
            state_batch[k] = state_batch[k].to(device)
        focus_index = focus_index.to(device)
        gt_focus_index = gt_focus_index.to(device)

        out = model(state_batch)
        loss = ordinal_regression_loss(
            out["logits"], gt_focus_index, focus_index, temperature=temperature,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: AutofocusActorCritic,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Compute MAE and accuracy metrics on the evaluation set."""
    model.eval()
    total_mae = 0.0
    total_count = 0
    hits = {0: 0, 1: 0, 2: 0, 4: 0}

    for state_batch, focus_index, gt_focus_index, _meta in loader:
        for k in state_batch:
            state_batch[k] = state_batch[k].to(device)
        gt_focus_index = gt_focus_index.to(device)

        out = model(state_batch)
        pred = out["probs"].argmax(dim=-1)  # predicted absolute position
        error = (pred - gt_focus_index).abs()
        total_mae += error.sum().item()
        total_count += error.numel()
        for thr in hits:
            hits[thr] += (error <= thr).sum().item()

    mae = total_mae / max(total_count, 1)
    acc = {f"<={k}": hits[k] / max(total_count, 1) for k in hits}
    return {"mae": mae, **acc}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 1: Ordinal Regression Pretraining")
    parser.add_argument("--txt_path", type=str, required=True, help="Annotation TXT path.")
    parser.add_argument("--val_txt_path", type=str, default=None, help="Validation TXT path.")
    parser.add_argument("--data_root", type=str, required=True, help="RAW image root dir.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=2.0, help="Ordinal regression T.")
    parser.add_argument("--pe_dim", type=int, default=16)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="checkpoints/phase1")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Dataset & DataLoader
    train_ds = AutofocusDataset(
        args.txt_path, args.data_root,
        patch_size=args.patch_size, pe_dim=args.pe_dim,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    val_loader = None
    if args.val_txt_path:
        val_ds = AutofocusDataset(
            args.val_txt_path, args.data_root,
            patch_size=args.patch_size, pe_dim=args.pe_dim,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
        )

    # Model
    model = AutofocusActorCritic(
        pe_dim=args.pe_dim, image_channels=2, pretrain=True,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_mae = float("inf")
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(
            model, train_loader, optimizer, device,
            temperature=args.temperature,
        )
        scheduler.step()

        log_msg = f"Epoch {epoch}/{args.epochs}  loss={loss:.4f}"

        if val_loader is not None:
            metrics = evaluate(model, val_loader, device)
            log_msg += f"  MAE={metrics['mae']:.3f}"
            for k, v in metrics.items():
                if k != "mae":
                    log_msg += f"  {k}={v:.3f}"
            if metrics["mae"] < best_mae:
                best_mae = metrics["mae"]
                ckpt_path = os.path.join(args.output_dir, "best_model.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "mae": best_mae,
                }, ckpt_path)
                log_msg += "  [BEST]"

        print(log_msg)

        # Periodic save
        if epoch % 10 == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.output_dir, f"model_epoch{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, ckpt_path)

    print(f"Training complete.  Best MAE = {best_mae:.3f}")


if __name__ == "__main__":
    main()
