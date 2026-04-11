"""
train_phase1.py – Paper-faithful phase-1 actor pretraining.

Phase 1 trains the actor with ordinal regression over relative lens
movement bins, matching the paper's reformulation from absolute focus
position prediction to relative movement prediction. The resulting actor
checkpoint is used directly to initialize phase-2 PPO training.
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import AutofocusDataset
from evaluation import evaluate_phase1_model
from models import ACTION_DIM, ACTION_RANGE, AutofocusActorCritic



def movement_target_indices(gt_focus_index: torch.Tensor, focus_index: torch.Tensor) -> torch.Tensor:
    offsets = gt_focus_index - focus_index
    return (offsets + ACTION_RANGE).clamp(0, ACTION_DIM - 1)



def ordinal_target_distribution(target_indices: torch.Tensor, temperature: float) -> torch.Tensor:
    ranks = torch.arange(ACTION_DIM, device=target_indices.device, dtype=torch.float32)
    target_ranks = target_indices.float().unsqueeze(1)
    logits = -((target_ranks - ranks.unsqueeze(0)) ** 2) / temperature
    return F.softmax(logits, dim=-1)



def ordinal_regression_loss(
    action_logits: torch.Tensor,
    gt_focus_index: torch.Tensor,
    focus_index: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    target_indices = movement_target_indices(gt_focus_index, focus_index)
    soft_target = ordinal_target_distribution(target_indices, temperature=temperature)
    log_probs = F.log_softmax(action_logits, dim=-1)
    return -(soft_target * log_probs).sum(dim=-1).mean()



def collate_fn(batch):
    states, focus_index, gt_focus_index, meta = zip(*batch)
    return {
        "image": torch.stack([state["image"] for state in states]),
        "lens_pe": torch.stack([state["lens_pe"] for state in states]),
        "roi_pe": torch.stack([state["roi_pe"] for state in states]),
        "temperature": torch.stack([state["temperature"] for state in states]),
    }, torch.tensor(focus_index, dtype=torch.long), torch.tensor(gt_focus_index, dtype=torch.long), meta



def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch



def save_checkpoint(path, model, optimizer, scheduler, iteration, best_mae, args) -> None:
    torch.save(
        {
            "iteration": iteration,
            "best_mae": best_mae,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "args": vars(args),
        },
        path,
    )



def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-1 pretraining with relative-movement ordinal regression.")
    parser.add_argument("--txt_path", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--val_txt_path", default=None)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--pe_dim", type=int, default=16)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--raw_suffix", type=str, default=".npy")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="checkpoints/phase1")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--imagenet_pretrained", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.iterations <= 0:
        raise ValueError(f"--iterations must be positive (got {args.iterations}).")

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    train_dataset = AutofocusDataset(
        args.txt_path,
        args.data_root,
        patch_size=args.patch_size,
        pe_dim=args.pe_dim,
        raw_suffix=args.raw_suffix,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    train_iter = infinite_loader(train_loader)

    model = AutofocusActorCritic(pe_dim=args.pe_dim, imagenet_pretrained=args.imagenet_pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.iterations, 1))

    start_iteration = 0
    best_mae = float("inf")
    if args.resume:
        checkpoint = model.load_training_checkpoint(args.resume)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_iteration = int(checkpoint.get("iteration", 0))
        best_mae = float(checkpoint.get("best_mae", best_mae))

    for iteration in range(start_iteration + 1, args.iterations + 1):
        state_batch, focus_index, gt_focus_index, _ = next(train_iter)
        state_batch = {key: value.to(device, non_blocking=True) for key, value in state_batch.items()}
        focus_index = focus_index.to(device)
        gt_focus_index = gt_focus_index.to(device)

        logits = model.actor_logits(state_batch)
        loss = ordinal_regression_loss(logits, gt_focus_index, focus_index, temperature=args.temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if iteration == 1 or iteration % 50 == 0:
            print(f"iter={iteration}/{args.iterations} loss={loss.item():.6f} lr={scheduler.get_last_lr()[0]:.6e}")

        if args.val_txt_path and (iteration % args.eval_interval == 0 or iteration == args.iterations):
            metrics = evaluate_phase1_model(
                model,
                txt_path=args.val_txt_path,
                data_root=args.data_root,
                device=device,
                patch_size=args.patch_size,
                pe_dim=args.pe_dim,
                raw_suffix=args.raw_suffix,
            )
            metric_log = " ".join([f"{key}={value:.4f}" for key, value in metrics.items()])
            print(f"[eval] iter={iteration} {metric_log}")
            if metrics["mae"] < best_mae:
                best_mae = metrics["mae"]
                save_checkpoint(os.path.join(args.output_dir, "best_model.pth"), model, optimizer, scheduler, iteration, best_mae, args)
                print(f"[eval] saved new best checkpoint with mae={best_mae:.4f}")

        if iteration % args.save_interval == 0 or iteration == args.iterations:
            save_checkpoint(
                os.path.join(args.output_dir, f"checkpoint_iter{iteration}.pth"),
                model,
                optimizer,
                scheduler,
                iteration,
                best_mae,
                args,
            )

    save_checkpoint(os.path.join(args.output_dir, "last_model.pth"), model, optimizer, scheduler, args.iterations, best_mae, args)
    print(f"training complete best_mae={best_mae:.4f}")


if __name__ == "__main__":
    main()
