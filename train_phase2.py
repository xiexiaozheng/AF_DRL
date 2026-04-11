"""
train_phase2.py – PPO-CLIP with expert trajectory regularization.

Phase 2 follows the paper's actor-critic training recipe: PPO-CLIP
optimizes the online policy while batches of offline expert trajectories
regularize the actor toward stable, non-hunting lens movements. The
environment reward also follows the paper's negative-distance plus focus
hunting penalty formulation.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from env import AutofocusEnv
from evaluation import rollout_actor_on_txt, summarise_rollouts
from models import AutofocusActorCritic
from trajectory_builder import (
    ExpertTrajectoryDataset,
    collate_expert_trajectories,
    load_trajectories_json,
)


@dataclass
class RolloutSample:
    state: Dict[str, torch.Tensor]
    action: int
    log_prob: float
    value: float
    reward: float
    done: bool


class RolloutBuffer:
    def __init__(self, gamma: float, gae_lambda: float) -> None:
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.samples: List[RolloutSample] = []

    def add(self, sample: RolloutSample) -> None:
        self.samples.append(sample)

    def clear(self) -> None:
        self.samples.clear()

    def __len__(self) -> int:
        return len(self.samples)

    def compute_returns_and_advantages(self, last_value: float) -> Tuple[torch.Tensor, torch.Tensor]:
        rewards = torch.tensor([sample.reward for sample in self.samples], dtype=torch.float32)
        values = torch.tensor([sample.value for sample in self.samples], dtype=torch.float32)
        dones = torch.tensor([sample.done for sample in self.samples], dtype=torch.float32)
        advantages = torch.zeros_like(rewards)
        rewards_list = rewards.tolist()
        values_list = values.tolist()
        dones_list = dones.tolist()
        last_gae = 0.0
        for index in reversed(range(len(self.samples))):
            next_value = last_value if index == len(self.samples) - 1 else values_list[index + 1]
            next_non_terminal = 1.0 - dones_list[index]
            delta = rewards_list[index] + self.gamma * next_value * next_non_terminal - values_list[index]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[index] = last_gae
        returns = advantages + values
        return returns, advantages



def fetch_expert_batch(loader: Optional[DataLoader], iterator):
    if loader is None:
        return None, iterator
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator



def ppo_update(
    model: AutofocusActorCritic,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    device: torch.device,
    clip_eps: float,
    entropy_coef: float,
    value_coef: float,
    expert_lambda: float,
    ppo_epochs: int,
    mini_batch_size: int,
    expert_loader: Optional[DataLoader],
    expert_iterator,
):
    with torch.no_grad():
        if buffer.samples:
            last_state = {key: value.unsqueeze(0).to(device) for key, value in buffer.samples[-1].state.items()}
            last_value = model.get_value(last_state).item()
        else:
            last_value = 0.0
    returns, advantages = buffer.compute_returns_and_advantages(last_value)

    all_images = torch.stack([sample.state["image"] for sample in buffer.samples])
    all_lens_pe = torch.stack([sample.state["lens_pe"] for sample in buffer.samples])
    all_roi_pe = torch.stack([sample.state["roi_pe"] for sample in buffer.samples])
    all_temperature = torch.stack([sample.state["temperature"] for sample in buffer.samples])
    all_actions = torch.tensor([sample.action for sample in buffer.samples], dtype=torch.long)
    all_old_log_probs = torch.tensor([sample.log_prob for sample in buffer.samples], dtype=torch.float32)

    metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "expert_loss": 0.0}
    indices = np.arange(len(buffer))
    updates = 0

    for _ in range(ppo_epochs):
        np.random.shuffle(indices)
        for start in range(0, len(buffer), mini_batch_size):
            mb_idx = indices[start:start + mini_batch_size]
            state_batch = {
                "image": all_images[mb_idx].to(device),
                "lens_pe": all_lens_pe[mb_idx].to(device),
                "roi_pe": all_roi_pe[mb_idx].to(device),
                "temperature": all_temperature[mb_idx].to(device),
            }
            actions = all_actions[mb_idx].to(device)
            old_log_probs = all_old_log_probs[mb_idx].to(device)
            mb_returns = returns[mb_idx].to(device)
            mb_advantages = advantages[mb_idx].to(device)
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            _, new_log_probs, entropy, values = model.get_action_and_value(state_batch, action=actions)
            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * mb_advantages
            surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, mb_returns)

            expert_loss = torch.tensor(0.0, device=device)
            expert_batch, expert_iterator = fetch_expert_batch(expert_loader, expert_iterator)
            if expert_batch is not None:
                expert_state, expert_actions = expert_batch
                expert_state = {key: value.to(device) for key, value in expert_state.items()}
                expert_actions = expert_actions.to(device)
                expert_logits = model.actor_logits(expert_state)
                expert_loss = F.cross_entropy(expert_logits, expert_actions)

            total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean() + expert_lambda * expert_loss
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            metrics["policy_loss"] += policy_loss.item()
            metrics["value_loss"] += value_loss.item()
            metrics["entropy"] += entropy.mean().item()
            metrics["expert_loss"] += expert_loss.item()
            updates += 1

    for key in metrics:
        metrics[key] /= max(updates, 1)
    return metrics, expert_iterator



def collect_rollout(
    model: AutofocusActorCritic,
    env: AutofocusEnv,
    device: torch.device,
    rollout_steps: int,
    buffer: RolloutBuffer,
) -> Dict[str, float]:
    model.eval()
    state, _ = env.reset()
    episode_rewards: List[float] = []
    current_episode_reward = 0.0
    for _ in range(rollout_steps):
        state_batch = {key: value.unsqueeze(0).to(device) for key, value in state.items()}
        with torch.no_grad():
            action, log_prob, _, value = model.get_action_and_value(state_batch, deterministic=False)
        next_state, reward, terminated, truncated, _ = env.step(int(action.item()))
        buffer.add(
            RolloutSample(
                state=state,
                action=int(action.item()),
                log_prob=float(log_prob.item()),
                value=float(value.item()),
                reward=float(reward),
                done=bool(terminated or truncated),
            )
        )
        current_episode_reward += float(reward)
        if terminated or truncated:
            episode_rewards.append(current_episode_reward)
            current_episode_reward = 0.0
            state, _ = env.reset()
        else:
            state = next_state
    return {"avg_episode_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0, "episodes": len(episode_rewards)}



def save_checkpoint(path, model, optimizer, update_step, best_mae, args):
    torch.save(
        {
            "update": update_step,
            "best_mae": best_mae,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
        },
        path,
    )



def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-2 PPO-CLIP training with expert trajectory regularization.")
    parser.add_argument("--txt_path", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--expert_json", required=True)
    parser.add_argument("--val_txt_path", default=None)
    parser.add_argument("--pretrained", default=None, help="Phase-1 checkpoint used to initialize the actor.")
    parser.add_argument("--resume", default=None, help="Resume a phase-2 checkpoint.")
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--rollout_steps", type=int, default=2048)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--mini_batch_size", type=int, default=32)
    parser.add_argument("--expert_batch_trajectories", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--expert_lambda", type=float, default=1e-3)
    parser.add_argument("--fh_penalty", type=float, default=-1.5)
    parser.add_argument("--max_env_steps", type=int, default=4)
    parser.add_argument("--pe_dim", type=int, default=16)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--raw_suffix", type=str, default=".npy")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--output_dir", type=str, default="checkpoints/phase2")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--imagenet_pretrained", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    model = AutofocusActorCritic(
        pe_dim=args.pe_dim,
        freeze_backbone=args.freeze_backbone,
        imagenet_pretrained=args.imagenet_pretrained,
    ).to(device)
    optimizer = torch.optim.Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
    )

    start_update = 0
    best_mae = float("inf")
    if args.resume:
        checkpoint = model.load_training_checkpoint(args.resume)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_update = int(checkpoint.get("update", 0))
        best_mae = float(checkpoint.get("best_mae", best_mae))
    elif args.pretrained:
        model.load_phase1_weights(args.pretrained, freeze_backbone=args.freeze_backbone, load_actor_head=True)

    env = AutofocusEnv(
        txt_path=args.txt_path,
        data_root=args.data_root,
        max_steps=args.max_env_steps,
        patch_size=args.patch_size,
        pe_dim=args.pe_dim,
        raw_suffix=args.raw_suffix,
        focus_hunting_penalty=args.fh_penalty,
    )

    expert_trajectories = load_trajectories_json(args.expert_json)
    expert_dataset = ExpertTrajectoryDataset(
        expert_trajectories,
        data_root=args.data_root,
        patch_size=args.patch_size,
        pe_dim=args.pe_dim,
        raw_suffix=args.raw_suffix,
    )
    expert_loader = DataLoader(
        expert_dataset,
        batch_size=args.expert_batch_trajectories,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_expert_trajectories,
    )
    expert_iterator = iter(expert_loader)
    print(f"loaded {len(expert_trajectories)} expert trajectories")

    for update_step in range(start_update + 1, args.updates + 1):
        buffer = RolloutBuffer(gamma=args.gamma, gae_lambda=args.gae_lambda)
        rollout_stats = collect_rollout(model, env, device, args.rollout_steps, buffer)
        update_metrics, expert_iterator = ppo_update(
            model,
            optimizer,
            buffer,
            device,
            clip_eps=args.clip_eps,
            entropy_coef=args.entropy_coef,
            value_coef=args.value_coef,
            expert_lambda=args.expert_lambda,
            ppo_epochs=args.ppo_epochs,
            mini_batch_size=args.mini_batch_size,
            expert_loader=expert_loader,
            expert_iterator=expert_iterator,
        )
        log = (
            f"update={update_step}/{args.updates} reward={rollout_stats['avg_episode_reward']:.4f} "
            f"policy_loss={update_metrics['policy_loss']:.6f} value_loss={update_metrics['value_loss']:.6f} "
            f"expert_loss={update_metrics['expert_loss']:.6f} entropy={update_metrics['entropy']:.6f}"
        )
        print(log)

        if args.val_txt_path and update_step % args.eval_interval == 0:
            traces = rollout_actor_on_txt(
                model,
                txt_path=args.val_txt_path,
                data_root=args.data_root,
                device=device,
                max_steps=args.max_env_steps,
                patch_size=args.patch_size,
                pe_dim=args.pe_dim,
                raw_suffix=args.raw_suffix,
                deterministic=True,
            )
            metrics = summarise_rollouts(traces, max_steps=args.max_env_steps)
            metric_log = " ".join([f"{key}={value:.4f}" for key, value in metrics.items()])
            print(f"[eval] update={update_step} {metric_log}")
            if metrics["mae"] < best_mae:
                best_mae = metrics["mae"]
                save_checkpoint(os.path.join(args.output_dir, "best_model.pth"), model, optimizer, update_step, best_mae, args)
                print(f"[eval] saved new best checkpoint with mae={best_mae:.4f}")

        if update_step % args.save_interval == 0 or update_step == args.updates:
            save_checkpoint(
                os.path.join(args.output_dir, f"checkpoint_update{update_step}.pth"),
                model,
                optimizer,
                update_step,
                best_mae,
                args,
            )

    save_checkpoint(os.path.join(args.output_dir, "last_model.pth"), model, optimizer, args.updates, best_mae, args)
    print(f"training complete best_mae={best_mae:.4f}")


if __name__ == "__main__":
    main()
