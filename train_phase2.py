"""
train_phase2.py – Phase 2: PPO + Expert Trajectory Regularization
===================================================================

Trains the Actor-Critic network using Proximal Policy Optimization (PPO)
with an additional expert trajectory regularization term.

Key components
--------------
* **PPO clipped surrogate loss** for the actor.
* **Expert regularization** ``L_expert``: cross-entropy between actor's
  output distribution and the expert action (from Algorithm 1/2/3).
* **Focus-hunting penalty** built into the environment reward.
* **Critic loss** (MSE on value function).

Total actor loss:
    L(θ) = L_clip(θ) + λ · L_expert(θ)

Usage
-----
    python train_phase2.py \\
        --txt_path data/train.txt \\
        --data_root data/raw/ \\
        --pretrained checkpoints/phase1/best_model.pth \\
        --expert_json expert_trajectories.json \\
        --epochs 100 \\
        --output_dir checkpoints/phase2
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import NUM_FOCUS_POSITIONS, parse_txt
from env import AutofocusEnv
from models import ACTION_DIM, ACTION_RANGE, AutofocusActorCritic
from trajectory_builder import (
    ExpertTrajectory,
    StepData,
    build_expert_trajectories,
    trajectories_to_flat,
    FlatStep,
)


# ---------------------------------------------------------------------------
# PPO rollout buffer
# ---------------------------------------------------------------------------

@dataclass
class RolloutSample:
    """One transition stored in the rollout buffer."""
    state: Dict[str, torch.Tensor]
    action: int
    log_prob: float
    value: float
    reward: float
    done: bool
    # Expert action (if available) for regularisation
    expert_action: Optional[int] = None


class RolloutBuffer:
    """Stores a fixed number of transitions for PPO updates."""

    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.samples: List[RolloutSample] = []

    def add(self, sample: RolloutSample) -> None:
        self.samples.append(sample)

    def clear(self) -> None:
        self.samples = []

    def __len__(self) -> int:
        return len(self.samples)

    def compute_returns_and_advantages(
        self, last_value: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and discounted returns.

        Returns
        -------
        returns    : (N,) tensor
        advantages : (N,) tensor
        """
        n = len(self.samples)
        rewards = torch.zeros(n)
        values = torch.zeros(n)
        dones = torch.zeros(n)

        for i, s in enumerate(self.samples):
            rewards[i] = s.reward
            values[i] = s.value
            dones[i] = float(s.done)

        # GAE
        advantages = torch.zeros(n)
        last_gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae

        returns = advantages + values
        return returns, advantages


# ---------------------------------------------------------------------------
# Expert regularisation loss
# ---------------------------------------------------------------------------

def expert_regularisation_loss(
    action_logits: torch.Tensor,
    expert_actions: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy between actor distribution and expert actions.

    Parameters
    ----------
    action_logits : (B, ACTION_DIM) raw logits from actor head.
    expert_actions : (B,) expert action indices ∈ [0, ACTION_DIM).

    Returns
    -------
    Scalar loss.
    """
    return F.cross_entropy(action_logits, expert_actions)


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def ppo_update(
    model: AutofocusActorCritic,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    device: torch.device,
    clip_eps: float = 0.2,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    expert_lambda: float = 1e-3,
    ppo_epochs: int = 4,
    mini_batch_size: int = 64,
) -> Dict[str, float]:
    """Run PPO update using the collected rollout buffer.

    Returns a dict of loss metrics.
    """
    model.train()

    # Compute returns & advantages
    with torch.no_grad():
        if buffer.samples:
            last_state = buffer.samples[-1].state
            last_state_dev = {k: v.unsqueeze(0).to(device) for k, v in last_state.items()}
            last_value = model.get_value(last_state_dev).item()
        else:
            last_value = 0.0
    returns, advantages = buffer.compute_returns_and_advantages(last_value)

    # Flatten buffer into tensors
    n = len(buffer)
    all_images = torch.stack([s.state["image"] for s in buffer.samples])
    all_lens_pe = torch.stack([s.state["lens_pe"] for s in buffer.samples])
    all_roi_pe = torch.stack([s.state["roi_pe"] for s in buffer.samples])
    all_temp = torch.stack([s.state["temperature"] for s in buffer.samples])
    all_actions = torch.tensor([s.action for s in buffer.samples], dtype=torch.long)
    all_old_log_probs = torch.tensor([s.log_prob for s in buffer.samples], dtype=torch.float32)
    all_expert_actions = torch.tensor(
        [s.expert_action if s.expert_action is not None else 0
         for s in buffer.samples],
        dtype=torch.long,
    )
    has_expert = torch.tensor(
        [s.expert_action is not None for s in buffer.samples],
        dtype=torch.bool,
    )

    indices = np.arange(n)
    metrics = {"policy_loss": 0, "value_loss": 0, "expert_loss": 0, "entropy": 0}
    update_count = 0

    for _ in range(ppo_epochs):
        np.random.shuffle(indices)
        for start in range(0, n, mini_batch_size):
            end = min(start + mini_batch_size, n)
            mb_idx = indices[start:end]

            state_batch = {
                "image": all_images[mb_idx].to(device),
                "lens_pe": all_lens_pe[mb_idx].to(device),
                "roi_pe": all_roi_pe[mb_idx].to(device),
                "temperature": all_temp[mb_idx].to(device),
            }
            mb_actions = all_actions[mb_idx].to(device)
            mb_old_log_probs = all_old_log_probs[mb_idx].to(device)
            mb_returns = returns[mb_idx].to(device)
            mb_advantages = advantages[mb_idx].to(device)
            mb_expert = all_expert_actions[mb_idx].to(device)
            mb_has_expert = has_expert[mb_idx].to(device)

            # Normalise advantages
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Forward
            _, new_log_probs, entropy, values = model.get_action_and_value(
                state_batch, action=mb_actions,
            )

            # Clipped surrogate loss
            ratio = (new_log_probs - mb_old_log_probs).exp()
            surr1 = ratio * mb_advantages
            surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(values, mb_returns)

            # Expert regularisation (only where expert actions exist)
            out = model.forward(state_batch)
            expert_loss = torch.tensor(0.0, device=device)
            if mb_has_expert.any():
                expert_logits = out["action_logits"][mb_has_expert]
                expert_targets = mb_expert[mb_has_expert]
                expert_loss = expert_regularisation_loss(expert_logits, expert_targets)

            # Total loss
            total_loss = (
                policy_loss
                + value_coef * value_loss
                - entropy_coef * entropy.mean()
                + expert_lambda * expert_loss
            )

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            metrics["policy_loss"] += policy_loss.item()
            metrics["value_loss"] += value_loss.item()
            metrics["expert_loss"] += expert_loss.item()
            metrics["entropy"] += entropy.mean().item()
            update_count += 1

    for k in metrics:
        metrics[k] /= max(update_count, 1)
    return metrics


# ---------------------------------------------------------------------------
# Expert dataset helper
# ---------------------------------------------------------------------------

def load_expert_dataset(
    json_path: Optional[str] = None,
    txt_path: Optional[str] = None,
    n_steps: int = 4,
    m: int = 5,
) -> List[FlatStep]:
    """Load or build expert trajectory data."""
    if json_path and os.path.exists(json_path):
        with open(json_path) as f:
            raw = json.load(f)
        flat: List[FlatStep] = []
        for traj_d in raw:
            for t, step_d in enumerate(traj_d["steps"]):
                flat.append(FlatStep(
                    focus_index=step_d["focus_index"],
                    gt_focus_index=step_d["gt_focus_index"],
                    expert_action=step_d["action"],
                    scene_name=traj_d["scene_name"],
                    patch_x=traj_d["patch_x"],
                    patch_y=traj_d["patch_y"],
                    temperature=step_d.get("temperature", 0.0),
                    left_raw_prefix=step_d.get("left_raw_prefix", ""),
                    right_raw_prefix=step_d.get("right_raw_prefix", ""),
                    step_in_traj=t,
                    algo_id=traj_d.get("algo_id", 0),
                ))
        return flat
    elif txt_path:
        trajs = build_expert_trajectories(txt_path, n_steps=n_steps, m=m)
        return trajectories_to_flat(trajs)
    else:
        return []


def sample_expert_action(
    expert_data: List[FlatStep],
    scene_name: str,
    patch_x: int,
    patch_y: int,
    focus_index: int,
    step_in_traj: int,
) -> Optional[int]:
    """Look up the expert action for the given state, if available.

    Returns the expert action offset converted to action index, or None.
    """
    for entry in expert_data:
        if (
            entry.scene_name == scene_name
            and entry.patch_x == patch_x
            and entry.patch_y == patch_y
            and abs(entry.focus_index - focus_index) <= 1
            and entry.step_in_traj == step_in_traj
        ):
            # Convert expert_action (offset) to action index
            action_idx = entry.expert_action + ACTION_RANGE
            action_idx = max(0, min(action_idx, ACTION_DIM - 1))
            return action_idx
    return None


# ---------------------------------------------------------------------------
# Collect rollout
# ---------------------------------------------------------------------------

def collect_rollout(
    model: AutofocusActorCritic,
    env: AutofocusEnv,
    device: torch.device,
    buffer: RolloutBuffer,
    n_steps: int,
    expert_data: List[FlatStep],
) -> Dict[str, float]:
    """Collect ``n_steps`` transitions by interacting with the environment.

    Returns episode statistics.
    """
    model.eval()
    state, info = env.reset()
    episode_reward = 0.0
    episode_count = 0
    total_reward = 0.0

    for _ in range(n_steps):
        state_dev = {k: v.unsqueeze(0).to(device) for k, v in state.items()}
        with torch.no_grad():
            action, log_prob, _, value = model.get_action_and_value(state_dev)

        action_int = action.item()
        next_state, reward, terminated, truncated, step_info = env.step(action_int)

        # Look up expert action
        stack_key = env._current_stack_key
        expert_act = None
        if stack_key and expert_data:
            expert_act = sample_expert_action(
                expert_data,
                stack_key[0], stack_key[1], stack_key[2],
                step_info.get("focus_index", 0),
                step_info.get("step", 0),
            )

        buffer.add(RolloutSample(
            state=state,
            action=action_int,
            log_prob=log_prob.item(),
            value=value.item(),
            reward=reward,
            done=terminated or truncated,
            expert_action=expert_act,
        ))

        episode_reward += reward
        if terminated or truncated:
            state, info = env.reset()
            total_reward += episode_reward
            episode_reward = 0.0
            episode_count += 1
        else:
            state = next_state

    avg_reward = total_reward / max(episode_count, 1)
    return {"avg_episode_reward": avg_reward, "episodes": episode_count}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 2: PPO + Expert Regularization")
    parser.add_argument("--txt_path", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Phase-1 checkpoint to load.")
    parser.add_argument("--expert_json", type=str, default=None,
                        help="Pre-built expert trajectory JSON.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--rollout_steps", type=int, default=2048,
                        help="Steps per rollout collection.")
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--mini_batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--expert_lambda", type=float, default=1e-3,
                        help="Weight for expert regularisation loss.")
    parser.add_argument("--fh_penalty", type=float, default=-1.5,
                        help="Focus-hunting penalty in the environment.")
    parser.add_argument("--max_env_steps", type=int, default=7)
    parser.add_argument("--n_traj_steps", type=int, default=4,
                        help="Trajectory length for expert generation.")
    parser.add_argument("--pe_dim", type=int, default=16)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze CNN backbone during RL training.")
    parser.add_argument("--output_dir", type=str, default="checkpoints/phase2")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ---- Model ----
    model = AutofocusActorCritic(
        pe_dim=args.pe_dim, image_channels=2,
        pretrain=False, freeze_backbone=args.freeze_backbone,
    ).to(device)

    if args.pretrained:
        model.load_pretrained_weights(args.pretrained, freeze_backbone=args.freeze_backbone)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )

    # ---- Environment ----
    env = AutofocusEnv(
        txt_path=args.txt_path,
        data_root=args.data_root,
        max_steps=args.max_env_steps,
        patch_size=args.patch_size,
        pe_dim=args.pe_dim,
        focus_hunting_penalty=args.fh_penalty,
    )

    # ---- Expert data ----
    expert_data = load_expert_dataset(
        json_path=args.expert_json,
        txt_path=args.txt_path,
        n_steps=args.n_traj_steps,
    )
    print(f"Loaded {len(expert_data)} expert step records.")

    # ---- Training loop ----
    best_reward = -float("inf")
    for epoch in range(1, args.epochs + 1):
        buffer = RolloutBuffer(gamma=args.gamma, gae_lambda=args.gae_lambda)

        # Collect rollout
        rollout_stats = collect_rollout(
            model, env, device, buffer,
            n_steps=args.rollout_steps,
            expert_data=expert_data,
        )

        # PPO update
        update_metrics = ppo_update(
            model, optimizer, buffer, device,
            clip_eps=args.clip_eps,
            entropy_coef=args.entropy_coef,
            value_coef=args.value_coef,
            expert_lambda=args.expert_lambda,
            ppo_epochs=args.ppo_epochs,
            mini_batch_size=args.mini_batch_size,
        )

        avg_reward = rollout_stats["avg_episode_reward"]
        log = (
            f"Epoch {epoch}/{args.epochs}  "
            f"reward={avg_reward:.3f}  "
            f"policy_loss={update_metrics['policy_loss']:.4f}  "
            f"value_loss={update_metrics['value_loss']:.4f}  "
            f"expert_loss={update_metrics['expert_loss']:.4f}  "
            f"entropy={update_metrics['entropy']:.4f}"
        )
        print(log)

        if avg_reward > best_reward:
            best_reward = avg_reward
            ckpt = os.path.join(args.output_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "reward": best_reward,
            }, ckpt)
            print(f"  [BEST] saved to {ckpt}")

        if epoch % 10 == 0 or epoch == args.epochs:
            ckpt = os.path.join(args.output_dir, f"model_epoch{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, ckpt)

    print(f"Training complete.  Best reward = {best_reward:.3f}")


if __name__ == "__main__":
    main()
