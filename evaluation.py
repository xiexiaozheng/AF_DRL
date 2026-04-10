"""
evaluation.py – Shared evaluation helpers for phase 1 and phase 2.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import sqrt
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from dataset import DEFAULT_PATCH_SIZE, group_focal_stacks, parse_txt
from models import ACTION_RANGE, AutofocusActorCritic
from trajectory_builder import build_state_from_record, _find_nearest_focus, _find_nearest_record


@dataclass
class RolloutTrace:
    scene_name: str
    patch_x: int
    patch_y: int
    gt_focus_index: int
    start_focus_index: int
    positions: List[int]
    actions: List[int]
    focus_hunting: bool
    hit_gt_step: Optional[int]



def compute_error_metrics(errors: Sequence[int]) -> Dict[str, float]:
    if not errors:
        return {"<=0": 0.0, "<=1": 0.0, "<=2": 0.0, "<=4": 0.0, "mae": 0.0, "rmse": 0.0}
    errors_np = np.asarray(errors, dtype=np.float32)
    return {
        "<=0": float((errors_np <= 0).mean()),
        "<=1": float((errors_np <= 1).mean()),
        "<=2": float((errors_np <= 2).mean()),
        "<=4": float((errors_np <= 4).mean()),
        "mae": float(errors_np.mean()),
        "rmse": float(np.sqrt(np.square(errors_np).mean())),
    }



def evaluate_phase1_model(
    model: AutofocusActorCritic,
    txt_path: str,
    data_root: str,
    device: torch.device,
    patch_size: int = DEFAULT_PATCH_SIZE,
    pe_dim: int = 16,
    raw_suffix: str = ".npy",
) -> Dict[str, float]:
    records = parse_txt(txt_path)
    errors: List[int] = []
    model.eval()
    for record in records:
        state = build_state_from_record(
            record,
            data_root=data_root,
            patch_size=patch_size,
            pe_dim=pe_dim,
            raw_suffix=raw_suffix,
        )
        state = {key: value.unsqueeze(0).to(device) for key, value in state.items()}
        with torch.no_grad():
            action_index = model.actor_logits(state).argmax(dim=-1)
        offset = int(action_index.item()) - ACTION_RANGE
        predicted_stop = int(np.clip(record.focus_index + offset, 0, ACTION_RANGE))
        errors.append(abs(predicted_stop - record.gt_focus_index))
    return compute_error_metrics(errors)



def rollout_actor_on_txt(
    model: AutofocusActorCritic,
    txt_path: str,
    data_root: str,
    device: torch.device,
    max_steps: int = 4,
    patch_size: int = DEFAULT_PATCH_SIZE,
    pe_dim: int = 16,
    raw_suffix: str = ".npy",
    deterministic: bool = True,
) -> List[RolloutTrace]:
    stacks = group_focal_stacks(parse_txt(txt_path))
    traces: List[RolloutTrace] = []
    model.eval()
    for stack in stacks.values():
        gt = stack[0].gt_focus_index
        for start_rec in stack:
            positions = [start_rec.focus_index]
            actions = []
            focus_hunting = False
            prev_direction = None
            hit_gt_step: Optional[int] = None
            current_focus = start_rec.focus_index
            for step_idx in range(max_steps):
                record = _find_nearest_record(stack, current_focus)
                state = build_state_from_record(
                    record,
                    data_root=data_root,
                    patch_size=patch_size,
                    pe_dim=pe_dim,
                    raw_suffix=raw_suffix,
                )
                state = {key: value.unsqueeze(0).to(device) for key, value in state.items()}
                with torch.no_grad():
                    logits = model.actor_logits(state)
                    if deterministic:
                        action_index = logits.argmax(dim=-1)
                    else:
                        action_index = torch.distributions.Categorical(logits=logits).sample()
                offset = int(action_index.item()) - ACTION_RANGE
                next_focus = int(np.clip(current_focus + offset, 0, ACTION_RANGE))
                next_focus = _find_nearest_focus(stack, next_focus)
                action = next_focus - current_focus
                actions.append(action)
                positions.append(next_focus)
                direction = 1 if action > 0 else (-1 if action < 0 else 0)
                if direction != 0 and prev_direction is not None and direction != prev_direction:
                    focus_hunting = True
                if direction != 0:
                    prev_direction = direction
                if hit_gt_step is None and next_focus == gt:
                    hit_gt_step = step_idx + 1
                current_focus = next_focus
            traces.append(
                RolloutTrace(
                    scene_name=start_rec.scene_name,
                    patch_x=start_rec.patch_x,
                    patch_y=start_rec.patch_y,
                    gt_focus_index=gt,
                    start_focus_index=start_rec.focus_index,
                    positions=positions,
                    actions=actions,
                    focus_hunting=focus_hunting,
                    hit_gt_step=hit_gt_step,
                )
            )
    return traces



def summarise_rollouts(traces: Sequence[RolloutTrace], max_steps: int) -> Dict[str, float]:
    errors = [abs(trace.positions[-1] - trace.gt_focus_index) for trace in traces]
    metrics = compute_error_metrics(errors)
    metrics["fh"] = float(np.mean([trace.focus_hunting for trace in traces])) if traces else 0.0
    metrics["avg_focus_speed"] = float(
        np.mean([trace.hit_gt_step if trace.hit_gt_step is not None else max_steps + 1 for trace in traces])
    ) if traces else 0.0
    return metrics



def save_rollout_traces(traces: Sequence[RolloutTrace], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump([asdict(trace) for trace in traces], handle, indent=2)
