"""
trajectory_builder.py – Exact expert trajectory generation pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset import (
    DEFAULT_PATCH_SIZE,
    AutofocusRecord,
    NUM_FOCUS_POSITIONS,
    _ensure_2d,
    crop_patch,
    group_focal_stacks,
    lens_position_encoding,
    load_raw_image,
    normalise_patch,
    parse_txt,
    roi_position_encoding,
)
from models import ACTION_DIM, ACTION_RANGE, AutofocusActorCritic


@dataclass
class StepData:
    focus_index: int
    action: int
    gt_focus_index: int
    scene_name: str
    patch_x: int
    patch_y: int
    temperature: float
    left_raw_prefix: str
    right_raw_prefix: str


@dataclass
class ExpertTrajectory:
    scene_name: str
    patch_x: int
    patch_y: int
    gt_focus_index: int
    steps: List[StepData] = field(default_factory=list)
    algo_id: int = 0
    record_bank: Dict[int, StepData] = field(default_factory=dict, repr=False, compare=False)

    @property
    def positions(self) -> List[int]:
        return [step.focus_index for step in self.steps]



def _find_nearest_record(stack: List[AutofocusRecord], target_focus: int) -> AutofocusRecord:
    return min(stack, key=lambda rec: abs(rec.focus_index - target_focus))



def _find_nearest_focus(stack: List[AutofocusRecord], target_focus: int) -> int:
    return _find_nearest_record(stack, target_focus).focus_index



def build_state_from_record(
    record: AutofocusRecord,
    data_root: str,
    patch_size: int = DEFAULT_PATCH_SIZE,
    pe_dim: int = 16,
    raw_suffix: str = ".npy",
) -> Dict[str, torch.Tensor]:
    left_path = os.path.join(data_root, record.left_raw_prefix + raw_suffix)
    right_path = os.path.join(data_root, record.right_raw_prefix + raw_suffix)
    left = _ensure_2d(normalise_patch(crop_patch(load_raw_image(left_path), record.patch_x, record.patch_y, patch_size)))
    right = _ensure_2d(normalise_patch(crop_patch(load_raw_image(right_path), record.patch_x, record.patch_y, patch_size)))
    image = torch.from_numpy(np.stack([left, right], axis=0)).float()
    lens_pe = torch.from_numpy(lens_position_encoding(record.focus_index, embed_dim=pe_dim)).float()
    roi_pe = torch.from_numpy(roi_position_encoding(record.patch_x, record.patch_y, embed_dim=pe_dim)).float()
    temperature = torch.tensor([record.temperature], dtype=torch.float32)
    return {
        "image": image,
        "lens_pe": lens_pe,
        "roi_pe": roi_pe,
        "temperature": temperature,
    }



def rollout_policy_trajectory(
    model: AutofocusActorCritic,
    stack: List[AutofocusRecord],
    start_rec: AutofocusRecord,
    data_root: str,
    max_steps: int,
    device: torch.device,
    patch_size: int = DEFAULT_PATCH_SIZE,
    pe_dim: int = 16,
    raw_suffix: str = ".npy",
    deterministic: bool = True,
) -> ExpertTrajectory:
    gt_focus = start_rec.gt_focus_index
    current_focus = start_rec.focus_index
    trajectory = ExpertTrajectory(
        scene_name=start_rec.scene_name,
        patch_x=start_rec.patch_x,
        patch_y=start_rec.patch_y,
        gt_focus_index=gt_focus,
        algo_id=0,
        record_bank={
            rec.focus_index: StepData(
                focus_index=rec.focus_index,
                action=0,
                gt_focus_index=rec.gt_focus_index,
                scene_name=rec.scene_name,
                patch_x=rec.patch_x,
                patch_y=rec.patch_y,
                temperature=rec.temperature,
                left_raw_prefix=rec.left_raw_prefix,
                right_raw_prefix=rec.right_raw_prefix,
            )
            for rec in stack
        },
    )

    positions = [current_focus]
    for _ in range(max_steps):
        current_record = _find_nearest_record(stack, current_focus)
        state = build_state_from_record(
            current_record,
            data_root=data_root,
            patch_size=patch_size,
            pe_dim=pe_dim,
            raw_suffix=raw_suffix,
        )
        state = {key: value.unsqueeze(0).to(device) for key, value in state.items()}
        with torch.no_grad():
            logits = model.actor_logits(state)
            action_index = logits.argmax(dim=-1) if deterministic else torch.distributions.Categorical(logits=logits).sample()
        offset = int(action_index.item()) - ACTION_RANGE
        next_focus = int(np.clip(current_focus + offset, 0, NUM_FOCUS_POSITIONS - 1))
        next_focus = _find_nearest_focus(stack, next_focus)
        positions.append(next_focus)
        current_focus = next_focus

    for step_idx, focus_index in enumerate(positions):
        rec = _find_nearest_record(stack, focus_index)
        if step_idx < len(positions) - 1:
            action = positions[step_idx + 1] - focus_index
        else:
            action = 0
        trajectory.steps.append(
            StepData(
                focus_index=focus_index,
                action=action,
                gt_focus_index=gt_focus,
                scene_name=rec.scene_name,
                patch_x=rec.patch_x,
                patch_y=rec.patch_y,
                temperature=rec.temperature,
                left_raw_prefix=rec.left_raw_prefix,
                right_raw_prefix=rec.right_raw_prefix,
            )
        )
    return trajectory



def build_policy_trajectories(
    txt_path: str,
    data_root: str,
    checkpoint_path: str,
    max_steps: int = 4,
    pe_dim: int = 16,
    patch_size: int = DEFAULT_PATCH_SIZE,
    raw_suffix: str = ".npy",
    device: str = "cpu",
    deterministic: bool = True,
    imagenet_pretrained: bool = False,
) -> List[ExpertTrajectory]:
    model = AutofocusActorCritic(pe_dim=pe_dim, imagenet_pretrained=imagenet_pretrained)
    model.load_phase1_weights(checkpoint_path, freeze_backbone=False, load_actor_head=True)
    model.to(device)
    model.eval()

    stacks = group_focal_stacks(parse_txt(txt_path))
    trajectories: List[ExpertTrajectory] = []
    for stack in stacks.values():
        for start_rec in stack:
            trajectories.append(
                rollout_policy_trajectory(
                    model,
                    stack,
                    start_rec,
                    data_root=data_root,
                    max_steps=max_steps,
                    device=torch.device(device),
                    patch_size=patch_size,
                    pe_dim=pe_dim,
                    raw_suffix=raw_suffix,
                    deterministic=deterministic,
                )
            )
    return trajectories



def algorithm1(original: ExpertTrajectory) -> ExpertTrajectory:
    gt = original.gt_focus_index
    positions = original.positions
    k0 = positions[0]
    if k0 == gt:
        return algorithm2(original)
    reflected = []
    lower = min(k0, gt)
    upper = max(k0, gt)
    for position in positions:
        mirrored = (1 if k0 > gt else -1) * abs(position - gt) + gt
        mirrored = int(np.clip(mirrored, lower, upper))
        reflected.append(mirrored)
    reflected = sorted(reflected, reverse=(k0 > gt))

    expert = ExpertTrajectory(
        scene_name=original.scene_name,
        patch_x=original.patch_x,
        patch_y=original.patch_y,
        gt_focus_index=gt,
        algo_id=1,
        record_bank=original.record_bank,
    )
    for index, step in enumerate(original.steps):
        focus_index = reflected[index]
        action = reflected[index + 1] - focus_index if index < len(reflected) - 1 else 0
        bank_step = original.record_bank.get(focus_index, step)
        expert.steps.append(
            StepData(
                focus_index=focus_index,
                action=action,
                gt_focus_index=gt,
                scene_name=bank_step.scene_name,
                patch_x=bank_step.patch_x,
                patch_y=bank_step.patch_y,
                temperature=bank_step.temperature,
                left_raw_prefix=bank_step.left_raw_prefix,
                right_raw_prefix=bank_step.right_raw_prefix,
            )
        )
    return expert



def algorithm2(original: ExpertTrajectory) -> ExpertTrajectory:
    gt = original.gt_focus_index
    k0 = original.steps[0].focus_index
    expert = ExpertTrajectory(
        scene_name=original.scene_name,
        patch_x=original.patch_x,
        patch_y=original.patch_y,
        gt_focus_index=gt,
        algo_id=2,
        record_bank=original.record_bank,
    )
    for index, step in enumerate(original.steps):
        focus_index = k0 if index == 0 else gt
        action = (gt - k0) if index == 0 else 0
        source_step = original.record_bank.get(focus_index, original.steps[min(index, len(original.steps) - 1)])
        expert.steps.append(
            StepData(
                focus_index=focus_index,
                action=action,
                gt_focus_index=gt,
                scene_name=source_step.scene_name,
                patch_x=source_step.patch_x,
                patch_y=source_step.patch_y,
                temperature=source_step.temperature,
                left_raw_prefix=source_step.left_raw_prefix,
                right_raw_prefix=source_step.right_raw_prefix,
            )
        )
    return expert



def algorithm3(original: ExpertTrajectory, m: int = 5) -> ExpertTrajectory:
    gt = original.gt_focus_index
    k0 = original.steps[0].focus_index
    positions = [k0]
    distance = k0 - gt
    for _ in range(1, len(original.steps) - 1):
        distance = int(distance / m)
        positions.append(gt + distance)
    positions.append(gt)

    expert = ExpertTrajectory(
        scene_name=original.scene_name,
        patch_x=original.patch_x,
        patch_y=original.patch_y,
        gt_focus_index=gt,
        algo_id=3,
        record_bank=original.record_bank,
    )
    for index, step in enumerate(original.steps):
        focus_index = int(np.clip(positions[index], 0, NUM_FOCUS_POSITIONS - 1))
        action = positions[index + 1] - focus_index if index < len(positions) - 1 else 0
        source_step = original.record_bank.get(focus_index, step)
        expert.steps.append(
            StepData(
                focus_index=focus_index,
                action=action,
                gt_focus_index=gt,
                scene_name=source_step.scene_name,
                patch_x=source_step.patch_x,
                patch_y=source_step.patch_y,
                temperature=source_step.temperature,
                left_raw_prefix=source_step.left_raw_prefix,
                right_raw_prefix=source_step.right_raw_prefix,
            )
        )
    return expert



def generate_expert_trajectories(
    originals: Iterable[ExpertTrajectory],
    algo_mix: Sequence[int] = (1, 2, 3),
    m: int = 5,
) -> List[ExpertTrajectory]:
    experts: List[ExpertTrajectory] = []
    for original in originals:
        if 1 in algo_mix:
            experts.append(algorithm1(original))
        if 2 in algo_mix:
            experts.append(algorithm2(original))
        if 3 in algo_mix:
            experts.append(algorithm3(original, m=m))
    return experts



def save_trajectories_json(trajectories: Sequence[ExpertTrajectory], output_path: str) -> None:
    payload = []
    for traj in trajectories:
        payload.append(
            {
                "scene_name": traj.scene_name,
                "patch_x": traj.patch_x,
                "patch_y": traj.patch_y,
                "gt_focus_index": traj.gt_focus_index,
                "algo_id": traj.algo_id,
                "steps": [asdict(step) for step in traj.steps],
            }
        )
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)



def load_trajectories_json(json_path: str) -> List[ExpertTrajectory]:
    with open(json_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    trajectories: List[ExpertTrajectory] = []
    for traj_data in raw:
        trajectories.append(
            ExpertTrajectory(
                scene_name=traj_data["scene_name"],
                patch_x=traj_data["patch_x"],
                patch_y=traj_data["patch_y"],
                gt_focus_index=traj_data["gt_focus_index"],
                algo_id=traj_data.get("algo_id", 0),
                steps=[StepData(**step_data) for step_data in traj_data["steps"]],
            )
        )
    return trajectories


class ExpertTrajectoryDataset(Dataset):
    def __init__(
        self,
        trajectories: Sequence[ExpertTrajectory],
        data_root: str,
        patch_size: int = DEFAULT_PATCH_SIZE,
        pe_dim: int = 16,
        raw_suffix: str = ".npy",
    ) -> None:
        self.trajectories = list(trajectories)
        self.data_root = data_root
        self.patch_size = patch_size
        self.pe_dim = pe_dim
        self.raw_suffix = raw_suffix

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, index: int):
        trajectory = self.trajectories[index]
        states = []
        actions = []
        for step in trajectory.steps:
            record = AutofocusRecord(
                scene_name=step.scene_name,
                left_raw_prefix=step.left_raw_prefix,
                right_raw_prefix=step.right_raw_prefix,
                focus_index=step.focus_index,
                gt_focus_index=step.gt_focus_index,
                patch_x=step.patch_x,
                patch_y=step.patch_y,
                temperature=step.temperature,
            )
            states.append(
                build_state_from_record(
                    record,
                    data_root=self.data_root,
                    patch_size=self.patch_size,
                    pe_dim=self.pe_dim,
                    raw_suffix=self.raw_suffix,
                )
            )
            actions.append(int(np.clip(step.action + ACTION_RANGE, 0, ACTION_DIM - 1)))
        return states, torch.tensor(actions, dtype=torch.long)



def collate_expert_trajectories(batch):
    images = []
    lens_pe = []
    roi_pe = []
    temperature = []
    actions = []
    for states, traj_actions in batch:
        for state, action in zip(states, traj_actions):
            images.append(state["image"])
            lens_pe.append(state["lens_pe"])
            roi_pe.append(state["roi_pe"])
            temperature.append(state["temperature"])
            actions.append(action)
    return {
        "image": torch.stack(images),
        "lens_pe": torch.stack(lens_pe),
        "roi_pe": torch.stack(roi_pe),
        "temperature": torch.stack(temperature),
    }, torch.stack(actions)



def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper-faithful expert trajectories.")
    parser.add_argument("--txt_path", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--policy_ckpt", required=True, help="Phase-1 checkpoint used to generate original trajectories.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--save_original_json", default=None)
    parser.add_argument("--max_steps", type=int, default=4)
    parser.add_argument("--m", type=int, default=5)
    parser.add_argument("--algos", type=str, default="1,2,3")
    parser.add_argument("--pe_dim", type=int, default=16)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--raw_suffix", type=str, default=".npy")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--stochastic_policy", action="store_true")
    args = parser.parse_args()

    algo_mix = [int(token) for token in args.algos.split(",") if token.strip()]
    originals = build_policy_trajectories(
        txt_path=args.txt_path,
        data_root=args.data_root,
        checkpoint_path=args.policy_ckpt,
        max_steps=args.max_steps,
        pe_dim=args.pe_dim,
        patch_size=args.patch_size,
        raw_suffix=args.raw_suffix,
        device=args.device,
        deterministic=not args.stochastic_policy,
    )
    experts = generate_expert_trajectories(originals, algo_mix=algo_mix, m=args.m)
    if args.save_original_json:
        save_trajectories_json(originals, args.save_original_json)
    save_trajectories_json(experts, args.output)
    print(f"Generated {len(experts)} expert trajectories from {len(originals)} original trajectories.")
    print(f"Saved expert trajectories to {args.output}")


if __name__ == "__main__":
    main()
