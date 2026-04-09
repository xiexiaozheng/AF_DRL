"""
trajectory_builder.py – Offline Expert Trajectory Generation
=============================================================

Implements the three expert trajectory generation algorithms described
in the paper's supplementary material:

  • **Algorithm 1** – Fix overshooting / focus-hunting trajectories by
    mirroring positions that crossed the GT back into a valid interval,
    then re-sorting to guarantee monotonicity.

  • **Algorithm 2** – For difficult / textureless scenes: keep the
    initial position, then jump directly to GT for all subsequent
    steps ("one-step-to-GT").

  • **Algorithm 3** – Smooth decaying approach: divide the initial
    distance *d* by a factor *m* at each step, yielding a progressively
    smaller step size that converges to GT.

All three algorithms produce state–action expert trajectories
``(s_0, a_0), ..., (s_n, a_n)`` where ``a_t = k_{t+1} - k_t`` is the
relative lens movement and ``a_n = 0`` (stop).

Public API
----------
``build_expert_trajectories(records, ...)``
    Main entry-point.  Takes the parsed TXT records, iterates over all
    focal stacks, and returns a list of ``ExpertTrajectory`` objects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from dataset import (
    AutofocusRecord,
    group_focal_stacks,
    parse_txt,
    NUM_FOCUS_POSITIONS,
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StepData:
    """A single (state-descriptor, action) pair in a trajectory."""
    focus_index: int            # lens position at this step
    action: int                 # relative movement  a_t = k_{t+1} - k_t
    gt_focus_index: int
    scene_name: str = ""
    patch_x: int = 0
    patch_y: int = 0
    temperature: float = 0.0
    left_raw_prefix: str = ""
    right_raw_prefix: str = ""


@dataclass
class ExpertTrajectory:
    """A complete expert trajectory of length *n_steps + 1*."""
    scene_name: str
    patch_x: int
    patch_y: int
    gt_focus_index: int
    steps: List[StepData] = field(default_factory=list)
    algo_id: int = 0           # which algorithm generated this


# ---------------------------------------------------------------------------
# Helper: find the record closest to a given focus_index in a stack
# ---------------------------------------------------------------------------

def _find_nearest_record(
    stack: List[AutofocusRecord],
    target_focus: int,
) -> AutofocusRecord:
    """Return the record in *stack* whose focus_index is closest to
    *target_focus*.  Ties are broken in favour of the lower index."""
    best = stack[0]
    best_dist = abs(best.focus_index - target_focus)
    for rec in stack[1:]:
        d = abs(rec.focus_index - target_focus)
        if d < best_dist:
            best = rec
            best_dist = d
    return best


# ---------------------------------------------------------------------------
# Algorithm 1  –  Fix overshooting / focus hunting
# ---------------------------------------------------------------------------

def algorithm1(
    stack: List[AutofocusRecord],
    n_steps: int,
) -> List[ExpertTrajectory]:
    """Generate expert trajectories by mirroring and re-sorting.

    For every record in *stack* used as a starting position:
      1. Select *n_steps + 1* positions from the stack, sampling only
         within the useful overshooting range [k_0, 2·GT − k_0] (or
         [2·GT − k_0, k_0] when k_0 > GT).  Positions outside this
         symmetric window around GT would collapse to k_0 after
         mirroring and clipping, so restricting to this range avoids
         degenerate trajectories.
      2. For each position *k_j*, compute the mirrored position:
             o_j = sign(k_0 - GT) · |k_j - GT| + GT
         This "reflects" any position that overshot GT back across GT.
      3. Clip each o_j to [min(k_0, GT), max(k_0, GT)].
      4. Force the last waypoint to GT so the trajectory always ends
         exactly at the ground-truth focus position.
      5. Sort the resulting positions monotonically towards GT.
      6. Derive actions as differences between consecutive positions;
         the final action is 0 (stop at GT).

    Returns one trajectory per starting position in the stack.
    """
    gt = stack[0].gt_focus_index
    trajectories: List[ExpertTrajectory] = []

    for start_rec in stack:
        k0 = start_rec.focus_index
        if k0 == gt:
            # Already at GT – trivial trajectory (all actions = 0)
            traj = ExpertTrajectory(
                scene_name=start_rec.scene_name,
                patch_x=start_rec.patch_x,
                patch_y=start_rec.patch_y,
                gt_focus_index=gt,
                algo_id=1,
            )
            for step_idx in range(n_steps + 1):
                traj.steps.append(StepData(
                    focus_index=gt,
                    action=0,
                    gt_focus_index=gt,
                    scene_name=start_rec.scene_name,
                    patch_x=start_rec.patch_x,
                    patch_y=start_rec.patch_y,
                    temperature=start_rec.temperature,
                    left_raw_prefix=start_rec.left_raw_prefix,
                    right_raw_prefix=start_rec.right_raw_prefix,
                ))
            trajectories.append(traj)
            continue

        sign_k0 = 1 if k0 > gt else -1
        lo, hi = min(k0, gt), max(k0, gt)

        # Collect n_steps+1 positions from the stack (starting from k0).
        # Only sample within the useful overshooting range: the mirror of k0
        # through GT is 2*gt - k0.  Positions beyond that boundary would clip
        # to k0 after mirroring, creating degenerate collapsed trajectories.
        n_total = n_steps + 1
        overshoot_limit = 2 * gt - k0  # symmetric mirror of k0 across GT
        if k0 < gt:
            valid_lo_pos = k0
            valid_hi_pos = min(overshoot_limit, NUM_FOCUS_POSITIONS - 1)
        else:
            valid_lo_pos = max(overshoot_limit, 0)
            valid_hi_pos = k0

        # Map valid focus-position range to sorted-stack indices
        valid_start_idx = next(
            (i for i, r in enumerate(stack) if r.focus_index >= valid_lo_pos), 0
        )
        valid_end_idx = next(
            (i - 1 for i, r in enumerate(stack) if r.focus_index > valid_hi_pos),
            len(stack) - 1,
        )
        valid_end_idx = max(valid_start_idx, valid_end_idx)

        indices = np.linspace(valid_start_idx, valid_end_idx, n_total, dtype=int)
        # The first position is always the starting record k0
        raw_positions = [k0]
        for idx in indices[1:]:
            raw_positions.append(stack[idx].focus_index)

        # Step 2 & 3: mirror & clip
        mirrored = []
        for kj in raw_positions:
            oj = sign_k0 * abs(kj - gt) + gt
            oj = max(lo, min(oj, hi))
            mirrored.append(oj)

        # Step 4: sort monotonically towards GT.
        # Force the last waypoint to GT so the trajectory always ends exactly
        # at the ground-truth focus position (last action = 0, i.e. stop).
        reverse = k0 > gt  # descending if starting above GT
        mirrored[-1] = gt
        mirrored.sort(reverse=reverse)

        # Step 5: build trajectory
        traj = ExpertTrajectory(
            scene_name=start_rec.scene_name,
            patch_x=start_rec.patch_x,
            patch_y=start_rec.patch_y,
            gt_focus_index=gt,
            algo_id=1,
        )
        for j in range(len(mirrored)):
            fj = int(round(mirrored[j]))
            fj = max(0, min(fj, NUM_FOCUS_POSITIONS - 1))
            nearest_rec = _find_nearest_record(stack, fj)
            if j < len(mirrored) - 1:
                action = int(round(mirrored[j + 1])) - fj
            else:
                action = 0  # last step
            traj.steps.append(StepData(
                focus_index=fj,
                action=action,
                gt_focus_index=gt,
                scene_name=nearest_rec.scene_name,
                patch_x=nearest_rec.patch_x,
                patch_y=nearest_rec.patch_y,
                temperature=nearest_rec.temperature,
                left_raw_prefix=nearest_rec.left_raw_prefix,
                right_raw_prefix=nearest_rec.right_raw_prefix,
            ))
        trajectories.append(traj)

    return trajectories


# ---------------------------------------------------------------------------
# Algorithm 2  –  Difficult / no-texture scenes (one-step-to-GT)
# ---------------------------------------------------------------------------

def algorithm2(
    stack: List[AutofocusRecord],
    n_steps: int,
) -> List[ExpertTrajectory]:
    """Generate expert trajectories: keep initial position, then jump to GT.

    Trajectory:  k_0  →  GT  →  GT  →  ...  →  GT
    Actions:     (GT-k_0), 0, 0, ..., 0
    """
    gt = stack[0].gt_focus_index
    gt_rec = _find_nearest_record(stack, gt)
    trajectories: List[ExpertTrajectory] = []

    for start_rec in stack:
        k0 = start_rec.focus_index
        traj = ExpertTrajectory(
            scene_name=start_rec.scene_name,
            patch_x=start_rec.patch_x,
            patch_y=start_rec.patch_y,
            gt_focus_index=gt,
            algo_id=2,
        )
        # Step 0: at k0, action = GT - k0
        traj.steps.append(StepData(
            focus_index=k0,
            action=gt - k0,
            gt_focus_index=gt,
            scene_name=start_rec.scene_name,
            patch_x=start_rec.patch_x,
            patch_y=start_rec.patch_y,
            temperature=start_rec.temperature,
            left_raw_prefix=start_rec.left_raw_prefix,
            right_raw_prefix=start_rec.right_raw_prefix,
        ))
        # Steps 1..n: at GT, action = 0
        for _ in range(n_steps):
            traj.steps.append(StepData(
                focus_index=gt,
                action=0,
                gt_focus_index=gt,
                scene_name=gt_rec.scene_name,
                patch_x=gt_rec.patch_x,
                patch_y=gt_rec.patch_y,
                temperature=gt_rec.temperature,
                left_raw_prefix=gt_rec.left_raw_prefix,
                right_raw_prefix=gt_rec.right_raw_prefix,
            ))
        trajectories.append(traj)

    return trajectories


# ---------------------------------------------------------------------------
# Algorithm 3  –  Smooth decaying approach
# ---------------------------------------------------------------------------

def algorithm3(
    stack: List[AutofocusRecord],
    n_steps: int,
    m: int = 5,
) -> List[ExpertTrajectory]:
    """Generate expert trajectories with decaying step size.

    Starting from k_0 with distance d = k_0 - GT:
      • At each step j, d ← int(d / m), k_j = GT + d
      • The last step lands exactly on GT.
    Actions are the differences between consecutive positions.
    """
    gt = stack[0].gt_focus_index
    trajectories: List[ExpertTrajectory] = []

    for start_rec in stack:
        k0 = start_rec.focus_index
        d = k0 - gt

        # Build position sequence
        positions = [k0]
        cur_d = d
        for j in range(1, n_steps):
            cur_d = int(cur_d / m)
            kj = gt + cur_d
            kj = max(0, min(kj, NUM_FOCUS_POSITIONS - 1))
            positions.append(kj)
        # Final step: land on GT
        positions.append(gt)

        # Trim or pad to exactly n_steps + 1
        while len(positions) < n_steps + 1:
            positions.append(gt)
        positions = positions[: n_steps + 1]

        # Build trajectory
        traj = ExpertTrajectory(
            scene_name=start_rec.scene_name,
            patch_x=start_rec.patch_x,
            patch_y=start_rec.patch_y,
            gt_focus_index=gt,
            algo_id=3,
        )
        for j in range(len(positions)):
            fj = positions[j]
            nearest_rec = _find_nearest_record(stack, fj)
            action = (positions[j + 1] - fj) if j < len(positions) - 1 else 0
            traj.steps.append(StepData(
                focus_index=fj,
                action=action,
                gt_focus_index=gt,
                scene_name=nearest_rec.scene_name,
                patch_x=nearest_rec.patch_x,
                patch_y=nearest_rec.patch_y,
                temperature=nearest_rec.temperature,
                left_raw_prefix=nearest_rec.left_raw_prefix,
                right_raw_prefix=nearest_rec.right_raw_prefix,
            ))
        trajectories.append(traj)

    return trajectories


# ---------------------------------------------------------------------------
# Public API – build all expert trajectories from TXT
# ---------------------------------------------------------------------------

def build_expert_trajectories(
    txt_path: str,
    n_steps: int = 4,
    m: int = 5,
    algo_mix: Sequence[int] = (1, 2, 3),
) -> List[ExpertTrajectory]:
    """Parse *txt_path* and generate expert trajectories with the selected
    algorithms.

    Parameters
    ----------
    txt_path : str
        Path to the annotation TXT file.
    n_steps : int
        Maximum number of movement steps per trajectory (trajectory
        length = n_steps + 1 including the initial position).
    m : int
        Division factor for Algorithm 3.
    algo_mix : sequence of int
        Which algorithms to include.  Default ``(1, 2, 3)`` generates
        trajectories from all three algorithms and concatenates them.

    Returns
    -------
    list of ExpertTrajectory
    """
    records = parse_txt(txt_path)
    stacks = group_focal_stacks(records)

    all_trajectories: List[ExpertTrajectory] = []
    for _key, stack_records in stacks.items():
        if 1 in algo_mix:
            all_trajectories.extend(algorithm1(stack_records, n_steps))
        if 2 in algo_mix:
            all_trajectories.extend(algorithm2(stack_records, n_steps))
        if 3 in algo_mix:
            all_trajectories.extend(algorithm3(stack_records, n_steps, m=m))

    return all_trajectories


# ---------------------------------------------------------------------------
# Trajectory → flat dataset for RL training
# ---------------------------------------------------------------------------

@dataclass
class FlatStep:
    """One (state-descriptor, expert_action) row ready for RL training."""
    focus_index: int
    gt_focus_index: int
    expert_action: int
    scene_name: str
    patch_x: int
    patch_y: int
    temperature: float
    left_raw_prefix: str
    right_raw_prefix: str
    step_in_traj: int
    algo_id: int


def trajectories_to_flat(
    trajectories: List[ExpertTrajectory],
) -> List[FlatStep]:
    """Flatten a list of trajectories into individual step records.

    Useful for building a ``torch.utils.data.Dataset`` that yields
    (state, expert_action) pairs for expert regularisation.
    """
    flat: List[FlatStep] = []
    for traj in trajectories:
        for t, step in enumerate(traj.steps):
            flat.append(FlatStep(
                focus_index=step.focus_index,
                gt_focus_index=step.gt_focus_index,
                expert_action=step.action,
                scene_name=step.scene_name,
                patch_x=step.patch_x,
                patch_y=step.patch_y,
                temperature=step.temperature,
                left_raw_prefix=step.left_raw_prefix,
                right_raw_prefix=step.right_raw_prefix,
                step_in_traj=t,
                algo_id=traj.algo_id,
            ))
    return flat


# ---------------------------------------------------------------------------
# CLI convenience – build & save trajectories to disk
# ---------------------------------------------------------------------------

def main():
    """Command-line interface for building expert trajectories."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Build offline expert trajectories from TXT annotation."
    )
    parser.add_argument("txt_path", help="Path to annotation TXT file.")
    parser.add_argument(
        "--n_steps", type=int, default=4,
        help="Max movement steps per trajectory (default: 4)."
    )
    parser.add_argument(
        "--m", type=int, default=5,
        help="Division factor for Algorithm 3 (default: 5)."
    )
    parser.add_argument(
        "--algos", type=str, default="1,2,3",
        help="Comma-separated algorithm IDs to use (default: '1,2,3')."
    )
    parser.add_argument(
        "--output", type=str, default="expert_trajectories.json",
        help="Output JSON path."
    )
    args = parser.parse_args()

    algo_mix = [int(x) for x in args.algos.split(",")]
    trajs = build_expert_trajectories(
        args.txt_path, n_steps=args.n_steps, m=args.m, algo_mix=algo_mix,
    )

    # Serialise
    data = []
    for traj in trajs:
        traj_dict = {
            "scene_name": traj.scene_name,
            "patch_x": traj.patch_x,
            "patch_y": traj.patch_y,
            "gt_focus_index": traj.gt_focus_index,
            "algo_id": traj.algo_id,
            "steps": [
                {
                    "focus_index": s.focus_index,
                    "action": s.action,
                    "gt_focus_index": s.gt_focus_index,
                    "temperature": s.temperature,
                    "left_raw_prefix": s.left_raw_prefix,
                    "right_raw_prefix": s.right_raw_prefix,
                }
                for s in traj.steps
            ],
        }
        data.append(traj_dict)

    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(trajs)} expert trajectories to {args.output}")


if __name__ == "__main__":
    main()
