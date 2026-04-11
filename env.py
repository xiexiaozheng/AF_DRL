"""
env.py – Gymnasium environment matching the paper reward definition.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from dataset import DEFAULT_PATCH_SIZE, AutofocusRecord, NUM_FOCUS_POSITIONS, group_focal_stacks, parse_txt
from models import ACTION_DIM, ACTION_RANGE
from trajectory_builder import build_state_from_record


class AutofocusEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        txt_path: str,
        data_root: str,
        max_steps: int = 4,
        patch_size: int = DEFAULT_PATCH_SIZE,
        pe_dim: int = 16,
        raw_suffix: str = ".npy",
        focus_hunting_penalty: float = -1.5,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.max_steps = max_steps
        self.patch_size = patch_size
        self.pe_dim = pe_dim
        self.raw_suffix = raw_suffix
        self.focus_hunting_penalty = focus_hunting_penalty

        records = parse_txt(txt_path)
        self.stacks = group_focal_stacks(records)
        self.stack_keys = list(self.stacks.keys())
        self.record_index: Dict[Tuple[str, int, int, int], AutofocusRecord] = {}
        for recs in self.stacks.values():
            for rec in recs:
                self.record_index[(rec.scene_name, rec.patch_x, rec.patch_y, rec.focus_index)] = rec

        obs_dim = 2 * patch_size * patch_size + pe_dim * 2 + 1
        self.action_space = spaces.Discrete(ACTION_DIM)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self._current_stack_key: Optional[Tuple[str, int, int]] = None
        self._current_stack: List[AutofocusRecord] = []
        self._focus_index = 0
        self._gt_focus_index = 0
        self._step_count = 0
        self._prev_direction: Optional[int] = None

    def _find_nearest_focus(self, target: int) -> int:
        available = [rec.focus_index for rec in self._current_stack]
        return min(available, key=lambda focus: abs(focus - target))

    def _get_state_dict(self, focus_index: int) -> Dict[str, torch.Tensor]:
        assert self._current_stack_key is not None
        scene, patch_x, patch_y = self._current_stack_key
        rec = self.record_index.get((scene, patch_x, patch_y, focus_index))
        if rec is None:
            rec = self.record_index[(scene, patch_x, patch_y, self._find_nearest_focus(focus_index))]
        return build_state_from_record(
            rec,
            data_root=self.data_root,
            patch_size=self.patch_size,
            pe_dim=self.pe_dim,
            raw_suffix=self.raw_suffix,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        super().reset(seed=seed)
        rng = self.np_random
        if options and "stack_key" in options:
            self._current_stack_key = tuple(options["stack_key"])
        else:
            self._current_stack_key = self.stack_keys[int(rng.integers(len(self.stack_keys)))]
        self._current_stack = self.stacks[self._current_stack_key]
        self._gt_focus_index = self._current_stack[0].gt_focus_index
        if options and "start_focus" in options:
            self._focus_index = int(options["start_focus"])
        else:
            random_rec = self._current_stack[int(rng.integers(len(self._current_stack)))]
            self._focus_index = random_rec.focus_index
        self._step_count = 0
        self._prev_direction = None
        state = self._get_state_dict(self._focus_index)
        return state, {
            "focus_index": self._focus_index,
            "gt_focus_index": self._gt_focus_index,
            "step": self._step_count,
        }

    def step(self, action: int):
        offset = int(action) - ACTION_RANGE
        new_focus = int(np.clip(self._focus_index + offset, 0, NUM_FOCUS_POSITIONS - 1))
        new_focus = self._find_nearest_focus(new_focus)
        self._step_count += 1

        actual_move = new_focus - self._focus_index
        direction = 1 if actual_move > 0 else (-1 if actual_move < 0 else 0)
        hunting = (
            direction != 0
            and self._prev_direction is not None
            and self._prev_direction != 0
            and direction != self._prev_direction
        )
        if direction != 0:
            self._prev_direction = direction

        self._focus_index = new_focus
        new_error = abs(self._focus_index - self._gt_focus_index)
        reward = -float(new_error)
        if hunting:
            reward += self.focus_hunting_penalty

        state = self._get_state_dict(self._focus_index)
        terminated = new_error == 0
        truncated = self._step_count >= self.max_steps
        info = {
            "focus_index": self._focus_index,
            "gt_focus_index": self._gt_focus_index,
            "error": new_error,
            "step": self._step_count,
            "hunting": hunting,
        }
        return state, reward, terminated, truncated, info
