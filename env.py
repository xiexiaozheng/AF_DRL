"""
env.py – Autofocus Gymnasium Environment
==========================================

Simulates the autofocus process as a Gymnasium-compatible environment.
At each time-step the agent observes a dual-pixel image patch at the
current lens position and selects a *relative* lens movement (action).
The environment then looks up the closest available focus-index in the
dataset to produce the next state.

Reward
------
* **Step reward** – proportional to reduction in absolute error.
* **Focus-hunting penalty** – large negative reward when the lens
  reverses direction (overshoots GT).

The episode terminates after ``max_steps`` or when the agent lands on
the GT focus index.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from dataset import (
    AutofocusRecord,
    NUM_FOCUS_POSITIONS,
    _ensure_2d,
    crop_patch,
    lens_position_encoding,
    load_raw_image,
    normalise_patch,
    parse_txt,
    roi_position_encoding,
    group_focal_stacks,
    DEFAULT_PATCH_SIZE,
)
from models import ACTION_DIM, ACTION_RANGE


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class AutofocusEnv(gym.Env):
    """Gymnasium environment for autofocus with offline data.

    The environment is *data-driven*: instead of a physical camera,
    it looks up images from a pre-captured dataset indexed by
    (scene, patch, focus_index).

    Parameters
    ----------
    txt_path : str
        Annotation TXT file.
    data_root : str
        Root directory for RAW image files.
    max_steps : int
        Maximum number of steps per episode.
    patch_size : int
        Crop size for each patch.
    pe_dim : int
        Positional-encoding dimensionality.
    raw_suffix : str
        File extension for RAW files.
    focus_hunting_penalty : float
        Penalty applied when the lens reverses direction (default -1.5).
    reward_scale : float
        Multiplier for the step reward based on MAE reduction.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        txt_path: str,
        data_root: str,
        max_steps: int = 7,
        patch_size: int = DEFAULT_PATCH_SIZE,
        pe_dim: int = 16,
        raw_suffix: str = ".npy",
        focus_hunting_penalty: float = -1.5,
        reward_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.max_steps = max_steps
        self.patch_size = patch_size
        self.pe_dim = pe_dim
        self.raw_suffix = raw_suffix
        self.focus_hunting_penalty = focus_hunting_penalty
        self.reward_scale = reward_scale

        # Parse & group data
        records = parse_txt(txt_path)
        self.stacks = group_focal_stacks(records)
        self.stack_keys = list(self.stacks.keys())

        # Build an index: (scene, px, py, focus_index) → record
        self.record_index: Dict[Tuple[str, int, int, int], AutofocusRecord] = {}
        for key, recs in self.stacks.items():
            for r in recs:
                self.record_index[(r.scene_name, r.patch_x, r.patch_y, r.focus_index)] = r

        # Gymnasium spaces
        self.action_space = spaces.Discrete(ACTION_DIM)
        # Observation is a dict; we use a flat Box as a placeholder for
        # compatibility, but the actual observation is returned as a dict.
        obs_dim = 2 * patch_size * patch_size + pe_dim * 2 + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )

        # Episode state
        self._current_stack_key: Optional[Tuple[str, int, int]] = None
        self._current_stack: List[AutofocusRecord] = []
        self._focus_index: int = 0
        self._gt_focus_index: int = 0
        self._step_count: int = 0
        self._prev_direction: Optional[int] = None  # +1 or -1
        self._initial_error: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _find_nearest_focus(self, target: int) -> int:
        """Return the closest available focus_index in the current stack."""
        available = [r.focus_index for r in self._current_stack]
        best = min(available, key=lambda f: abs(f - target))
        return best

    def _get_state_dict(self, focus_index: int) -> Dict[str, torch.Tensor]:
        """Build the state dict for the given focus_index in the current stack."""
        scene, px, py = self._current_stack_key  # type: ignore[misc]
        # Find the record
        rec = self.record_index.get((scene, px, py, focus_index))
        if rec is None:
            # Fallback to nearest
            focus_index = self._find_nearest_focus(focus_index)
            rec = self.record_index.get((scene, px, py, focus_index))
        if rec is None:
            rec = self._current_stack[0]

        # Load images
        import os
        path_l = os.path.join(self.data_root, rec.left_raw_prefix + self.raw_suffix)
        path_r = os.path.join(self.data_root, rec.right_raw_prefix + self.raw_suffix)
        try:
            left = _ensure_2d(normalise_patch(crop_patch(load_raw_image(path_l), px, py, self.patch_size)))
            right = _ensure_2d(normalise_patch(crop_patch(load_raw_image(path_r), px, py, self.patch_size)))
            image = torch.from_numpy(np.stack([left, right], axis=0))
        except Exception:
            image = torch.zeros(2, self.patch_size, self.patch_size)

        lens_pe = torch.from_numpy(
            lens_position_encoding(focus_index, embed_dim=self.pe_dim)
        )
        roi_pe = torch.from_numpy(
            roi_position_encoding(px, py, embed_dim=self.pe_dim)
        )
        temperature = torch.tensor([rec.temperature], dtype=torch.float32)

        return {
            "image": image,
            "lens_pe": lens_pe,
            "roi_pe": roi_pe,
            "temperature": temperature,
        }

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Reset the environment to a random (scene, patch, start_focus).

        ``options`` may contain:
            ``"stack_key"``    – specific (scene, px, py) tuple.
            ``"start_focus"``  – specific starting focus_index.
        """
        super().reset(seed=seed)
        rng = self.np_random

        if options and "stack_key" in options:
            self._current_stack_key = options["stack_key"]
        else:
            idx = rng.integers(len(self.stack_keys))
            self._current_stack_key = self.stack_keys[idx]

        self._current_stack = self.stacks[self._current_stack_key]
        self._gt_focus_index = self._current_stack[0].gt_focus_index

        if options and "start_focus" in options:
            self._focus_index = options["start_focus"]
        else:
            random_rec = self._current_stack[rng.integers(len(self._current_stack))]
            self._focus_index = random_rec.focus_index

        self._step_count = 0
        self._prev_direction = None
        self._initial_error = abs(self._focus_index - self._gt_focus_index)

        state = self._get_state_dict(self._focus_index)
        info = {
            "focus_index": self._focus_index,
            "gt_focus_index": self._gt_focus_index,
            "initial_error": self._initial_error,
        }
        return state, info

    # ------------------------------------------------------------------
    def step(
        self, action: int,
    ) -> Tuple[Dict[str, torch.Tensor], float, bool, bool, Dict[str, Any]]:
        """Execute one step.

        Parameters
        ----------
        action : int
            Index into the action space [0, ACTION_DIM).
            Converted to offset ∈ [-ACTION_RANGE, ACTION_RANGE].

        Returns
        -------
        state, reward, terminated, truncated, info
        """
        offset = int(action) - ACTION_RANGE
        old_error = abs(self._focus_index - self._gt_focus_index)

        # Move lens
        new_focus = self._focus_index + offset
        new_focus = max(0, min(new_focus, NUM_FOCUS_POSITIONS - 1))
        # Snap to nearest available index in the stack
        new_focus = self._find_nearest_focus(new_focus)

        new_error = abs(new_focus - self._gt_focus_index)
        self._step_count += 1

        # --- Reward ---
        # Base reward: proportional to error reduction
        reward = float(old_error - new_error) * self.reward_scale

        # Focus hunting penalty: detect direction reversal
        direction = 0
        actual_move = new_focus - self._focus_index
        if actual_move > 0:
            direction = 1
        elif actual_move < 0:
            direction = -1

        hunting = False
        if (
            direction != 0
            and self._prev_direction is not None
            and self._prev_direction != 0
            and direction != self._prev_direction
        ):
            reward += self.focus_hunting_penalty
            hunting = True

        if direction != 0:
            self._prev_direction = direction

        # Update state
        self._focus_index = new_focus
        state = self._get_state_dict(self._focus_index)

        # Termination
        terminated = new_error == 0
        truncated = self._step_count >= self.max_steps

        info = {
            "focus_index": self._focus_index,
            "gt_focus_index": self._gt_focus_index,
            "error": new_error,
            "old_error": old_error,
            "hunting": hunting,
            "step": self._step_count,
        }
        return state, reward, terminated, truncated, info
