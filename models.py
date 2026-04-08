"""
models.py – Actor-Critic Network with MobileNetV2 Backbone
============================================================

Network architecture for the DRL-based autofocus system:

  1. **Feature extractor** (MobileNetV2):
     Takes the dual-pixel image (2-channel input) and produces a
     spatial feature map, which is then globally average-pooled.

  2. **Position & temperature encoding fusion**:
     The CNN feature vector is concatenated with Lens-PE, RoI-PE, and
     a temperature scalar in the MLP stage (not in the CNN stage, as
     recommended).

  3. **Output heads**:
     • *Pretrain head* – Outputs a probability distribution over all
       ``NUM_FOCUS_POSITIONS`` (70) absolute positions (ordinal regression).
     • *Actor head (RL)* – Outputs a probability distribution over
       ``ACTION_DIM`` relative movement offsets (e.g. [-10, +10] → 21 bins).
     • *Critic head (RL)* – Outputs a scalar state-value V(s).

Public classes
--------------
``AutofocusActorCritic``
    Full model with switchable pretrain / RL heads.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2

from dataset import NUM_FOCUS_POSITIONS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTION_RANGE = 10               # relative offset in [-ACTION_RANGE, ACTION_RANGE]
ACTION_DIM = 2 * ACTION_RANGE + 1  # 21 bins for action space

# ---------------------------------------------------------------------------
# Helper – adapt first conv for 2-channel input
# ---------------------------------------------------------------------------

def _adapt_first_conv(model: nn.Module, in_channels: int = 2) -> nn.Module:
    """Replace the first convolution layer of a MobileNetV2 so that it
    accepts ``in_channels`` instead of the default 3 (RGB).

    The existing weights are averaged over the channel dimension and
    tiled to the new number of channels to provide a reasonable
    initialisation.
    """
    old_conv = model.features[0][0]  # Conv2dNormActivation → Conv2d
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    with torch.no_grad():
        # Average over 3 RGB channels → (out, 1, kH, kW), tile
        w = old_conv.weight.mean(dim=1, keepdim=True)
        new_conv.weight.copy_(w.repeat(1, in_channels, 1, 1))
        if old_conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    model.features[0][0] = new_conv
    return model


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class AutofocusActorCritic(nn.Module):
    """Actor-Critic model for autofocus.

    Parameters
    ----------
    pe_dim : int
        Dimensionality of each positional encoding vector (Lens-PE and
        RoI-PE).  Default 16.
    image_channels : int
        Number of input image channels (2 for dual-pixel L, R).
    pretrain : bool
        If ``True`` (default), the model exposes the pretrain head
        (``NUM_FOCUS_POSITIONS`` classes with ordinal regression).
        Set to ``False`` for the RL phase to use the actor/critic heads.
    freeze_backbone : bool
        If ``True``, freeze the CNN backbone parameters (useful after
        switching from pretrain to RL).
    """

    def __init__(
        self,
        pe_dim: int = 16,
        image_channels: int = 2,
        pretrain: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.pe_dim = pe_dim
        self.pretrain = pretrain

        # ---- 1. CNN backbone ----
        backbone = mobilenet_v2(weights=None)
        backbone = _adapt_first_conv(backbone, in_channels=image_channels)
        # Remove the original classifier
        self.features = backbone.features           # (B, 1280, H', W')
        self.pool = nn.AdaptiveAvgPool2d(1)          # → (B, 1280)

        cnn_feat_dim = 1280  # MobileNetV2 last channel dim

        # ---- 2. MLP fusion (CNN feat + Lens-PE + RoI-PE + temperature) ----
        fusion_in = cnn_feat_dim + pe_dim + pe_dim + 1  # +1 for temperature
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_in, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )

        # ---- 3. Output heads ----
        # Pretrain: ordinal regression over NUM_FOCUS_POSITIONS
        self.pretrain_head = nn.Linear(256, NUM_FOCUS_POSITIONS)

        # RL Actor: distribution over ACTION_DIM relative movements
        self.actor_head = nn.Linear(256, ACTION_DIM)

        # RL Critic: scalar value
        self.critic_head = nn.Linear(256, 1)

        if freeze_backbone:
            self._freeze_backbone()

    # ------------------------------------------------------------------
    def _freeze_backbone(self) -> None:
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.features.parameters():
            p.requires_grad = True

    # ------------------------------------------------------------------
    def _extract_features(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Run the image through the CNN and fuse with PE + temperature.

        Parameters
        ----------
        state : dict with keys "image", "lens_pe", "roi_pe", "temperature"

        Returns
        -------
        Tensor of shape (B, 256)
        """
        x = state["image"]                  # (B, 2, H, W)
        x = self.features(x)                # (B, 1280, H', W')
        x = self.pool(x).flatten(1)          # (B, 1280)

        lens_pe = state["lens_pe"]           # (B, pe_dim)
        roi_pe = state["roi_pe"]             # (B, pe_dim)
        temp = state["temperature"]          # (B, 1)

        fused = torch.cat([x, lens_pe, roi_pe, temp], dim=1)
        return self.fusion_mlp(fused)        # (B, 256)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Returns
        -------
        dict with:
            In pretrain mode:
                ``"logits"`` – (B, NUM_FOCUS_POSITIONS)
                ``"probs"``  – softmax probabilities
            In RL mode:
                ``"action_logits"`` – (B, ACTION_DIM)
                ``"action_probs"``  – softmax probabilities over actions
                ``"value"``         – (B, 1) state value
        """
        feat = self._extract_features(state)

        if self.pretrain:
            logits = self.pretrain_head(feat)
            probs = F.softmax(logits, dim=-1)
            return {"logits": logits, "probs": probs}
        else:
            action_logits = self.actor_head(feat)
            action_probs = F.softmax(action_logits, dim=-1)
            value = self.critic_head(feat)
            return {
                "action_logits": action_logits,
                "action_probs": action_probs,
                "value": value,
            }

    # ------------------------------------------------------------------
    # Convenience methods for RL
    # ------------------------------------------------------------------
    def get_action_and_value(
        self,
        state: Dict[str, torch.Tensor],
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action and compute log-prob, entropy, and value.

        Parameters
        ----------
        state : dict
        action : optional pre-selected action indices (B,)

        Returns
        -------
        action      : (B,) sampled or given action index
        log_prob    : (B,) log-probability of the action
        entropy     : (B,) entropy of the action distribution
        value       : (B,) state value
        """
        out = self.forward(state)
        dist = torch.distributions.Categorical(probs=out["action_probs"])
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = out["value"].squeeze(-1)
        return action, log_prob, entropy, value

    def get_value(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the state value V(s).  Shape (B,)."""
        feat = self._extract_features(state)
        return self.critic_head(feat).squeeze(-1)

    # ------------------------------------------------------------------
    # Action ↔ offset conversion
    # ------------------------------------------------------------------
    @staticmethod
    def action_to_offset(action_index: torch.Tensor) -> torch.Tensor:
        """Convert action index ∈ [0, ACTION_DIM) to lens offset ∈ [-ACTION_RANGE, ACTION_RANGE]."""
        return action_index - ACTION_RANGE

    @staticmethod
    def offset_to_action(offset: torch.Tensor) -> torch.Tensor:
        """Convert lens offset ∈ [-ACTION_RANGE, ACTION_RANGE] to action index ∈ [0, ACTION_DIM)."""
        action = offset + ACTION_RANGE
        return action.clamp(0, ACTION_DIM - 1)

    # ------------------------------------------------------------------
    # Weight loading helper for pretrain → RL transition
    # ------------------------------------------------------------------
    def load_pretrained_weights(
        self,
        ckpt_path: str,
        freeze_backbone: bool = True,
    ) -> None:
        """Load a Phase-1 (pretrain) checkpoint and switch to RL mode.

        The CNN backbone and fusion MLP weights are copied.  The pretrain
        head is discarded and the actor / critic heads are randomly
        initialised (or optionally initialised from the pretrain head).
        """
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        # Load backbone + fusion weights (ignore head mismatches)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"[load_pretrained_weights] missing={len(missing)}, "
              f"unexpected={len(unexpected)}")

        self.pretrain = False
        if freeze_backbone:
            self._freeze_backbone()
