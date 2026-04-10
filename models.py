"""
models.py – Paper-faithful actor-critic network for AF_DRL.
"""

from __future__ import annotations

import sys
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

try:
    from torchvision.models import MobileNet_V2_Weights
except ImportError:  # pragma: no cover
    MobileNet_V2_Weights = None

from dataset import NUM_FOCUS_POSITIONS

# The paper expands the action space to [-kmax, kmax], where kmax is the
# maximum discrete focus index. With 70 focus positions indexed as 0..69,
# this yields 139 relative-movement bins.
ACTION_RANGE = NUM_FOCUS_POSITIONS - 1
ACTION_DIM = 2 * ACTION_RANGE + 1



def _build_backbone(imagenet_pretrained: bool) -> nn.Module:
    weights = None
    if imagenet_pretrained and MobileNet_V2_Weights is not None:
        weights = MobileNet_V2_Weights.DEFAULT
    try:
        return mobilenet_v2(weights=weights)
    except (RuntimeError, ValueError, TypeError):
        if weights is not None:
            print("Warning: failed to load ImageNet-pretrained MobileNetV2 weights; falling back to random init.", file=sys.stderr)
        return mobilenet_v2(weights=None)



def _adapt_first_conv(model: nn.Module, in_channels: int = 2) -> nn.Module:
    old_conv = model.features[0][0]
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    with torch.no_grad():
        weight = old_conv.weight.mean(dim=1, keepdim=True)
        new_conv.weight.copy_(weight.repeat(1, in_channels, 1, 1))
        if old_conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    model.features[0][0] = new_conv
    return model


class AutofocusActorCritic(nn.Module):
    def __init__(
        self,
        pe_dim: int = 16,
        image_channels: int = 2,
        freeze_backbone: bool = False,
        imagenet_pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.pe_dim = pe_dim

        backbone = _build_backbone(imagenet_pretrained=imagenet_pretrained)
        backbone = _adapt_first_conv(backbone, in_channels=image_channels)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        fusion_in = 1280 + pe_dim + pe_dim + 1
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_in, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )
        self.actor_head = nn.Linear(256, ACTION_DIM)
        self.critic_head = nn.Linear(256, 1)

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.features.parameters():
            param.requires_grad = True

    def _extract_features(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        image = state["image"]
        feat = self.features(image)
        feat = self.pool(feat).flatten(1)
        fused = torch.cat(
            [feat, state["lens_pe"], state["roi_pe"], state["temperature"]],
            dim=1,
        )
        return self.fusion_mlp(fused)

    def actor_logits(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.actor_head(self._extract_features(state))

    def actor_distribution(self, state: Dict[str, torch.Tensor]) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.actor_logits(state))

    def forward(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        feat = self._extract_features(state)
        actor_logits = self.actor_head(feat)
        critic_value = self.critic_head(feat).squeeze(-1)
        return {
            "action_logits": actor_logits,
            "action_probs": actor_logits.softmax(dim=-1),
            "value": critic_value,
        }

    def get_action_and_value(
        self,
        state: Dict[str, torch.Tensor],
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.forward(state)
        dist = torch.distributions.Categorical(logits=out["action_logits"])
        if action is None:
            action = out["action_logits"].argmax(dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, out["value"]

    def get_value(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.forward(state)["value"]

    @staticmethod
    def action_to_offset(action_index: torch.Tensor) -> torch.Tensor:
        return action_index - ACTION_RANGE

    @staticmethod
    def offset_to_action(offset: torch.Tensor) -> torch.Tensor:
        return (offset + ACTION_RANGE).clamp(0, ACTION_DIM - 1)

    def load_phase1_weights(
        self,
        ckpt_path: str,
        freeze_backbone: bool = False,
        load_actor_head: bool = True,
    ) -> None:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if not load_actor_head:
            state_dict = {
                key: value
                for key, value in state_dict.items()
                if not key.startswith("actor_head.")
            }
        self.load_state_dict(state_dict, strict=False)
        if freeze_backbone:
            self.freeze_backbone()

    def load_training_checkpoint(self, ckpt_path: str) -> Dict[str, object]:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        self.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint
