"""CNN / ViT baseline architectures reported in the MorphoFormer paper.

All baselines share MorphoFormer's input tensor, split, loss and evaluation
pipeline; only the encoder differs. Each returns the same tuple contract as
MorphoFormer with ``use_uncertainty=False``-style decoding: (bh, bf) or
(bh, bf, log_sigma_h, log_sigma_f).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import timm


class ResNetMTL(nn.Module):
    """ResNet-18 backbone with dual regression heads."""

    def __init__(self, in_chans: int = 8, use_uncertainty: bool = True, pretrained: bool = False):
        super().__init__()
        self.use_uncertainty = use_uncertainty

        if self.use_uncertainty:
            self.log_sigma_h = nn.Parameter(torch.zeros(1))
            self.log_sigma_f = nn.Parameter(torch.zeros(1))

        self.backbone = timm.create_model(
            'resnet18', pretrained=pretrained, in_chans=in_chans,
            num_classes=0, global_pool='avg',
        )

        out_dim = self.backbone.num_features
        self.head_h = nn.Linear(out_dim, 1)
        self.head_f = nn.Linear(out_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        feat = self.backbone(x)
        bh_pred = F.relu(self.head_h(feat))
        bf_pred = torch.sigmoid(self.head_f(feat))

        if self.use_uncertainty:
            return bh_pred, bf_pred, self.log_sigma_h, self.log_sigma_f
        return bh_pred, bf_pred


class SENetMTL(nn.Module):
    """SE-ResNet-18 backbone with channel attention."""

    def __init__(self, in_chans: int = 8, use_uncertainty: bool = True, pretrained: bool = False):
        super().__init__()
        self.use_uncertainty = use_uncertainty

        if self.use_uncertainty:
            self.log_sigma_h = nn.Parameter(torch.zeros(1))
            self.log_sigma_f = nn.Parameter(torch.zeros(1))

        self.backbone = timm.create_model(
            'seresnet18', pretrained=pretrained, in_chans=in_chans,
            num_classes=0, global_pool='avg',
        )

        out_dim = self.backbone.num_features
        self.head_h = nn.Linear(out_dim, 1)
        self.head_f = nn.Linear(out_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        feat = self.backbone(x)
        bh_pred = F.relu(self.head_h(feat))
        bf_pred = torch.sigmoid(self.head_f(feat))

        if self.use_uncertainty:
            return bh_pred, bf_pred, self.log_sigma_h, self.log_sigma_f
        return bh_pred, bf_pred




class ViTMTL(nn.Module):
    """Vision Transformer (ViT-Small) backbone with dual regression heads."""

    def __init__(self, in_chans: int = 8, img_size: int = 50,
                 use_uncertainty: bool = True, pretrained: bool = False):
        super().__init__()
        self.use_uncertainty = use_uncertainty

        if self.use_uncertainty:
            self.log_sigma_h = nn.Parameter(torch.zeros(1))
            self.log_sigma_f = nn.Parameter(torch.zeros(1))

        self.backbone = timm.create_model(
            'vit_small_patch8_224', pretrained=pretrained, in_chans=in_chans,
            img_size=img_size, num_classes=0, global_pool='avg',
        )

        out_dim = self.backbone.num_features
        self.head_h = nn.Linear(out_dim, 1)
        self.head_f = nn.Linear(out_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        feat = self.backbone(x)
        bh_pred = F.relu(self.head_h(feat))
        bf_pred = torch.sigmoid(self.head_f(feat))

        if self.use_uncertainty:
            return bh_pred, bf_pred, self.log_sigma_h, self.log_sigma_f
        return bh_pred, bf_pred


def build_baseline(name: str, in_chans: int = 8, **kwargs) -> nn.Module:
    """Factory for baseline models."""
    models = {
        'resnet': ResNetMTL,
        'senet': SENetMTL,
        'vit': ViTMTL,
    }

    if name not in models:
        raise ValueError(f"Unknown baseline: {name}. Choose from {list(models.keys())}")

    return models[name](in_chans=in_chans, **kwargs)
