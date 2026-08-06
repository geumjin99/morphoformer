"""Swin-MTL baseline: shared single-stage Swin encoder + two independent
regression heads, with none of MorphoFormer's cross-task modules.

This is the architecture behind the paper's Swin-MTL reference row. It was
trained under the first-generation codebase (see docs/reproducibility.md for
what that means for re-running it).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.swin_transformer import SwinTransformer
from typing import Tuple, Optional


class SwinMTL(nn.Module):
    """Swin Transformer backbone with center-crop pooling and dual regression heads."""

    def __init__(
        self,
        img_size: int = 50,
        in_chans: int = 8,
        patch_size: int = 2,
        embed_dim: int = 96,
        depths: Tuple[int, ...] = (2,),
        num_heads: Tuple[int, ...] = (3,),
        window_size: int = 5,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        use_uncertainty: bool = True,
        center_size: int = 5,
    ):
        super().__init__()

        self.use_uncertainty = use_uncertainty
        self.center_size = center_size

        if self.use_uncertainty:
            self.log_sigma_h = nn.Parameter(torch.zeros(1))
            self.log_sigma_f = nn.Parameter(torch.zeros(1))

        self.backbone = SwinTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=nn.LayerNorm,
        )

        out_dim = self.backbone.num_features
        self.head_h = nn.Linear(out_dim, 1)
        self.head_f = nn.Linear(out_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        feat = self.backbone.forward_features(x)  # [B, H', W', C]
        B, H, W, C = feat.shape

        # Extract center region
        start = (H - self.center_size) // 2
        center_feat = feat[:, start:start+self.center_size,
                          start:start+self.center_size, :]

        center_mean = center_feat.mean(dim=(1, 2))  # [B, C]

        bh_raw = self.head_h(center_mean)
        bf_raw = self.head_f(center_mean)

        bh_pred = F.relu(bh_raw)
        bf_pred = torch.sigmoid(bf_raw)

        if self.use_uncertainty:
            return bh_pred, bf_pred, self.log_sigma_h, self.log_sigma_f
        else:
            return bh_pred, bf_pred


class SwinMTLLarge(SwinMTL):
    """Enlarged SwinMTL variant (doubled embed_dim and depth)."""

    def __init__(self, img_size=50, in_chans=8, **kwargs):
        super().__init__(
            img_size=img_size,
            in_chans=in_chans,
            embed_dim=128,
            depths=(2, 2),
            num_heads=(4, 8),
            **kwargs
        )


def build_swin_mtl(
    variant: str = 'base',
    in_chans: int = 8,
    patch_grid_size: int = 5,
    **kwargs
) -> SwinMTL:
    """Factory for SwinMTL models. variant: 'base' or 'large'."""
    img_size = patch_grid_size * 10

    if variant == 'base':
        return SwinMTL(
            img_size=img_size,
            in_chans=in_chans,
            embed_dim=96,
            depths=(2,),
            num_heads=(3,),
            **kwargs
        )
    elif variant == 'large':
        return SwinMTLLarge(
            img_size=img_size,
            in_chans=in_chans,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")
