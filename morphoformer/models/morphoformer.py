"""
MorphoFormer — Lightweight Swin Transformer with Adaptive Modality Gating,
Multi-Scale Morphology Pooling, and BF-Guided Task Decoding.

Full pipeline:
  Input [B, C, H, W]
      │
  AdaptiveModalityGatingEncoder (AMGE)
      │   Per-modality SE re-weighting (SAR / Optical / DEM)
      │
  SwinTransformer backbone
      │   forward_features() → [B, H', W', C]
      │
  MultiScaleMorphologyPool (MSMP)
      │   3×3 / 5×5 / 9×9 center crops → scale-attention fusion → [B, C]
      │
  BFGuidedTaskDecoder (BGTD)
      ├── BF branch → bf_pred, morphology context
      ├── BH branch (BF-gated) → bh_pred
      └── Consistency branch → bh_from_bf  (for MCL auxiliary loss)
      │
  Uncertainty params  log_σ_h, log_σ_f
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.swin_transformer import SwinTransformer
from typing import Tuple, Sequence

from .components import (
    AdaptiveModalityGatingEncoder,
    MultiScaleMorphologyPool,
    BFGuidedTaskDecoder,
    SimpleTaskDecoder,
)


class MorphoFormer(nn.Module):
    """
    MorphoFormer: joint building height (BH) and footprint (BF) estimation
    from multi-modal Sentinel-1 / Sentinel-2 / SRTM-DEM imagery.

    Args:
        img_size:       Input spatial size in pixels (= patch_grid_size × 10).
        in_chans:       Total input channels including mask channel.
        sar_ch:         SAR input channels (default 2: VH, VV).
        opt_ch:         Optical input channels (default 4: B,G,R,NIR).
        dem_ch:         DEM input channels (default 1).
        patch_size:     Swin patch size (default 2).
        embed_dim:      Swin embedding dimension (default 96 for base).
        depths:         Swin stage depths (default (2,) for lightweight base).
        num_heads:      Attention heads per stage.
        window_size:    Swin window size.
        center_sizes:   Center crop sizes for multi-scale pooling.
        use_uncertainty: Enable Kendall-style learnable uncertainty weighting.
        use_amge:       Enable Adaptive Modality Gating Encoder.
    """

    def __init__(
        self,
        img_size: int = 90,
        in_chans: int = 8,
        sar_ch: int = 2,
        opt_ch: int = 4,
        dem_ch: int = 1,
        patch_size: int = 10,
        embed_dim: int = 96,
        depths: Tuple[int, ...] = (2,),
        num_heads: Tuple[int, ...] = (3,),
        window_size: int = 9,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        center_sizes: Sequence[int] = (3, 5, 9),
        use_uncertainty: bool = True,
        use_amge: bool = True,
        use_bgtd: bool = True,
    ):
        super().__init__()
        self.use_uncertainty = use_uncertainty

        if use_uncertainty:
            self.log_sigma_h = nn.Parameter(torch.zeros(1))
            self.log_sigma_f = nn.Parameter(torch.zeros(1))

        # ── 1. Adaptive Modality Gating Encoder ──
        self.amge = (
            AdaptiveModalityGatingEncoder(sar_ch=sar_ch, opt_ch=opt_ch, dem_ch=dem_ch)
            if use_amge else nn.Identity()
        )

        # ── 2. Swin backbone ──
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

        out_dim = self.backbone.num_features  # 96 for base config

        # ── 3. Multi-Scale Morphology Pooling ──
        self.msmp = MultiScaleMorphologyPool(out_dim, center_sizes=center_sizes)

        # ── 4. Task Decoder ──
        self.bgtd = (
            BFGuidedTaskDecoder(out_dim) if use_bgtd
            else SimpleTaskDecoder(out_dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            x: [B, in_chans, H, W]
        Returns (use_uncertainty=True):
            bh_pred, bf_pred, log_sigma_h, log_sigma_f, bh_from_bf
        Returns (use_uncertainty=False):
            bh_pred, bf_pred, bh_from_bf
        """
        x = self.amge(x)                                    # modality re-weighting
        feat_map = self.backbone.forward_features(x)        # [B, H', W', C]
        feat = self.msmp(feat_map)                          # [B, C]
        bh_pred, bf_pred, bh_from_bf = self.bgtd(feat)     # [B,1], [B,1], [B,1]

        if self.use_uncertainty:
            return bh_pred, bf_pred, self.log_sigma_h, self.log_sigma_f, bh_from_bf
        return bh_pred, bf_pred, bh_from_bf


class MorphoFormerLarge(MorphoFormer):
    """Larger variant with doubled depth and embedding dimension."""

    def __init__(self, img_size: int = 90, in_chans: int = 8, **kwargs):
        super().__init__(
            img_size=img_size,
            in_chans=in_chans,
            embed_dim=128,
            depths=(2, 2),
            num_heads=(4, 8),
            **kwargs,
        )


def build_morphoformer(
    variant: str = 'base',
    in_chans: int = 8,
    patch_grid_size: int = 9,
    patch_size: int = 10,
    center_sizes: Sequence[int] = (3, 5, 9),
    use_bgtd: bool = True,
    **kwargs,
) -> MorphoFormer:
    """
    Factory for MorphoFormer.

    Args:
        variant:         'base' or 'large'
        patch_grid_size: Number of 10×10 px patches per side (input = size×10 px).
                         Recommend 9 for richer spatial context.
        patch_size:      Swin patch embedding stride. Use 10 (default) for one token
                         per 100 m cell; use 2 for finer 20 m features (45×45 tokens).
        center_sizes:    Crop sizes for MSMP (must all be ≤ feature map spatial size).
    """
    img_size = patch_grid_size * 10
    cls = MorphoFormerLarge if variant == 'large' else MorphoFormer
    return cls(
        img_size=img_size,
        in_chans=in_chans,
        patch_size=patch_size,
        center_sizes=list(center_sizes),
        use_bgtd=use_bgtd,
        **kwargs,
    )
