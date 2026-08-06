"""
MorphoFormer Novel Components.

Three architecture modules, each independently ablatable:

  AMGE  — Adaptive Modality Gating Encoder
           Per-modality SE-style channel attention before the Swin backbone.
           Physical motivation: SAR (double-bounce geometry), Optical (rooftop
           spectral signature), and DEM (terrain baseline) encode fundamentally
           different cues; their relative importance is scene-dependent.

  MSMP  — Multi-Scale Morphology Pooling
           Crops the Swin token map at 3×3 / 5×5 / 9×9 *tokens* around the
           centre, applies scale-specific projections, and fuses via learned
           attention. The crops are expressed in tokens, so the ground extent
           they cover depends on ``patch_size`` — see the class docstring of
           MultiScaleMorphologyPool for the conversion.

  BGTD  — BF-Guided Task Decoder
           Two-branch decoder where the BF branch first encodes a 'morphology
           context', which then gates the BH branch via cross-attention.
           Physical motivation: building footprint ratio is a compact proxy for
           urban morphology type (high-rise dense vs suburban sparse), which
           determines the plausible building height range (Floor Area Ratio).
           Also exposes a secondary head (bh_from_bf) for the consistency loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Sequence


# ---------------------------------------------------------------------------
# Adaptive Modality Gating Encoder (AMGE)
# ---------------------------------------------------------------------------

class ModalitySEGate(nn.Module):
    """Squeeze-and-Excitation gate for a single modality group.

    Global average pool → FC squeeze → ReLU → FC excite → Sigmoid → scale.
    Reduction ratio 'r' controls capacity; default 2 is lightweight.
    """

    def __init__(self, channels: int, reduction: int = 2):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.gate = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        s = x.mean(dim=(2, 3))       # [B, C]
        w = self.gate(s).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        return x * w


class AdaptiveModalityGatingEncoder(nn.Module):
    """
    Apply independent SE gating to SAR, Optical, and DEM channel groups
    before concatenation, then restore the original channel layout.

    Assumes input channel order: [SAR(sar_ch), Optical(opt_ch), DEM(dem_ch), Mask(1)].
    """

    def __init__(
        self,
        sar_ch: int = 2,
        opt_ch: int = 4,
        dem_ch: int = 1,
    ):
        super().__init__()
        self.sar_ch = sar_ch
        self.opt_ch = opt_ch
        self.dem_ch = dem_ch

        self.sar_gate = ModalitySEGate(sar_ch)
        self.opt_gate = ModalitySEGate(opt_ch)
        self.dem_gate = ModalitySEGate(dem_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, sar_ch + opt_ch + dem_ch + 1, H, W]
        Returns:
            x_gated: same shape — modality channels are re-weighted, mask untouched
        """
        s = self.sar_ch
        o = self.opt_ch
        d = self.dem_ch

        sar   = self.sar_gate(x[:, :s])
        opt   = self.opt_gate(x[:, s:s + o])
        dem   = self.dem_gate(x[:, s + o:s + o + d])
        mask  = x[:, s + o + d:]                # pass through unchanged

        return torch.cat([sar, opt, dem, mask], dim=1)


# ---------------------------------------------------------------------------
# Multi-Scale Morphology Pooling (MSMP)
# ---------------------------------------------------------------------------

class MultiScaleMorphologyPool(nn.Module):
    """
    Extract centre features from the Swin token map at multiple spatial scales,
    then fuse them with a learned scale-attention gate.

    ``center_sizes`` is measured in **tokens**, not in 100 m label cells. The
    ground extent of each crop therefore depends on the patch-embedding stride:
    for a 9x9-cell (900 m) input window the token grid is ``90 / patch_size``
    on a side, so one token spans ``900 / (90 / patch_size)`` metres.

        patch_size=10  ->  9x9 tokens,  1 token = 100 m
                           crops 3/5/9 tokens =  300 /  500 /  900 m
                           (the crops span the whole token map)

        patch_size=2   -> 45x45 tokens, 1 token =  20 m
                           crops 3/5/9 tokens =   60 /  100 /  180 m
                           (the crops read the central 9x9 of 45x45 tokens,
                            i.e. 4 % of the map; context outside that radius
                            reaches the pooled vector only through the Swin
                            blocks' windowed and shifted-window attention)

    The configuration used for the paper's main results is ``patch_size=2``.

    Fusion: each scale gets a linear projection, then a softmax attention gate
    determines per-sample scale importance before summation.
    """

    def __init__(self, in_dim: int, center_sizes: Sequence[int] = (3, 5, 9)):
        super().__init__()
        self.center_sizes = list(center_sizes)
        n = len(center_sizes)

        # Scale-specific projections (allow scale-specialised features)
        self.scale_projs = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, in_dim), nn.LayerNorm(in_dim))
            for _ in center_sizes
        ])

        # Scale attention: learns to weight scales based on content
        self.scale_attn = nn.Sequential(
            nn.Linear(in_dim * n, n),
            nn.Softmax(dim=-1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: [B, H, W, C]  (Swin output — spatial × channel)
        Returns:
            fused: [B, C]
        """
        B, H, W, C = feat.shape
        scale_feats = []

        for cs, proj in zip(self.center_sizes, self.scale_projs):
            sh = max((H - cs) // 2, 0)
            sw = max((W - cs) // 2, 0)
            crop = feat[:, sh:sh + cs, sw:sw + cs, :]  # [B, cs, cs, C]
            pooled = crop.mean(dim=(1, 2))              # [B, C]
            scale_feats.append(proj(pooled))            # [B, C]

        stacked = torch.stack(scale_feats, dim=1)       # [B, n, C]
        concat  = torch.cat(scale_feats, dim=-1)        # [B, n*C]
        attn    = self.scale_attn(concat).unsqueeze(-1) # [B, n, 1]

        fused = (stacked * attn).sum(dim=1)             # [B, C]
        return fused


# ---------------------------------------------------------------------------
# BF-Guided Task Decoder (BGTD)
# ---------------------------------------------------------------------------

class BFGuidedTaskDecoder(nn.Module):
    """
    Geometry-Aware Cross-Task Decoder.

    Two-stage decoding exploiting the physical BF → BH relationship:

    Stage 1 — BF branch:
        shared_feat → bf_encoder → bf_feat → head_f → bf_pred
        bf_feat also drives a 'morphology context' via cross_proj.

    Stage 2 — BH branch (BF-guided):
        shared_feat → bh_encoder → bh_feat
        morphology context gates bh_feat: gate = σ(W[bh_feat ‖ morph_ctx])
        bh_feat_guided = bh_feat ⊙ gate + morph_ctx ⊙ (1 − gate)
        bh_feat_guided → head_h → bh_pred

    Secondary head (for consistency loss):
        bf_feat → consistency_head → bh_from_bf
        This secondary BH predictor, supervised by ground-truth BH, forces
        the BF branch to encode height-correlated morphology representations.

    Args:
        in_dim: Dimension of the shared feature vector from MSMP.
        hidden_dim: Dimension of task-specific features (default in_dim // 2).
    """

    def __init__(self, in_dim: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or in_dim // 2

        # ── BF branch ──
        self.bf_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.head_f = nn.Linear(hidden_dim, 1)

        # Morphology context projection (from BF features)
        self.morph_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # ── BH branch ──
        self.bh_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Cross-task gate: [bh_feat ‖ morph_ctx] → gate weights
        self.cross_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        self.head_h = nn.Linear(hidden_dim, 1)

        # ── Secondary consistency head (on BF features) ──
        self.consistency_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.ReLU(),   # BH is non-negative
        )

    def forward(
        self, feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            feat: shared feature [B, in_dim]
        Returns:
            bh_pred:    [B, 1]
            bf_pred:    [B, 1]
            bh_from_bf: [B, 1]  secondary BH estimate from BF features only
        """
        # ── BF branch ──
        bf_feat   = self.bf_encoder(feat)                   # [B, D]
        bf_pred   = torch.sigmoid(self.head_f(bf_feat))     # [B, 1]
        morph_ctx = self.morph_proj(bf_feat)                # [B, D]

        # ── BH branch (BF-guided) ──
        bh_feat   = self.bh_encoder(feat)                   # [B, D]
        gate      = self.cross_gate(
            torch.cat([bh_feat, morph_ctx], dim=-1)
        )                                                   # [B, D]
        bh_guided = bh_feat * gate + morph_ctx * (1 - gate)# [B, D]
        bh_pred   = F.relu(self.head_h(bh_guided))         # [B, 1]

        # ── Secondary consistency prediction ──
        bh_from_bf = self.consistency_head(bf_feat)         # [B, 1]

        return bh_pred, bf_pred, bh_from_bf


# ---------------------------------------------------------------------------
# Simple Independent Task Decoder (ablation: no cross-task gating)
# ---------------------------------------------------------------------------

class SimpleTaskDecoder(nn.Module):
    """
    Independent BH and BF heads with no cross-task coupling: two separate MLP
    towers operating directly on the shared feature vector, matching the
    Swin-MTL baseline decoder.

    Used as the ``--no-bgtd`` ablation. Note that this decoder has no
    ``bh_from_bf`` output, so selecting it also switches off the MCL
    consistency term (see morphoformer/losses/mcl_loss.py, which requires
    ``bh_from_bf is not None``). ``--no-bgtd`` therefore removes BGTD *and*
    MCL together; use ``--lambda-consist 0`` for a decoder-identical, loss-only
    ablation of MCL alone.
    """

    def __init__(self, in_dim: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or in_dim // 2

        self.bh_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.bf_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.head_h = nn.Linear(hidden_dim, 1)
        self.head_f = nn.Linear(hidden_dim, 1)

    def forward(
        self, feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, None]:
        bh_pred = F.relu(self.head_h(self.bh_mlp(feat)))       # [B, 1]
        bf_pred = torch.sigmoid(self.head_f(self.bf_mlp(feat))) # [B, 1]
        return bh_pred, bf_pred, None
