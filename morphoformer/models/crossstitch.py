"""
Cross-Stitch MTL baseline for MorphoFormer.

This is the *symmetric, undirected* task-interaction counterpart to MorphoFormer's
directed BF->BH coupling. It is deliberately built on the **identical encoder**
(AMGE + single-stage Swin + MSMP) so that the only difference from the full model
is the cross-task coupling mechanism in the decoder:

  MorphoFormer (full)   : BGTD  — directed, dual-level (feature-gate + MCL output
                                   consistency), prior-anchored BF -> BH.
  Cross-Stitch (this)   : symmetric 2x2 learned mixing of the two task towers at
                          every layer (Misra et al., CVPR 2016); no direction
                          prior, no consistency term.

Comparing the two under the same backbone, receptive field, data and split
isolates the effect of the directed, prior-anchored coupling against generic
symmetric task interaction.

The loss is the plain uncertainty-weighted MTL loss (MorphologyConsistentMTLLoss
with bh_from_bf=None, so the consistency term is never active); training, data and
evaluation reuse the exact MorphoFormer pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Sequence

from .morphoformer import MorphoFormer


# ---------------------------------------------------------------------------
# Cross-stitch unit (Misra et al., CVPR 2016)
# ---------------------------------------------------------------------------

class CrossStitchUnit(nn.Module):
    """Learned symmetric linear combination of two task activations.

        out_a = a_aa * x_a + a_ab * x_b
        out_b = a_ba * x_a + a_bb * x_b

    The 2x2 matrix is shared across feature channels (the standard scalar
    cross-stitch unit) and initialised to favour task identity:
    [[init_same, init_cross], [init_cross, init_same]].

    Crucially the coupling is *symmetric and undirected*: nothing fixes which
    task informs which — the contrast with MorphoFormer's fixed BF->BH direction.
    """

    def __init__(self, init_same: float = 0.9, init_cross: float = 0.1):
        super().__init__()
        self.alpha = nn.Parameter(
            torch.tensor([[init_same, init_cross],
                          [init_cross, init_same]])
        )

    def forward(
        self, x_a: torch.Tensor, x_b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        a = self.alpha
        out_a = a[0, 0] * x_a + a[0, 1] * x_b
        out_b = a[1, 0] * x_a + a[1, 1] * x_b
        return out_a, out_b


class CrossStitchDecoder(nn.Module):
    """Two task-specific MLP towers with a cross-stitch unit after each layer.

    Mirrors BGTD's depth/width (hidden = in_dim // 2, two transformation layers)
    so that capacity is comparable; the only structural difference is symmetric
    cross-stitch mixing in place of the directed BF-gate + consistency head.
    """

    def __init__(self, in_dim: int, hidden_dim: int = None, n_layers: int = 2):
        super().__init__()
        hidden_dim = hidden_dim or in_dim // 2
        self.n_layers = n_layers

        def block(d_in, d_out):
            return nn.Sequential(
                nn.Linear(d_in, d_out),
                nn.LayerNorm(d_out),
                nn.GELU(),
            )

        dims = [in_dim] + [hidden_dim] * n_layers
        self.bh_layers = nn.ModuleList(
            [block(dims[i], dims[i + 1]) for i in range(n_layers)]
        )
        self.bf_layers = nn.ModuleList(
            [block(dims[i], dims[i + 1]) for i in range(n_layers)]
        )
        self.cross_stitch = nn.ModuleList(
            [CrossStitchUnit() for _ in range(n_layers)]
        )

        self.head_h = nn.Linear(hidden_dim, 1)
        self.head_f = nn.Linear(hidden_dim, 1)

    def forward(
        self, feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_bh = feat
        h_bf = feat
        for i in range(self.n_layers):
            h_bh = self.bh_layers[i](h_bh)
            h_bf = self.bf_layers[i](h_bf)
            h_bh, h_bf = self.cross_stitch[i](h_bh, h_bf)

        bh_pred = F.relu(self.head_h(h_bh))         # BH non-negative
        bf_pred = torch.sigmoid(self.head_f(h_bf))  # BF in [0, 1]
        return bh_pred, bf_pred


# ---------------------------------------------------------------------------
# Cross-Stitch MTL model (same encoder as MorphoFormer)
# ---------------------------------------------------------------------------

class CrossStitchMTL(MorphoFormer):
    """MorphoFormer encoder (AMGE + Swin + MSMP) with a symmetric cross-stitch
    decoder replacing BGTD. Returns the 4-tuple consumed by the 'legacy
    4-output' branch of MorphoFormerLitModule when uncertainty is enabled.
    """

    def __init__(self, *args, **kwargs):
        # Build the full encoder via MorphoFormer (use_bgtd irrelevant — replaced).
        super().__init__(*args, **kwargs)
        # Drop the directed decoder; install the symmetric cross-stitch one.
        del self.bgtd
        out_dim = self.backbone.num_features
        self.decoder = CrossStitchDecoder(out_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x = self.amge(x)
        feat_map = self.backbone.forward_features(x)
        feat = self.msmp(feat_map)
        bh_pred, bf_pred = self.decoder(feat)

        if self.use_uncertainty:
            # 4-tuple: handled by LitModule 'legacy 4-output' branch; the v2 loss
            # gets bh_from_bf=None so the MCL consistency term stays inactive.
            return bh_pred, bf_pred, self.log_sigma_h, self.log_sigma_f
        return bh_pred, bf_pred


def build_crossstitch(
    variant: str = 'base',
    in_chans: int = 8,
    patch_grid_size: int = 9,
    patch_size: int = 10,
    center_sizes: Sequence[int] = (3, 5, 9),
    use_amge: bool = True,
    use_uncertainty: bool = True,
    **kwargs,
) -> CrossStitchMTL:
    """Factory mirroring build_morphoformer so CLI flags map 1:1."""
    img_size = patch_grid_size * 10
    return CrossStitchMTL(
        img_size=img_size,
        in_chans=in_chans,
        patch_size=patch_size,
        center_sizes=list(center_sizes),
        use_amge=use_amge,
        use_uncertainty=use_uncertainty,
        # use_bgtd is accepted by MorphoFormer.__init__ but the decoder it builds
        # is deleted in CrossStitchMTL.__init__; pass False to skip BGTD build.
        use_bgtd=False,
        **kwargs,
    )
