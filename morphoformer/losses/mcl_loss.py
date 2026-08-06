"""
Morphology-Consistent MTL Loss (MCL) for MorphoFormer.

Combines three loss terms:

  L_total = L_h + L_f + λ_c · L_consist

  L_h      — Kendall uncertainty-weighted regression for building height (BH).
  L_f      — Kendall uncertainty-weighted regression for building footprint (BF).
  L_consist — Morphology consistency: the secondary BH predictor that operates
               solely on BF features (bh_from_bf) is also supervised by ground-
               truth BH. This forces the BF branch to encode height-correlated
               morphology representations, explicitly tightening the cross-task
               coupling beyond shared backbone weights.

Physical motivation
-------------------
Building footprint ratio (BF) is a proxy for urban morphology type
(e.g., dense high-rise blocks have high BF AND high BH; suburban areas have
low BF AND low BH). Jointly supervising the BF-feature-derived BH estimate
teaches the network to encode the Floor-Area-Ratio pattern implicitly present
in the data, yielding more physically consistent joint predictions.

Consistency weight schedule
---------------------------
To prevent the consistency term from dominating early training (before the BF
branch has learned meaningful representations), λ_c is linearly warmed up from
0 to its target value over `warmup_epochs` epochs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MorphologyConsistentMTLLoss(nn.Module):
    """
    Primary uncertainty-weighted MTL loss + morphology consistency regulariser.

    Args:
        loss_type:      Base regression loss — 'huber' | 'mse' | 'l1'.
        beta_h:         Huber δ for BH (metres scale; default 2.0).
        beta_f:         Huber δ for BF (ratio scale; default 0.05).
        lambda_consist: Weight for the consistency term (default 0.2).
        warmup_epochs:  Number of epochs to linearly ramp λ_consist (default 10).
    """

    def __init__(
        self,
        loss_type: str = 'huber',
        beta_h: float = 2.0,
        beta_f: float = 0.05,
        lambda_consist: float = 0.2,
        warmup_epochs: int = 10,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.beta_h = beta_h
        self.beta_f = beta_f
        self.lambda_consist = lambda_consist
        self.warmup_epochs = warmup_epochs
        self._current_epoch = 0

    # ── called by LightningModule ──────────────────────────────────────────

    def set_epoch(self, epoch: int):
        self._current_epoch = epoch

    @property
    def effective_lambda(self) -> float:
        if self.warmup_epochs <= 0:
            return self.lambda_consist
        progress = min(self._current_epoch / self.warmup_epochs, 1.0)
        return self.lambda_consist * progress

    # ── regression base ────────────────────────────────────────────────────

    def _reg(self, pred: torch.Tensor, true: torch.Tensor, beta: float) -> torch.Tensor:
        if self.loss_type == 'huber':
            return F.smooth_l1_loss(pred, true, beta=beta)
        if self.loss_type == 'mse':
            return F.mse_loss(pred, true)
        return F.l1_loss(pred, true)

    # ── main forward ───────────────────────────────────────────────────────

    def forward(
        self,
        bh_pred: torch.Tensor,
        bh_true: torch.Tensor,
        bf_pred: torch.Tensor,
        bf_true: torch.Tensor,
        log_sigma_h: Optional[torch.Tensor] = None,
        log_sigma_f: Optional[torch.Tensor] = None,
        bh_from_bf: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            bh_pred:     Predicted building height          [B, 1]
            bh_true:     Ground-truth building height       [B, 1]
            bf_pred:     Predicted building footprint       [B, 1]
            bf_true:     Ground-truth building footprint    [B, 1]
            log_sigma_h: Log uncertainty for BH head        [1]  (optional)
            log_sigma_f: Log uncertainty for BF head        [1]  (optional)
            bh_from_bf:  Secondary BH estimate from BF feat [B, 1] (optional)
        Returns:
            Scalar total loss.
        """
        loss_h = self._reg(bh_pred, bh_true, self.beta_h)
        loss_f = self._reg(bf_pred, bf_true, self.beta_f)

        # ── Uncertainty weighting (Kendall et al., 2018) ──
        if log_sigma_h is not None and log_sigma_f is not None:
            # L = exp(-σ²) * L_task + σ²  (log-space for numerical stability)
            w_h = 0.5 * torch.exp(-log_sigma_h)
            w_f = 0.5 * torch.exp(-log_sigma_f)
            total = w_h * loss_h + 0.5 * log_sigma_h \
                  + w_f * loss_f + 0.5 * log_sigma_f
        else:
            total = loss_h + loss_f

        # ── Morphology consistency regulariser ──
        if bh_from_bf is not None and self.effective_lambda > 0:
            # Supervise the secondary predictor (BF-features → BH) with BH GT.
            # Effect: forces BF encoder to learn height-correlated representations.
            l_consist = self._reg(bh_from_bf, bh_true, self.beta_h)
            total = total + self.effective_lambda * l_consist

        return total


def build_mcl_loss(**kwargs) -> MorphologyConsistentMTLLoss:
    return MorphologyConsistentMTLLoss(**kwargs)
