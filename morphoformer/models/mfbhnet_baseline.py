"""MF-BHNet re-implemented as a recent hybrid CNN-Transformer baseline.

MF-BHNet = Wang et al., "A Hybrid Multimodal Fusion Network for Building Height
Estimation Using Sentinel-1 and Sentinel-2 Imagery", IEEE TGRS 2024,
DOI 10.1109/TGRS.2024.3477588. This is our own re-implementation; no code was
released by the original authors.

Why the adaptation below is faithful:
  The paper's named contributions are the **encoder + fusion**, not the decoder:
    1. Hybrid multimodal encoder — IME (dual-branch CNN with SC-RU = SCConv's
       Spatial+Channel Reconstruction Units, Li et al. CVPR 2023) + CME
       (transformer-based cross-modal encoder on the highest features).
    2. Coarse-fine progressive fusion — MFF (multiscale feature fusion) + MSF
       (multimodal semantic fusion via SMM).
  The decoder is explicitly a generic U-Net-style upsampler ("follows the same
  twinned design ... skip connection similar to U-Net"), NOT a contribution.

  Native MF-BHNet is dense (per-pixel height + footprint, MSE+CE+Dice on rasters).
  Our task is scene-level scalar BH/BF (8-ch 90x90 patch -> two scalars). We keep
  100% of the contributions (IME + CME + MFF + MSF) and replace ONLY the generic
  decoder with global pooling + dual scalar regression heads, trained with v2's
  Kendall-uncertainty Huber. This is the standard "keep the published
  architecture, swap the task head" baseline — fully faithful, because none of
  MF-BHNet's contributions depend on dense supervision.

Input adaptation: MF-BHNet is bimodal (SAR 2ch, optical 4ch). Our 8-ch stack is
SAR(2)+Optical(4)+DEM(1)+mask(1). To preserve the bimodal architecture while
giving it the same information MorphoFormer sees, DEM joins the SAR branch
(both geometric) and the validity mask joins the optical branch:
  SAR branch in = 3 (VV,VH,DEM), optical branch in = 5 (R,G,B,NIR,mask).

Hyperparameters not stated in the paper (channel widths D1-D4, CME depth/dim,
patch size, SCConv alpha/groups) use SCConv defaults + a moderate ViT config;
flagged here as reasonable-inference reproduction choices.
"""
from __future__ import annotations

from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── SCConv units (SRU + CRU), Li et al. CVPR 2023 — the SC-RU building block ──
class SRU(nn.Module):
    """Spatial Reconstruction Unit (paper Fig. 3): GN scale-factor gating into
    informative / uninformative features + cross-reconstruction."""

    def __init__(self, channels: int, group_num: int = 4, gate_thresh: float = 0.5):
        super().__init__()
        group_num = max(1, min(group_num, channels))
        while channels % group_num != 0:
            group_num -= 1
        self.gn = nn.GroupNorm(group_num, channels)
        self.gate_thresh = gate_thresh

    def forward(self, x):
        gn_x = self.gn(x)
        w = (self.gn.weight / self.gn.weight.sum()).view(1, -1, 1, 1)
        reweight = torch.sigmoid(gn_x * w)
        # SCConv soft gate: above threshold -> 1/0, else keep soft weight so
        # gradient flows back to the GN scale factors.
        w1 = torch.where(reweight > self.gate_thresh, torch.ones_like(reweight), reweight)
        w2 = torch.where(reweight > self.gate_thresh, torch.zeros_like(reweight), reweight)
        info, noninfo = w1 * x, w2 * x
        i1, i2 = torch.chunk(info, 2, dim=1)
        n1, n2 = torch.chunk(noninfo, 2, dim=1)
        return torch.cat([i1 + n2, i2 + n1], dim=1)


class CRU(nn.Module):
    """Channel Reconstruction Unit (paper Fig. 4): split-transform-fuse to cut
    channel redundancy (SCConv)."""

    def __init__(self, channels: int, alpha: float = 0.5, squeeze_ratio: int = 2,
                 groups: int = 4):
        super().__init__()
        self.up_ch = max(1, int(alpha * channels))
        self.low_ch = channels - self.up_ch
        up_sq = max(1, self.up_ch // squeeze_ratio)
        low_sq = max(1, self.low_ch // squeeze_ratio)
        self.squeeze_up = nn.Conv2d(self.up_ch, up_sq, 1)
        self.squeeze_low = nn.Conv2d(self.low_ch, low_sq, 1)
        g = max(1, min(groups, up_sq))
        while up_sq % g != 0:
            g -= 1
        self.gwc = nn.Conv2d(up_sq, channels, 3, padding=1, groups=g)
        self.pwc1 = nn.Conv2d(up_sq, channels, 1)
        self.pwc2 = nn.Conv2d(low_sq, channels - low_sq, 1)
        self.low_sq = low_sq
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        up, low = torch.split(x, [self.up_ch, self.low_ch], dim=1)
        up = self.squeeze_up(up)
        low = self.squeeze_low(low)
        y1 = self.gwc(up) + self.pwc1(up)                       # rich-channel path
        y2 = torch.cat([self.pwc2(low), low], dim=1)            # scarce-channel path
        s = torch.softmax(torch.cat([self.pool(y1), self.pool(y2)], dim=1), dim=1)
        s1, s2 = torch.chunk(s, 2, dim=1)
        return y1 * s1 + y2 * s2


class SCRU(nn.Module):
    """SC-RU residual block (paper Fig. 2): 1x1 -> 3x3 -> SRU -> CRU -> 1x1,
    residual add then ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.sru = SRU(out_ch)
        self.cru = CRU(out_ch)
        self.conv3 = nn.Sequential(nn.Conv2d(out_ch, out_ch, 1), nn.BatchNorm2d(out_ch))
        self.skip = (nn.Identity() if in_ch == out_ch
                     else nn.Sequential(nn.Conv2d(in_ch, out_ch, 1), nn.BatchNorm2d(out_ch)))

    def forward(self, x):
        out = self.conv3(self.cru(self.sru(self.conv2(self.conv1(x)))))
        return F.relu(out + self.skip(x))


# ── IME: intramodal encoder (dual-branch; one instance per modality) ────────
class IMEBranch(nn.Module):
    """7x7 conv (H/2) + 3 SC-RU stages -> 4 multiscale features {F1..F4} at
    H/2, H/4, H/8, H/16 (paper eq. 1)."""

    def __init__(self, in_ch: int, dims: List[int]):
        super().__init__()
        d1, d2, d3, d4 = dims
        self.stem = nn.Sequential(nn.Conv2d(in_ch, d1, 7, stride=2, padding=3),
                                  nn.BatchNorm2d(d1), nn.ReLU(inplace=True))
        self.stage1 = self._stage(d1, d2)
        self.stage2 = self._stage(d2, d3)
        self.stage3 = self._stage(d3, d4)

    @staticmethod
    def _stage(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            SCRU(out_ch, out_ch),
        )

    def forward(self, x):
        f1 = self.stem(x)
        f2 = self.stage1(f1)
        f3 = self.stage2(f2)
        f4 = self.stage3(f3)
        return [f1, f2, f3, f4]


# ── MFF: multiscale feature fusion (paper eq. 5) ────────────────────────────
class MFF(nn.Module):
    def __init__(self, dims: List[int]):
        super().__init__()
        self.fuse = nn.ModuleList([
            nn.Sequential(nn.Conv2d(2 * d, d, 1), nn.BatchNorm2d(d), nn.ReLU(inplace=True))
            for d in dims
        ])

    def forward(self, feats_o, feats_s):
        return [self.fuse[i](torch.cat([fo, fs], dim=1))
                for i, (fo, fs) in enumerate(zip(feats_o, feats_s))]


# ── MSF: multimodal semantic fusion via SMM (paper Fig. 5, eq. 6-9) ─────────
class SMM(nn.Module):
    """Semantic Mining Module: channel + spatial attention refinement of one
    modality's highest feature."""

    def __init__(self, channels: int, group_num: int = 4):
        super().__init__()
        g = max(1, min(group_num, channels))
        while channels % g != 0:
            g -= 1
        self.gn = nn.GroupNorm(g, channels)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc_ch = nn.Conv2d(channels, channels, 1)
        self.fc_sp = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        ch = torch.sigmoid(self.fc_ch(self.gap(x))) * x          # channel-refined
        sp = torch.sigmoid(self.fc_sp(self.gn(x))) * x           # spatial-refined
        return ch, sp


class MSF(nn.Module):
    """Cascade channel-fusion + spatial-fusion of the two modalities' highest
    features, then channel shuffle -> S (2C channels)."""

    def __init__(self, channels: int, groups: int = 4):
        super().__init__()
        self.smm_s = SMM(channels)
        self.smm_o = SMM(channels)
        self.groups = max(1, min(groups, 2 * channels))
        while (2 * channels) % self.groups != 0:
            self.groups -= 1

    @staticmethod
    def _shuffle(x, groups):
        n, c, h, w = x.shape
        return x.view(n, groups, c // groups, h, w).transpose(1, 2).reshape(n, c, h, w)

    def forward(self, f_s, f_o):
        ch_s, sp_s = self.smm_s(f_s)
        ch_o, sp_o = self.smm_o(f_o)
        chan = ch_s * ch_o                                       # channel fusion
        spat = sp_s * sp_o                                       # spatial fusion
        s = torch.cat([chan, spat], dim=1)                       # 2C
        return self._shuffle(s, self.groups)


# ── CME: transformer cross-modal encoder on highest features (paper eq. 2-4) ─
class CME(nn.Module):
    """Patchify {optical F4, SAR F4, MSF S4} -> linear proj + pos/modality
    embed -> L self-attention blocks -> mean-pooled global multimodal context."""

    def __init__(self, ch_o: int, ch_s: int, ch_m: int, dim: int = 256,
                 depth: int = 4, num_heads: int = 4, max_tokens: int = 512):
        super().__init__()
        self.proj_o = nn.Conv2d(ch_o, dim, 1)
        self.proj_s = nn.Conv2d(ch_s, dim, 1)
        self.proj_m = nn.Conv2d(ch_m, dim, 1)
        self.modality_embed = nn.Parameter(torch.zeros(3, 1, dim))
        self.pos_embed = nn.Parameter(torch.rand(max_tokens, dim))
        layer = nn.TransformerEncoderLayer(dim, num_heads, dim_feedforward=4 * dim,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.dim = dim

    def _tokens(self, feat, proj, mod_idx):
        t = proj(feat).flatten(2).transpose(1, 2)                # N, S, dim
        return t + self.modality_embed[mod_idx]

    def forward(self, f_o, f_s, s_m):
        to = self._tokens(f_o, self.proj_o, 0)
        ts = self._tokens(f_s, self.proj_s, 1)
        tm = self._tokens(s_m, self.proj_m, 2)
        tok = torch.cat([to, ts, tm], dim=1)                     # N, S_total, dim
        tok = tok + self.pos_embed[:tok.shape[1]].unsqueeze(0)
        g = self.encoder(tok)                                    # N, S_total, dim
        return g.mean(dim=1)                                     # N, dim  (global context)


# ── Full MF-BHNet adapted to scene-level scalar BH/BF regression ────────────
class MFBHNetMTL(nn.Module):
    """Output (use_uncertainty=True): (bh[N,1], bf[N,1], log_sigma_h, log_sigma_f)
    -> MorphoFormerLitModule._compute_loss len==4 branch."""

    def __init__(self, in_chans: int = 8, sar_dem_ch: int = 3, opt_mask_ch: int = 5,
                 dims: List[int] = (64, 128, 256, 512),
                 cme_dim: int = 256, cme_depth: int = 4,
                 use_uncertainty: bool = True):
        super().__init__()
        assert sar_dem_ch + opt_mask_ch == in_chans, "branch split must sum to in_chans"
        self.sar_dem_ch = sar_dem_ch
        self.opt_mask_ch = opt_mask_ch
        self.use_uncertainty = use_uncertainty
        dims = list(dims)
        d4 = dims[-1]

        self.ime_s = IMEBranch(sar_dem_ch, dims)
        self.ime_o = IMEBranch(opt_mask_ch, dims)
        self.mff = MFF(dims)
        self.msf = MSF(d4)
        self.cme = CME(ch_o=d4, ch_s=d4, ch_m=2 * d4, dim=cme_dim, depth=cme_depth)

        head_in = cme_dim + sum(dims)                            # CME global ctx + pooled all-scale MFF
        self.head_h = nn.Sequential(nn.Linear(head_in, head_in // 2), nn.ReLU(inplace=True),
                                    nn.Linear(head_in // 2, 1))
        self.head_f = nn.Sequential(nn.Linear(head_in, head_in // 2), nn.ReLU(inplace=True),
                                    nn.Linear(head_in // 2, 1))
        if use_uncertainty:
            self.log_sigma_h = nn.Parameter(torch.zeros(1))
            self.log_sigma_f = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x_s, x_o = torch.split(x, [self.sar_dem_ch, self.opt_mask_ch], dim=1)
        feats_s = self.ime_s(x_s)                                # [F1..F4]
        feats_o = self.ime_o(x_o)
        fused = self.mff(feats_o, feats_s)                       # MFF [F'1..F'4]
        s4 = self.msf(feats_s[-1], feats_o[-1])                  # MSF top -> 2*d4
        g = self.cme(feats_o[-1], feats_s[-1], s4)               # N, cme_dim
        # pool every MFF scale (faithful to "multiscale" fusion) + global context
        local = torch.cat([f.mean(dim=(2, 3)) for f in fused], dim=1)   # N, sum(dims)
        feat = torch.cat([g, local], dim=1)
        bh = F.relu(self.head_h(feat))
        bf = torch.sigmoid(self.head_f(feat))
        if self.use_uncertainty:
            return bh, bf, self.log_sigma_h, self.log_sigma_f
        return bh, bf


def build_mfbhnet(in_chans: int = 8, use_uncertainty: bool = True, **kwargs) -> MFBHNetMTL:
    return MFBHNetMTL(in_chans=in_chans, use_uncertainty=use_uncertainty, **kwargs)
