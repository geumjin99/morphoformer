"""MorphoFormer — morphology-guided cross-task coupling for joint building
height and footprint estimation.

Reference implementation for:

    Han et al., "Morphology-Guided Cross-Task Coupling for Joint Building
    Height and Footprint Estimation", Science of Remote Sensing, 2026.
"""

from .data.dataset import (
    H5PatchDataset,
    CachedPatchDataset,
    MorphoFormerDataModule,
)
from .lit_module import MorphoFormerLitModule
from .losses.mcl_loss import MorphologyConsistentMTLLoss, build_mcl_loss
from .models import build_morphoformer

__all__ = [
    'H5PatchDataset',
    'CachedPatchDataset',
    'MorphoFormerDataModule',
    'MorphoFormerLitModule',
    'MorphologyConsistentMTLLoss',
    'build_mcl_loss',
    'build_morphoformer',
]

__version__ = '1.0.0'
