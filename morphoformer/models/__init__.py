"""Model zoo.

``MorphoFormer`` is the proposed model. The remaining entries are the
baselines reported in the paper, each re-trained under MorphoFormer's own
data pipeline, split, loss and evaluation code so that every number in the
comparison tables is commensurable.
"""
from .morphoformer import MorphoFormer, MorphoFormerLarge, build_morphoformer
from .crossstitch import CrossStitchMTL, build_crossstitch
from .mfbhnet_baseline import MFBHNetMTL, build_mfbhnet
from .swin_mtl_baseline import SwinMTL, build_swin_mtl
from .cnn_baselines import ResNetMTL, SENetMTL, ViTMTL, build_baseline

__all__ = [
    'MorphoFormer', 'MorphoFormerLarge', 'build_morphoformer',
    'CrossStitchMTL', 'build_crossstitch',
    'MFBHNetMTL', 'build_mfbhnet',
    'SwinMTL', 'build_swin_mtl',
    'ResNetMTL', 'SENetMTL', 'ViTMTL', 'build_baseline',
]
