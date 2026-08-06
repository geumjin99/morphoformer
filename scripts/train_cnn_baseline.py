"""Train the CNN baselines (ResNet-MTL / SENet-MTL) at MorphoFormer's 9x9
receptive field on identical data, for a head-to-head comparison.

The baseline encoders are receptive-field agnostic; they simply consume the
same 90x90 input tensor, wrapped in MorphoFormer's LitModule + DataModule +
loss. lambda_consist is 0 because these models have no consistency head.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from morphoformer import MorphoFormerDataModule, MorphoFormerLitModule, build_mcl_loss
from morphoformer.models.cnn_baselines import ResNetMTL, SENetMTL
from morphoformer.paths import cache_root, checkpoint_root, split_paths

BASELINES = {
    'resnet': ResNetMTL,
    'senet':  SENetMTL,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', required=True, choices=list(BASELINES.keys()))
    p.add_argument('--exp-name', required=True)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--patch-grid-size', type=int, default=9)
    p.add_argument('--patch-size', type=int, default=2)
    p.add_argument('--early-stopping', type=int, default=15)
    p.add_argument('--seed', type=int, default=42)
    _S = split_paths()
    p.add_argument('--cache-dir',  default=str(cache_root()))
    p.add_argument('--train-path', default=str(_S['train']))
    p.add_argument('--val-path',   default=str(_S['val']))
    p.add_argument('--test-path',  default=str(_S['test']))
    p.add_argument('--ckpt-root',  default=str(checkpoint_root()))
    p.add_argument('--preload-cache', action='store_true', default=True)
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision('high')

    # in_chans = 8 explanatory bands + 1 mask = 9
    modality_channels = {'sar': 2, 'optical': 4, 'dem': 1}
    modalities = ['sar', 'optical', 'dem']
    in_chans = sum(modality_channels[m] for m in modalities) + 1

    # Build baseline (no MCL/BGTD; just CNN encoder + dual heads with uncertainty)
    Model = BASELINES[args.baseline]
    model = Model(in_chans=in_chans, use_uncertainty=True)
    print(f'[{args.baseline}] params: {sum(p.numel() for p in model.parameters())/1e6:.2f} M')

    # Loss: same Kendall-uncertainty Huber as v2; lambda_consist=0 since no surrogate
    loss_fn = build_mcl_loss(loss_type='huber', lambda_consist=0.0, warmup_epochs=0)

    dm = MorphoFormerDataModule(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patch_grid_size=args.patch_grid_size,
        modalities=modalities,
        preload=False,
        cache_dir=args.cache_dir,
        preload_cache=args.preload_cache,
        chunked=False,
    )

    lit = MorphoFormerLitModule(
        model=model, loss_fn=loss_fn,
        learning_rate=args.lr, weight_decay=1e-2,
        scheduler_T_max=args.epochs, scheduler_eta_min=1e-6,
    )

    exp_dir = Path(args.ckpt_root) / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    csv_logger = CSVLogger(str(exp_dir.parent), name=args.exp_name)

    ckpt_cb = ModelCheckpoint(
        dirpath=str(exp_dir),
        filename='epoch_{epoch:03d}_mae_{val/combined_mae:.4f}',
        monitor='val/combined_mae', mode='min',
        save_top_k=3, save_last=True,
    )
    early = EarlyStopping(monitor='val/combined_mae', patience=args.early_stopping, mode='min')

    trainer = pl.Trainer(
        accelerator='gpu', devices=1, precision='16-mixed',
        max_epochs=args.epochs,
        callbacks=[ckpt_cb, early],
        logger=csv_logger,
        check_val_every_n_epoch=1,
        log_every_n_steps=50,
        gradient_clip_val=1.0,
    )

    trainer.fit(lit, datamodule=dm)
    print(f'[{args.baseline}] best val ckpt: {ckpt_cb.best_model_path}')

    # Test on best ckpt
    trainer.test(lit, datamodule=dm, ckpt_path='best')


if __name__ == '__main__':
    main()
