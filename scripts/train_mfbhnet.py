"""Train MF-BHNet (Wang et al., IEEE TGRS 2024) as a recent hybrid
CNN-Transformer baseline for building height estimation.

MF-BHNet (hybrid IME+CME multimodal encoder + MFF/MSF fusion) with its
generic U-Net decoder replaced by global pooling + dual scalar heads, on the *same* data / split /
loss / eval pipeline as MorphoFormer and the other baselines. See
morphoformer/models/mfbhnet_baseline.py for the adaptation rationale.

Example:
    python scripts/train_mfbhnet.py --exp-name mfbhnet_patch2 \
        --batch-size 256 --lr 2e-4 --epochs 100 --early-stopping 15 \
        --num-workers 8 --preload-cache
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor, Callback)
from pytorch_lightning.loggers import CSVLogger

sys.path.insert(0, str(Path(__file__).parent.parent))

from morphoformer import MorphoFormerDataModule, MorphoFormerLitModule, build_mcl_loss
from morphoformer.models import build_mfbhnet
from morphoformer.paths import cache_root, checkpoint_root, data_root


class EpochSummaryCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        m = trainer.callback_metrics
        ep = trainer.current_epoch

        def g(key):
            v = m.get(key, float('nan'))
            return float(v) if v is not None else float('nan')

        def fmt(v, w=7, d=3):
            return f'{v:{w}.{d}f}' if not math.isnan(v) else ' ' * (w - 3) + '---'

        print(
            f'\n[Ep {ep:03d}]'
            f'  val  MAE_h={fmt(g("val/mae_h"))} R²={fmt(g("val/r2_h"))}'
            f' MAE_f={fmt(g("val/mae_f"), 8, 4)}'
            f'  combined={fmt(g("val/combined_mae"))}',
            flush=True,
        )


def parse_args():
    p = argparse.ArgumentParser(description='MF-BHNet transformer baseline')
    p.add_argument('--exp-name', required=True)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--patch-grid-size', type=int, default=9)
    p.add_argument('--patch-size', type=int, default=2)  # kept for parity; data is 90x90
    p.add_argument('--early-stopping', type=int, default=15)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cache-dir', default=str(cache_root()))
    _DATA = str(data_root())
    p.add_argument('--train-path', default=f'{_DATA}/train.h5')
    p.add_argument('--val-path',   default=f'{_DATA}/valid.h5')
    p.add_argument('--test-path',  default=f'{_DATA}/test2.h5')
    p.add_argument('--ckpt-root', default=str(checkpoint_root()))
    p.add_argument('--preload-cache', action='store_true', default=False)
    p.add_argument('--resume', default=None)
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision('high')

    modality_channels = {'sar': 2, 'optical': 4, 'dem': 1}
    modalities = ['sar', 'optical', 'dem']
    in_chans = sum(modality_channels[m] for m in modalities) + 1  # +1 mask = 9

    model = build_mfbhnet(in_chans=in_chans, use_uncertainty=True)
    print(f'[mfbhnet] in_chans={in_chans}  params: '
          f'{sum(p.numel() for p in model.parameters())/1e6:.2f} M', flush=True)

    # same Kendall-uncertainty Huber as the other baselines; no MCL surrogate
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

    resume_ckpt = None
    if args.resume:
        resume_ckpt = str(exp_dir / 'last.ckpt') if args.resume == 'last' else args.resume

    trainer = pl.Trainer(
        accelerator='gpu', devices=1, precision='16-mixed',
        max_epochs=args.epochs,
        callbacks=[ckpt_cb, early, LearningRateMonitor(logging_interval='epoch'),
                   EpochSummaryCallback()],
        logger=csv_logger,
        check_val_every_n_epoch=1,
        log_every_n_steps=50,
        gradient_clip_val=1.0,
    )

    trainer.fit(lit, datamodule=dm, ckpt_path=resume_ckpt)
    print(f'[mfbhnet] best val ckpt: {ckpt_cb.best_model_path}', flush=True)
    trainer.test(lit, datamodule=dm, ckpt_path='best')


if __name__ == '__main__':
    main()
