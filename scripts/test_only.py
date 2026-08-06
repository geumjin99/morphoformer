"""Test-only evaluation for a saved MorphoFormer checkpoint."""

import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from morphoformer import MorphoFormerDataModule, MorphoFormerLitModule, build_morphoformer, build_mcl_loss
from morphoformer.models import build_crossstitch
from morphoformer.paths import cache_root, data_root


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--model', default='morphoformer',
                   choices=['morphoformer', 'crossstitch'],
                   help='Architecture matching the checkpoint (default morphoformer)')
    p.add_argument('--patch-size', type=int, default=10)
    p.add_argument('--patch-grid-size', type=int, default=9)
    p.add_argument('--center-sizes', type=int, nargs='+', default=[3, 5, 9])
    p.add_argument('--modalities', nargs='+', default=['sar', 'optical', 'dem'])
    p.add_argument('--no-amge', action='store_true')
    p.add_argument('--no-bgtd', action='store_true')
    p.add_argument('--no-uncertainty', action='store_true')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--preload-cache', action='store_true')
    p.add_argument('--cache-dir', default=str(cache_root()))
    _DATA = str(data_root())
    p.add_argument('--train-path', default=f'{_DATA}/train.h5')
    p.add_argument('--val-path', default=f'{_DATA}/valid.h5')
    p.add_argument('--test-path', default=f'{_DATA}/test2.h5')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision('high')

    modality_channels = {'sar': 2, 'optical': 4, 'dem': 1}
    in_chans = sum(modality_channels[m] for m in args.modalities) + 1

    if args.model == 'crossstitch':
        model = build_crossstitch(
            variant='base',
            in_chans=in_chans,
            patch_grid_size=args.patch_grid_size,
            patch_size=args.patch_size,
            center_sizes=tuple(args.center_sizes),
            use_uncertainty=not args.no_uncertainty,
            use_amge=not args.no_amge,
        )
    else:
        model = build_morphoformer(
            variant='base',
            in_chans=in_chans,
            patch_grid_size=args.patch_grid_size,
            patch_size=args.patch_size,
            center_sizes=tuple(args.center_sizes),
            use_uncertainty=not args.no_uncertainty,
            use_amge=not args.no_amge,
            use_bgtd=not args.no_bgtd,
        )

    loss_fn = build_mcl_loss(loss_type='huber', lambda_consist=0.2, warmup_epochs=10)

    dm = MorphoFormerDataModule(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patch_grid_size=args.patch_grid_size,
        modalities=args.modalities,
        preload=False,
        cache_dir=args.cache_dir,
        preload_cache=args.preload_cache,
        chunked=False,
    )

    lit = MorphoFormerLitModule.load_from_checkpoint(
        args.ckpt, model=model, loss_fn=loss_fn, strict=True
    )

    trainer = pl.Trainer(accelerator='gpu', devices=1, precision='16-mixed', logger=False)
    print(f"[Test-only] ckpt={args.ckpt}")
    trainer.test(lit, datamodule=dm)


if __name__ == '__main__':
    main()
