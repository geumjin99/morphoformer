"""MorphoFormer training entry point.

The configuration behind the paper's main results is

    python scripts/train.py --exp-name morphoformer_full \
        --patch-size 2 --batch-size 512 --lr 2e-4 \
        --epochs 150 --early-stopping 20 --num-workers 8 --preload-cache

Ablations reuse the same command with exactly one switch changed; see
scripts/run_paper_experiments.sh.
"""

import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl
import math
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, Callback
from pytorch_lightning.loggers import TensorBoardLogger

sys.path.insert(0, str(Path(__file__).parent.parent))

from morphoformer import (
    MorphoFormerDataModule,
    MorphoFormerLitModule,
    build_morphoformer,
    build_mcl_loss,
)
from morphoformer.paths import cache_root, checkpoint_root, data_root


class EpochSummaryCallback(Callback):
    """Print a new line after each validation epoch with val + test metrics."""

    def on_validation_epoch_end(self, trainer, pl_module):
        # sanity check runs at epoch -1; skip
        if trainer.sanity_checking:
            return
        m = trainer.callback_metrics
        ep = trainer.current_epoch

        def g(key):
            v = m.get(key, float('nan'))
            return float(v) if v is not None else float('nan')

        v_maeh = g('val/mae_h');  v_r2h = g('val/r2_h');  v_maef = g('val/mae_f')
        t_maeh = g('test_epoch/mae_h'); t_r2h = g('test_epoch/r2_h'); t_maef = g('test_epoch/mae_f')
        lr = g('lr-AdamW')

        # build line (use --- for missing test metrics)
        def fmt(v, w=7, d=3):
            return f'{v:{w}.{d}f}' if not math.isnan(v) else ' ' * (w - 3) + '---'

        print(
            f'\n[Ep {ep:03d}]'
            f'  val  MAE_h={fmt(v_maeh)} R²={fmt(v_r2h)} MAE_f={fmt(v_maef, 8, 4)}'
            f'  |  test MAE_h={fmt(t_maeh)} R²={fmt(t_r2h)} MAE_f={fmt(t_maef, 8, 4)}'
            f'  lr={lr:.2e}',
            flush=True,
        )


def parse_args():
    p = argparse.ArgumentParser(description='MorphoFormer Training')

    # ── Data ──
    _DATA = str(data_root())
    p.add_argument('--train-path', default=f'{_DATA}/train.h5')
    p.add_argument('--val-path',   default=f'{_DATA}/valid.h5')
    p.add_argument('--test-path',  default=f'{_DATA}/test2.h5')

    # ── Model ──
    p.add_argument('--variant',         default='base', choices=['base', 'large'])
    p.add_argument('--patch-grid-size', type=int, default=9, choices=[3, 5, 7, 9],
                   help='Neighbourhood size in 100 m patches')
    p.add_argument('--patch-size',      type=int, default=10,
                   help='Swin patch embedding stride (10=one token/100m, 2=finer 20m features)')
    p.add_argument('--center-sizes',    type=int, nargs='+', default=[3, 5, 9],
                   help='MSMP crop sizes, e.g. --center-sizes 3 5 9')
    p.add_argument('--modalities',      nargs='+', default=['sar', 'optical', 'dem'])
    p.add_argument('--no-uncertainty',  action='store_true')
    p.add_argument('--no-amge',         action='store_true',
                   help='Disable Adaptive Modality Gating Encoder (ablation)')
    p.add_argument('--no-bgtd',         action='store_true',
                   help='Replace BF-Guided Task Decoder with independent heads (ablation)')

    # ── Loss ──
    p.add_argument('--loss-type',       default='huber', choices=['mse', 'huber', 'l1'])
    p.add_argument('--lambda-consist',  type=float, default=0.2,
                   help='Weight for morphology consistency term (MCL)')
    p.add_argument('--warmup-epochs',   type=int, default=10,
                   help='Epochs to linearly ramp lambda-consist from 0')

    # ── Training ──
    p.add_argument('--batch-size',    type=int,   default=256)
    p.add_argument('--epochs',        type=int,   default=150)
    p.add_argument('--lr',            type=float, default=1e-4)
    p.add_argument('--weight-decay',  type=float, default=1e-2)
    p.add_argument('--num-workers',   type=int,   default=4)
    p.add_argument('--preload',       action='store_true',
                   help='Preload full H5 dataset into RAM (H5PatchDataset only)')
    p.add_argument('--preload-cache', action='store_true',
                   help='Preload all .npy cache files into RAM (~6 GB); '
                        'eliminates mmap random-read bottleneck')
    p.add_argument('--chunked', action='store_true',
                   help='Use ChunkedIterableDataset: background threads pre-assemble '
                        'batches with vectorized city-level fancy-index; '
                        'implies --preload-cache and num_workers=0')
    p.add_argument('--chunk-size', type=int, default=8192,
                   help='Samples per pre-assembled chunk (default 8192)')
    p.add_argument('--chunk-threads', type=int, default=4,
                   help='Background assembly threads (default 4)')
    p.add_argument('--cache-dir',     default=str(cache_root()),
                   help='Pre-computed npy cache dir (scripts/precompute_cache.py). '
                        'Falls back to H5 if not found.')
    p.add_argument('--early-stopping',type=int,   default=15)
    p.add_argument('--no-augment',    action='store_true',
                   help='Disable train-set rot90/hflip augmentation')
    p.add_argument('--no-track-test', action='store_true',
                   help='Do not score the test split during training. Test '
                        'metrics are diagnostic only -- model selection and '
                        'early stopping monitor val/combined_mae either way -- '
                        'so this flag does not change the trained model.')

    # ── Misc ──
    p.add_argument('--checkpoint-dir', default=str(checkpoint_root()))
    p.add_argument('--exp-name',  default='morphoformer')
    p.add_argument('--resume',    default=None,
                   help='Path to checkpoint, or "last" to auto-find last.ckpt')
    p.add_argument('--seed',      type=int, default=42)

    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed)

    modality_channels = {'sar': 2, 'optical': 4, 'dem': 1}
    in_chans = sum(modality_channels[m] for m in args.modalities) + 1  # +1 mask

    print(f"[Model] MorphoFormer-{args.variant} | "
          f"patch_grid={args.patch_grid_size} | patch_size={args.patch_size} | "
          f"center_sizes={args.center_sizes} | "
          f"AMGE={'on' if not args.no_amge else 'off'} | "
          f"BGTD={'on' if not args.no_bgtd else 'off'}")

    model = build_morphoformer(
        variant=args.variant,
        in_chans=in_chans,
        patch_grid_size=args.patch_grid_size,
        patch_size=args.patch_size,
        center_sizes=tuple(args.center_sizes),
        use_uncertainty=not args.no_uncertainty,
        use_amge=not args.no_amge,
        use_bgtd=not args.no_bgtd,
    )

    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Trainable params: {n:,} ({n/1e6:.3f} M)")

    loss_fn = build_mcl_loss(
        loss_type=args.loss_type,
        lambda_consist=args.lambda_consist,
        warmup_epochs=args.warmup_epochs,
    )
    print(f"[Loss] MCL | type={args.loss_type} | "
          f"lambda_consist={args.lambda_consist} | warmup={args.warmup_epochs}")

    # --chunked implies preloading into RAM
    preload_cache = args.preload_cache or args.chunked

    datamodule = MorphoFormerDataModule(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patch_grid_size=args.patch_grid_size,
        modalities=args.modalities,
        preload=args.preload,
        cache_dir=args.cache_dir,
        preload_cache=preload_cache,
        chunked=args.chunked,
        chunk_size=args.chunk_size,
        chunk_threads=args.chunk_threads,
        augment=not args.no_augment,
        track_test_during_training=not args.no_track_test,
    )

    lit = MorphoFormerLitModule(
        model=model,
        loss_fn=loss_fn,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        scheduler_T_max=args.epochs,
    )

    exp_dir = Path(args.checkpoint_dir) / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # ── auto-resume from last.ckpt ──
    resume_path = None
    if args.resume == 'last':
        last_ckpt = exp_dir / 'last.ckpt'
        if last_ckpt.exists():
            resume_path = str(last_ckpt)
            print(f'[Resume] {resume_path}')
        else:
            print('[Resume] last.ckpt not found, starting fresh')
    elif args.resume:
        resume_path = args.resume
        print(f'[Resume] {resume_path}')

    callbacks = [
        EpochSummaryCallback(),
        # Model selection is validation-only. save_top_k=-1 keeps every epoch
        # so that the val-optimal checkpoint can be identified after the fact;
        # the reported checkpoint is always argmin of val/combined_mae.
        ModelCheckpoint(
            dirpath=exp_dir,
            filename='epoch_{epoch:03d}_mae_{val/combined_mae:.4f}',
            monitor='val/combined_mae', mode='min',
            save_top_k=-1, save_last=True,
        ),
        LearningRateMonitor(logging_interval='epoch'),
    ]
    if args.early_stopping > 0:
        callbacks.append(EarlyStopping(monitor='val/combined_mae', patience=args.early_stopping, mode='min'))

    from pytorch_lightning.loggers import CSVLogger
    logger_tb  = TensorBoardLogger(save_dir=args.checkpoint_dir, name=args.exp_name)
    logger_csv = CSVLogger(save_dir=args.checkpoint_dir, name=args.exp_name)

    import torch
    torch.set_float32_matmul_precision('high')
    if torch.cuda.is_available():
        acc, dev, prec = 'gpu', 1, '16-mixed'
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
    else:
        acc, dev, prec = 'cpu', 'auto', '32'

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=acc, devices=dev, precision=prec,
        callbacks=callbacks, logger=[logger_tb, logger_csv],
        gradient_clip_val=1.0, log_every_n_steps=50,
    )

    trainer.fit(lit, datamodule=datamodule, ckpt_path=resume_path)
    print("[Test] Final evaluation on best checkpoint")
    trainer.test(lit, datamodule=datamodule, ckpt_path='best')
    print("Done.")


if __name__ == '__main__':
    main()
