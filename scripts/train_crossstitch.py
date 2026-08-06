"""Cross-Stitch MTL baseline training.

Identical data / split / loss / evaluation pipeline as scripts/train.py, but
builds the symmetric cross-stitch model (same AMGE + Swin + MSMP encoder, BGTD
replaced by cross-stitch coupling). The MCL consistency term is structurally
inactive (the model returns no bh_from_bf), so this is a pure symmetric
task-interaction baseline.

Example (matches the full-model patch=2 config):
    python scripts/train_crossstitch.py --exp-name crossstitch_patch2 \
        --patch-size 2 --batch-size 512 --lr 2e-4 --epochs 100 \
        --early-stopping 15 --num-workers 8 --preload-cache
"""

import argparse
import sys
import math
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor, Callback)
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

sys.path.insert(0, str(Path(__file__).parent.parent))

from morphoformer import MorphoFormerDataModule, MorphoFormerLitModule, build_mcl_loss
from morphoformer.models import build_crossstitch
from morphoformer.paths import cache_root, data_root


class EpochSummaryCallback(Callback):
    """Print val + test metrics after each validation epoch (mirrors train.py)."""

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
            f'  |  test MAE_h={fmt(g("test_epoch/mae_h"))} R²={fmt(g("test_epoch/r2_h"))}'
            f' MAE_f={fmt(g("test_epoch/mae_f"), 8, 4)}'
            f'  lr={g("lr-AdamW"):.2e}',
            flush=True,
        )


def parse_args():
    p = argparse.ArgumentParser(description='Cross-Stitch MTL baseline training')

    _DATA = str(data_root())
    p.add_argument('--train-path', default=f'{_DATA}/train.h5')
    p.add_argument('--val-path',   default=f'{_DATA}/valid.h5')
    p.add_argument('--test-path',  default=f'{_DATA}/test2.h5')

    # ── Model (kept identical to train.py where shared) ──
    p.add_argument('--variant',         default='base', choices=['base', 'large'])
    p.add_argument('--patch-grid-size', type=int, default=9, choices=[3, 5, 7, 9])
    p.add_argument('--patch-size',      type=int, default=10)
    p.add_argument('--center-sizes',    type=int, nargs='+', default=[3, 5, 9])
    p.add_argument('--modalities',      nargs='+', default=['sar', 'optical', 'dem'])
    p.add_argument('--no-uncertainty',  action='store_true')
    p.add_argument('--no-amge',         action='store_true',
                   help='Disable AMGE (default keeps it, matching the full model)')

    # ── Loss (consistency term is inactive for cross-stitch regardless) ──
    p.add_argument('--loss-type',       default='huber', choices=['mse', 'huber', 'l1'])

    # ── Training ──
    p.add_argument('--batch-size',    type=int,   default=256)
    p.add_argument('--epochs',        type=int,   default=100)
    p.add_argument('--lr',            type=float, default=1e-4)
    p.add_argument('--weight-decay',  type=float, default=1e-2)
    p.add_argument('--num-workers',   type=int,   default=4)
    p.add_argument('--preload-cache', action='store_true')
    p.add_argument('--cache-dir',
                   default=str(cache_root()))
    p.add_argument('--early-stopping', type=int,  default=15)
    p.add_argument('--no-augment',    action='store_true')

    p.add_argument('--checkpoint-dir',
                   default=str(Path(__file__).parent.parent / 'checkpoints'))
    p.add_argument('--exp-name',  default='crossstitch_patch2')
    p.add_argument('--resume',    default=None, help='Path or "last"')
    p.add_argument('--seed',      type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed)

    modality_channels = {'sar': 2, 'optical': 4, 'dem': 1}
    in_chans = sum(modality_channels[m] for m in args.modalities) + 1  # +1 mask

    print(f"[Model] CrossStitch-MTL | patch_grid={args.patch_grid_size} | "
          f"patch_size={args.patch_size} | center_sizes={args.center_sizes} | "
          f"AMGE={'on' if not args.no_amge else 'off'}")

    model = build_crossstitch(
        variant=args.variant,
        in_chans=in_chans,
        patch_grid_size=args.patch_grid_size,
        patch_size=args.patch_size,
        center_sizes=tuple(args.center_sizes),
        use_amge=not args.no_amge,
        use_uncertainty=not args.no_uncertainty,
    )

    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Trainable params: {n:,} ({n/1e6:.3f} M)")

    # Same loss object; with bh_from_bf=None the consistency term never fires.
    loss_fn = build_mcl_loss(loss_type=args.loss_type, lambda_consist=0.0)
    print(f"[Loss] uncertainty-weighted MTL | type={args.loss_type} "
          f"| consistency inactive (cross-stitch has no bh_from_bf)")

    datamodule = MorphoFormerDataModule(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patch_grid_size=args.patch_grid_size,
        modalities=args.modalities,
        cache_dir=args.cache_dir,
        preload_cache=args.preload_cache,
        augment=not args.no_augment,
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
        ModelCheckpoint(
            dirpath=exp_dir,
            filename='epoch_{epoch:03d}_mae_{val/combined_mae:.4f}',
            monitor='val/combined_mae', mode='min',
            save_top_k=-1, save_last=True,
        ),
        EarlyStopping(monitor='val/combined_mae',
                      patience=args.early_stopping, mode='min'),
        LearningRateMonitor(logging_interval='epoch'),
    ]

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
