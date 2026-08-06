"""ViT-Small MTL baseline training: global attention vs. windowed (Swin).

Identical data / split / loss / evaluation pipeline as scripts/train.py, but
builds a global-attention ViT-Small backbone (timm vit_small_patch8) with two
parallel regression heads, at the same 9x9 / 90x90 receptive field as every
other row of the comparison table. The MCL consistency term is structurally inactive (the model
returns a 4-tuple with no bh_from_bf), so this is a plain global-attention
Transformer baseline directly comparable to the windowed-attention Swin-MTL.

Example (same 9x9 RF, matching the baseline protocol):
    python scripts/train_vit.py --exp-name vit_patchgrid9 \
        --patch-grid-size 9 --batch-size 512 --lr 2e-4 --epochs 100 \
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

# v2 package (parent of scripts/) + repo root (for the v1 ViTMTL definition)
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from morphoformer import MorphoFormerDataModule, MorphoFormerLitModule, build_mcl_loss
from morphoformer.models.cnn_baselines import ViTMTL
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
    p = argparse.ArgumentParser(description='ViT-Small MTL baseline training')

    _DATA = str(data_root())
    p.add_argument('--train-path', default=f'{_DATA}/train.h5')
    p.add_argument('--val-path',   default=f'{_DATA}/valid.h5')
    p.add_argument('--test-path',  default=f'{_DATA}/test2.h5')

    # ── Model ──
    p.add_argument('--patch-grid-size', type=int, default=9, choices=[3, 5, 7, 9])
    p.add_argument('--modalities',      nargs='+', default=['sar', 'optical', 'dem'])
    p.add_argument('--no-uncertainty',  action='store_true')

    # ── Loss (consistency term is inactive for ViT regardless) ──
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
    p.add_argument('--exp-name',  default='vit_patchgrid9')
    p.add_argument('--resume',    default=None, help='Path or "last"')
    p.add_argument('--seed',      type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed)

    modality_channels = {'sar': 2, 'optical': 4, 'dem': 1}
    in_chans = sum(modality_channels[m] for m in args.modalities) + 1  # +1 mask
    img_size = args.patch_grid_size * 10  # 9 cells x 10 px = 90x90 (same RF as Swin)

    print(f"[Model] ViT-S MTL | patch_grid={args.patch_grid_size} | "
          f"img_size={img_size} | in_chans={in_chans}")

    model = ViTMTL(
        in_chans=in_chans,
        img_size=img_size,
        use_uncertainty=not args.no_uncertainty,
        pretrained=False,
    )

    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Trainable params: {n:,} ({n/1e6:.3f} M)")

    # Same loss object; with bh_from_bf absent the consistency term never fires.
    loss_fn = build_mcl_loss(loss_type=args.loss_type, lambda_consist=0.0)
    print(f"[Loss] uncertainty-weighted MTL | type={args.loss_type} "
          f"| consistency inactive (ViT has no bh_from_bf)")

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
