"""Dump per-cell test-set predictions to NPZ.

This is the single evaluation routine: MorphoFormer, its ablations, the extra
seeds and every re-trained baseline are all pushed through *this* function, so
that all reported metrics are strictly commensurable. `analysis/metrics.py`
then computes the tables from the resulting `.npz` files.

Exception: the Swin-MTL row was produced under the first-generation codebase,
which used its own band normalisation, so it cannot be regenerated here. Its
prediction dump is shipped directly in `results/`. See docs/reproducibility.md.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import torch
import pytorch_lightning as pl

sys.path.insert(0, str(Path(__file__).parent.parent))
from morphoformer import MorphoFormerDataModule, MorphoFormerLitModule, build_morphoformer, build_mcl_loss
from morphoformer.models import build_crossstitch, build_mfbhnet, build_baseline
from morphoformer.paths import cache_root, data_root


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--model', default='morphoformer',
                   choices=['morphoformer', 'crossstitch', 'mfbhnet', 'resnet', 'senet', 'vit'])
    p.add_argument('--patch-size', type=int, default=2)
    p.add_argument('--patch-grid-size', type=int, default=9)
    p.add_argument('--center-sizes', type=int, nargs='+', default=[3, 5, 9])
    p.add_argument('--modalities', nargs='+', default=['sar', 'optical', 'dem'])
    p.add_argument('--no-amge', action='store_true')
    p.add_argument('--no-bgtd', action='store_true')
    p.add_argument('--no-uncertainty', action='store_true')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--precision', default='32', choices=['32', '16-mixed'],
                   help='Inference precision. fp32 is the default; the difference '
                        'between the two is well under 0.01 m of BH RMSE.')
    p.add_argument('--cache-dir', default=str(cache_root()))
    _DATA = str(data_root())
    p.add_argument('--test-path', default=f'{_DATA}/test2.h5')
    p.add_argument('--train-path', default=f'{_DATA}/train.h5')
    p.add_argument('--val-path', default=f'{_DATA}/valid.h5')
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
            variant='base', in_chans=in_chans, patch_grid_size=args.patch_grid_size,
            patch_size=args.patch_size, center_sizes=tuple(args.center_sizes),
            use_uncertainty=not args.no_uncertainty, use_amge=not args.no_amge)
    elif args.model == 'mfbhnet':
        model = build_mfbhnet(in_chans=in_chans, use_uncertainty=not args.no_uncertainty)
    elif args.model in ('resnet', 'senet', 'vit'):
        extra = {'img_size': args.patch_grid_size * 10} if args.model == 'vit' else {}
        model = build_baseline(args.model, in_chans=in_chans,
                               use_uncertainty=not args.no_uncertainty, **extra)
    else:
        model = build_morphoformer(
            variant='base', in_chans=in_chans, patch_grid_size=args.patch_grid_size,
            patch_size=args.patch_size, center_sizes=tuple(args.center_sizes),
            use_uncertainty=not args.no_uncertainty, use_amge=not args.no_amge,
            use_bgtd=not args.no_bgtd)

    loss_fn = build_mcl_loss(loss_type='huber', lambda_consist=0.2, warmup_epochs=10)
    dm = MorphoFormerDataModule(
        train_path=args.train_path, val_path=args.val_path, test_path=args.test_path,
        batch_size=args.batch_size, num_workers=args.num_workers,
        patch_grid_size=args.patch_grid_size, modalities=args.modalities,
        preload=False, cache_dir=args.cache_dir, preload_cache=False, chunked=False)
    dm.setup(stage='test')
    loader = dm.test_dataloader()

    lit = MorphoFormerLitModule.load_from_checkpoint(
        args.ckpt, model=model, loss_fn=loss_fn, strict=True)
    lit.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lit = lit.to(device)

    use_half = args.precision == '16-mixed'
    bh_p, bf_p, bh_t, bf_t = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            if use_half:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    out = lit.model(x)
            else:
                out = lit.model(x)
            bh_p.append(out[0].squeeze().float().cpu().numpy())
            bf_p.append(out[1].squeeze().float().cpu().numpy())
            bh_t.append(batch[1].numpy())
            bf_t.append(batch[2].numpy())

    np.savez(args.out,
             bh_true=np.concatenate(bh_t).astype(np.float32),
             bh_pred=np.concatenate(bh_p).astype(np.float32),
             bf_true=np.concatenate(bf_t).astype(np.float32),
             bf_pred=np.concatenate(bf_p).astype(np.float32))
    n = len(np.concatenate(bh_t))
    print(f"Saved {n} cells -> {args.out} (precision={args.precision})")


if __name__ == '__main__':
    main()
