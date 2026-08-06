"""Run inference on the test split with three model configurations
(full / no-BGTD / no-MCL), save per-sample predictions, then stratify
test residuals by lambda_p (BF) bins and compute per-bin BH RMSE.

Outputs:
  - predictions cache .npz for each model (full / nobgtd / nomcl)
  - q2_stratification.png : 2-panel figure
       (a) per-bin BH RMSE for each model (line plot)
       (b) BGTD/MCL gain (Δ vs full) per bin
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from morphoformer import MorphoFormerDataModule, MorphoFormerLitModule, build_morphoformer, build_mcl_loss
from morphoformer.paths import cache_root, checkpoint_root, results_root, split_paths

CKPT_BASE = str(checkpoint_root())
CKPTS = {
    'full':   f'{CKPT_BASE}/v2_patch2_aug/epoch_epoch=102_mae_val/combined_mae=1.9262.ckpt',
    'nobgtd': f'{CKPT_BASE}/v2_patch2_aug_noBGTD/epoch_epoch=077_mae_val/combined_mae=1.9401.ckpt',
    'nomcl':  f'{CKPT_BASE}/v2_patch2_aug_noMCL/epoch_epoch=075_mae_val/combined_mae=1.9393.ckpt',
}
OUT_DIR = results_root()
OUT_DIR.mkdir(exist_ok=True)


def build_model(no_bgtd: bool):
    in_chans = 2 + 4 + 1 + 1  # SAR + Optical + DEM + mask
    return build_morphoformer(
        variant='base',
        in_chans=in_chans,
        patch_grid_size=9,
        patch_size=2,
        center_sizes=(3, 5, 9),
        use_uncertainty=True,
        use_amge=True,
        use_bgtd=not no_bgtd,
    )


def run_inference(ckpt: str, no_bgtd: bool, dm, save_path: Path):
    if save_path.exists():
        print(f'  cache hit: {save_path}')
        d = np.load(save_path)
        return d['bh_pred'], d['bf_pred'], d['bh_true'], d['bf_true']

    model = build_model(no_bgtd=no_bgtd)
    loss_fn = build_mcl_loss(loss_type='huber', lambda_consist=0.2, warmup_epochs=10)
    lit = MorphoFormerLitModule.load_from_checkpoint(ckpt, model=model, loss_fn=loss_fn, strict=True)
    lit = lit.cuda().eval()

    bh_p, bf_p, bh_t, bf_t = [], [], [], []
    dm.setup('test')
    with torch.no_grad():
        for batch in dm.test_dataloader():
            x, bh, bf = batch
            x = x.cuda(non_blocking=True)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outs = lit.model(x)
            # Outputs: 5-tuple if BGTD+uncertainty, 3-tuple if no BGTD+uncertainty
            bh_pred = outs[0].squeeze(-1).float().cpu().numpy()
            bf_pred = outs[1].squeeze(-1).float().cpu().numpy()
            bh_p.append(bh_pred); bf_p.append(bf_pred)
            bh_t.append(bh.numpy()); bf_t.append(bf.numpy())
    bh_p = np.concatenate(bh_p)
    bf_p = np.concatenate(bf_p)
    bh_t = np.concatenate(bh_t)
    bf_t = np.concatenate(bf_t)

    np.savez_compressed(save_path,
                        bh_pred=bh_p, bf_pred=bf_p,
                        bh_true=bh_t, bf_true=bf_t)
    print(f'  saved: {save_path}  n={len(bh_p):,}')
    return bh_p, bf_p, bh_t, bf_t


def main():
    pl.seed_everything(42)
    torch.set_float32_matmul_precision('high')

    dm = MorphoFormerDataModule(
        train_path=str(split_paths()['train']),
        val_path=str(split_paths()['val']),
        test_path=str(split_paths()['test']),
        batch_size=256, num_workers=4,
        patch_grid_size=9, modalities=['sar', 'optical', 'dem'],
        preload=False,
        cache_dir=str(cache_root()),
        preload_cache=True,
        chunked=False,
    )

    preds = {}
    for name, ckpt in CKPTS.items():
        print(f'== {name} ==')
        cache = OUT_DIR / f'preds_{name}.npz'
        no_bgtd = (name == 'nobgtd')
        preds[name] = run_inference(ckpt, no_bgtd, dm, cache)

    # ── stratify by lambda_p (BF) into bins ──
    bh_t = preds['full'][2]   # ground truth identical across configs
    bf_t = preds['full'][3]
    bf_bins = [(0.01, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.35),
               (0.35, 0.55), (0.55, 1.0)]
    bin_labels = [f'{lo:.2f}-{hi:.2f}' for lo, hi in bf_bins]
    bin_centers = [0.5*(lo+hi) for lo, hi in bf_bins]
    n_bins = len(bf_bins)

    rmse = {name: np.zeros(n_bins) for name in preds}
    n_per_bin = np.zeros(n_bins, dtype=int)
    for k, (lo, hi) in enumerate(bf_bins):
        sel = (bf_t >= lo) & (bf_t < hi)
        n_per_bin[k] = sel.sum()
        for name, (bh_p, _, _, _) in preds.items():
            err = bh_p[sel] - bh_t[sel]
            rmse[name][k] = np.sqrt((err**2).mean()) if sel.sum() else float('nan')

    # ── plot ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2),
                                   gridspec_kw={'width_ratios': [1.1, 1.0]})
    colors = {'full': '#1f77b4', 'nobgtd': '#d62728', 'nomcl': '#2ca02c'}
    labels = {'full': 'MorphoFormer (full)',
              'nobgtd': 'w/o BGTD',
              'nomcl': 'w/o MCL'}

    for name in ['full', 'nobgtd', 'nomcl']:
        ax1.plot(bin_centers, rmse[name], 'o-', color=colors[name],
                 linewidth=1.7, markersize=7, label=labels[name])
    ax1.set_xlabel('Footprint ratio $\\lambda_p$ (test bin centre)', fontsize=10)
    ax1.set_ylabel('BH RMSE (m)', fontsize=10)
    ax1.set_title(f'(a) BH RMSE stratified by $\\lambda_p$', fontsize=10)
    ax1.legend(fontsize=9, loc='upper left', frameon=False)
    ax1.grid(True, alpha=0.3)
    # annotate sample counts
    for k, (xc, n) in enumerate(zip(bin_centers, n_per_bin)):
        ax1.annotate(f'n={n//1000}k', (xc, rmse['full'][k]),
                     textcoords='offset points', xytext=(0, -16),
                     fontsize=7.5, ha='center', alpha=0.65)

    # delta plot
    delta_bgtd = rmse['nobgtd'] - rmse['full']
    delta_mcl  = rmse['nomcl']  - rmse['full']
    width = 0.35
    x = np.arange(n_bins)
    ax2.bar(x - width/2, delta_bgtd, width, color=colors['nobgtd'],
            alpha=0.85, label='Δ from removing BGTD')
    ax2.bar(x + width/2, delta_mcl, width, color=colors['nomcl'],
            alpha=0.85, label='Δ from removing MCL')
    ax2.axhline(0, color='black', linewidth=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(bin_labels, fontsize=8.5, rotation=20)
    ax2.set_xlabel('Footprint ratio $\\lambda_p$ bin', fontsize=10)
    ax2.set_ylabel('BH RMSE increase (m)', fontsize=10)
    ax2.set_title('(b) Per-bin contribution of BGTD and MCL', fontsize=10)
    ax2.legend(fontsize=9, loc='upper right', frameon=False)
    ax2.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    out_png = OUT_DIR / 'q2_stratification.png'
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    print(f'\nsaved {out_png}')

    # print numerical summary
    print('\n=== Per-bin BH RMSE ===')
    print(f'{"BF bin":<14s} {"n":>8s}  {"full":>7s}  {"noBGTD":>7s} {"+":>5s}  {"noMCL":>7s} {"+":>5s}')
    for k, (lab, n) in enumerate(zip(bin_labels, n_per_bin)):
        print(f'{lab:<14s} {n:>8d}  {rmse["full"][k]:7.3f}  '
              f'{rmse["nobgtd"][k]:7.3f} {delta_bgtd[k]:+5.3f}  '
              f'{rmse["nomcl"][k]:7.3f} {delta_mcl[k]:+5.3f}')


if __name__ == '__main__':
    main()
