"""Extract the height-from-footprint surrogate (BH-from-BF) on the
test split and compare:
  - main BH prediction vs surrogate BH-from-BF (correlation)
  - surrogate vs ground truth BH (R², RMSE, scatter)

Outputs:
  - q4_surrogate.png : 2-panel figure
       (a) hexbin scatter: ground-truth BH vs BH-from-BF surrogate
       (b) main-pred vs surrogate (do they agree?), separated by BF bin
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from morphoformer import MorphoFormerDataModule, MorphoFormerLitModule, build_morphoformer, build_mcl_loss
from morphoformer.paths import cache_root, checkpoint_root, results_root, split_paths

CKPT = str(checkpoint_root() / 'morphoformer_full.ckpt')
OUT_DIR = results_root()
SURR_NPZ = OUT_DIR / 'preds_surrogate.npz'
PNG = OUT_DIR / 'q4_surrogate.png'


def main():
    pl.seed_everything(42)
    torch.set_float32_matmul_precision('high')

    if SURR_NPZ.exists():
        d = np.load(SURR_NPZ)
        bh_main = d['bh_main']; bh_surr = d['bh_surr']; bh_t = d['bh_true']; bf_t = d['bf_true']
        print(f'cache hit: n={len(bh_main):,}')
    else:
        in_chans = 2 + 4 + 1 + 1
        model = build_morphoformer(
            variant='base', in_chans=in_chans,
            patch_grid_size=9, patch_size=2, center_sizes=(3, 5, 9),
            use_uncertainty=True, use_amge=True, use_bgtd=True,
        )
        loss_fn = build_mcl_loss(loss_type='huber', lambda_consist=0.2, warmup_epochs=10)
        lit = MorphoFormerLitModule.load_from_checkpoint(CKPT, model=model, loss_fn=loss_fn, strict=True)
        lit = lit.cuda().eval()

        dm = MorphoFormerDataModule(
            train_path=str(split_paths()['train']),
            val_path=str(split_paths()['val']),
            test_path=str(split_paths()['test']),
            batch_size=256, num_workers=4,
            patch_grid_size=9, modalities=['sar', 'optical', 'dem'],
            preload=False,
            cache_dir=str(cache_root()),
            preload_cache=True, chunked=False,
        )
        dm.setup('test')

        bh_main, bh_surr, bh_t, bf_t = [], [], [], []
        with torch.no_grad():
            for batch in dm.test_dataloader():
                x, bh, bf = batch
                x = x.cuda(non_blocking=True)
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    outs = lit.model(x)
                # outs = (bh_pred, bf_pred, log_sigma_h, log_sigma_f, bh_from_bf)
                bh_main.append(outs[0].squeeze(-1).float().cpu().numpy())
                bh_surr.append(outs[4].squeeze(-1).float().cpu().numpy())
                bh_t.append(bh.numpy()); bf_t.append(bf.numpy())
        bh_main = np.concatenate(bh_main)
        bh_surr = np.concatenate(bh_surr)
        bh_t = np.concatenate(bh_t)
        bf_t = np.concatenate(bf_t)
        np.savez_compressed(SURR_NPZ, bh_main=bh_main, bh_surr=bh_surr,
                            bh_true=bh_t, bf_true=bf_t)
        print(f'saved {SURR_NPZ}: n={len(bh_main):,}')

    # ─── analysis ───
    surr_rmse = np.sqrt(((bh_surr - bh_t)**2).mean())
    surr_mae  = np.abs(bh_surr - bh_t).mean()
    ss_res = ((bh_t - bh_surr)**2).sum()
    ss_tot = ((bh_t - bh_t.mean())**2).sum()
    surr_r2 = 1 - ss_res / ss_tot
    rho_main_surr, _ = stats.pearsonr(bh_main, bh_surr)
    rho_surr_truth, _ = stats.pearsonr(bh_surr, bh_t)
    rho_main_truth, _ = stats.pearsonr(bh_main, bh_t)

    print(f'\nSurrogate (BH-from-BF) vs ground truth BH:')
    print(f'  RMSE = {surr_rmse:.3f} m   MAE = {surr_mae:.3f} m   R² = {surr_r2:.3f}')
    print(f'  corr(surrogate, truth)  = {rho_surr_truth:.3f}')
    print(f'  corr(surrogate, main)   = {rho_main_surr:.3f}')
    print(f'  corr(main,      truth)  = {rho_main_truth:.3f}')

    # ─── plot ───
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.7),
                                   gridspec_kw={'width_ratios': [1.0, 1.0]})

    # (a) hexbin truth vs surrogate
    bound = 35
    h = ax1.hexbin(bh_t, bh_surr, gridsize=70, bins='log', cmap='viridis',
                   extent=(2, bound, 0, bound), mincnt=1, linewidths=0)
    ax1.plot([0, bound], [0, bound], 'r--', lw=1, alpha=0.85, label='y = x')
    ax1.set_xlabel('Ground-truth $H_{\\mathrm{ave}}$ (m)', fontsize=10)
    ax1.set_ylabel('Surrogate $\\widehat{H}_{\\mathrm{from\\,BF}}$ (m)', fontsize=10)
    ax1.set_title(f'(a) Surrogate vs ground truth\n'
                  f'RMSE={surr_rmse:.2f} m,  $R^2$={surr_r2:.2f},  '
                  f'$\\rho$={rho_surr_truth:.2f}', fontsize=10)
    cb = plt.colorbar(h, ax=ax1, fraction=0.046, pad=0.04)
    cb.set_label('log$_{10}$ count', fontsize=9)
    cb.ax.tick_params(labelsize=8)
    ax1.set_xlim(2, bound); ax1.set_ylim(0, bound)
    ax1.legend(loc='lower right', fontsize=9, frameon=False)
    ax1.tick_params(labelsize=8.5)

    # (b) surrogate vs main pred
    h2 = ax2.hexbin(bh_main, bh_surr, gridsize=70, bins='log', cmap='plasma',
                    extent=(0, bound, 0, bound), mincnt=1, linewidths=0)
    ax2.plot([0, bound], [0, bound], 'cyan', linestyle='--', lw=1, alpha=0.95, label='y = x')
    ax2.set_xlabel('Main BH prediction $\\hat{H}_{\\mathrm{ave}}$ (m)', fontsize=10)
    ax2.set_ylabel('Surrogate $\\widehat{H}_{\\mathrm{from\\,BF}}$ (m)', fontsize=10)
    ax2.set_title(f'(b) Surrogate vs main prediction\n'
                  f'$\\rho$ = {rho_main_surr:.2f}', fontsize=10)
    cb = plt.colorbar(h2, ax=ax2, fraction=0.046, pad=0.04)
    cb.set_label('log$_{10}$ count', fontsize=9)
    cb.ax.tick_params(labelsize=8)
    ax2.set_xlim(0, bound); ax2.set_ylim(0, bound)
    ax2.legend(loc='lower right', fontsize=9, frameon=False)
    ax2.tick_params(labelsize=8.5)

    fig.tight_layout()
    fig.savefig(PNG, dpi=200, bbox_inches='tight')
    print(f'\nsaved {PNG}')


if __name__ == '__main__':
    main()
