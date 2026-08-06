"""Hexbin scatter comparison: predicted vs ground truth for each ablation
on the test split. Replicates the panel-of-hexbins style with metric box
in each panel.

Outputs:
  - hexbin_compare.png : 2 x 3 grid
       row 1: BH predictions for full / no-BGTD / no-MCL
       row 2: BF predictions for full / no-BGTD / no-MCL
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from morphoformer.paths import results_root

OUT_DIR = results_root()
PNG = OUT_DIR / 'hexbin_compare.png'

CONFIGS = [
    ('full',   'MorphoFormer (full)'),
    ('nobgtd', 'w/o BGTD'),
    ('nomcl',  'w/o MCL'),
    ('resnet', 'ResNet-MTL'),
    ('senet',  'SENet-MTL'),
]


def load(name):
    d = np.load(OUT_DIR / f'preds_{name}.npz')
    return d['bh_pred'], d['bf_pred'], d['bh_true'], d['bf_true']


def metric_box(y_true, y_pred):
    rmse = np.sqrt(((y_pred - y_true)**2).mean())
    mae = np.abs(y_pred - y_true).mean()
    me = (y_pred - y_true).mean()
    ss_res = ((y_true - y_pred)**2).sum()
    ss_tot = ((y_true - y_true.mean())**2).sum()
    r2 = 1 - ss_res / ss_tot
    cc = np.corrcoef(y_true, y_pred)[0, 1]
    return rmse, mae, me, r2, cc


def hexbin_panel(ax, x, y, *, log=False, ylim=None, xlabel='', ylabel='',
                 title=''):
    if log:
        ax.set_xscale('log'); ax.set_yscale('log')
        h = ax.hexbin(x, y, xscale='log', yscale='log',
                      gridsize=70, bins='log', cmap='turbo',
                      mincnt=1, linewidths=0)
        lim = (1.5, 80)
    else:
        h = ax.hexbin(x, y, gridsize=70, bins='log', cmap='turbo',
                      mincnt=1, linewidths=0)
        lim = (0, 1) if ylim == (0, 1) else (0, max(x.max(), y.max()))
    ax.plot(lim, lim, 'r--', lw=0.9, alpha=0.85)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(xlabel, fontsize=9.5)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.set_title(title, fontsize=10.5)
    ax.tick_params(labelsize=8.5)
    return h


def add_metric_box(ax, rmse, mae, me, r2, cc, *, fmt='{:.2f}'):
    txt = (f'RMSE: {fmt.format(rmse)}, MAE: {fmt.format(mae)}\n'
           f'ME: {fmt.format(me)}\n'
           f'CC: {cc:.2f}, R²: {r2:.2f}')
    ax.text(0.04, 0.96, txt, transform=ax.transAxes,
            fontsize=8, va='top', ha='left',
            bbox=dict(facecolor='white', edgecolor='gray',
                      boxstyle='round,pad=0.3', alpha=0.92))


fig, axes = plt.subplots(2, 5, figsize=(20.0, 8.5),
                         gridspec_kw={'hspace': 0.30, 'wspace': 0.22})

# preload all
preds = {name: load(name) for name, _ in CONFIGS}

# row 1: BH (log scale)
for col, (name, label) in enumerate(CONFIGS):
    bh_p, _, bh_t, _ = preds[name]
    # filter for log plot (BH ≥ 2)
    sel = (bh_t >= 2) & (bh_p >= 1)
    h = hexbin_panel(
        axes[0, col], bh_t[sel], bh_p[sel], log=True,
        xlabel='Ground-truth $H_{\\mathrm{ave}}$ (m)' if col == 0 else '',
        ylabel='Predicted $H_{\\mathrm{ave}}$ (m)' if col == 0 else '',
        title=f'BH: {label}',
    )
    rmse, mae, me, r2, cc = metric_box(bh_t, bh_p)
    add_metric_box(axes[0, col], rmse, mae, me, r2, cc, fmt='{:.2f}')

# row 2: BF (linear scale)
for col, (name, label) in enumerate(CONFIGS):
    _, bf_p, _, bf_t = preds[name]
    h = hexbin_panel(
        axes[1, col], bf_t, bf_p, log=False, ylim=(0, 1),
        xlabel='Ground-truth $\\lambda_p$' if col == 0 else '',
        ylabel='Predicted $\\lambda_p$' if col == 0 else '',
        title=f'BF: {label}',
    )
    axes[1, col].set_xlim(0, 1); axes[1, col].set_ylim(0, 1)
    rmse, mae, me, r2, cc = metric_box(bf_t, bf_p)
    add_metric_box(axes[1, col], rmse, mae, me, r2, cc, fmt='{:.3f}')

# shared colorbar (just for the last hexbin object)
cbar_ax = fig.add_axes([0.30, 0.04, 0.40, 0.018])
cb = fig.colorbar(h, cax=cbar_ax, orientation='horizontal')
cb.set_label('log$_{10}$ count per bin', fontsize=9)
cb.ax.tick_params(labelsize=8)

fig.subplots_adjust(bottom=0.13, top=0.95, left=0.07, right=0.97)
fig.savefig(PNG, dpi=200, bbox_inches='tight')
print(f'saved {PNG}')
