"""Error source analysis: 2D heatmaps of MorphoFormer's BH/BF errors
binned over the (lambda_p, H_ave) plane, alongside the empirical sample
density. Argues that high errors concentrate where training samples are
sparse (large+tall buildings), implicating the dataset rather than the
model.

Outputs:
  - error_source.png : 1 x 3 grid
       (a) sample count log-scale heatmap (per (BH bin, BF bin))
       (b) BH RMSE per bin (m)
       (c) BF RMSE per bin
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from morphoformer.paths import results_root

OUT_DIR = results_root()
PNG = OUT_DIR / 'error_source.png'

d = np.load(OUT_DIR / 'preds_full.npz')
bh_p = d['bh_pred']; bf_p = d['bf_pred']
bh_t = d['bh_true']; bf_t = d['bf_true']

# 2D bin edges
bf_edges = np.array([0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 1.00])
bh_edges = np.array([2, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 200])
n_bf = len(bf_edges) - 1
n_bh = len(bh_edges) - 1

count = np.zeros((n_bh, n_bf), dtype=int)
bh_rmse = np.full((n_bh, n_bf), np.nan)
bf_rmse = np.full((n_bh, n_bf), np.nan)

bf_idx = np.clip(np.digitize(bf_t, bf_edges[1:-1]), 0, n_bf - 1)
bh_idx = np.clip(np.digitize(bh_t, bh_edges[1:-1]), 0, n_bh - 1)

for i in range(n_bh):
    for j in range(n_bf):
        m = (bh_idx == i) & (bf_idx == j)
        n = m.sum()
        count[i, j] = n
        if n >= 5:
            bh_rmse[i, j] = np.sqrt(((bh_p[m] - bh_t[m])**2).mean())
            bf_rmse[i, j] = np.sqrt(((bf_p[m] - bf_t[m])**2).mean())

# ── plot ──
fig, axes = plt.subplots(1, 3, figsize=(17.5, 4.9),
                         gridspec_kw={'wspace': 0.55})

bf_centers_lab = [f'{bf_edges[i]:.2f}\n-{bf_edges[i+1]:.2f}' for i in range(n_bf)]
bh_centers_lab = [f'{bh_edges[i]}-{bh_edges[i+1]}' for i in range(n_bh)]

# (a) sample count (log scale)
im0 = axes[0].imshow(np.where(count > 0, count, 0.5),
                     origin='lower', aspect='auto', cmap='Blues',
                     norm=LogNorm(vmin=1, vmax=max(count.max(), 1)))
cb0 = plt.colorbar(im0, ax=axes[0], fraction=0.045, pad=0.03)
cb0.set_label('Test sample count (log scale)', fontsize=9)
axes[0].set_title('(a) Sample density in $(\\lambda_p, H_{\\mathrm{ave}})$',
                  fontsize=10.5)

# (b) BH RMSE
im1 = axes[1].imshow(bh_rmse, origin='lower', aspect='auto', cmap='YlOrRd',
                     vmin=0, vmax=np.nanpercentile(bh_rmse, 95))
cb1 = plt.colorbar(im1, ax=axes[1], fraction=0.045, pad=0.03)
cb1.set_label('BH RMSE (m)', fontsize=9)
axes[1].set_title('(b) BH RMSE per bin', fontsize=10.5)

# (c) BF RMSE
im2 = axes[2].imshow(bf_rmse, origin='lower', aspect='auto', cmap='YlGn',
                     vmin=0, vmax=np.nanpercentile(bf_rmse, 95))
cb2 = plt.colorbar(im2, ax=axes[2], fraction=0.045, pad=0.03)
cb2.set_label('BF RMSE', fontsize=9)
axes[2].set_title('(c) BF RMSE per bin', fontsize=10.5)

for ax in axes:
    ax.set_xticks(np.arange(n_bf))
    ax.set_xticklabels(bf_centers_lab, fontsize=7.5, rotation=0)
    ax.set_yticks(np.arange(n_bh))
    ax.set_yticklabels(bh_centers_lab, fontsize=7.5)
    ax.set_xlabel('Footprint ratio $\\lambda_p$ bin', fontsize=10)
    ax.set_ylabel('Building height $H_{\\mathrm{ave}}$ bin (m)', fontsize=10)
    # mark bins with too few samples
    for i in range(n_bh):
        for j in range(n_bf):
            if 0 < count[i, j] < 5:
                ax.text(j, i, '·', ha='center', va='center',
                        color='lightgray', fontsize=10)
            elif count[i, j] == 0:
                ax.text(j, i, '×', ha='center', va='center',
                        color='lightgray', fontsize=8)

# annotate sample counts on (a)
for i in range(n_bh):
    for j in range(n_bf):
        if count[i, j] >= 1:
            axes[0].text(j, i, f'{count[i, j]:,}'.replace(',', ''),
                         ha='center', va='center', fontsize=6.5,
                         color='black' if count[i, j] < 1000 else 'white')

fig.savefig(PNG, dpi=200, bbox_inches='tight')
print(f'saved {PNG}')
print(f'\n=== Sample count totals ===')
print(f'  total cells   : {count.sum():,}')
print(f'  cells in [BH>30, BF>0.3]   : {count[bh_edges[:-1]>30][:, bf_edges[:-1]>0.3].sum():,}')
print(f'  cells in [BH>50, BF>0.5]   : {count[bh_edges[:-1]>50][:, bf_edges[:-1]>0.5].sum():,}')
