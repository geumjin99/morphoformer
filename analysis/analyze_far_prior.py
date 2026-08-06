"""Empirical characterization of the (BH, BF) cross-task prior on the
training split, for paper_sors Section 'The Cross-Task Coupling Prior'.

Outputs:
  - far_prior_joint.png : 2-panel figure
       (a) hexbin of (BF, BH) over all training cells, with FAR iso-curves
       (b) conditional H_ave distributions sliced by BF bin
  - prints: mutual information I(BH;BF), Pearson rho, Spearman rho,
           variance reduction Var(BH)/E[Var(BH|BF)] - 1
"""
from __future__ import annotations

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.feature_selection import mutual_info_regression
from scipy import stats
from morphoformer.paths import results_root, split_paths

H5 = str(split_paths()['train'])
OUT = str(results_root() / 'far_prior_joint.png')

ASSUMED_FLOOR_HEIGHT_M = 3.0  # standard residential floor height for FAR

bh_all, bf_all = [], []
n_per_city = {}
with h5py.File(H5, 'r') as f:
    for city in f.keys():
        bh = f[city]['BuildingHeight'][:]
        bf = f[city]['BuildingFootprint'][:]
        bh_all.append(bh)
        bf_all.append(bf)
        n_per_city[city] = len(bh)

bh = np.concatenate(bh_all).astype(np.float64)
bf = np.concatenate(bf_all).astype(np.float64)

# match the paper's filtering: BH in [2,500], BF >= 0.01, no sliver-with-tall
mask = (bh >= 2.0) & (bh <= 500.0) & (bf > 0.01)
mask &= ~((bf < 0.04) & (bh >= 20.0))
bh, bf = bh[mask], bf[mask]
n_total = len(bh)
print(f'training cells retained: {n_total:,}  (across {len(n_per_city)} cities)')
print(f'  BH range: [{bh.min():.2f}, {bh.max():.2f}] m,  median {np.median(bh):.2f}')
print(f'  BF range: [{bf.min():.3f}, {bf.max():.3f}],     median {np.median(bf):.3f}')

# ---------------------------------------------------------------
# Quantify the coupling
# ---------------------------------------------------------------
rho_p, _ = stats.pearsonr(bh, bf)
rho_s, _ = stats.spearmanr(bh, bf)
mi = float(mutual_info_regression(bf.reshape(-1, 1), bh, random_state=0)[0])

# variance reduction: Var(BH) - E[Var(BH | BF bin)]   /  Var(BH)
n_bins = 20
bin_edges = np.quantile(bf, np.linspace(0, 1, n_bins + 1))
bin_idx = np.clip(np.digitize(bf, bin_edges[1:-1]), 0, n_bins - 1)
var_bh = bh.var()
cond_vars = np.array([bh[bin_idx == k].var() for k in range(n_bins)])
cond_weights = np.array([(bin_idx == k).sum() for k in range(n_bins)]) / len(bh)
expected_cond_var = (cond_vars * cond_weights).sum()
var_reduction = (var_bh - expected_cond_var) / var_bh

print(f'\nGlobal coupling diagnostics:')
print(f'  Pearson  rho(BH, BF) = {rho_p:.3f}')
print(f'  Spearman rho(BH, BF) = {rho_s:.3f}')
print(f'  Mutual information  I(BH; BF) = {mi:.3f} nats')
print(f'  Var(BH | BF) reduces total Var(BH) by {var_reduction*100:.1f} %')
print(f'  (Var(BH) = {var_bh:.2f},  E[Var(BH|BF)] = {expected_cond_var:.2f})')

# ---------------------------------------------------------------
# Within-city coupling: is the prior context-conditional?
# ---------------------------------------------------------------
city_rhos = []
city_var_reductions = []
city_n = []
city_names = []
with h5py.File(H5, 'r') as f:
    for city in f.keys():
        cb = f[city]['BuildingHeight'][:].astype(np.float64)
        cf = f[city]['BuildingFootprint'][:].astype(np.float64)
        m = (cb >= 2.0) & (cb <= 500.0) & (cf > 0.01)
        m &= ~((cf < 0.04) & (cb >= 20.0))
        cb, cf = cb[m], cf[m]
        if len(cb) < 200:
            continue
        rho, _ = stats.pearsonr(cb, cf)
        # within-city var reduction
        be = np.quantile(cf, np.linspace(0, 1, 11))
        bi = np.clip(np.digitize(cf, be[1:-1]), 0, 9)
        v = cb.var()
        if v < 1e-6:
            continue
        cv = np.array([cb[bi == k].var() if (bi == k).sum() > 5 else v
                       for k in range(10)])
        cw = np.array([(bi == k).sum() for k in range(10)]) / len(cb)
        vr = (v - (cv * cw).sum()) / v
        city_rhos.append(rho)
        city_var_reductions.append(vr)
        city_n.append(len(cb))
        city_names.append(city)

city_rhos = np.array(city_rhos)
city_var_reductions = np.array(city_var_reductions)
city_n = np.array(city_n)
n_eff_weighted_rho = np.average(np.abs(city_rhos), weights=city_n)
n_eff_weighted_vr = np.average(city_var_reductions, weights=city_n)

print(f'\nWithin-city coupling (n={len(city_rhos)} cities, sample-weighted):')
print(f'  median |rho|       = {np.median(np.abs(city_rhos)):.3f}')
print(f'  weighted mean |rho|= {n_eff_weighted_rho:.3f}')
print(f'  rho range          = [{city_rhos.min():.2f}, {city_rhos.max():.2f}]')
print(f'  median Var-reduction = {np.median(city_var_reductions)*100:.1f} %')
print(f'  weighted mean Var-r  = {n_eff_weighted_vr*100:.1f} %')
print(f'  ratio (within / global) = {n_eff_weighted_vr / var_reduction:.2f}x')

# ---------------------------------------------------------------
# Plot
# ---------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4),
                                gridspec_kw={'width_ratios': [1.25, 1.0]})

# (a) hexbin with FAR iso-curves
hb = ax1.hexbin(bf, bh, gridsize=55, bins='log',
                cmap='viridis', mincnt=1, linewidths=0)
cb = plt.colorbar(hb, ax=ax1, fraction=0.045, pad=0.03)
cb.set_label('log$_{10}$ cell count', fontsize=9)
cb.ax.tick_params(labelsize=8)

# FAR iso-curves: FAR = lambda_p * H_ave / floor_height
far_levels = [0.5, 1.0, 2.0, 4.0, 8.0]
bf_grid = np.linspace(0.01, bf.max(), 400)
for FAR in far_levels:
    bh_iso = FAR * ASSUMED_FLOOR_HEIGHT_M / bf_grid
    valid = (bh_iso >= 2) & (bh_iso <= 80)
    ax1.plot(bf_grid[valid], bh_iso[valid],
             color='white', lw=1.0, ls='--', alpha=0.85)
    # label near the right side of each curve
    bf_lab = bf_grid[valid][-1] if valid.any() else None
    if bf_lab is not None:
        ax1.text(bf_lab + 0.005, FAR * ASSUMED_FLOOR_HEIGHT_M / bf_lab,
                 f'FAR={FAR}', color='white', fontsize=7.5,
                 ha='left', va='center')

ax1.set_xlim(0, min(0.85, bf.max()))
ax1.set_ylim(0, 60)
ax1.set_xlabel('Footprint ratio $\\lambda_p$ (BF)', fontsize=10)
ax1.set_ylabel('Average building height $H_{\\mathrm{ave}}$ (m)', fontsize=10)
ax1.set_title(f'(a) Joint distribution over {n_total:,} training cells '
              f'across {len(n_per_city)} cities', fontsize=10)
ax1.tick_params(labelsize=8.5)

# (b) conditional H_ave histograms by BF bin
bf_bins = [(0.05, 0.15), (0.15, 0.30), (0.30, 0.50), (0.50, 0.80)]
colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(bf_bins)))
for (lo, hi), c in zip(bf_bins, colors):
    sel = (bf >= lo) & (bf < hi)
    if sel.sum() < 100:
        continue
    h = bh[sel]
    ax2.hist(h, bins=np.linspace(2, 50, 49), density=True,
             histtype='step', lw=1.6, color=c,
             label=f'$\\lambda_p \\in [{lo:.2f}, {hi:.2f})$  '
                   f'(n={sel.sum():,})')

# also overlay the marginal for comparison
ax2.hist(bh, bins=np.linspace(2, 50, 49), density=True,
         histtype='step', lw=1.2, color='gray', ls='--',
         label='Marginal (all cells)')

ax2.set_xlabel('$H_{\\mathrm{ave}}$ (m)', fontsize=10)
ax2.set_ylabel('Density', fontsize=10)
ax2.set_title('(b) Height distribution conditional on footprint ratio',
              fontsize=10)
ax2.legend(fontsize=8, loc='upper right', frameon=False)
ax2.set_xlim(2, 50)
ax2.tick_params(labelsize=8.5)

# small annotation summarizing the diagnostics inside panel (a)
diag = (f'$\\rho_P$ = {rho_p:.2f}    '
        f'$I(\\mathrm{{BH}};\\mathrm{{BF}})$ = {mi:.2f} nats\n'
        f'Var$(H \\mid \\lambda_p)$ reduces $\\sigma^2_H$ by '
        f'{var_reduction*100:.0f}\\,\\%')
ax1.text(0.03, 0.97, diag, transform=ax1.transAxes,
         fontsize=8.5, va='top', ha='left',
         bbox=dict(facecolor='white', edgecolor='gray',
                   boxstyle='round,pad=0.35', alpha=0.9))

fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'\nsaved {OUT}')
