"""Cleaner GeoSplit visualization: three morphologically distinct cities
(New York, Los Angeles, London) plotted side-by-side, each showing the
radial-wedge train/valid/test assignment over its 100 m cells.

Replaces v1's single-city `newyork_sectors.png` with a multi-city figure
that demonstrates the split strategy across different urban shapes.
"""
from __future__ import annotations

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from morphoformer.paths import results_root, split_paths

H5 = {
    'train':  str(split_paths()['train']),
    'valid':  str(split_paths()['val']),
    'test2':  str(split_paths()['test']),
}
OUT = str(results_root() / 'geosplit.png')

CITIES = ['london', 'toronto', 'sanfrancisco']
LABELS = ['London',  'Toronto', 'San Francisco']

# unified palette (matches blue/orange/red used elsewhere in the paper)
COLORS = {
    'train':  '#3B82F6',  # blue
    'valid':  '#F59E0B',  # amber
    'test2':  '#EF4444',  # red
}
DISPLAY = {'train': 'Train', 'valid': 'Valid', 'test2': 'Test'}


def load_city(city: str):
    """Return dict split -> (rows, cols) and the bounding box of the city."""
    out = {}
    all_r, all_c = [], []
    for split, path in H5.items():
        with h5py.File(path, 'r') as f:
            if city not in f:
                out[split] = (np.array([], int), np.array([], int))
                continue
            co = f[city]['coords'][:]
            r, c = co[:, 0].astype(int), co[:, 1].astype(int)
        out[split] = (r, c)
        all_r.append(r); all_c.append(c)
    if not all_r:
        return out, None
    rs = np.concatenate(all_r); cs = np.concatenate(all_c)
    return out, (rs.min(), rs.max(), cs.min(), cs.max())


def urban_core(splits):
    """Estimate urban core as the median of all cells (resilient to shape)."""
    rs, cs = [], []
    for s in splits.values():
        rs.append(s[0]); cs.append(s[1])
    rs = np.concatenate(rs); cs = np.concatenate(cs)
    return float(np.median(rs)), float(np.median(cs))


fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.7),
                         gridspec_kw={'wspace': 0.22})

for ax, city, name in zip(axes, CITIES, LABELS):
    splits, bbox = load_city(city)
    if bbox is None:
        ax.text(0.5, 0.5, f'{name}: data missing',
                transform=ax.transAxes, ha='center')
        continue
    r0, r1, c0, c1 = bbox
    # plot in km units relative to urban core (each cell = 100 m = 0.1 km)
    cr, cc = urban_core(splits)
    n_total = 0
    for split in ['train', 'valid', 'test2']:
        r, c = splits[split]
        if len(r) == 0:
            continue
        # x = (c - cc) * 0.1 km; y = -(r - cr) * 0.1 km   (flip rows so north is up)
        xs = (c - cc) * 0.1
        ys = -(r - cr) * 0.1
        ax.scatter(xs, ys, s=4, c=COLORS[split], alpha=0.85,
                   linewidths=0, marker='s')
        n_total += len(r)

    # urban-core marker
    ax.plot(0, 0, marker='*', markersize=15, color='black',
            markerfacecolor='white', markeredgewidth=1.6, zorder=5)

    # split-ratio in title
    n_train = len(splits['train'][0])
    n_valid = len(splits['valid'][0])
    n_test  = len(splits['test2'][0])
    ratios = (f'{n_train/n_total:.0%}/{n_valid/n_total:.0%}/{n_test/n_total:.0%}'
              if n_total else '0/0/0')
    ax.set_title(f'{name}  ($n$={n_total:,}, {ratios})',
                 fontsize=11)
    ax.set_xlabel('East–west (km from core)', fontsize=9)
    if ax is axes[0]:
        ax.set_ylabel('North–south (km from core)', fontsize=9)
    ax.set_aspect('equal')
    ax.grid(True, lw=0.3, alpha=0.4)
    ax.tick_params(labelsize=8.5)

# legend in lower-right corner of the rightmost subplot
legend_elems = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['train'],
           markersize=8, label='Train'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['valid'],
           markersize=8, label='Validation'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['test2'],
           markersize=8, label='Test'),
    Line2D([0], [0], marker='*', color='black', markerfacecolor='white',
           markeredgewidth=1.4, markersize=11, label='Urban core',
           linestyle='None'),
]
fig.legend(handles=legend_elems, loc='upper center', ncol=4,
           frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.04))

fig.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'saved {OUT}')
