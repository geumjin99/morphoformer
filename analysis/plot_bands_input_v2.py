"""Render the v2 8-band multi-source input tensor as a 5-panel figure
for the paper_sors Method section.

Picks a *boundary* sample (city edge, coastline, or other no-data region)
so that the validity mask carries genuine information — illustrating why
the mask channel is part of the input. Plots:
    Sentinel-1 (VV+VH avg) | Sentinel-2 RGB | Sentinel-2 NIR | DEM | Mask
The centre 10x10 px (1 grid cell) is outlined in red; cell boundaries are
shown as light dashed grid; invalid cells (mask=0) are obvious in the
final panel.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import label as cc_label

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from morphoformer.data.dataset import H5PatchDataset
from morphoformer.paths import results_root, split_paths

H5 = str(split_paths()['test'])
OUT = str(results_root() / 'bands_input_v2.png')

# Cities likely to expose interesting boundaries (coastline, lake, river).
PRIORITY_CITIES = [
    'sanfrancisco', 'newyork', 'wellington', 'toronto', 'boston',
    'london', 'miamidadecounty', 'porirua', 'moncton', 'launceston',
]

ds = H5PatchDataset(
    h5_path=H5,
    patch_grid_size=9,
    modalities=['sar', 'optical', 'dem'],
    augment=False,
    preload=False,
)

# search criterion
#   - centre cell valid (otherwise the sample wouldn't exist)
#   - 12..40 invalid neighbours out of 81  (roughly 15-50% of the window)
#   - the invalid region is a single contiguous component covering >= 70%
#     of the invalid cells (water / off-city), not random scatter
#   - centre BH in (4, 25) m so the panel is also visually meaningful
def search_city(city: str, bh_arr, bf_arr):
    info = ds.city_data[city]
    grid = info['grid']
    coords = info['coords']
    r_min, c_min = info['r_min'], info['c_min']
    n_rows, n_cols = info['n_rows'], info['n_cols']

    best = None
    for local_idx, coord in enumerate(coords):
        row_c = int(coord[0]) - r_min
        col_c = int(coord[1]) - c_min
        if not (4 <= row_c < n_rows - 4 and 4 <= col_c < n_cols - 4):
            continue
        block = grid[row_c - 4:row_c + 5, col_c - 4:col_c + 5]
        if block[4, 4] < 0:
            continue
        invalid = block < 0
        n_inv = int(invalid.sum())
        if not (12 <= n_inv <= 40):
            continue
        # contiguity test
        cc, n_cc = cc_label(invalid)
        if n_cc == 0:
            continue
        sizes = np.bincount(cc.ravel())[1:]
        largest = int(sizes.max())
        if largest < 0.70 * n_inv:
            continue
        bh = float(bh_arr[local_idx])
        bf = float(bf_arr[local_idx])
        if not (4.0 <= bh <= 25.0):
            continue
        # score: prefer larger contiguous masked region (more visually obvious)
        score = largest * 100 + n_inv
        if best is None or score > best[0]:
            best = (score, local_idx, n_inv, largest, bh, bf, coord)
    return best

target_city = None
target_local = None
target_meta = None
for city in PRIORITY_CITIES:
    if city not in ds.city_data:
        continue
    with h5py.File(H5, 'r') as f:
        bh_arr = f[city]['BuildingHeight'][:]
        bf_arr = f[city]['BuildingFootprint'][:]
    res = search_city(city, bh_arr, bf_arr)
    if res is None:
        print(f'  {city:24s}  no boundary sample found')
        continue
    score, local_idx, n_inv, largest, bh, bf, coord = res
    print(f'  {city:24s}  inv={n_inv:2d}  cc={largest:2d}  '
          f'bh={bh:5.2f} bf={bf:.3f}  coord={coord.tolist()}')
    if target_city is None or score > target_meta[0]:
        target_city = city
        target_local = local_idx
        target_meta = res

assert target_city is not None, 'no suitable boundary sample found in priority cities'
print(f'\nselected: {target_city} local_idx={target_local}  '
      f'invalid={target_meta[2]}  largest_cc={target_meta[3]}  '
      f'bh={target_meta[4]:.2f}  bf={target_meta[5]:.3f}')

# locate the dataset.index_map slot for (target_city, target_local)
target_idx = None
for idx, (city, local) in enumerate(ds.index_map):
    if city == target_city and local == target_local:
        target_idx = idx
        break
assert target_idx is not None, 'index_map lookup failed'

chosen_bh = target_meta[4]
chosen_bf = target_meta[5]
chosen_coord = target_meta[6]

sample, bh_label, bf_label = ds[target_idx]
tensor = sample.numpy()           # (C, 90, 90), normalized
mask = tensor[-1]                 # (90, 90)
explan = tensor[:-1]              # (8, 90, 90)

means = ds._means.squeeze()
stds = ds._stds.squeeze()
explan_raw = explan * stds[:, None, None] + means[:, None, None]

#   0: s1_VV, 1: s1_VH
#   2..5: B1..B4 (Blue, Green, Red, NIR)
#   6: DEM
sar_avg = 0.5 * (explan_raw[0] + explan_raw[1])
opt_R = explan_raw[4]
opt_G = explan_raw[3]
opt_B = explan_raw[2]
opt_NIR = explan_raw[5]
dem = explan_raw[6]

# Robust scaling, but compute percentiles only on valid pixels so that the
# zero-filled invalid regions don't compress the dynamic range.
def robust_scale(x, m, lo=2, hi=98):
    valid_pix = x[m > 0.5]
    if valid_pix.size == 0:
        return np.zeros_like(x)
    a, b = np.percentile(valid_pix, [lo, hi])
    return np.clip((x - a) / (b - a + 1e-6), 0, 1)

sar_disp = robust_scale(sar_avg, mask)
nir_disp = robust_scale(opt_NIR, mask)
rgb_disp = np.stack(
    [robust_scale(opt_R, mask),
     robust_scale(opt_G, mask),
     robust_scale(opt_B, mask)],
    axis=-1,
)
dem_disp = np.where(mask > 0.5, dem, np.nan)

H = W = 90
P = 10
G = 9
center_x = (G // 2) * P
center_y = (G // 2) * P

fig, axes = plt.subplots(1, 5, figsize=(16, 3.4))

panels = [
    ('Sentinel-1 (VV+VH avg)', sar_disp,         'gray',     None),
    ('Sentinel-2 (R,G,B)',     rgb_disp,         None,       None),
    ('Sentinel-2 NIR',         nir_disp,         'gray',     None),
    ('DEM (m)',                dem_disp,         'viridis',  True),
    ('Mask (1=valid)',         mask,             'gray',     None),
]

for ax, (title, img, cmap, show_cbar) in zip(axes, panels):
    if cmap is None:
        ax.imshow(img, origin='upper')
    else:
        im = ax.imshow(img, cmap=cmap, origin='upper')
        if show_cbar:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    ax.set_title(title, fontsize=11)

    for k in range(1, G):
        ax.axhline(k * P - 0.5, color='red', lw=0.4, ls='--', alpha=0.55)
        ax.axvline(k * P - 0.5, color='red', lw=0.4, ls='--', alpha=0.55)

    rect = Rectangle((center_x - 0.5, center_y - 0.5), P, P,
                     linewidth=1.6, edgecolor='red', facecolor='none')
    ax.add_patch(rect)

    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle(
    f'{target_city.title()}  |  centre cell BH = {chosen_bh:.2f} m,  BF = {chosen_bf:.3f}',
    fontsize=11, y=1.02,
)

fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'\nsaved {OUT}')
