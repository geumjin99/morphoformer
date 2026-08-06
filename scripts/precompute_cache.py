"""
Precompute combined band arrays from per-band H5 into flat numpy files.

H5 layout (slow):  city/sentinel_1_50pt_B1[N,1,10,10], city/sentinel_2_..., ...  (7 reads/sample)
Cache layout (fast): bands[N, 7, 10, 10] float32 + bh[N] + bf[N] + coords[N,2] per city, one file per split

Loading then becomes 1 read instead of 7, and numpy memmap hits OS page cache efficiently.
"""

import h5py
import numpy as np
import os
import time
from pathlib import Path
from morphoformer.paths import cache_root, data_root

BANDS = [
    'sentinel_1_50pt_B1', 'sentinel_1_50pt_B2',
    'sentinel_2_50pt_B1', 'sentinel_2_50pt_B2',
    'sentinel_2_50pt_B3', 'sentinel_2_50pt_B4',
    'DEM',
]

DATA_DIR  = data_root()
CACHE_DIR = cache_root()
SPLITS    = {'train': 'train.h5', 'valid': 'valid.h5', 'test2': 'test2.h5'}


def precompute_split(split_name: str, h5_file: str):
    src = DATA_DIR / h5_file
    dst = CACHE_DIR / split_name
    dst.mkdir(parents=True, exist_ok=True)

    print(f'\n=== {split_name} ({src}) ===')
    t0 = time.time()
    total = 0

    with h5py.File(src, 'r') as f:
        cities = list(f.keys())
        for ci, city in enumerate(cities):
            g = f[city]
            N = g['BuildingHeight'].shape[0]

            # Stack all 7 bands: (N, 7, 10, 10) float32
            bands_arr = np.stack(
                [g[b][:].astype(np.float32)[:, 0, :, :] for b in BANDS],
                axis=1
            )  # (N, 7, 10, 10)

            bh     = g['BuildingHeight'][:].astype(np.float32)
            bf     = g['BuildingFootprint'][:].astype(np.float32)
            coords = g['coords'][:].astype(np.int32)

            city_dir = dst / city
            city_dir.mkdir(exist_ok=True)
            np.save(city_dir / 'bands.npy',  bands_arr)
            np.save(city_dir / 'bh.npy',     bh)
            np.save(city_dir / 'bf.npy',     bf)
            np.save(city_dir / 'coords.npy', coords)

            total += N
            elapsed = time.time() - t0
            print(f'  [{ci+1:3d}/{len(cities)}] {city:<20s} N={N:7,}  total={total:,}  {elapsed:.0f}s',
                  flush=True)

    print(f'Done {split_name}: {total:,} samples in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for name, fname in SPLITS.items():
        precompute_split(name, fname)
    print('\nAll splits cached.')
