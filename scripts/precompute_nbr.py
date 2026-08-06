"""Precompute per-sample neighbour indices for CachedPatchDataset.

For each sample, pre-sorts the up-to-81 neighbourhood slots by array index and
stores the result so __getitem__ can skip all the per-sample numpy discovery
work (bounds check, grid lookup, argsort, inv_sort) and go straight to the
one expensive call: bands_all[sorted_idx].

Output per city directory (inside cache_dir/<split>/<city>/):
    nbr_sorted.npy  (N, 81) int32   sorted array indices; -1 = padding
    nbr_gi.npy      (N, 81) uint8   grid row (0-8) in same order as sorted idx
    nbr_gj.npy      (N, 81) uint8   grid col (0-8) in same order as sorted idx
    nbr_k.npy       (N,)    uint8   number of valid neighbours

Storage: ~487 bytes/sample → ~900 MB for the full train split.

Usage:
    python scripts/precompute_nbr.py                  # all splits
    python scripts/precompute_nbr.py --split train    # one split
    python scripts/precompute_nbr.py --overwrite      # recompute even if exists
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from morphoformer.paths import cache_root

CACHE_ROOT = cache_root()
PATCH_GRID_SIZE = 9   # must match training config


def build_offsets(G: int) -> np.ndarray:
    half = G // 2
    di, dj = np.meshgrid(np.arange(-half, half + 1, dtype=np.int32),
                          np.arange(-half, half + 1, dtype=np.int32),
                          indexing='ij')
    return np.stack([di.ravel(), dj.ravel()], axis=1)   # (G², 2)


def precompute_city(city_dir: Path, offsets: np.ndarray, overwrite: bool) -> int:
    """Precompute and save nbr_*.npy for one city. Returns N."""
    G = int(offsets.shape[0] ** 0.5)
    half = G // 2

    out_files = [city_dir / f for f in
                 ('nbr_sorted.npy', 'nbr_gi.npy', 'nbr_gj.npy', 'nbr_k.npy')]
    if all(f.exists() for f in out_files) and not overwrite:
        coords = np.load(city_dir / 'coords.npy')
        return len(coords)

    coords = np.load(city_dir / 'coords.npy')   # (N, 2) int32
    N = len(coords)

    rows, cols = coords[:, 0], coords[:, 1]
    r_min, c_min = int(rows.min()), int(cols.min())
    n_rows = int(rows.max()) - r_min + 1
    n_cols = int(cols.max()) - c_min + 1

    grid = np.full((n_rows, n_cols), -1, dtype=np.int32)
    grid[rows - r_min, cols - c_min] = np.arange(N, dtype=np.int32)

    # ── vectorised over all N samples at once ─────────────────────────────
    # Shape: (N, G²)
    ri = (coords[:, 0:1] + offsets[np.newaxis, :, 0] - r_min)   # (N, 81)
    ci = (coords[:, 1:2] + offsets[np.newaxis, :, 1] - c_min)

    in_bounds = ((ri >= 0) & (ri < n_rows) &
                 (ci >= 0) & (ci < n_cols))                       # (N, 81) bool
    ri_c = ri.clip(0, n_rows - 1).astype(np.int32)
    ci_c = ci.clip(0, n_cols - 1).astype(np.int32)

    arr_all = grid[ri_c.ravel(), ci_c.ravel()].reshape(N, 81)    # (N, 81)
    valid   = in_bounds & (arr_all >= 0)                          # (N, 81) bool

    k_arr = valid.sum(axis=1).astype(np.uint8)                   # (N,)

    # Sort each row by array index; put INT_MAX for invalid so they go last
    INT_MAX = np.iinfo(np.int32).max
    arr_sortable = np.where(valid, arr_all, INT_MAX)
    sort_order = np.argsort(arr_sortable, axis=1, kind='stable')  # (N, 81)

    # sorted array indices: -1 for padding slots
    sorted_idx = np.take_along_axis(arr_all,  sort_order, axis=1).astype(np.int32)
    sorted_idx[~np.take_along_axis(valid, sort_order, axis=1)] = -1

    # gi/gj in same order as sorted_idx (offsets → 0-based grid coords)
    gi_base = (offsets[:, 0] + half).astype(np.uint8)   # (81,)
    gj_base = (offsets[:, 1] + half).astype(np.uint8)
    gi_out  = gi_base[sort_order]                         # (N, 81)
    gj_out  = gj_base[sort_order]

    np.save(city_dir / 'nbr_sorted.npy', sorted_idx)
    np.save(city_dir / 'nbr_gi.npy',     gi_out)
    np.save(city_dir / 'nbr_gj.npy',     gj_out)
    np.save(city_dir / 'nbr_k.npy',      k_arr)

    return N


def precompute_split(split: str, overwrite: bool):
    split_dir = CACHE_ROOT / split
    if not split_dir.exists():
        print(f'  [skip] {split}: cache dir not found')
        return

    offsets = build_offsets(PATCH_GRID_SIZE)
    city_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())
    total_n, t_start = 0, time.time()

    print(f'\n=== {split} ({len(city_dirs)} cities) ===')
    for city_dir in city_dirs:
        t0 = time.time()
        N = precompute_city(city_dir, offsets, overwrite)
        elapsed = time.time() - t0
        total_n += N
        print(f'  {city_dir.name:<20s}  {N:>7,} samples  {elapsed:5.1f}s')

    total_elapsed = time.time() - t_start
    size_mb = sum((city_dir / 'nbr_sorted.npy').stat().st_size
                  for city_dir in city_dirs
                  if (city_dir / 'nbr_sorted.npy').exists()) / 1024**2
    size_mb += sum((city_dir / 'nbr_gi.npy').stat().st_size
                   for city_dir in city_dirs
                   if (city_dir / 'nbr_gi.npy').exists()) / 1024**2
    size_mb += sum((city_dir / 'nbr_gj.npy').stat().st_size
                   for city_dir in city_dirs
                   if (city_dir / 'nbr_gj.npy').exists()) / 1024**2
    size_mb += sum((city_dir / 'nbr_k.npy').stat().st_size
                   for city_dir in city_dirs
                   if (city_dir / 'nbr_k.npy').exists()) / 1024**2
    print(f'  → {total_n:,} samples, {size_mb:.0f} MB, {total_elapsed:.1f}s total')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', choices=['train', 'valid', 'test2'], default=None,
                   help='Process one split only (default: all)')
    p.add_argument('--overwrite', action='store_true',
                   help='Recompute even if nbr_*.npy already exist')
    args = p.parse_args()

    splits = [args.split] if args.split else ['train', 'valid', 'test2']
    for s in splits:
        precompute_split(s, args.overwrite)
    print('\nDone.')


if __name__ == '__main__':
    main()
