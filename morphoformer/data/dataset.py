"""HDF5 dataset loader for multi-modal Sentinel imagery.

Two backends:
  H5PatchDataset      — reads directly from H5 (slow, no precompute needed)
  CachedPatchDataset — reads from pre-computed .npy files (fast, run
                           scripts/precompute_cache.py once first)
"""

import h5py
import numpy as np
import torch
import threading
import queue as Q
from torch.utils.data import Dataset, IterableDataset, DataLoader
import pytorch_lightning as pl
from typing import Optional, List, Dict, Tuple
from pathlib import Path


def _augment(input_tensor: np.ndarray, mask_tensor: np.ndarray):
    """Dihedral augmentation: rot90 x {0,1,2,3} then optional horizontal flip.

    Eight orientations in total. Both regression targets (mean building height
    and footprint ratio of the centre cell) are invariant under these
    transforms, so no label adjustment is needed.
    """
    k = int(np.random.randint(4))
    if k:
        input_tensor = np.rot90(input_tensor, k, axes=(-2, -1))
        mask_tensor = np.rot90(mask_tensor, k, axes=(-2, -1))
    if np.random.rand() < 0.5:
        input_tensor = input_tensor[..., ::-1]
        mask_tensor = mask_tensor[..., ::-1]
    return np.ascontiguousarray(input_tensor), np.ascontiguousarray(mask_tensor)


class H5PatchDataset(Dataset):
    """Loads neighbourhood patches straight from HDF5.

    Correct but slow (one read per band per sample). ``CachedPatchDataset`` is
    the backend used for every result in the paper; this one is the fallback
    when the ``.npy`` cache has not been built.
    """

    BAND_STATS = {
        'sentinel_1_50pt_B1': {'mean': -8.5614, 'std': 3.5339},   # VH
        'sentinel_1_50pt_B2': {'mean': -15.6317, 'std': 3.1389},  # VV
        'sentinel_2_50pt_B1': {'mean': 0.4858, 'std': 0.2354},    # Blue
        'sentinel_2_50pt_B2': {'mean': 0.5587, 'std': 0.2422},    # Green
        'sentinel_2_50pt_B3': {'mean': 0.5113, 'std': 0.2621},    # Red
        'sentinel_2_50pt_B4': {'mean': 0.6108, 'std': 0.2353},    # NIR
        'DEM': {'mean': 247.3289, 'std': 391.9973}
    }

    MODALITY_BANDS = {
        'sar': ['sentinel_1_50pt_B1', 'sentinel_1_50pt_B2'],
        'optical': ['sentinel_2_50pt_B1', 'sentinel_2_50pt_B2',
                   'sentinel_2_50pt_B3', 'sentinel_2_50pt_B4'],
        'dem': ['DEM']
    }

    def __init__(self, h5_path: str, patch_grid_size: int = 5,
                 modalities: List[str] = ['sar', 'optical', 'dem'],
                 augment: bool = False, preload: bool = False):
        self.h5_path = h5_path
        self.patch_grid_size = patch_grid_size
        self.patch_size = 10
        self.augment = augment
        self.preload = preload

        self.bands = []
        for mod in modalities:
            self.bands.extend(self.MODALITY_BANDS[mod])
        self.num_channels = len(self.bands)

        self._load_index()

        if preload:
            self._preload_data()

    def _load_index(self):
        """Build global index mapping (city, local_idx).

        Uses a 2-D numpy grid for O(1) neighbour lookup instead of a Python
        dict, which is ~10x faster to build and avoids per-lookup Python
        overhead in __getitem__.
        """
        self.city_data = {}
        self.index_map = []

        with h5py.File(self.h5_path, 'r') as f:
            for city_name in f.keys():
                group = f[city_name]
                coords = group['coords'][:]          # (N, 2) int64
                rows, cols = coords[:, 0], coords[:, 1]
                r_min, c_min = int(rows.min()), int(cols.min())
                n_rows = int(rows.max()) - r_min + 1
                n_cols = int(cols.max()) - c_min + 1

                # grid[r, c] = array index, -1 means no sample there
                grid = np.full((n_rows, n_cols), -1, dtype=np.int32)
                grid[rows - r_min, cols - c_min] = np.arange(len(coords), dtype=np.int32)

                self.city_data[city_name] = {
                    'n_samples': len(coords),
                    'coords': coords,
                    'r_min': r_min,
                    'c_min': c_min,
                    'n_rows': n_rows,
                    'n_cols': n_cols,
                    'grid': grid,
                }

                self.index_map.extend(
                    (city_name, i) for i in range(len(coords))
                )

    def _preload_data(self):
        """Load all data into memory."""
        with h5py.File(self.h5_path, 'r') as f:
            for city_name in f.keys():
                group = f[city_name]
                self.city_data[city_name]['bh'] = group['BuildingHeight'][:]
                self.city_data[city_name]['bf'] = group['BuildingFootprint'][:]
                self.city_data[city_name]['bands'] = {
                    band: group[band][:] for band in self.bands
                }

    def __len__(self) -> int:
        return len(self.index_map)

    # band normalization arrays built once at init
    _means: np.ndarray
    _stds: np.ndarray
    _OFFSET_CACHE: Dict[int, np.ndarray] = {}

    @classmethod
    def _get_offsets(cls, half: int) -> np.ndarray:
        if half not in cls._OFFSET_CACHE:
            g = 2 * half + 1
            di, dj = np.meshgrid(np.arange(-half, half + 1, dtype=np.int32),
                                  np.arange(-half, half + 1, dtype=np.int32),
                                  indexing='ij')
            cls._OFFSET_CACHE[half] = np.stack([di.ravel(), dj.ravel()], axis=1)
        return cls._OFFSET_CACHE[half]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        city_name, local_idx = self.index_map[idx]
        city_info = self.city_data[city_name]

        center_coord = city_info['coords'][local_idx]
        row_c, col_c = int(center_coord[0]), int(center_coord[1])
        half = self.patch_grid_size // 2
        G, P, C = self.patch_grid_size, self.patch_size, self.num_channels

        grid   = city_info['grid']
        r_min, c_min = city_info['r_min'], city_info['c_min']
        n_rows, n_cols = city_info['n_rows'], city_info['n_cols']

        # ── Vectorized neighbor discovery ────────────────────────────────────
        offsets = self._get_offsets(half)
        ri_all  = row_c + offsets[:, 0] - r_min
        ci_all  = col_c + offsets[:, 1] - c_min
        in_bounds = ((ri_all >= 0) & (ri_all < n_rows) &
                     (ci_all >= 0) & (ci_all < n_cols))
        ri_c = ri_all.clip(0, n_rows - 1)
        ci_c = ci_all.clip(0, n_cols - 1)
        arr_all = grid[ri_c, ci_c]
        valid   = in_bounds & (arr_all >= 0)
        valid_flat  = np.where(valid)[0]
        arr_indices = arr_all[valid_flat]

        sort_order = np.argsort(arr_indices)
        sorted_idx = arr_indices[sort_order].tolist()
        inv_sort   = np.empty(len(sort_order), dtype=np.int32)
        inv_sort[sort_order] = np.arange(len(sort_order), dtype=np.int32)

        # ── Load from H5 or preloaded cache ──────────────────────────────────
        if self.preload:
            bh = float(city_info['bh'][local_idx])
            bf = float(city_info['bf'][local_idx])
            # stack bands: (K, C, P, P)
            bands_loaded = np.stack(
                [city_info['bands'][b][sorted_idx][:, 0] for b in self.bands], axis=1
            ).astype(np.float32)
        else:
            with h5py.File(self.h5_path, 'r') as f:
                group = f[city_name]
                bh = float(group['BuildingHeight'][local_idx])
                bf = float(group['BuildingFootprint'][local_idx])
                bands_loaded = np.stack(
                    [group[b][sorted_idx][:, 0] for b in self.bands], axis=1
                ).astype(np.float32)  # (K, C, P, P)

        # ── Vectorized assembly ──────────────────────────────────────────────
        loaded_ordered = bands_loaded[inv_sort]   # (K, C, P, P) in slot order
        gi_arr = offsets[valid_flat, 0] + half
        gj_arr = offsets[valid_flat, 1] + half

        input_grid = np.zeros((C, G, G, P, P), dtype=np.float32)
        mask_grid  = np.zeros((1, G, G, P, P), dtype=np.float32)
        input_grid[:, gi_arr, gj_arr] = loaded_ordered.transpose(1, 0, 2, 3)
        mask_grid [0, gi_arr, gj_arr] = 1.0

        input_tensor = input_grid.transpose(0, 1, 3, 2, 4).reshape(C, G * P, G * P)
        mask_tensor  = mask_grid.transpose(0, 1, 3, 2, 4).reshape(1, G * P, G * P)

        # normalize
        if not hasattr(self, '_means'):
            self._means = np.array([self.BAND_STATS[b]['mean'] for b in self.bands],
                                   dtype=np.float32).reshape(-1, 1, 1)
            self._stds  = np.array([self.BAND_STATS[b]['std']  for b in self.bands],
                                   dtype=np.float32).reshape(-1, 1, 1)
        input_tensor = (input_tensor - self._means) / self._stds

        if self.augment:
            input_tensor, mask_tensor = _augment(input_tensor, mask_tensor)

        full_input = np.concatenate([input_tensor, mask_tensor], axis=0)
        return (
            torch.from_numpy(full_input),
            torch.tensor(bh, dtype=torch.float32),
            torch.tensor(bf, dtype=torch.float32),
        )


class CachedPatchDataset(Dataset):
    """Fast dataset backed by pre-computed .npy files (see scripts/precompute_cache.py).

    Cache layout (one directory per split):
        <cache_dir>/<city>/bands.npy   (N, 7, 10, 10)  float32, all 7 bands stacked
        <cache_dir>/<city>/bh.npy      (N,)             float32
        <cache_dir>/<city>/bf.npy      (N,)             float32
        <cache_dir>/<city>/coords.npy  (N, 2)           int32

    At load time coords are used to build the same 2-D numpy grid as
    H5PatchDataset so neighbour lookup is identical (O(1) array indexing).
    The key speedup: one numpy fancy-index read for all 7 bands instead of
    7 separate H5 reads.
    """

    # band order must match precompute_cache.py BANDS list
    BAND_ORDER = [
        'sentinel_1_50pt_B1', 'sentinel_1_50pt_B2',
        'sentinel_2_50pt_B1', 'sentinel_2_50pt_B2',
        'sentinel_2_50pt_B3', 'sentinel_2_50pt_B4',
        'DEM',
    ]
    BAND_STATS = H5PatchDataset.BAND_STATS
    MODALITY_BANDS = H5PatchDataset.MODALITY_BANDS

    def __init__(self, cache_dir: str, patch_grid_size: int = 9,
                 modalities: List[str] = None, preload: bool = False,
                 augment: bool = False):
        self.cache_dir = Path(cache_dir)
        self.patch_grid_size = patch_grid_size
        self.patch_size = 10
        self.preload = preload
        self.augment = augment
        modalities = modalities or ['sar', 'optical', 'dem']

        # which band indices to use (subset of the 7 stored)
        requested = []
        for mod in modalities:
            requested.extend(self.MODALITY_BANDS[mod])
        self.band_indices = [self.BAND_ORDER.index(b) for b in requested]
        self.bands_requested = requested
        self.num_channels = len(requested)

        # normalization arrays: shape (C, 1, 1) for broadcasting
        means = np.array([self.BAND_STATS[b]['mean'] for b in requested],
                         dtype=np.float32).reshape(-1, 1, 1)
        stds  = np.array([self.BAND_STATS[b]['std']  for b in requested],
                         dtype=np.float32).reshape(-1, 1, 1)
        self._means = means
        self._stds  = stds

        self._load_index()

    def _load_index(self):
        self.city_data = {}
        self.index_map = []

        for city_dir in sorted(self.cache_dir.iterdir()):
            if not city_dir.is_dir():
                continue
            city = city_dir.name
            coords = np.load(city_dir / 'coords.npy')   # (N, 2) int32
            rows, cols = coords[:, 0], coords[:, 1]
            r_min, c_min = int(rows.min()), int(cols.min())
            n_rows = int(rows.max()) - r_min + 1
            n_cols = int(cols.max()) - c_min + 1
            grid = np.full((n_rows, n_cols), -1, dtype=np.int32)
            grid[rows - r_min, cols - c_min] = np.arange(len(coords), dtype=np.int32)

            entry = {
                'dir': city_dir,
                'n_samples': len(coords),
                'coords': coords,
                'r_min': r_min, 'c_min': c_min,
                'n_rows': n_rows, 'n_cols': n_cols,
                'grid': grid,
                '_bands': None, '_bh': None, '_bf': None,
                '_nbr': None,   # precomputed neighbour indices (or None)
            }
            if self.preload:
                entry['_bands'] = np.load(city_dir / 'bands.npy')
                entry['_bh']    = np.load(city_dir / 'bh.npy')
                entry['_bf']    = np.load(city_dir / 'bf.npy')
            # Auto-detect precomputed neighbour index files.
            # These are small (~18 bytes/sample) — always load into RAM for O(1) access.
            nbr_sorted = city_dir / 'nbr_sorted.npy'
            if nbr_sorted.exists():
                entry['_nbr'] = {
                    'sorted': np.load(city_dir / 'nbr_sorted.npy'),
                    'gi':     np.load(city_dir / 'nbr_gi.npy'),
                    'gj':     np.load(city_dir / 'nbr_gj.npy'),
                    'k':      np.load(city_dir / 'nbr_k.npy'),
                }
            self.city_data[city] = entry
            self.index_map.extend((city, i) for i in range(len(coords)))

    def _get_arrays(self, city):
        """Return city arrays; uses RAM if preloaded, else mmap fallback."""
        cd = self.city_data[city]
        if cd['_bands'] is None:
            cd['_bands'] = np.load(cd['dir'] / 'bands.npy', mmap_mode='r')
            cd['_bh']    = np.load(cd['dir'] / 'bh.npy',    mmap_mode='r')
            cd['_bf']    = np.load(cd['dir'] / 'bf.npy',    mmap_mode='r')
        return cd['_bands'], cd['_bh'], cd['_bf']

    @property
    def total_ram_bytes(self) -> int:
        """Estimated RAM usage of preloaded arrays (for logging)."""
        total = 0
        for cd in self.city_data.values():
            for key in ('_bands', '_bh', '_bf'):
                arr = cd.get(key)
                if arr is not None and not isinstance(arr, np.memmap):
                    total += arr.nbytes
        return total

    def __len__(self):
        return len(self.index_map)

    # pre-build offset grid once per class instance (shared across workers via fork)
    _OFFSET_CACHE: Dict[int, np.ndarray] = {}

    @classmethod
    def _get_offsets(cls, half: int) -> np.ndarray:
        """Return (G*G, 2) int32 array of (di, dj) offsets, cached."""
        if half not in cls._OFFSET_CACHE:
            g = 2 * half + 1
            di, dj = np.meshgrid(np.arange(-half, half + 1, dtype=np.int32),
                                  np.arange(-half, half + 1, dtype=np.int32),
                                  indexing='ij')
            cls._OFFSET_CACHE[half] = np.stack([di.ravel(), dj.ravel()], axis=1)  # (G²,2)
        return cls._OFFSET_CACHE[half]

    def __getitem__(self, idx):
        city, local_idx = self.index_map[idx]
        cd = self.city_data[city]
        bands_all, bh_arr, bf_arr = self._get_arrays(city)

        G, P = self.patch_grid_size, self.patch_size

        nbr = cd['_nbr']
        if nbr is not None:
            # ── Fast path: precomputed indices, skip all discovery ops ──────
            K          = int(nbr['k'][local_idx])
            sorted_idx = np.array(nbr['sorted'][local_idx, :K])   # copy out of mmap
            gi_arr     = nbr['gi'][local_idx, :K].astype(np.int32)
            gj_arr     = nbr['gj'][local_idx, :K].astype(np.int32)
            # gi/gj are already in sorted_idx order → no inv_sort needed
            loaded = bands_all[sorted_idx][:, self.band_indices, :, :]   # (K, C, P, P)
        else:
            # ── Slow path: compute on the fly ────────────────────────────────
            half   = G // 2
            grid   = cd['grid']
            r_min, c_min = cd['r_min'], cd['c_min']
            n_rows, n_cols = cd['n_rows'], cd['n_cols']
            row_c, col_c = int(cd['coords'][local_idx, 0]), int(cd['coords'][local_idx, 1])

            offsets = self._get_offsets(half)
            ri_all  = row_c + offsets[:, 0] - r_min
            ci_all  = col_c + offsets[:, 1] - c_min
            in_bounds = ((ri_all >= 0) & (ri_all < n_rows) &
                         (ci_all >= 0) & (ci_all < n_cols))
            ri_clamp = ri_all.clip(0, n_rows - 1)
            ci_clamp = ci_all.clip(0, n_cols - 1)
            arr_all  = grid[ri_clamp, ci_clamp]
            valid    = in_bounds & (arr_all >= 0)
            valid_flat  = np.where(valid)[0]
            arr_indices = arr_all[valid_flat]
            sort_order  = np.argsort(arr_indices)
            sorted_idx  = arr_indices[sort_order]
            inv_sort    = np.empty(len(sort_order), dtype=np.int32)
            inv_sort[sort_order] = np.arange(len(sort_order), dtype=np.int32)
            gi_arr = offsets[valid_flat, 0] + half
            gj_arr = offsets[valid_flat, 1] + half

            loaded = bands_all[sorted_idx][:, self.band_indices, :, :]
            loaded = loaded[inv_sort]   # restore slot order

        input_grid = np.zeros((self.num_channels, G, G, P, P), dtype=np.float32)
        mask_grid  = np.zeros((1, G, G, P, P), dtype=np.float32)
        input_grid[:, gi_arr, gj_arr, :, :] = loaded.transpose(1, 0, 2, 3)
        mask_grid [0, gi_arr, gj_arr, :, :] = 1.0

        # (C, G, G, P, P) → (C, G, P, G, P) → (C, G*P, G*P)
        input_tensor = input_grid.transpose(0, 1, 3, 2, 4).reshape(
            self.num_channels, G * P, G * P)
        mask_tensor  = mask_grid.transpose(0, 1, 3, 2, 4).reshape(1, G * P, G * P)

        # vectorized normalize
        input_tensor = (input_tensor - self._means) / self._stds

        if self.augment:
            input_tensor, mask_tensor = _augment(input_tensor, mask_tensor)

        full_input = np.concatenate([input_tensor, mask_tensor], axis=0)
        return (
            torch.from_numpy(full_input),
            torch.tensor(float(bh_arr[local_idx]), dtype=torch.float32),
            torch.tensor(float(bf_arr[local_idx]), dtype=torch.float32),
        )


class ChunkedIterableDataset(IterableDataset):
    """Double-buffered pre-assembler for CachedPatchDataset.

    Background threads pre-assemble large chunks of samples using a single
    vectorized fancy-index per city (deduplicating shared neighbours). The
    main DataLoader (num_workers=0) consumes from a shared queue — reads are
    sequential pointer-chasing in RAM rather than random mmap pages.

    Key ideas
    ---------
    * Groups chunk members by city → ONE ``bands_all[unique_indices]`` call
      replaces N per-sample calls, cutting random-access overhead.
    * ``n_threads`` assembly threads run concurrently with GPU training
      (numpy releases the GIL; CUDA also releases the GIL).
    * No inter-process communication — threads share the preloaded RAM
      arrays directly (no pickling, no shared-memory setup).

    Usage
    -----
    Use with ``DataLoader(num_workers=0, pin_memory=True)``.
    """

    def __init__(self, base_ds: 'CachedPatchDataset',
                 chunk_size: int = 8192,
                 n_threads: int = 4,
                 queue_depth: int = 3,
                 shuffle: bool = True):
        self.ds = base_ds
        self.chunk_size = chunk_size
        self.n_threads = n_threads
        self.queue_depth = queue_depth
        self.shuffle = shuffle

    def __len__(self):
        return len(self.ds)

    # ------------------------------------------------------------------ #
    #  Vectorized chunk assembly                                           #
    # ------------------------------------------------------------------ #
    def _assemble_chunk(self, chunk_indices: np.ndarray) -> list:
        """Assemble a chunk of samples, grouped by city for bulk IO."""
        ds = self.ds
        G, P, C = ds.patch_grid_size, ds.patch_size, ds.num_channels
        half = G // 2
        offsets = ds._get_offsets(half)   # (G², 2) — cached, no alloc

        # ── group chunk by city ──────────────────────────────────────────
        city_groups: Dict[str, List[Tuple[int, int]]] = {}
        for chunk_pos, global_idx in enumerate(chunk_indices.tolist()):
            city, local_idx = ds.index_map[global_idx]
            city_groups.setdefault(city, []).append((chunk_pos, local_idx))

        results = [None] * len(chunk_indices)

        for city, members in city_groups.items():
            cd = ds.city_data[city]
            bands_all, bh_arr, bf_arr = ds._get_arrays(city)
            grid   = cd['grid']
            r_min, c_min   = cd['r_min'],  cd['c_min']
            n_rows, n_cols = cd['n_rows'], cd['n_cols']
            coords = cd['coords']

            # ── per-sample neighbour discovery (vectorized, no Python loop) ──
            sample_meta = []   # (chunk_pos, local_idx, K, inv_sort, gi, gj)
            acc_indices  = []  # all sorted neighbour indices (concatenated)

            for chunk_pos, local_idx in members:
                row_c = int(coords[local_idx, 0])
                col_c = int(coords[local_idx, 1])

                ri = row_c + offsets[:, 0] - r_min
                ci = col_c + offsets[:, 1] - c_min
                in_bounds = ((ri >= 0) & (ri < n_rows) &
                             (ci >= 0) & (ci < n_cols))
                ri_c = ri.clip(0, n_rows - 1)
                ci_c = ci.clip(0, n_cols - 1)
                arr_all  = grid[ri_c, ci_c]
                valid    = in_bounds & (arr_all >= 0)
                vf       = np.where(valid)[0]          # valid flat positions
                arr_idx  = arr_all[vf]                 # array indices

                sort_order = np.argsort(arr_idx)
                sorted_idx = arr_idx[sort_order]
                inv_sort   = np.empty(len(sort_order), dtype=np.int32)
                inv_sort[sort_order] = np.arange(len(sort_order), dtype=np.int32)

                gi = offsets[vf, 0] + half
                gj = offsets[vf, 1] + half

                sample_meta.append((chunk_pos, local_idx,
                                    len(sorted_idx), inv_sort, gi, gj))
                acc_indices.append(sorted_idx)

            if not acc_indices:
                continue

            # ── ONE bulk fancy-index per city (deduplicated) ─────────────
            all_idx = np.concatenate(acc_indices)
            unique_idx, inverse = np.unique(all_idx, return_inverse=True)
            # big_loaded: (U, C, P, P)  — U = number of unique neighbours
            big_loaded = bands_all[unique_idx][:, ds.band_indices, :, :]

            # ── scatter back to individual samples ────────────────────────
            inv_offset = 0
            for chunk_pos, local_idx, K, inv_sort, gi, gj in sample_meta:
                # positions in big_loaded for this sample's K neighbours
                pos = inverse[inv_offset: inv_offset + K]
                inv_offset += K

                loaded_ordered = big_loaded[pos][inv_sort]   # (K, C, P, P)

                input_grid = np.zeros((C, G, G, P, P), dtype=np.float32)
                mask_grid  = np.zeros((1, G, G, P, P), dtype=np.float32)
                input_grid[:, gi, gj, :, :] = loaded_ordered.transpose(1, 0, 2, 3)
                mask_grid [0,  gi, gj, :, :] = 1.0

                inp = input_grid.transpose(0, 1, 3, 2, 4).reshape(C,     G*P, G*P)
                msk = mask_grid .transpose(0, 1, 3, 2, 4).reshape(1,     G*P, G*P)
                inp = (inp - ds._means) / ds._stds
                full = np.concatenate([inp, msk], axis=0)

                results[chunk_pos] = (
                    torch.from_numpy(full),
                    torch.tensor(float(bh_arr[local_idx]), dtype=torch.float32),
                    torch.tensor(float(bf_arr[local_idx]), dtype=torch.float32),
                )

        return results

    # ------------------------------------------------------------------ #
    #  Iterator                                                            #
    # ------------------------------------------------------------------ #
    def __iter__(self):
        n = len(self.ds)
        indices = (np.random.permutation(n) if self.shuffle
                   else np.arange(n, dtype=np.int64))
        chunks = [indices[i: i + self.chunk_size]
                  for i in range(0, n, self.chunk_size)]

        chunk_q  = Q.Queue()
        result_q = Q.Queue(maxsize=self.queue_depth)
        err_box  = [None]

        for c in chunks:
            chunk_q.put(c)

        def worker():
            while True:
                try:
                    chunk = chunk_q.get_nowait()
                except Q.Empty:
                    return
                try:
                    assembled = self._assemble_chunk(chunk)
                    result_q.put(assembled)
                except Exception as e:
                    err_box[0] = e
                    result_q.put(None)   # unblock consumer
                    return

        threads = [threading.Thread(target=worker, daemon=True)
                   for _ in range(self.n_threads)]
        for t in threads:
            t.start()

        for _ in chunks:
            assembled = result_q.get()
            if assembled is None or err_box[0] is not None:
                raise RuntimeError(
                    f"ChunkedIterableDataset worker error: {err_box[0]}"
                ) from err_box[0]
            yield from assembled

        for t in threads:
            t.join()


class MorphoFormerDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule wrapper."""

    def __init__(self, train_path: str, val_path: str, test_path: str,
                 batch_size: int = 64, num_workers: int = 4,
                 patch_grid_size: int = 5,
                 modalities: List[str] = ['sar', 'optical', 'dem'],
                 preload: bool = False,
                 cache_dir: Optional[str] = None,
                 preload_cache: bool = False,
                 chunked: bool = False,
                 chunk_size: int = 8192,
                 chunk_threads: int = 4,
                 augment: bool = True,
                 track_test_during_training: bool = True):
        super().__init__()
        self.track_test_during_training = track_test_during_training
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_grid_size = patch_grid_size
        self.modalities = modalities
        self.preload = preload
        self.preload_cache = preload_cache
        self.chunked = chunked
        self.chunk_size = chunk_size
        self.chunk_threads = chunk_threads
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.augment = augment

    def _make_dataset(self, path_or_split, augment=False):
        """Return CachedPatchDataset if cache exists, else H5PatchDataset."""
        if self.cache_dir is not None:
            split = Path(path_or_split).stem   # 'train', 'valid', 'test2'
            split_cache = self.cache_dir / split
            if split_cache.exists():
                ds = CachedPatchDataset(
                    str(split_cache),
                    patch_grid_size=self.patch_grid_size,
                    modalities=self.modalities,
                    preload=self.preload_cache,
                    augment=augment,
                )
                if self.preload_cache:
                    mb = ds.total_ram_bytes / 1024**2
                    print(f"[preload] {split}: {mb:.0f} MB loaded into RAM")
                return ds
        return H5PatchDataset(
            path_or_split, patch_grid_size=self.patch_grid_size,
            modalities=self.modalities, augment=augment, preload=self.preload,
        )

    def setup(self, stage: Optional[str] = None):
        if stage in ('fit', None):
            self.train_dataset = self._make_dataset(self.train_path, augment=self.augment)
            self.val_dataset   = self._make_dataset(self.val_path)
            self.test_dataset  = self._make_dataset(self.test_path)

        if stage == 'test':
            if not hasattr(self, 'test_dataset'):
                self.test_dataset = self._make_dataset(self.test_path)

    def _make_dl(self, dataset, shuffle=False, persistent=False):
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=(persistent and self.num_workers > 0),
            prefetch_factor=(4 if self.num_workers > 0 else None),
        )

    def _make_chunked_dl(self, base_ds, shuffle=False):
        """DataLoader backed by ChunkedIterableDataset (num_workers=0)."""
        chunked_ds = ChunkedIterableDataset(
            base_ds,
            chunk_size=self.chunk_size,
            n_threads=self.chunk_threads,
            shuffle=shuffle,
        )
        return DataLoader(
            chunked_ds, batch_size=self.batch_size,
            num_workers=0, pin_memory=True,
        )

    def train_dataloader(self):
        if self.chunked:
            return self._make_chunked_dl(self.train_dataset, shuffle=True)
        return self._make_dl(self.train_dataset, shuffle=True, persistent=True)

    def val_dataloader(self):
        """Returns [validation, test].

        The test split is scored every epoch purely so that training curves can
        be inspected; those numbers are logged under the ``test_epoch/`` prefix
        and are never used to make a decision. Model selection and early
        stopping monitor ``val/combined_mae`` only (see scripts/train.py), and
        every reported result is taken from the checkpoint with the lowest
        ``val/combined_mae`` for that configuration.

        Pass ``track_test_during_training=False`` to drop the second loader if
        you would rather not have the test split touched during fitting; this
        changes nothing about training itself.
        """
        if not self.track_test_during_training:
            return (self._make_chunked_dl(self.val_dataset, shuffle=False)
                    if self.chunked else self._make_dl(self.val_dataset))
        if self.chunked:
            return [
                self._make_chunked_dl(self.val_dataset,  shuffle=False),
                self._make_chunked_dl(self.test_dataset, shuffle=False),
            ]
        return [
            self._make_dl(self.val_dataset),
            self._make_dl(self.test_dataset),
        ]

    def test_dataloader(self):
        if self.chunked:
            return self._make_chunked_dl(self.test_dataset, shuffle=False)
        return self._make_dl(self.test_dataset)
