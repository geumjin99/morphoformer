"""Filesystem defaults for data, cache and checkpoints.

No path in this repository is hard-coded to a particular machine. Every entry
point resolves its locations through this module, in the following order:

1. an explicit command-line flag (``--data-root``, ``--cache-dir``, ...);
2. the corresponding environment variable;
3. a directory relative to the repository root.

So a fresh clone works with either

    export MORPHOFORMER_DATA=/somewhere/h5
    export MORPHOFORMER_CACHE=/somewhere/npy_cache

or by simply placing the files under ``<repo>/data`` and ``<repo>/data_cache``.
"""

import os
from pathlib import Path

#: Repository root (this file lives at <root>/morphoformer/paths.py).
REPO_ROOT = Path(__file__).resolve().parent.parent


def _from_env(var: str, default: Path) -> Path:
    value = os.environ.get(var)
    return Path(value).expanduser() if value else default


def data_root() -> Path:
    """Directory holding ``train.h5`` / ``valid.h5`` / ``test2.h5``."""
    return _from_env('MORPHOFORMER_DATA', REPO_ROOT / 'data')


def cache_root() -> Path:
    """Directory holding the per-city ``.npy`` cache (see scripts/precompute_cache.py)."""
    return _from_env('MORPHOFORMER_CACHE', REPO_ROOT / 'data_cache')


def checkpoint_root() -> Path:
    """Directory under which per-experiment checkpoint folders are created."""
    return _from_env('MORPHOFORMER_CHECKPOINTS', REPO_ROOT / 'checkpoints')


def results_root() -> Path:
    """Directory holding the released ``preds_*.npz`` prediction dumps."""
    return _from_env('MORPHOFORMER_RESULTS', REPO_ROOT / 'results')


def split_paths() -> dict:
    """Default H5 paths for the three splits."""
    root = data_root()
    return {
        'train': root / 'train.h5',
        'val': root / 'valid.h5',
        'test': root / 'test2.h5',
    }
