import sys, random
import numpy as np
sys.path.insert(0, '.')
from morphoformer.data.dataset import CachedPatchDataset
from morphoformer.paths import cache_root

cache = str(cache_root() / 'valid')

ds_fast = CachedPatchDataset(cache)
ds_slow = CachedPatchDataset(cache)
for cd in ds_slow.city_data.values():
    cd['_nbr'] = None

has_nbr = sum(1 for cd in ds_fast.city_data.values() if cd['_nbr'] is not None)
print(f'Cities with precomputed nbr: {has_nbr}/{len(ds_fast.city_data)}')

idxs = random.sample(range(len(ds_fast)), 500)
max_diff = 0.0
for i in idxs:
    x_fast, bh_f, bf_f = ds_fast[i]
    x_slow, bh_s, bf_s = ds_slow[i]
    diff = float((x_fast - x_slow).abs().max())
    max_diff = max(max_diff, diff)

print(f'Max abs diff (fast vs slow): {max_diff:.2e}')
print('PASS' if max_diff < 1e-5 else 'FAIL - MISMATCH')
