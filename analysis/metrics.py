"""Recompute the paper's comparison and ablation tables from results/*.npz.

Every row is computed here by one metric function applied to one prediction
dump, so the tables cannot drift from the stored predictions.

    python analysis/metrics.py              # all shipped dumps
    python analysis/metrics.py --csv out.csv

Each ``.npz`` holds four float32 vectors of length 207,579 (the test split):
``bh_true``, ``bh_pred``, ``bf_true``, ``bf_pred``.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from morphoformer.paths import results_root

# Display order and labels; keys are file stems under results/.
ROWS = [
    ('preds_swin',        'Swin-MTL (baseline)'),
    ('preds_resnet',      'ResNet-MTL (baseline)'),
    ('preds_senet',       'SENet-MTL (baseline)'),
    ('preds_mfbhnet',     'MF-BHNet (re-implemented)'),
    ('preds_crossstitch', 'Cross-stitch MTL'),
    ('preds_full',        'MorphoFormer (full, seed 42)'),
    ('preds_seed43',      'MorphoFormer (full, seed 43)'),
    ('preds_seed44',      'MorphoFormer (full, seed 44)'),
    ('preds_noamge',      '  w/o AMGE'),
    ('preds_nomsmp',      '  w/o MSMP (single scale)'),
    ('preds_nobgtd',      '  w/o BGTD (also disables MCL)'),
    ('preds_nomcl',       '  w/o MCL (lambda_consist = 0)'),
]


def metrics(path: Path) -> dict:
    d = np.load(path)
    ht, hp = d['bh_true'].ravel(), d['bh_pred'].ravel()
    ft, fp = d['bf_true'].ravel(), d['bf_pred'].ravel()

    def r2(t, p):
        return 1.0 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum()

    return {
        'n': len(ht),
        'rmse_h': float(np.sqrt(((hp - ht) ** 2).mean())),
        'mae_h': float(np.abs(hp - ht).mean()),
        'r2_h': float(r2(ht, hp)),
        'rmse_f': float(np.sqrt(((fp - ft) ** 2).mean())),
        'mae_f': float(np.abs(fp - ft).mean()),
        'r2_f': float(r2(ft, fp)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', default=str(results_root()))
    ap.add_argument('--csv', help='Also write the table to this CSV file')
    args = ap.parse_args()

    root = Path(args.results_dir)
    header = (f"{'configuration':32s} {'RMSE_h':>7s} {'MAE_h':>7s} {'R2_h':>6s}"
              f" {'RMSE_f':>7s} {'MAE_f':>7s} {'R2_f':>6s}")
    print(header)
    print('-' * len(header))

    collected = []
    for stem, label in ROWS:
        path = root / f'{stem}.npz'
        if not path.exists():
            print(f'{label:32s} {"(missing)":>7s}')
            continue
        m = metrics(path)
        collected.append((label, m))
        print(f"{label:32s} {m['rmse_h']:7.3f} {m['mae_h']:7.3f} {m['r2_h']:6.3f}"
              f" {m['rmse_f']:7.4f} {m['mae_f']:7.4f} {m['r2_f']:6.3f}")

    seeds = [m for label, m in collected if 'full, seed' in label]
    if len(seeds) > 1:
        rmse = np.array([m['rmse_h'] for m in seeds])
        r2 = np.array([m['r2_h'] for m in seeds])
        print('-' * len(header))
        print(f'MorphoFormer over {len(seeds)} seeds: '
              f'BH RMSE {rmse.mean():.3f} +/- {rmse.std(ddof=1):.3f} m, '
              f'BH R2 {r2.mean():.3f} +/- {r2.std(ddof=1):.4f}')

    if args.csv:
        import csv
        with open(args.csv, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['configuration', 'n', 'rmse_h', 'mae_h', 'r2_h',
                        'rmse_f', 'mae_f', 'r2_f'])
            for label, m in collected:
                w.writerow([label.strip(), m['n'],
                            f"{m['rmse_h']:.4f}", f"{m['mae_h']:.4f}", f"{m['r2_h']:.4f}",
                            f"{m['rmse_f']:.5f}", f"{m['mae_f']:.5f}", f"{m['r2_f']:.4f}"])
        print(f'\nwrote {args.csv}')


if __name__ == '__main__':
    main()
