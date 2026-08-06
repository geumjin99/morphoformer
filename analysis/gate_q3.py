"""Extract BGTD cross-gate activations on the test split, then characterise:
  - distribution of per-sample mean gate value (closer to 0 = lean on
    morph_ctx; closer to 1 = lean on bh_feat)
  - stratification of gate behaviour by lambda_p bin
  - per-channel gate heterogeneity (do channels specialise?)

Outputs:
  - gates_full.npz     : per-sample gate vectors (n × D)
  - q3_gate.png        : 2-panel figure
       (a) histogram of mean(gate) over test set
       (b) mean gate value per BF bin
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from morphoformer import MorphoFormerDataModule, MorphoFormerLitModule, build_morphoformer, build_mcl_loss
from morphoformer.paths import cache_root, checkpoint_root, results_root, split_paths

CKPT = str(checkpoint_root() / 'morphoformer_full.ckpt')
OUT_DIR = results_root()
OUT_DIR.mkdir(exist_ok=True)
GATES_NPZ = OUT_DIR / 'gates_full.npz'
PNG = OUT_DIR / 'q3_gate.png'


def main():
    pl.seed_everything(42)
    torch.set_float32_matmul_precision('high')

    if GATES_NPZ.exists():
        d = np.load(GATES_NPZ)
        gates = d['gates']; bf_t = d['bf_true']; bh_t = d['bh_true']
        print(f'loaded cache: gates={gates.shape}')
    else:
        in_chans = 2 + 4 + 1 + 1
        model = build_morphoformer(
            variant='base', in_chans=in_chans,
            patch_grid_size=9, patch_size=2,
            center_sizes=(3, 5, 9),
            use_uncertainty=True, use_amge=True, use_bgtd=True,
        )
        loss_fn = build_mcl_loss(loss_type='huber', lambda_consist=0.2, warmup_epochs=10)
        lit = MorphoFormerLitModule.load_from_checkpoint(CKPT, model=model, loss_fn=loss_fn, strict=True)
        lit = lit.cuda().eval()

        decoder = lit.model.bgtd  # BFGuidedTaskDecoder

        # Hook the cross_gate output
        captured = []
        def hook(module, inp, out):
            captured.append(out.detach().float().cpu().numpy())
        h = decoder.cross_gate.register_forward_hook(hook)

        dm = MorphoFormerDataModule(
            train_path=str(split_paths()['train']),
            val_path=str(split_paths()['val']),
            test_path=str(split_paths()['test']),
            batch_size=256, num_workers=4,
            patch_grid_size=9, modalities=['sar', 'optical', 'dem'],
            preload=False,
            cache_dir=str(cache_root()),
            preload_cache=True,
            chunked=False,
        )
        dm.setup('test')

        bh_t, bf_t = [], []
        with torch.no_grad():
            for batch in dm.test_dataloader():
                x, bh, bf = batch
                x = x.cuda(non_blocking=True)
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    _ = lit.model(x)
                bh_t.append(bh.numpy()); bf_t.append(bf.numpy())
        h.remove()
        gates = np.concatenate(captured, axis=0)
        bh_t = np.concatenate(bh_t)
        bf_t = np.concatenate(bf_t)
        np.savez_compressed(GATES_NPZ, gates=gates, bh_true=bh_t, bf_true=bf_t)
        print(f'saved {GATES_NPZ} : gates={gates.shape}')

    # ─── analysis ───
    g_mean_per_sample = gates.mean(axis=1)
    print(f'\nGate statistics over {len(g_mean_per_sample):,} test samples:')
    print(f'  mean(gate) sample-wise mean: {g_mean_per_sample.mean():.3f}')
    print(f'  mean(gate) sample-wise std : {g_mean_per_sample.std():.3f}')
    print(f'  per-channel mean: min {gates.mean(0).min():.3f} max {gates.mean(0).max():.3f}')
    print(f'  per-channel std : min {gates.std(0).min():.3f} max {gates.std(0).max():.3f}')

    # stratify by BF
    bf_bins = [(0.01, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.35),
               (0.35, 0.55), (0.55, 1.0)]
    bin_labels = [f'{lo:.2f}-{hi:.2f}' for lo, hi in bf_bins]
    bin_centers = [0.5*(lo+hi) for lo, hi in bf_bins]
    g_per_bin_mean = []
    g_per_bin_std = []
    n_per_bin = []
    for lo, hi in bf_bins:
        sel = (bf_t >= lo) & (bf_t < hi)
        if sel.sum() == 0:
            g_per_bin_mean.append(np.nan); g_per_bin_std.append(np.nan); n_per_bin.append(0)
            continue
        gm = g_mean_per_sample[sel]
        g_per_bin_mean.append(gm.mean())
        g_per_bin_std.append(gm.std())
        n_per_bin.append(int(sel.sum()))
    g_per_bin_mean = np.array(g_per_bin_mean)
    g_per_bin_std = np.array(g_per_bin_std)

    print('\n--- per-BF-bin gate statistics ---')
    for lab, n, m, s in zip(bin_labels, n_per_bin, g_per_bin_mean, g_per_bin_std):
        print(f'  {lab:<14s}  n={n:>7d}  gate mean={m:.3f}  std={s:.3f}')

    # ─── plot ───
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 4.0),
                                        gridspec_kw={'width_ratios': [1.0, 1.0, 1.0]})

    # (a) histogram of sample-mean gate
    ax1.hist(g_mean_per_sample, bins=60, color='#1f77b4', alpha=0.8, edgecolor='white', linewidth=0.4)
    ax1.axvline(0.5, color='black', ls='--', lw=1, alpha=0.7,
                label='neutral (g=0.5)')
    ax1.axvline(g_mean_per_sample.mean(), color='crimson', ls='-', lw=1.4,
                label=f'sample-wise mean = {g_mean_per_sample.mean():.2f}')
    ax1.set_xlabel('Per-sample mean gate value $\\bar{g}$', fontsize=10)
    ax1.set_ylabel('Number of test samples', fontsize=10)
    ax1.set_title(f'(a) Distribution of mean BGTD gate over {len(g_mean_per_sample):,} cells', fontsize=10)
    ax1.legend(fontsize=9, frameon=False, loc='upper left')
    ax1.set_xlim(0, 1)
    ax1.tick_params(labelsize=8.5)

    # (b) stratified by BF
    ax2.errorbar(bin_centers, g_per_bin_mean, yerr=g_per_bin_std,
                 fmt='o-', color='#2ca02c', linewidth=1.7, markersize=7,
                 capsize=3, capthick=1, ecolor='gray', alpha=0.85)
    ax2.axhline(0.5, color='black', ls='--', lw=0.8, alpha=0.6)
    ax2.set_xlabel('Footprint ratio $\\lambda_p$ (test bin centre)', fontsize=10)
    ax2.set_ylabel('Mean BGTD gate value', fontsize=10)
    ax2.set_title('(b) Gate value vs footprint ratio (mean ± std)', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    for k, (xc, n) in enumerate(zip(bin_centers, n_per_bin)):
        ax2.annotate(f'n={n//1000}k', (xc, g_per_bin_mean[k] + g_per_bin_std[k] + 0.02),
                     textcoords='offset points', xytext=(0, 0),
                     fontsize=7.5, ha='center', alpha=0.65)
    ax2.tick_params(labelsize=8.5)

    # (c) per-channel gate mean (sorted) — show channel specialisation
    per_chan_mean = gates.mean(axis=0)
    per_chan_std  = gates.std(axis=0)
    order = np.argsort(per_chan_mean)
    pcm_s = per_chan_mean[order]
    pcs_s = per_chan_std[order]
    x_idx = np.arange(len(pcm_s))
    ax3.fill_between(x_idx, pcm_s - pcs_s, pcm_s + pcs_s,
                     alpha=0.25, color='#9467bd', label='± 1 std across samples')
    ax3.plot(x_idx, pcm_s, '-', color='#9467bd', lw=1.6, label='per-channel mean')
    ax3.axhline(0.5, color='black', ls='--', lw=0.8, alpha=0.6)
    ax3.set_xlabel('Gate channel index (sorted by mean)', fontsize=10)
    ax3.set_ylabel('Gate value', fontsize=10)
    ax3.set_title(f'(c) Per-channel gate specialisation (D={len(pcm_s)})', fontsize=10)
    ax3.legend(fontsize=9, frameon=False, loc='upper left')
    ax3.set_xlim(-0.5, len(pcm_s)-0.5)
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelsize=8.5)

    fig.tight_layout()
    fig.savefig(PNG, dpi=200, bbox_inches='tight')
    print(f'\nsaved {PNG}')


if __name__ == '__main__':
    main()
