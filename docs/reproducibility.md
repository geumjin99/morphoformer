# Reproducibility notes

What in this repository regenerates the reported numbers exactly, what only
regenerates them approximately, and what cannot be regenerated here at all.

## Evaluation protocol

Every reported row is produced the same way:

1. train a configuration, checkpointing every epoch;
2. take the checkpoint with the lowest validation `combined_mae`
   (`mae_h + mae_f`) — never a test-set quantity;
3. push that checkpoint through `scripts/dump_preds.py`, which writes a
   per-cell `.npz` for the 207,579 test cells;
4. compute metrics from the `.npz` with `analysis/metrics.py`.

Steps 3 and 4 are shared by all configurations, so the numbers in the tables
are commensurable by construction. `metrics.py` recomputes the tables from
`results/` alone and needs neither data nor a GPU.

## Exact vs. approximate

**Exact.** `analysis/metrics.py` on the shipped `results/*.npz` reproduces every
number in the comparison and ablation tables bit-for-bit.

**Approximate at the fourth decimal.** Regenerating a `.npz` from a shipped
checkpoint reproduces the metrics to within a few thousandths of a metre. From
`checkpoints/morphoformer_full.ckpt`:

| | BH RMSE | BH R² | BF RMSE | BF R² |
|---|---|---|---|---|
| shipped `results/preds_full.npz` | 3.1454 | 0.6707 | 0.05088 | 0.8030 |
| regenerated via `dump_preds.py` | 3.1472 | 0.6703 | 0.05088 | 0.8030 |

The residual (mean |Δ| ≈ 0.005 m per cell) is floating-point precision: some of
the shipped dumps were written under mixed-precision inference, whereas
`dump_preds.py` now defaults to fp32. Both round to the reported 3.15 / 0.67.
`--precision 16-mixed` reproduces the other side of that difference.

**Not bitwise across machines.** Retraining from scratch with the same seed
will not reproduce a checkpoint bit-for-bit: cuDNN kernel selection, GPU model
and mixed-precision accumulation order all vary. The seed-42/43/44 runs give a
direct read on that spread — BH RMSE 3.134 ± 0.017 m, BH R² 0.673 ± 0.0035 —
and differences smaller than that band should not be interpreted.

**Not regenerable here.** The Swin-MTL reference row was trained under the
first-generation codebase, which applied its own band normalisation and its own
training loop. Its predictions are shipped as `results/preds_swin.npz` and
cannot be reproduced by the scripts in this repository. The ResNet-MTL and
SENet-MTL baselines, by contrast, were trained through the pipeline included
here and retrain from `scripts/run_paper_experiments.sh`.

## Things worth knowing before drawing conclusions

**`--no-bgtd` also removes MCL.** The ablated decoder (`SimpleTaskDecoder`)
returns no height-from-footprint surrogate, and the consistency term is
skipped when that surrogate is absent
(`morphoformer/losses/mcl_loss.py`, `if bh_from_bf is not None`). So the
w/o-BGTD row is *BGTD and MCL removed together*, not BGTD alone, and the
w/o-BGTD and w/o-MCL deltas do not add. `--lambda-consist 0.0` is the clean
ablation of MCL by itself: identical decoder, identical parameter count, only
the consistency weight set to zero.

**The w/o-MSMP row is nominally better than the full model** (3.128 vs 3.145 m).
The difference is smaller than the ±0.017 m seed band above — one of the three
full-model seeds reaches 3.115 m — so this experiment does not separate the two
configurations. The paper's ablation table reports the same figures.

**MSMP crop sizes are token counts.** At the reported `--patch-size 2` the token
grid is 45×45 (one token = 20 m), so the 3/5/9 crops cover 60/100/180 m and read
the central 9×9 of 45×45 tokens — 4 % of the map. At `--patch-size 10` the token
grid is 9×9 (one token = one 100 m cell) and the same crops cover 300/500/900 m.
The 900 m figure describes the *input* window in both cases; anything outside
the crop radius reaches the pooled feature only through the two Swin blocks'
windowed and shifted-window attention.

**The test split is scored during training.** It is logged under `test_epoch/`
for curve inspection and is not consulted by checkpoint selection or early
stopping, both of which monitor `val/combined_mae`. `--no-track-test` removes
the second validation loader entirely; the trained model is unaffected.

**Augmentation** is the 8-element dihedral group (rot90 ×4 × horizontal flip).
Both targets are scalars describing the centre cell and are invariant under it,
so labels need no adjustment. It is on by default; `--no-augment` disables it.

## Re-implemented baselines

`MFBHNetMTL` and `CrossStitchMTL` are our own re-implementations, not code
released by those authors.

* **MF-BHNet** (Wang et al., IEEE TGRS 2024) is natively a dense per-pixel
  model. We keep its stated contributions — the IME/CME encoder and the
  MFF/MSF fusion — and replace only its generic U-Net-style decoder with global
  pooling plus two scalar heads, since our task is a scene-level scalar
  regression. Widths, CME depth and SCConv hyper-parameters are not given in
  the paper and were chosen as documented in the module docstring.
* **Cross-stitch** (Misra et al., CVPR 2016) uses MorphoFormer's own encoder so
  that the decoder coupling is the only difference.

Both should be read as faithful-in-spirit reproductions under our protocol, not
as authoritative reproductions of the original results.
