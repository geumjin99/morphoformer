# MorphoFormer

Reference implementation of

> **Morphology-Guided Cross-Task Coupling for Joint Building Height and Footprint Estimation**
> Jinzhen Han, JinByeong Lee, Jisung Kim, HongSik Yun · *Science of Remote Sensing*, 2026

MorphoFormer estimates mean building height (BH, metres) and building footprint
ratio (BF, unitless) for a 100 m grid cell from a 9×9-cell neighbourhood of
Sentinel-1 SAR, Sentinel-2 optical and SRTM DEM imagery. Its two mechanisms
target the floor-area-ratio coupling between the two targets rather than the
encoder:

* **BGTD** — a BF-Guided Task Decoder, in which a footprint-derived morphology
  context gates the height branch;
* **MCL** — a Morphology Consistency Loss, which supervises a
  height-from-footprint surrogate against ground-truth height.

The encoder is a single-stage Swin backbone preceded by per-modality gating
(**AMGE**) and followed by multi-scale centre pooling (**MSMP**). The whole
model is 0.373 M parameters.

---

## What is in this repository

```
morphoformer/          model, data pipeline, losses, Lightning module
scripts/               data preparation, training, evaluation entry points
analysis/              the scripts behind the paper's figures and tables
checkpoints/           trained weights (see below)
results/               per-cell test-set predictions for every reported row
docs/reproducibility.md  what reproduces exactly, what does not, and why
```

**No source imagery or label raster is redistributed here.** See *Data* below.

## Installation

```bash
git clone https://github.com/geumjin99/morphoformer.git
cd morphoformer
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Results were produced with Python 3.11, PyTorch 2.5.1 + CUDA 12.1 and a single
NVIDIA RTX 3090 (24 GB).

## Quickest way to check the paper's numbers

The per-cell predictions behind every reported row are in `results/`, so the
comparison and ablation tables can be recomputed in seconds, without data,
GPU or training:

```bash
python analysis/metrics.py
```

```
configuration                     RMSE_h   MAE_h   R2_h  RMSE_f   MAE_f   R2_f
------------------------------------------------------------------------------
Swin-MTL (baseline)                3.386   1.604  0.618  0.0525  0.0319  0.790
ResNet-MTL (baseline)              3.398   1.597  0.616  0.0576  0.0357  0.747
SENet-MTL (baseline)               3.416   1.599  0.612  0.0570  0.0359  0.753
MF-BHNet (re-implemented)          3.159   1.529  0.668  0.0539  0.0338  0.779
Cross-stitch MTL                   3.177   1.515  0.664  0.0514  0.0316  0.799
MorphoFormer (full, seed 42)       3.145   1.483  0.671  0.0509  0.0314  0.803
  w/o AMGE                         3.187   1.518  0.662  0.0521  0.0323  0.794
  w/o MSMP (single scale)          3.128   1.487  0.674  0.0520  0.0320  0.794
  w/o BGTD (also disables MCL)     3.260   1.540  0.646  0.0511  0.0315  0.801
  w/o MCL (lambda_consist = 0)     3.254   1.534  0.648  0.0506  0.0313  0.805
------------------------------------------------------------------------------
MorphoFormer over 3 seeds: BH RMSE 3.134 +/- 0.017 m, BH R2 0.673 +/- 0.0035
```

## Data

This repository does **not** contain the training data, and none of the source
products may be redistributed from here.

The 100 m building-height and building-footprint reference labels come from the
**SHAFTS** product:

> Li, R., Sun, T., Tian, F., Ni, G.-H. (2023). SHAFTS (v2022.3): a deep-learning-based
> Python package for simultaneous extraction of building height and footprint from
> Sentinel imagery. *Geoscientific Model Development* **16**(2), 751–778.
> https://doi.org/10.5194/gmd-16-751-2023
> Raster products: https://doi.org/10.5281/zenodo.6370003

Predictors are Sentinel-1 (VH, VV) and Sentinel-2 (B2, B3, B4, B8) annual
50th-percentile composites plus SRTM DEM resampled to 100 m.

To rebuild the HDF5 splits from those sources:

```bash
python scripts/prepare_h5_data.py \
    --data-info-json  /path/to/data_info.json \
    --satellite-dir   /path/to/SatelliteData \
    --building-dir    /path/to/BuildingInfo \
    --srtm-dir        /path/to/SRTM \
    --output-dir      ./data
python scripts/precompute_cache.py     # ~20 GB of .npy, ~10x faster loading
```

The study uses 51 cities that carry building-height attributes, split
geographically within each city (contiguous blocks assigned wholly to train,
validation or test) so that no test cell is adjacent to a training cell:
1,879,513 training / 223,322 validation / 207,579 test cells.

No path is hard-coded. Either place the files under `./data` and `./data_cache`,
or point the code elsewhere:

```bash
export MORPHOFORMER_DATA=/elsewhere/h5
export MORPHOFORMER_CACHE=/elsewhere/npy_cache
```

## Training

```bash
python scripts/train.py --exp-name morphoformer_full \
    --patch-size 2 --batch-size 512 --lr 2e-4 \
    --epochs 150 --early-stopping 20 --num-workers 8 --preload-cache
```

That is the configuration behind the reported results (~15 min/epoch on one
RTX 3090). Each ablation changes exactly one switch:

| ablation | flag |
|---|---|
| w/o AMGE | `--no-amge` |
| w/o MSMP | `--center-sizes 5` |
| w/o BGTD | `--no-bgtd` — note this also disables MCL, see below |
| w/o MCL | `--lambda-consist 0.0` |

`scripts/run_paper_experiments.sh` runs the whole sweep, baselines included.

**On `--no-bgtd`:** the ablated decoder emits no height-from-footprint
surrogate, and MCL is defined on that surrogate, so removing BGTD necessarily
removes MCL with it. The two ablations are therefore not independent, and
their effects are not additive. `--lambda-consist 0.0` is the clean loss-only
ablation: the decoder is bit-identical to the full model and only the
consistency weight changes.

**Model selection** monitors `val/combined_mae` and nothing else. Every epoch
is checkpointed (`save_top_k=-1`) and each reported result is the checkpoint
with the lowest validation `combined_mae` for that configuration. The test
split is also scored each epoch for curve inspection only, logged under
`test_epoch/`; pass `--no-track-test` to switch that off — it changes nothing
about the trained model.

## Evaluation

```bash
# metrics for one checkpoint
python scripts/test_only.py --ckpt checkpoints/morphoformer_full.ckpt --patch-size 2

# per-cell predictions -> npz (the single routine used for every reported row)
python scripts/dump_preds.py --ckpt checkpoints/morphoformer_full.ckpt \
    --patch-size 2 --out results/preds_full.npz
```

`--patch-size` must match the checkpoint or the shapes will not load.

## Checkpoints

`checkpoints/` holds the validation-optimal weights for the MorphoFormer
family (~3.8 MB each):

| file | configuration | BH RMSE | BH R² |
|---|---|---|---|
| `morphoformer_full.ckpt` | full model, seed 42 — **the paper's model** | 3.145 | 0.671 |
| `morphoformer_full_seed43.ckpt` | full model, seed 43 | 3.143 | 0.671 |
| `morphoformer_full_seed44.ckpt` | full model, seed 44 | 3.115 | 0.677 |
| `morphoformer_no_amge.ckpt` | w/o AMGE | 3.187 | 0.662 |
| `morphoformer_no_msmp.ckpt` | w/o MSMP | 3.128 | 0.674 |
| `morphoformer_no_bgtd.ckpt` | w/o BGTD | 3.260 | 0.646 |
| `morphoformer_no_mcl.ckpt` | w/o MCL | 3.254 | 0.648 |
| `crossstitch.ckpt` | cross-stitch MTL | 3.177 | 0.664 |

Baseline weights are far larger (ResNet/SENet ≈ 135 MB, MF-BHNet ≈ 224 MB) and
are not committed; their predictions are in `results/` and they retrain from
`scripts/run_paper_experiments.sh`.

## Notes on the released configuration

**MSMP crop sizes are in tokens, not in 100 m cells.** For the reported
configuration (`--patch-size 2`) the Swin token grid is 45×45 and one token
spans 20 m, so the 3/5/9 crops cover 60/100/180 m and read the central 9×9 of
the 45×45 token map. At `--patch-size 10` the token grid is 9×9, one token is
a 100 m cell, and the same crops cover 300/500/900 m. The 900 m figure is the
extent of the *input* window in both cases; context beyond the crop radius
reaches the pooled feature through the Swin blocks' windowed and
shifted-window attention. See the docstring of `MultiScaleMorphologyPool`.

`docs/reproducibility.md` records what regenerates exactly and what does not.

## Citation

```bibtex
@article{han2026morphoformer,
  title   = {Morphology-Guided Cross-Task Coupling for Joint Building Height
             and Footprint Estimation},
  author  = {Han, Jinzhen and Lee, JinByeong and Kim, Jisung and Yun, HongSik},
  journal = {Science of Remote Sensing},
  year    = {2026}
}
```

Please also cite SHAFTS (above) if you rebuild the dataset.

## License

MIT for the code; see `LICENSE`, which also states the terms attaching to the
reference values inside `results/`.
