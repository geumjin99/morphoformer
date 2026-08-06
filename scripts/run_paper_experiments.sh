#!/usr/bin/env bash
# Trains every configuration reported in the paper, sequentially.
#
# Prerequisites:
#   1. HDF5 splits built by scripts/prepare_h5_data.py  (or MORPHOFORMER_DATA set)
#   2. npy cache built by scripts/precompute_cache.py   (or MORPHOFORMER_CACHE set)
#
# Wall-clock on a single RTX 3090: roughly 15 min/epoch for MorphoFormer at
# patch_size=2, so about 25-30 h per run and several days for the whole sweep.
#
# Every run passes --resume last, so an interrupted sweep can simply be
# restarted with the same command.

set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"

COMMON="--patch-size 2 --batch-size 512 --lr 2e-4 --epochs 150 \
        --early-stopping 20 --num-workers 8 --preload-cache --resume last"

run () {  # run <exp-name> [extra flags...]
    local name=$1; shift
    echo "=== ${name} — $(date '+%F %T') ==="
    python scripts/train.py --exp-name "$name" $COMMON "$@"
}

# ── Proposed model, three seeds ────────────────────────────────────────────
run morphoformer_full                       --seed 42
run morphoformer_full_seed43                --seed 43
run morphoformer_full_seed44                --seed 44

# ── Per-module ablations (exactly one switch changed each) ─────────────────
run morphoformer_no_amge   --no-amge            # drop AMGE
run morphoformer_no_msmp   --center-sizes 5     # single-scale pooling
run morphoformer_no_bgtd   --no-bgtd            # drop BGTD (also disables MCL)
run morphoformer_no_mcl    --lambda-consist 0.0 # drop MCL only, decoder unchanged

# ── Baselines, identical data / split / loss / evaluation ──────────────────
python scripts/train_cnn_baseline.py --baseline resnet --exp-name baseline_resnet_9x9
python scripts/train_cnn_baseline.py --baseline senet  --exp-name baseline_senet_9x9
python scripts/train_vit.py          --exp-name baseline_vit_9x9   --patch-grid-size 9 \
    --batch-size 512 --lr 2e-4 --epochs 100 --early-stopping 15 --num-workers 8 --preload-cache
python scripts/train_mfbhnet.py      --exp-name baseline_mfbhnet   --patch-size 2 \
    --batch-size 256 --num-workers 8 --preload-cache
python scripts/train_crossstitch.py  --exp-name baseline_crossstitch --patch-size 2 \
    --batch-size 512 --lr 2e-4 --num-workers 8 --preload-cache

echo "All runs finished — $(date '+%F %T')"
echo "Next: dump predictions with scripts/dump_preds.py, then analysis/metrics.py"
