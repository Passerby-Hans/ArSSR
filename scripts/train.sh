#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# train.sh — build HR patches + train (fine-tune) ArSSR on the H100.
#
#   HR_DIR=/data/hr_abdomen bash scripts/train.sh
#
# Env vars:
#   HR_DIR    (required) dir of HIGH-res abdominal NIfTI volumes (HR patches source)
#   VENV      venv path (default ~/.venvs/arssr, created by deploy.sh)
#   EPOCH     epochs (default 1000; paper uses 2500)
#   ENCODER   RDN | ResCNN | SRResNet (default RDN)
#   PATCHES_PER_VOL  random 40^3 crops per volume (default 6)
#   FROM_SCRATCH=1   don't load pretrained (train from scratch instead of fine-tune)
#   GPU       device id (default 0)
#
# NOTE: ArSSR is self-supervised (HR target, LR downsampled in data.py). The HR
# volumes must be genuinely high-res; our anisotropic clinical data is the thing
# we want to SR, so point HR_DIR at proper high-res abdominal scans.
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

VENV="${VENV:-$HOME/.venvs/arssr}"
# shellcheck disable=SC1091
. "$VENV/bin/activate"

HR_DIR="${HR_DIR:?need HR_DIR (dir of high-res abdominal NIfTI volumes)}"
EPOCH="${EPOCH:-1000}"
ENCODER="${ENCODER:-RDN}"
PATCHES_PER_VOL="${PATCHES_PER_VOL:-6}"
GPU="${GPU:-0}"

case "$ENCODER" in
    RDN)      PT="pre_trained_models/ArSSR_RDN.pkl" ;;
    ResCNN)   PT="pre_trained_models/ArSSR_ResCNN.pkl" ;;
    SRResNet) PT="pre_trained_models/ArSSR_SRResnet.pkl" ;;
    *) echo "unknown ENCODER $ENCODER"; exit 1 ;;
esac

echo "=== build HR patches from $HR_DIR ==="
python scripts/build_data.py --hr_dir "$HR_DIR" --out_dir data --patches_per_vol "$PATCHES_PER_VOL"

FT_ARGS=()
if [ -z "${FROM_SCRATCH:-}" ]; then
    echo "(fine-tune from $PT; set FROM_SCRATCH=1 for from-scratch)"
    FT_ARGS=(-pre_trained_model "$PT")
else
    echo "(training from scratch)"
fi

echo "=== train: encoder=$ENCODER epoch=$EPOCH gpu=$GPU ==="
python train.py -encoder_name "$ENCODER" -epoch "$EPOCH" -bs 15 -ss 8000 -gpu "$GPU" "${FT_ARGS[@]}"

echo "=== done. checkpoints in ./model/, tensorboard in ./log/ (tensorboard --logdir log) ==="
