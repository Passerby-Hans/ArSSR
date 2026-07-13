#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# infer.sh — super-resolve LR volumes with ArSSR at an arbitrary (float) scale.
#
#   LR_DIR=/data/lr SCALE=2.67 bash scripts/infer.sh
#
# Env vars:
#   LR_DIR   (required) dir of LR NIfTI volumes to super-resolve
#   SCALE    up-sampling factor, int or float (default 2.67  -> T1map 20mm ~ T2 7.5mm)
#   ENCODER  RDN | ResCNN | SRResNet (default RDN)
#   MODEL    .pkl weights (default: pretrained HCP-brain; set to your trained ./model/xxx.pkl)
#   GPU      device id (default 0)
#
# Output: test/output/ArSSR_<encoder>_recon_<scale>x_<name>.nii.gz
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

VENV="${VENV:-$HOME/.venvs/arssr}"
# shellcheck disable=SC1091
. "$VENV/bin/activate"

LR_DIR="${LR_DIR:?need LR_DIR (dir of LR NIfTI to super-resolve)}"
SCALE="${SCALE:-2.67}"
ENCODER="${ENCODER:-RDN}"
GPU="${GPU:-0}"

case "$ENCODER" in
    RDN)      DEFAULT_PT="pre_trained_models/ArSSR_RDN.pkl" ;;
    ResCNN)   DEFAULT_PT="pre_trained_models/ArSSR_ResCNN.pkl" ;;
    SRResNet) DEFAULT_PT="pre_trained_models/ArSSR_SRResnet.pkl" ;;
    *) echo "unknown ENCODER $ENCODER"; exit 1 ;;
esac
MODEL="${MODEL:-$DEFAULT_PT}"

mkdir -p test/input test/output
shopt -s nullglob
copied=0
for f in "$LR_DIR"/*.nii.gz "$LR_DIR"/*.nii; do
    cp -u "$f" test/input/; copied=$((copied+1))
done
if [ "$copied" -eq 0 ]; then echo "no NIfTI in $LR_DIR"; exit 1; fi

echo "=== ArSSR infer: scale=$SCALE encoder=$ENCODER model=$MODEL  ($copied volumes) ==="
python test.py -input_path test/input -output_path test/output \
    -encoder "$ENCODER" -pre_trained_model "$MODEL" -scale "$SCALE" -is_gpu 1 -gpu "$GPU"
echo "=== outputs in test/output/ ==="
