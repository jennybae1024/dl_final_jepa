#!/bin/bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <checkpoint_file_or_dir> [hydra_overrides...]"
    exit 2
fi

CHECKPOINT="$1"
shift

OUT_PATH="${OUT_PATH:-./checkpoints/active_matter_knn_eval}"
CONFIG="${CONFIG:-configs/experiments/active_matter_knn_eval.yaml}"

python -m physics_jepa.finetune \
    "${CONFIG}" \
    --trained_model_path "${CHECKPOINT}" \
    out_path="${OUT_PATH}" \
    "$@"
