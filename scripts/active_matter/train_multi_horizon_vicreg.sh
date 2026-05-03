#!/bin/bash
set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
OUT_PATH="${OUT_PATH:-./checkpoints/jepa-vicreg-multi-offset-weighted-multi-predictor}"
CONFIG="${CONFIG:-configs/experiments/active_matter_multi_horizon_vicreg.yaml}"

torchrun --nproc_per_node="${NPROC_PER_NODE}" --standalone \
    -m physics_jepa.train_jepa \
    "${CONFIG}" \
    out_path="${OUT_PATH}" \
    "$@"
