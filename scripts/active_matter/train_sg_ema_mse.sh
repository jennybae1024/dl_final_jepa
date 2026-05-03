#!/bin/bash
set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
OUT_PATH="${OUT_PATH:-./checkpoints/ijepa_ema_mse}"
CONFIG="${CONFIG:-configs/experiments/active_matter_sg_ema_mse.yaml}"

torchrun --nproc_per_node="${NPROC_PER_NODE}" --standalone \
    -m physics_jepa.train_jepa \
    "${CONFIG}" \
    out_path="${OUT_PATH}" \
    "$@"
