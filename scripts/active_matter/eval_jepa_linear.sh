#!/bin/bash
source "$(dirname "$0")/../env_setup.sh"

# Pass the trained encoder checkpoint file or checkpoint directory as $1.
python -m physics_jepa.finetune \
    configs/train_activematter_small.yaml \
    --trained_model_path "$1" \
    ft.use_attentive_pooling=false \
    ft.head_type=linear \
    ft.run_name=activematter-jepa-linear-probe \
    "${@:2}"
