#!/bin/bash
# source "$(dirname "$0")/../env_setup.sh"

torchrun --nproc_per_node=2 --standalone \
    -m physics_jepa.train_jepa \
    configs/train_activematter_small.yaml \
    dataset.target_offsets=[1,2] \
    train.horizon_specific_predictors=true \
    model.loss=cosine \
    train.run_name=jepa-horizon-heads-cosine \
    "$@"
