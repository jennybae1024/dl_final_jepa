#!/bin/bash
# source "$(dirname "$0")/../env_setup.sh"

torchrun --nproc_per_node=2 --standalone \
    -m physics_jepa.train_jepa \
    configs/train_activematter_small.yaml \
    train.target_encoder_mode=shared \
    model.loss=vicreg \
    dataset.target_offsets=[1,2] \
    train.target_offset_weights=[0.75,0.25] \
    train.horizon_specific_predictors=true \
    train.run_name=jepa-vicreg-multi-offset-weighted-multi-predictor \
    "$@"
