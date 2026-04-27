#!/bin/bash
# source "$(dirname "$0")/../env_setup.sh"

torchrun --nproc_per_node=2 --standalone \
    -m physics_jepa.train_jepa \
    configs/train_activematter_small.yaml \
    train.target_encoder_mode=ema \
    model.loss=mse \
    dataset.target_offsets=[1,2] \
    train.target_offset_weights=null \
    train.run_name=ijepa-ema-mse-multi-offset \
    "$@"
