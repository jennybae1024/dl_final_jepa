#!/bin/bash
# source "$(dirname "$0")/../env_setup.sh"

torchrun --nproc_per_node=1 --standalone \
    -m physics_jepa.train_jepa \
    configs/train_activematter_small.yaml \
    train.target_encoder_mode=ema \
    train.lr=5e-4 \
    train.noise_std=0.0 \
    model.loss=vicreg \
    +train.context_masking.enabled=true \
    +train.context_masking.mode=spatiotemporal_block \
    +train.context_masking.mask_ratio=0.10 \
    +train.context_masking.block_size=[2,32,32] \
    +train.context_masking.mask_value=channel_mean \
    train.run_name=jepa-ema-vicreg-context-st-block-mask-r010 \
    "$@"
