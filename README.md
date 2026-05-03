# Active Matter JEPA Experiments

This repository is used for Active Matter JEPA pretraining and representation evaluation.

Reproducible experiment configurations and runnable training/evaluation scripts are detailed in [`EXPERIMENTS.md`](EXPERIMENTS.md).

## Installation

Requirements:

- Python 3.10+
- PyTorch 2.0+ with CUDA
- The Well dataset package and Active Matter data

Install the project dependencies:

```bash
pip install torch torchvision einops omegaconf wandb tqdm h5py psutil scikit-learn timm the-well
pip install -e .
```

## Active Matter JEPA Pretraining

Use:

```bash
scripts/active_matter/run_train_jepa.sh
```

The script runs:

```bash
torchrun --nproc_per_node=2 --standalone \
  -m physics_jepa.train_jepa \
  configs/train_activematter_small.yaml
```

Config values can be changed from the command line with Hydra-style overrides:

```bash
scripts/active_matter/run_train_jepa.sh \
  train.num_epochs=6 \
  train.lr=5e-4 \
  train.save_every=1 \
  out_path=./checkpoints/jepa_six
```

## Experiment Design Switches

Use these overrides to turn each experiment component on or off.

| Component | Off / baseline | On / experimental |
|---|---|---|
| EMA target encoder | `+train.target_encoder_mode=shared` | `+train.target_encoder_mode=ema` |
| VicReg loss | `+model.loss=mse` or `+model.loss=cosine` | `+model.loss=vicreg` |
| Multi-horizon targets | `dataset.target_offsets=[1]` | `dataset.target_offsets=[1,2] +train.target_offset_weights=[0.75,0.25]` |
| Horizon-specific predictors | `+train.horizon_specific_predictors=false` | `+train.horizon_specific_predictors=true` |
| Context masking | omit `train.context_masking` overrides | use the masking overrides below |
| Channel-wise encoding | `+model.channel_wise_encoding=false` | `+model.channel_wise_encoding=true model.dims=[22,32,64,128,128]` |

### EMA Target Encoder + VicReg

```bash
scripts/active_matter/run_train_jepa.sh \
  train.target_encoder_mode=ema \
  model.loss=vicreg \
  train.num_epochs=6 \
  train.lr=5e-4 \
  train.save_every=1 \
  out_path=./checkpoints/jepa_ema_vicreg
```

### Multi-Horizon Prediction

Add these overrides:

```bash
dataset.target_offsets=[1,2] \
train.target_offset_weights=[0.75,0.25]
```

To use separate predictors for each horizon, also add:

```bash
train.horizon_specific_predictors=true
```

### Context Masking

Add these overrides:

```bash
+train.context_masking.enabled=true \
+train.context_masking.mode=spatiotemporal_block \
+train.context_masking.mask_ratio=0.10 \
+train.context_masking.block_size=[2,32,32] \
+train.context_masking.mask_value=channel_mean
```

### Channel-Wise Encoding

Channel-wise encoding requires the first model dimension to be divisible by the number of input channels. Active Matter uses 11 input channels, so the default `model.dims=[16,32,64,128,128]` does not work for channel-wise encoding because 16 is not divisible by 11.

Example channel-wise encoder dimensions:

```bash
model.channel_wise_encoding=true \
model.dims=[22,32,64,128,128]
```

With `model.dims=[22,32,64,128,128]`, the first encoder layer allocates 2 channels of embedding capacity per input channel because `22 / 11 = 2`. Channel-wise encoding changes the first encoder stem to grouped per-channel convolution and adds learned channel tokens. Keep it off when comparing against the original dense encoder.

To reproduce the channel-wise experiment design setup, use both the channel-wise flag and the compatible dimensions:

```bash
scripts/active_matter/run_train_jepa.sh \
  model.channel_wise_encoding=true \
  model.dims=[22,32,64,128,128] \
  train.num_epochs=6 \
  train.lr=5e-4 \
  train.save_every=1 \
  out_path=./checkpoints/jepa_channel_six
```

To reproduce probing for channel-wise, add this argument to the scripts for both linear and knn

## Full Example

EMA target encoder, VicReg loss, multi-horizon targets, context masking, and channel-wise encoding:

```bash
scripts/active_matter/run_train_jepa.sh \
  train.target_encoder_mode=ema \
  model.loss=vicreg \
  dataset.target_offsets=[1,2] \
  train.target_offset_weights=[0.75,0.25] \
  +train.context_masking.enabled=true \
  +train.context_masking.mode=spatiotemporal_block \
  +train.context_masking.mask_ratio=0.10 \
  +train.context_masking.block_size=[2,32,32] \
  +train.context_masking.mask_value=channel_mean \
  model.channel_wise_encoding=true \
  model.dims=[22,32,64,128,128] \
  train.num_epochs=6 \
  train.lr=5e-4 \
  train.save_every=1 \
  train.run_name=jepa-ema-vicreg-mh-mask-channelwise \
  out_path=./checkpoints/jepa-ema-vicreg-mh-mask-channelwise
```

## Representation Evaluation

Use a saved JEPA encoder checkpoint for frozen representation evaluation.

Linear probe:

```bash
scripts/active_matter/run_finetune_jepa.sh /path/to/checkpoint \
  ft.head_type=linear \
  ft.use_attentive_pooling=false \
  ft.task=regression
```

kNN regression:

```bash
scripts/active_matter/run_finetune_jepa.sh /path/to/checkpoint \
  ft=knn \
  ft.task=regression
```

To evaluate on the test split, add:

```bash
ft.eval_split=test
```

Metrics include overall MSE and per-target MSE for Active Matter labels such as `alpha` and `zeta`.
