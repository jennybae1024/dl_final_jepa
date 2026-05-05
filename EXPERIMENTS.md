# Active Matter JEPA Experiments

This file documents the reproducible training and evaluation commands used for the Active Matter representation learning experiments.

## Representation Learning

All representation learning runs use the Active Matter dataset, the convolutional JEPA backbone, and save checkpoints under the configured `out_path`.
The scripts below default to two processes per node. Override this with `NPROC_PER_NODE=1` or another value if needed.

### Spatiotemporal Masking with VICReg

Config:

```text
configs/experiments/active_matter_masked_vicreg.yaml
```

Run:

```bash
scripts/active_matter/train_masked_vicreg.sh
```

Main settings:

```text
model.loss: vicreg
train.target_encoder_mode: shared
train.lr: 4e-5
train.batch_size: 4
train.target_global_batch_size: 8
train.num_epochs: 20
train.noise_std: 0.0
train.context_masking.enabled: true
train.context_masking.mode: spatiotemporal_block
train.context_masking.mask_ratio: 0.05
train.context_masking.block_size: [2, 32, 32]
train.context_masking.mask_value: channel_mean
```

### Multi-Horizon VICReg

Config:

```text
configs/experiments/active_matter_multi_horizon_vicreg.yaml
```

Run:

```bash
scripts/active_matter/train_multi_horizon_vicreg.sh
```

Main settings:

```text
model.loss: vicreg
dataset.target_offsets: [1, 2]
train.target_offset_weights: [0.75, 0.25]
train.horizon_specific_predictors: true
train.target_encoder_mode: ema
train.lr: 5e-4
train.batch_size: 4
train.num_epochs: 20
```

### Stop-Gradient EMA with MSE

Config:

```text
configs/experiments/active_matter_sg_ema_mse.yaml
```

Run:

```bash
scripts/active_matter/train_sg_ema_mse.sh
```

Main settings:

```text
model.loss: mse
train.target_encoder_mode: ema
train.lr: 5e-4
train.batch_size: 8
train.num_epochs: 20
```
### Channel-wise Encoding

Config:

```text
configs/train_activematter_small.yaml
```

```bash
scripts/active_matter/run_train_jepa.sh
```

Main settings:

```text
model.dims: [22, 32, 64, 128, 128]
model.channel_wise_encoding: false
```



## Evaluation

The evaluation scripts take either a checkpoint file or a checkpoint directory as the first argument.

### Linear Probe Regression

Config:

```text
configs/experiments/active_matter_linear_eval.yaml
```

Run:

```bash
scripts/active_matter/eval_linear_probe.sh [checkpoint_file_or_dir]
```

Main settings:

```text
ft.head_type: linear
ft.use_attentive_pooling: false
ft.task: regression
ft.batch_size: 8
ft.noise_std: 0.0
ft.eval_split: val
```

### kNN Regression

Config:

```text
configs/experiments/active_matter_knn_eval.yaml
```

Run:

```bash
scripts/active_matter/eval_knn_regression.sh [checkpoint_file_or_dir]
```

Main settings:

```text
ft.head_type: knn
ft.use_attentive_pooling: false
ft.task: regression
ft.batch_size: 8
ft.noise_std: 0.0
ft.eval_split: test
```

## Overriding Output Paths

Each script accepts an `OUT_PATH` environment variable:

```bash
OUT_PATH=./checkpoints/my_run scripts/active_matter/train_masked_vicreg.sh
```

Evaluation example:

```bash
OUT_PATH=./checkpoints/my_linear_eval scripts/active_matter/eval_linear_probe.sh ./checkpoints/my_run
```

Additional Hydra overrides can be appended after the script arguments:

```bash
scripts/active_matter/train_masked_vicreg.sh train.context_masking.mask_ratio=0.10
```
