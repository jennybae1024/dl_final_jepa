# Predictive Representation Learning for Active Matter Systems

This repository contains the code and artifacts for our project, **A Systematic Study of Design Choices in Predictive Representation Learning for Active Matter Systems**.

We study predictive representation learning for active matter systems using a joint-embedding framework. The project compares several design choices, including predictive objectives, EMA versus shared target encoders, multi-horizon prediction, spatiotemporal masking, and representation evaluation with linear and kNN regression probes.

Reproducible experiment configurations and runnable training/evaluation scripts are detailed in [`EXPERIMENTS.md`](EXPERIMENTS.md).

Model weights from our training and experiments can be found in this link: [Model Weights](https://drive.google.com/drive/folders/1QEaZ47_yjfuZsSUu6LmVWmaXDElv-zAz?usp=sharing)

## Repository Structure

```text
.
├── configs/                  # Dataset, model, finetuning, and experiment configs
│   └── experiments/          # Clean configs for submitted experiments
├── physics_jepa/             # Main source code
├── scripts/                  # Training and evaluation scripts
│   └── active_matter/        # Active Matter experiment entry points
├── EXPERIMENTS.md            # Detailed experiment configs and commands
├── ENV.md                    # Environment and cache setup instructions
├── requirements.txt          # Essential Python dependencies
└── README.md                 # This file
```

Saved backbone weights, logs, and result tables should be submitted with the artifact bundle, or placed under paths such as `checkpoints/`, `logs/`, and `results/` if included locally.

## Environment Setup

Install the essential Python dependencies:

```bash
pip install -r requirements.txt
```

Before training or evaluation on the cluster, set the dataset/cache paths:

```bash
export HF_HOME=/scratch/$USER/huggingface
export HF_DATASETS_CACHE=/scratch/$USER/huggingface
export THE_WELL_DATA_DIR=/scratch/$USER/huggingface
```

See [`ENV.md`](ENV.md) for the full environment setup notes.

## Training

Run the submitted Active Matter representation learning experiments with:

```bash
scripts/active_matter/train_masked_vicreg.sh
scripts/active_matter/train_multi_horizon_vicreg.sh
scripts/active_matter/train_sg_ema_mse.sh
scripts/active_matter/run_train_jepa.sh
```

Each script accepts `OUT_PATH` and `NPROC_PER_NODE` environment variables. For example:

```bash
OUT_PATH=./checkpoints/shared_masked_vicreg_r005_2 \
NPROC_PER_NODE=2 \
scripts/active_matter/train_masked_vicreg.sh
```

## Evaluation

Run frozen representation evaluation from a saved checkpoint or checkpoint directory:

```bash
scripts/active_matter/eval_linear_probe.sh [checkpoint_file_or_dir]
scripts/active_matter/eval_knn_regression.sh [checkpoint_file_or_dir]
```

The linear probe config evaluates on the validation split. The kNN regression config evaluates on the test split, matching the submitted experiment commands.

## Experiment Details

The exact configs and scripts for each submitted experiment are listed in [`EXPERIMENTS.md`](EXPERIMENTS.md), including:

- Spatiotemporal masking with VICReg
- Multi-horizon VICReg
- Stop-gradient EMA with MSE
- Channel-wise Encoding
- Linear probe regression
- kNN regression

Additional Hydra overrides can be appended to the scripts when needed.
