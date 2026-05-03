# Environment Setup

Use Python 3.10+ with a CUDA-compatible PyTorch installation.

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Before training or evaluation, set the Hugging Face and The Well dataset cache directories:

```bash
export HF_HOME=/scratch/$USER/huggingface
export HF_DATASETS_CACHE=/scratch/$USER/huggingface
export THE_WELL_DATA_DIR=/scratch/$USER/huggingface
```

These variables keep downloaded Active Matter data and Hugging Face cache files on scratch storage rather than in the home directory.

Run all training and evaluation commands from the repository root so that `python -m physics_jepa...` entry points resolve correctly.

For multi-GPU training, the provided scripts use `torchrun` and default to:

```bash
NPROC_PER_NODE=2
```

Override this for a different GPU count:

```bash
NPROC_PER_NODE=1 scripts/active_matter/train_masked_vicreg.sh
```
