# Flower CNN

## Overview

`flower_cnn` is a research codebase for experimenting with convolutional models on flower species datasets. It separates data processing, model training, and deployment to make experimentation and reproducibility easier.

## Setup

1. Install required packages: `pip install -r requirements.txt`.
2. Prepare datasets under `data/raw/`. Each split should follow the structure expected by `experiments/` scripts (typically `images/<label>/*.jpg`).
3. Run any preprocessing you need into `data/processed/` using scripts in `datasets/` or `training/`.

## Usage

- **Training:** Use the scripts under `training/` to start a run. Look for entry points such as `training/train.py` (you may need to create one) and pass config files or CLI args describing the model, dataset path, and hyperparameters.
- **Evaluation:** Use `experiments/` to score checkpoints. For example, you can write a script under `experiments/ablation/` that loads a model from `models/` and runs inference on a validation split.
- **Logging:** Outputs, checkpoints, and metrics go into `training/logs/` and `experiments/logs/`. Keep each run in a timestamped subdirectory.

## Experiments

1. Create a directory beneath `experiments/` (e.g., `experiments/ablation/`) for each study.
2. Version dataset splits in `data/annotations/` if you need consistent label mappings.
3. Save model artifacts under `models/` with a naming convention like `flowercnn-{backbone}-{date}.pth`.
4. Track experiment metadata (hyperparameters, dataset checksum) inside `experiments/logs/metadata.json` or similar files so later runs can be reproduced.

## Deployment / Serving

- `web/` contains a minimal backend/frontend for demoing models. Put serialization code in `web/backend/`, static assets in `web/frontend/`, and templates in `web/templates/`.
- The `model_web/` folder can host API wrappers or Flask/Django apps that integrate with the trained models.

## Contributing

1. Keep changes scoped to directories aligned with each task (data, training, web, etc.).
2. Describe each experiment you run in `docs/thesis/` or `docs/diagrams/` with cross-reference to the code where the configuration lives.
3. Run `git status` before committing to ensure new files are accounted for.

Feel free to open issues or pull requests with new experiments or improvements.
