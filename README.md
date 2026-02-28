# Flower CNN

## Overview

`flower_cnn` is a research codebase for experimenting with convolutional models on flower species datasets. It separates data processing, model training, and deployment to make experimentation and reproducibility easier.

## Setup

1. Install required packages: `pip install -r requirements.txt`.
2. Prepare datasets under `data/raw/`. Each split should follow the structure expected by `experiments/` scripts (typically `images/<label>/*.jpg`).
3. Run any preprocessing you need into `data/processed/` using scripts in `datasets/` or `training/`.

## Usage

- **Training:** Use [training/train_visible.py](training/train_visible.py) to start a run. Pass config files or CLI args describing the model, dataset path, and hyperparameters if you extend that entry point.
- **Evaluation:** Use `experiments/` to score checkpoints. For example, you can write a script under `experiments/ablation/` that loads a model from `models/` and runs inference on a validation split.
- **Logging:** Outputs, checkpoints, and metrics go into `training/logs/` and `experiments/logs/`. Keep each run in a timestamped subdirectory.

## Monitoring

[training/train_visible.py](training/train_visible.py), [train_ir.py](train_ir.py), and [train_multimodal.py](train_multimodal.py) all write TensorBoard summaries under `experiments/logs/`. Start `tensorboard --logdir experiments/logs --host 0.0.0.0 --port 6006` and open http://localhost:6006 to compare modality-specific runs.

## Experiments

1. Create a directory beneath `experiments/` (e.g., `experiments/ablation/`) for each study.
2. Version dataset splits in `data/annotations/` if you need consistent label mappings.
3. Save model artifacts under `models/` with a naming convention like `flowercnn-{backbone}-{date}.pth`.
4. Track experiment metadata (hyperparameters, dataset checksum) inside `experiments/logs/metadata.json` or similar files so later runs can be reproduced.

## Deployment / Serving

- `web/` contains a minimal backend/frontend for demoing models. Put serialization code in `web/backend/`, static assets in `web/frontend/`, and templates in `web/templates/`.
- The `model_web/` folder can host API wrappers or Flask/Django apps that integrate with the trained models.

## Scripts

1. `train_ir.py` mirrors `train_visible.py` but focuses on the infrared split (`data/processed/ir` and `data/annotations/*_ir.txt`). It uses `ReduceLROnPlateau`, TensorBoard (`experiments/logs/ir`), and `experiments/checkpoints/best_ir.pth` for the best weights.
2. `train_multimodal.py` learns from paired visible/IR annotations (CSV lines per example: `vis.jpg,ir.jpg,label`) and the fusion model in `models/multi_modal.py`; it logs to `experiments/logs/multimodal` and saves `best_multimodal.pth`.
3. `inference.py` runs a single image through a visible or infrared model or pairs through the multimodal fusion checkpoint. Use `--mode` to switch, provide `--checkpoint`, and pass `--image`/`--vis-image` + `--ir-image` accordingly.
4. `export_onnx.py` bundles `models/flower_model.py` into an ONNX asset (`models/flower_model.onnx`) that matches the trained checkpoint and exposes dynamic batch size support.

## Manual Demo

1. Drop a checkpoint at [model_web/model.pth](model_web/model.pth) so [web/backend/inference.py](web/backend/inference.py) can load it.
2. Launch the Flask server from the repo root with `FLASK_APP=web.backend.app flask run --host 0.0.0.0 --port 5000` so [web/backend/app.py](web/backend/app.py) serves both `/predict` and the UI.
3. Open http://localhost:5000 in a browser to render [web/frontend/index.html](web/frontend/index.html), upload a flower image, and inspect the prediction returned by `/predict`.

## Contributing

1. Keep changes scoped to directories aligned with each task (data, training, web, etc.).
2. Describe each experiment you run in `docs/thesis/` or `docs/diagrams/` with cross-reference to the code where the configuration lives.
3. Run `git status` before committing to ensure new files are accounted for.

Feel free to open issues or pull requests with new experiments or improvements.
