# Project Layout

## Active code

- `main.py`, `eval.py`: model training and evaluation entry points.
- `datasets/`, `models/`, `utils/`, `wsi_core/`: reusable training and WSI components.
- `feature_extraction/`: patch feature extraction utilities.
- `create_*.py`, `patch_generation_*.py`: preprocessing and split generation.
- `*_heatmaps*.py`, `regenerate_heatmaps.py`: attention and visualization tools.

## Data and experiments

- `data/`: source WSIs, patches, coordinates, and extracted features.
- `dataset_csv/`: dataset manifests and labels.
- `text_prompt/`: prompt definitions used by the VLM experiments.
- `splits/`: train/validation/test split files.
- `results/`: training checkpoints and fold-level training metrics.
- `eval_results/`: evaluation reports and misclassification tables.
- `trained_models/`: downloaded or externally supplied model weights.

These directories can be large and should not be committed to source control unless a specific artifact is required.

## Documentation and reference material

- `README.md`: upstream project overview and baseline instructions.
- `docs/`: pipeline instructions, experiment notes, and project documentation.
- `docs/project-notes/`: historical integration notes and session summaries.
- `history_code/`: old implementations retained for reference only; it is not part of the active import path.

## Local-only files

Model caches and Python bytecode are disposable. They are intentionally excluded from the project structure and can be regenerated.
