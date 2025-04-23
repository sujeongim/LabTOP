# Folder Structure

- `src/`: Source code
  - `core/models/`: Model definitions, training, and inference logic
  - `core/data/`: Dataset loading and preprocessing
  - `core/utils/`: Utility functions (logging, metrics, helpers)
  - `scripts/`: Entry-point scripts (main.py, evaluate.py, preprocess.py)
- `data/`: Datasets
  - `raw/`: Original datasets (e.g., ALFRED)
  - `processed/`: Processed datasets
- `configs/`: Configuration files (e.g., config.yaml)
- `experiments/`: Checkpoints and logs
  - `checkpoints/`: Saved model weights
  - `logs/`: Training and evaluation logs
- `scripts/`: Shell scripts for running experiments
- `tests/`: Unit tests
- `docs/`: Documentation