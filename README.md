# MAPLE: Self-Supervised Learning-Enhanced Nonlinear Dimensionality Reduction for Visual Analysis
DOI: 10.1109/TVCG.2026.3694000

## Info
- `main.py` — example script to run the algorithm
- **`maple/`** — algorithm and config  
  - `models/mmcr_model.py` 
  - `..config/default_config.py` — Datasets, training and MMCR/BT hyperparameters  
  - `training/training.py` — Training loop
  - `training/ce_umap.py` — UMAP embedding
  - `training/loss_functions.py` — Loss functions
- **`utils/`** — Data I/O, MMCR view-building, UMAP helpers  
- **`eval_metrics/`** — Evaluation (Jaccard, KNN classification accuracy, neighborhood hit, local label trustworthiness, silhouette, 2AFC, etc.)  
- **`visualization/`** — Getting all the plots   

## Requirements
- Python 3  
- PyTorch  
- UMAP
- Other dependencies as used in the scripts

## Under construction... Contact
zeyang.huang [at] liu.se
