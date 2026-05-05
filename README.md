# hierarchical_capsule_network_framework_for_supporting_expert_sensory_evaluation_of_herbal_medicines

This repository contains a hierarchical image dataset for herbal medicines and training scripts for hierarchical deep learning models, including ConditionCNN, BCNN, and H-CapsNet. The current training code supports a 3-level taxonomy built from folder names in the order `masterCategory / subCategory / articleType`. [Source](https://www.genspark.ai/api/files/s/Ym3LosEF)

## Repository overview

The dataset is organized as a hierarchical image tree. The training pipeline in `Train.py` scans three nested directory levels and converts them into labels for `masterCategory`, `subCategory`, and `articleType`. It then creates train/validation/test splits, one-hot labels, training history files, plots, saved weights, and hierarchical evaluation CSV outputs. [Source](https://www.genspark.ai/api/files/s/Ym3LosEF)

## Expected dataset structure

```text
hierarchy_image_dataset_aug/
└── herbal/
    ├── Aerial vegetative organs/
    │   ├── leaves/
    │   │   ├── CLASS_NAME/
    │   │   │   ├── image_[augmentation_type]_1.jpg
    │   │   │   └── ...
    │   └── stems/
    ├── Reproductive organs/
    ├── Subterranean vegetative organs/
    └── non-plant_organs/
```

For `scripts/Train.py`, the practical dataset argument is:

```bash
hierarchy_image_dataset_aug/herbal
```

This is because `Train.py` expects the image root to begin directly from the first taxonomy level. [Source](https://www.genspark.ai/api/files/s/Ym3LosEF)

## Files

- `scripts/Train.py`: main training entry point for multiple models, including `Condition` and `BCNN`. [Source](https://www.genspark.ai/api/files/s/Ym3LosEF)
- `scripts/ConditionCNN.py`: conditional hierarchical CNN model definition with master/sub/article outputs. [Source](https://www.genspark.ai/api/files/s/EEjlzb22)
- `scripts/BCNN.py`: branching CNN baseline with three hierarchical outputs. [Source](https://www.genspark.ai/api/files/s/9NQNhnpx)
- `scripts/H-CapsNet_only.py`: standalone H-CapsNet training script. It imports `src.MLmodel`, `src.datasets`, `src.metrics`, and `src.sysenv`. [Source](https://www.genspark.ai/api/files/s/37dgRBh5)

## Installation

Install the basic dependencies first:

```bash
pip install tensorflow numpy pandas matplotlib treelib
```

Optional tools:

- Graphviz is useful if you want model architecture plots from H-CapsNet.
- A CUDA-enabled TensorFlow environment is recommended for training. [Source](https://www.genspark.ai/api/files/s/37dgRBh5)

## How to run

### 1) ConditionCNN

`ConditionCNN` is launched through `Train.py` using the model name `Condition`:

```bash
python scripts/Train.py Condition 50 8 8 hierarchy_image_dataset_aug/herbal
```

Argument order:

```text
python scripts/Train.py <model_name> <epochs> <train_batch> <val_batch> <data_dir>
```

`Train.py` chooses the `ConditionCNN` branch when `model_type == 'Condition'`. [Source](https://www.genspark.ai/api/files/s/Ym3LosEF)

### 2) BCNN

`BCNN` is also launched through `Train.py`:

```bash
python scripts/Train.py BCNN 50 8 8 hierarchy_image_dataset_aug/herbal
```

`Train.py` chooses the `BCNN` branch when `model_type == 'BCNN'`. [Source](https://www.genspark.ai/api/files/s/Ym3LosEF)

### 3) H-CapsNet

`H-CapsNet_only.py` is a standalone script:

```bash
PYTHONPATH=. python scripts/H-CapsNet_only.py
```

Important notes:

1. The script imports modules from `src`, so the repository root must be visible in `PYTHONPATH`, or an equivalent package layout must be prepared. [Source](https://www.genspark.ai/api/files/s/37dgRBh5)
2. The script currently contains a fixed `DATA_ROOT` path:

```python
DATA_ROOT = "/mnt/mydisk/hyein/paper_exp_0929/hierarchy_image_dataset_aug"
```

If your dataset is stored elsewhere, edit that variable before running. [Source](https://www.genspark.ai/api/files/s/37dgRBh5)

## Output files

### `Train.py`

Running `Condition` or `BCNN` produces:

- training history CSV files in `history/`
- loss plots in `plots/`
- trained weights in `weights/`
- hierarchical metric CSV files for validation and test sets [Source](https://www.genspark.ai/api/files/s/Ym3LosEF)

### `H-CapsNet_only.py`

Running H-CapsNet produces:

- TensorBoard logs
- CSV logs
- best checkpoint weights
- final model weights
- evaluation metrics and confusion-matrix-related outputs [Source](https://www.genspark.ai/api/files/s/37dgRBh5)

## Notes

- `Train.py` uses image size `(128, 128)` for training. [Source](https://www.genspark.ai/api/files/s/Ym3LosEF)
- `ConditionCNN.py` and `BCNN.py` define default class counts as 4 master classes, 21 sub-classes, and 45 article classes, but `Train.py` can infer class counts dynamically from the dataset folders when a valid dataset path is provided. [Source](https://www.genspark.ai/api/files/s/EEjlzb22) [Source](https://www.genspark.ai/api/files/s/9NQNhnpx) [Source](https://www.genspark.ai/api/files/s/Ym3LosEF)
- Small batch sizes such as `8` or `4` are recommended first if GPU memory is limited. `H-CapsNet_only.py` already uses reduced batch settings to avoid OOM errors. [Source](https://www.genspark.ai/api/files/s/37dgRBh5)

## Example Git update

After adding or modifying files in this repository:

```bash
git add scripts/Train.py scripts/ConditionCNN.py scripts/BCNN.py scripts/H-CapsNet_only.py README.md
git commit -m "Add training scripts and update README"
git push origin main
```
