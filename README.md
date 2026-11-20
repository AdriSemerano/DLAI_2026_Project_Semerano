# DLAI_2026_Project_Semerano

This repository investigates from-scratch convolutional architectures for avian Sound Event Detection (**BirdCLEF+ 2026**) without pre-trained backbones. A controlled ablation study quantifies the individual contributions of architectural design, training components, and ensembling. Additionally, the impact of spatial data leakage on internal validation and out-of-sample generalization is isolated.

---

## Repository Structure

```text
DLAI_2026_Project_Semerano/
├── birdclef-training.ipynb             # EDA, dataset splits, model definitions, and parameterized 2-phase training
├── birdclef-submission.ipynb           # Single-model CPU inference with 3-window TTA and Gaussian smoothing
├── birdclef-ensemble-submission.ipynb  # Weighted multi-model ensemble inference with TTA and Gaussian smoothing
├── Models_Weights/                     # Saved Phase B checkpoints (.pth) for reproducing report results
├── DLAI_Project_Semerano.pdf           # Final 2-page project report + AI statement, references, and appendix
└── README.md                           # Project documentation and reproducibility guide
```

---

## Environment & Dataset Setup

All notebooks were developed and executed in a **Kaggle Notebook** environment using the official **BirdCLEF+ 2026** competition dataset mounted at:

```text
/kaggle/input/competitions/birdclef-2026/
```

### Hardware & Dependencies
* **Training (`birdclef-training.ipynb`):** GPU recommended (e.g., NVIDIA T4/P100). If no GPU is detected, the notebook automatically falls back to a smoke test (`epochs_a=2`, `epochs_b=1`).
* **Inference (`birdclef-submission.ipynb`, `birdclef-ensemble-submission.ipynb`):** Runs on CPU (`device = torch.device('cpu')`) to comply with Kaggle CPU-only submission constraints.
* **Core Libraries:** `torch`, `torchaudio`, `librosa`, `scikit-learn`, `scipy`, `pandas`, `numpy`, `matplotlib`, `seaborn`.

---

## 1. Reproducing Training & Ablation Studies (`birdclef-training.ipynb`)

All experiments in **Table 1** and **Appendix Table 2** are executed through a single parameterized entry point, `run_experiment(...)`, which keeps preprocessing, optimization (`AdamW` + `OneCycleLR`), and training schedules fixed (`epochs_a=30`, `epochs_b=15`, `batch_size=32`) while toggling individual design choices via boolean flags:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model_type` | `str` | `'Asymmetric'` | Backbone architecture: `'Standard'` (4.92M), `'Asymmetric'` (2.83M), or `'ResNet'` (5.09M) |
| `use_pcen` | `bool` | `True` | `True`: PCEN frontend (`s=0.025, alpha=0.98, delta=2.0, r=0.5`); `False`: Log-Mel + z-score |
| `use_mixup_lss` | `bool` | `True` | Enables Phase A Local Soundscape (LSS) Mixup with label capping at `0.5` |
| `use_specaugment` | `bool` | `True` | Enables SpecAugment (`freq_mask_param=36`, `time_mask_param=64`) during training |
| `use_focal` | `bool` | `True` | `True`: Multilabel Focal Loss (`gamma=2.0, alpha=1.0`); `False`: `BCEWithLogitsLoss` |
| `use_se` | `bool` | `True` | Adds Squeeze-and-Excitation blocks (`+0.04M` parameters) to convolutional stages |
| `use_group_split` | `bool` | `False` | `True`: Purified site-grouped split (S22 vs. other 8 sites); `False`: Naive 80/20 random split |

### Automatic Checkpoint Naming
Each run automatically saves minimum-validation-loss checkpoints using the naming convention:

```text
best_phaseB_[Leak_]<model_type>[_PCEN][_SE][_Mixup][_Spec][_Focal].pth
```

### Exact Flag Mapping for All 21 Runs (Table 2)

To reproduce any row from **Table 1** or **Appendix Table 2**, invoke `run_experiment(...)` with the corresponding flags below (`use_group_split = not Leak`):

| Backbone | Config Name | `use_pcen` | `use_mixup_lss` | `use_specaugment` | `use_focal` | `use_se` | `use_group_split` | Val ROC-AUC | Private LB |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard** | `Leak_Standard_PCEN_Mixup` | `True` | `True` | `False` | `False` | `False` | `False` | 0.9996 | 0.80395 |
| **Standard** | `Standard_PCEN_SE_Mixup` | `True` | `True` | `False` | `False` | `True` | `True` | 0.7451 | 0.77419 |
| **Standard** | `Standard_PCEN_Mixup_Focal` | `True` | `True` | `False` | `True` | `False` | `True` | 0.7444 | 0.75779 |
| **Standard** | `Standard_PCEN_Mixup_Spec` | `True` | `True` | `True` | `False` | `False` | `True` | 0.7641 | 0.76618 |
| **Standard** | `Standard_PCEN_Mixup` | `True` | `True` | `False` | `False` | `False` | `True` | 0.7229 | 0.77425 |
| **Standard** | `Standard_PCEN` | `True` | `False` | `False` | `False` | `False` | `True` | 0.7359 | 0.74054 |
| **Standard** | `Standard` (Log-Mel baseline) | `False` | `False` | `False` | `False` | `False` | `True` | 0.6190 | 0.68858 |
| **Asymmetric** | `Leak_Asymmetric_PCEN_Mixup` | `True` | `True` | `False` | `False` | `False` | `False` | 0.9994 | 0.80242 |
| **Asymmetric** | `Asymmetric_PCEN_SE_Mixup` | `True` | `True` | `False` | `False` | `True` | `True` | 0.7852 | 0.75919 |
| **Asymmetric** | `Asymmetric_PCEN_Mixup_Focal` | `True` | `True` | `False` | `True` | `False` | `True` | 0.7491 | 0.75426 |
| **Asymmetric** | `Asymmetric_PCEN_Mixup_Spec` | `True` | `True` | `True` | `False` | `False` | `True` | 0.7498 | 0.76723 |
| **Asymmetric** | `Asymmetric_PCEN_Mixup` | `True` | `True` | `False` | `False` | `False` | `True` | 0.8107 | 0.77565 |
| **Asymmetric** | `Asymmetric_PCEN` | `True` | `False` | `False` | `False` | `False` | `True` | 0.7421 | 0.76164 |
| **Asymmetric** | `Asymmetric` (Log-Mel baseline) | `False` | `False` | `False` | `False` | `False` | `True` | 0.6399 | 0.71310 |
| **ResNet** | `Leak_ResNet_PCEN_Mixup_Spec_Focal` | `True` | `True` | `True` | `True` | `False` | `False` | 0.9996 | 0.77718 |
| **ResNet** | `ResNet_PCEN_SE_Mixup_Spec_Focal` | `True` | `True` | `True` | `True` | `True` | `True` | 0.7279 | 0.69963 |
| **ResNet** | `ResNet_PCEN_Mixup_Spec_Focal` | `True` | `True` | `True` | `True` | `False` | `True` | 0.7606 | 0.74735 |
| **ResNet** | `ResNet_PCEN_Mixup_Spec` | `True` | `True` | `True` | `False` | `False` | `True` | 0.7768 | 0.73361 |
| **ResNet** | `ResNet_PCEN_Mixup` | `True` | `True` | `False` | `False` | `False` | `True` | 0.7170 | 0.71172 |
| **ResNet** | `ResNet_PCEN` | `True` | `False` | `False` | `False` | `False` | `True` | 0.6478 | 0.67958 |
| **ResNet** | `ResNet` (Log-Mel baseline) | `False` | `False` | `False` | `False` | `False` | `True` | 0.5416 | 0.50306 |

---

## 2. Reproducing Single-Model Inference (`birdclef-submission.ipynb`)

To evaluate any saved checkpoint from `Models_Weights/` and generate `submission.csv` (using 3-window Test-Time Augmentation at `0.0 s`, `-0.5 s`, `+0.5 s` and 1D Gaussian temporal smoothing with `sigma=1.0`):

1. Open `birdclef-submission.ipynb`.
2. Set the checkpoint path and matching architectural flags at the top of the notebook:

```python
model_weights_path = 'Models_Weights/best_phaseB_Asymmetric_PCEN_Mixup.pth'
model_type = 'Asymmetric'  # 'Standard', 'Asymmetric', or 'ResNet'
use_pcen = True
use_se = False
```

3. Run all cells to generate `submission.csv`.

---

## 3. Reproducing Ensemble Results (`birdclef-ensemble-submission.ipynb`)

To reproduce the multi-model ensembles reported in **Table 1**, open `birdclef-ensemble-submission.ipynb` and configure `ensemble_configs` as follows:

### A. `Ensemble_Leak (Standard + Asymmetric)` — Private LB: `0.8107`

```python
ensemble_configs = [
    {'weights_path': 'Models_Weights/best_phaseB_Leak_Asymmetric_PCEN_Mixup.pth', 'model_type': 'Asymmetric', 'use_pcen': True, 'use_se': False, 'weight': 1.0},
    {'weights_path': 'Models_Weights/best_phaseB_Leak_Standard_PCEN_Mixup.pth',   'model_type': 'Standard',   'use_pcen': True, 'use_se': False, 'weight': 1.0},
]
```

### B. `Ensemble_Clean (Standard + Asymmetric)` — Private LB: `0.7813`

```python
ensemble_configs = [
    {'weights_path': 'Models_Weights/best_phaseB_Asymmetric_PCEN_Mixup.pth', 'model_type': 'Asymmetric', 'use_pcen': True, 'use_se': False, 'weight': 1.0},
    {'weights_path': 'Models_Weights/best_phaseB_Standard_PCEN_Mixup.pth',   'model_type': 'Standard',   'use_pcen': True, 'use_se': False, 'weight': 1.0},
]
```

### C. `Ensemble_Clean (Std + Asym + 0.4 ResNet)` — Private LB: `0.7668`

```python
ensemble_configs = [
    {'weights_path': 'Models_Weights/best_phaseB_Asymmetric_PCEN_Mixup.pth',        'model_type': 'Asymmetric', 'use_pcen': True, 'use_se': False, 'weight': 1.0},
    {'weights_path': 'Models_Weights/best_phaseB_Standard_PCEN_Mixup.pth',          'model_type': 'Standard',   'use_pcen': True, 'use_se': False, 'weight': 1.0},
    {'weights_path': 'Models_Weights/best_phaseB_ResNet_PCEN_Mixup_Spec_Focal.pth', 'model_type': 'ResNet',     'use_pcen': True, 'use_se': False, 'weight': 0.4},
]
```
