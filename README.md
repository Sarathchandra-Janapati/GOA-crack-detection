# GOA-UNet: Enhanced Crack Detection using Grasshopper Optimization and Connected Component Analysis

Implementation of the paper:

> **"Enhanced crack detection using GOA-UNet segmentation and Connected Component Analysis based aspect ratio estimation"**
> Abinaya S., Arikumar K. Selvaraj, Sahaya Beni Prathiba, Eshaan Rithesh Adyanthaya, Sarathchandra Janapati, Praveen Kumar Donta, Thippa Reddy Gadekallu
> *Measurement, Vol. 263, 2026* — [DOI: 10.1016/j.measurement.2025.120197](https://doi.org/10.1016/j.measurement.2025.120197)

---

## Overview

A hybrid deep learning framework for automated crack detection in civil infrastructure (bridges, roads, concrete surfaces). Combines four techniques:

| Component | Role |
|---|---|
| **Power Law Transformation** (γ = 2.5) | Contrast enhancement — makes faint cracks more visible |
| **LSGAN** | Data augmentation — generates 4 synthetic crack images per real sample |
| **GOA-optimised ResUNet** | Pixel-level crack segmentation with automated hyperparameter tuning |
| **Connected Component Analysis** | Post-processing — extracts width, height, aspect ratio of detected cracks |

---

## Results

| Metric | GOA-UNet | Baseline UNet |
|---|---|---|
| Accuracy | 98.5% | — |
| Precision | 92.14% | 68.52% |
| Recall | 77.39% | 75.41% |
| **F1-Score** | **84.12%** | 71.80% |
| **IoU** | **72.92%** | 56.00% |
| Dice Coefficient | 71.22% | 52.20% |

Cross-dataset generalisation (zero-shot): **95.27% performance retention** on CrackTree200 and CFD datasets.

---

## Architecture

```
Input (448×448×3)
    ↓  Power Law Transformation (γ=2.5)
    ↓  Stem Block
    ↓  Encoder: 4 Residual Blocks (stride=2) — filters [16→32→64→128→256]
    ↓  Bottleneck: 2 Conv Blocks
    ↓  Decoder: 4 Residual Blocks + Upsample + Skip Concatenation
    ↓  Conv2D (1×1, sigmoid)
Output: Binary Crack Mask (448×448×1)
    ↓
Connected Component Analysis → width, height, aspect ratio per crack
```

GOA optimises the learning rate (search space [0.001, 0.1]) using Dice coefficient as fitness metric.

---

## Repository Structure

```
├── notebookf7ccc02329.ipynb   # Main notebook — full pipeline
├── mask.py                    # Custom Keras Sequence data generator
├── CLAUDE.md                  # Project documentation for Claude Code
└── .gitignore
```

### Notebook Sections

| Section | Description |
|---|---|
| Dataset Loading | Kaggle dataset paths, shuffle, 80/20 split |
| Power Law Transformation | γ=2.5 visualisation across different values |
| CrackLSGAN | DCGAN-style LSGAN training on crack images, synthetic image generation |
| Data Generator | Albumentations augmentation + power law preprocessing |
| ResUNet Architecture | Residual encoder-decoder, 448×448 input |
| Loss Functions | Dice loss, IoU metric |
| GOA Optimisation | `target_function` + GOA (N=50, L=50, cmax=1.0, cmin=0.4) |
| Training | 100 epochs with LR scheduling, ModelCheckpoint |
| Evaluation | Accuracy, Precision, Recall, F1, IoU, Dice, AUC, Sensitivity, Specificity |
| Connected Component Analysis | Per-crack width, height, aspect ratio with visualisation |
| Model Comparison | Table 6 from paper — GOA-UNet vs FCN/SegNet/UNet/TransUNet etc. |

---

## Setup

### Dataset

This notebook uses the **Conglomerate Concrete Crack Detection** dataset on Kaggle.

```
kaggle datasets download -d <dataset-slug>
```

Update the paths in the dataset loading cell if needed:

```python
train_image_dir = r'/kaggle/input/concrete-crack-dataset/...'
```

### Dependencies

```bash
pip install tensorflow opencv-python albumentations scikit-image scikit-learn pyMetaheuristic
```

### Running on Kaggle / Colab

1. Upload `notebookf7ccc02329.ipynb` to Kaggle or Colab
2. Attach the crack detection dataset
3. Enable GPU accelerator
4. Run all cells in order

> **Note:** GOA with N=50 grasshoppers × 50 iterations takes ~12 hours on an NVIDIA RTX 3080. Reduce `grasshoppers` and `iterations` in the GOA parameters cell for a faster run.

---

## Key Hyperparameters (Paper Table 1)

| Parameter | Value |
|---|---|
| Image size | 448 × 448 |
| Gamma (γ) | 2.5 |
| GOA population (N) | 50 |
| GOA iterations (L) | 50 |
| cmax / cmin | 1.0 / 0.4 |
| LR search range | [0.001, 0.1] |
| Batch size | 8 |
| Epochs | 100 |
| Optimizer | Adam (β1=0.9, β2=0.999, amsgrad=True) |
| Loss | Dice loss |

---

## Citation

```bibtex
@article{abinaya2026goa,
  title   = {Enhanced crack detection using GOA-UNet segmentation and Connected Component Analysis based aspect ratio estimation},
  author  = {Abinaya S. and Selvaraj, Arikumar K. and Prathiba, Sahaya Beni and Adyanthaya, Eshaan Rithesh and Janapati, Sarathchandra and Donta, Praveen Kumar and Gadekallu, Thippa Reddy},
  journal = {Measurement},
  volume  = {263},
  pages   = {120197},
  year    = {2026},
  doi     = {10.1016/j.measurement.2025.120197}
}
```
