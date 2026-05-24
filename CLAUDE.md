# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Concrete crack detection and segmentation system using deep learning. Two complementary approaches:
1. **ResUNet semantic segmentation** — primary implementation (`Deep_crack.ipynb`)
2. **Mask R-CNN instance segmentation** — reference/alternative (`Capstone Home/project.ipynb`)

Designed for Google Colab execution with GPU acceleration (Tesla K80/T4).

## Current State

- ResUNet model fully trained: Accuracy 98.59%, F1 84.12%, IoU 72.60%, AUC-ROC 91.15%
- Mask R-CNN implementation present but requires dataset format adjustments for full integration
- No automated test suite — validation is notebook-based via held-out test set (237 images)

## Key Files

| File | Purpose |
|------|---------|
| `Deep_crack.ipynb` | Main ResUNet training/evaluation notebook |
| `mask.py` | Custom `keras.utils.Sequence` data generator |
| `Capstone Home/project.ipynb` | Mask R-CNN training notebook |
| `Capstone Home/mrcnn/` | Custom Mask R-CNN module |
| `Capstone Home/Mask_RCNN-master/` | Official Mask R-CNN v2 reference |

## Commands & Setup Notes

**Environment (Colab):**
```python
!pip install tensorflow opencv-python albumentations scikit-image scikit-learn imgaug
```

**Compile and train (ResUNet):**
```python
model.compile(optimizer=optimiser, loss=dice_coef_loss,
              metrics=['accuracy', IOU, dice_coef])

history = model.fit(tg, steps_per_epoch=train_steps,
                    epochs=100, validation_data=vg,
                    validation_steps=valid_steps, callbacks=callbacks)
```

**Evaluate on test set:**
```python
y_pred = model.predict(x_test)
# Threshold at 0.5 for binary mask, then compute metrics
from sklearn.metrics import classification_report, roc_auc_score
```

## Architecture

### ResUNet (Deep_crack.ipynb)

```
Input (256×256×3)
→ Stem Block [Conv2D + BN + ReLU + residual skip]
→ Encoder: 4 residual blocks, stride=2 downsampling, filters [16→32→64→128→256]
→ Bottleneck: 2 conv blocks
→ Decoder: 4 residual blocks, bilinear upsampling + skip concatenation
→ Output Conv2D (1 channel, sigmoid) → binary crack mask
```

**Loss:** Dice loss (`1 - dice_coef`). Optimizer: Adam (lr=0.0035, β₁=0.9, β₂=0.999).

**Callbacks:** `ReduceLROnPlateau` + exponential `LearningRateScheduler` + `ModelCheckpoint`.

### Data Generator (`mask.py`)

Subclasses `keras.utils.Sequence`. Per-batch pipeline:
1. Load image + mask pairs, resize to 256×256
2. Apply power law (gamma) correction for illumination normalization
3. Albumentations augmentation: flips (70%), rotations, elastic/grid/optical distortion, CLAHE, contrast/brightness/RGB shifts
4. Normalize to [0, 1]

Dataset split: train=270, val=30, test=237.

### Mask R-CNN (`Capstone Home/`)

FPN + ResNet101 backbone, initialized from COCO weights. Annotations in VIA JSON polyline format. Single class: `crack`.

## Key Decisions & Conventions

- **256×256 image dimensions** used throughout; changing requires updating all generators and model input shape
- **Power law transformation** applied before augmentation — accounts for illumination variation in concrete surface images
- **Dice loss** preferred over BCE for crack segmentation due to class imbalance (cracks are sparse)
- **Batch size 10** balances memory and gradient stability on Colab GPUs
- **GOA (Grasshopper Optimization Algorithm)** used for hyperparameter search in some experiments
- Mask R-CNN uses **polyline** annotations (not polygon), requiring custom `draw_segment` in the dataset class

## Known Issues & Fixes

- **TF1 vs TF2 Mask R-CNN compatibility:** `Capstone Home/Mask-R-CNN-using-Tensorflow2-main/` is the TF2-compatible fork; do not mix with `Mask_RCNN-master/` (TF1)
- **Colab paths:** Notebooks hardcode Google Drive paths (`/content/drive/...`); update mount point if Drive structure differs
- **imgaug deprecation warnings** with newer NumPy — use `albumentations` instead where possible
