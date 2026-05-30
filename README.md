[README.md](https://github.com/user-attachments/files/28411686/README.md)
# Retinal Vessel Segmentation with U-Net++

A PyTorch implementation of U-Net and U-Net++ (nested U-Net) architectures for binary segmentation of retinal blood vessels. Supports training and validation on the **DRIVE**, **STARE**, and **CHASE_DB1** datasets.

---

## Table of Contents

- [Requirements](#requirements)
- [File Structure](#file-structure)
- [Imports](#imports)
- [Usage](#usage)
- [CLI Flags](#cli-flags)
- [Models](#models)
- [Metrics](#metrics)
- [Outputs](#outputs)

---
## Requirements

Install dependencies with:

```bash
pip install torch torchvision opencv-python pillow scikit-learn numpy matplotlib scipy
```

---
## Models
Models: https://drive.google.com/drive/folders/1XMOiqnan7a5JxifsDSUbqo4RdrTx7RDm?usp=sharing
## File Structure

```
project_root/
│
├── Code.py                        # Main script (training + validation)
│
├── model_weights.pth              # Saved model weights (generated after training)
├── history_<name>.json            # Training history per run (generated after training)
├── Summary_<name>.json            # Validation metrics summary (generated after validation)
│
└── <Database>/                    # Root directory passed via --Database
    │
    ├── DRIVE/
    │   └── train/
    │       ├── images/            # Retinal images (.png / .tif)
    │       ├── mask/              # Ground-truth segmentation masks
    │       └── filter/            # Field-of-view binary masks
    │
    ├── STARE/
    │   ├── images/                # Retinal images
    │   ├── masks_1/               # Ground-truth masks — Annotator 1
    │   └── masks_2/               # Ground-truth masks — Annotator 2
    │
    └── CHASE_DB1/
        ├── images/                # Retinal images
        └── mask/                  # Ground-truth masks (JSON bitmap format)
```

> **Note:** The `--Database` flag must point to the folder that contains the `DRIVE/`, `STARE/`, and/or `CHASE_DB1/` subdirectories, with a trailing `/` (e.g. `./data/`).

---

## Imports

| Library | Purpose |
|---|---|
| `argparse` | CLI argument parsing |
| `os` | File system operations |
| `cv2` (OpenCV) | Image I/O, resizing, flipping, rotation, augmentation |
| `json` | Saving/loading training history and metrics summaries; reading CHASE_DB1 bitmap masks |
| `PIL.Image` | Image loading |
| `base64`, `zlib`, `io.BytesIO` | Decoding compressed bitmap masks in CHASE_DB1 |
| `torch`, `torch.nn`, `torch.nn.functional` | Neural network definition, loss functions, inference |
| `torch.utils.data` | `Dataset` and `DataLoader` abstractions |
| `sklearn.metrics` | AUC-ROC computation |
| `numpy` | Array manipulation and metric calculations |
| `matplotlib.pyplot`, `matplotlib.gridspec` | Validation result plots |
| `scipy.ndimage` | Elastic deformation augmentation (Gaussian filter + coordinate mapping) |

---

## Usage

### Training

```bash
python Code.py \
  --Database ./data/ \
  --Dataset_name 0 \
  --Model 1 \
  --Mode 0 \
  --Features 32 \
  --Epochs 50 \
  --LR 1e-3 \
  --BatchSize 4 \
  --AugmentSize 5 \
  --DeepSV False
```

### Validation

```bash
python Code.py \
  --Database ./data/ \
  --Dataset_name 1 \
  --Model 1 \
  --Mode 1 \
  --DeepSV False \
  --Enhance_images True
```

---

## CLI Flags

| Flag | Type | Required | Default | Description |
|---|---|---|---|---|
| `--Database` | `str` | ✅ | `.` | Root directory containing the dataset folders |
| `--Dataset_name` | `int` | ✅ | `0` | Dataset selection: `0` = DRIVE, `1` = STARE, `2` = CHASE_DB1 |
| `--Model` | `int` | ✅ | `1` | Model variant: `1` = UNet++(4), `2` = UNet++(3), `3` = UNet++(2), `4` = UNet++(1) |
| `--Mode` | `int` | ✅ | `0` | Run mode: `0` = Training, `1` = Validation |
| `--DeepSV` | `bool` | ❌ | `False` | Enable deep supervision (multi-output training) |
| `--Features` | `int` | ❌ | `32` | Base feature channels for the U-Net encoder |
| `--Epochs` | `int` | ❌ | `20` | Number of training epochs |
| `--LR` | `float` | ❌ | `1e-3` | Learning rate for the Adam optimizer |
| `--BatchSize` | `int` | ❌ | `4` | Mini-batch size for training and validation |
| `--AugmentSize` | `int` | ❌ | `5` | Number of augmented copies per original image (DRIVE only) |
| `--Enhance_images` | `bool` | ❌ | `False` | Apply brightness/contrast enhancement (STARE and CHASE_DB1 only; not used during training) |

---

## Models

All models are U-Net++ variants with nested dense skip connections. Select the depth via `--Model`:

| `--Model` | Class | Encoder Depth | Feature list |
|---|---|---|---|
| `1` | `UNetpp_4` | 4 levels + bottleneck | `[f, 2f, 4f, 8f, 16f]` |
| `2` | `UNetpp_3` | 3 levels + bottleneck | `[f, 2f, 4f, 8f]` |
| `3` | `UNetpp_2` | 2 levels + bottleneck | `[f, 2f, 4f]` |
| `4` | `UNetpp_1` | 1 level + bottleneck | `[f, 2f]` |

Where `f` is the value of `--Features`. All models accept 3-channel RGB input and produce 2-class (background / vessel) logits.

When `--DeepSV True` is set, each model outputs a list of intermediate predictions (one per decoder level), and the training loss is averaged across all outputs.

The loss function combines **Cross-Entropy** and **Dice Loss** equally:

```
Loss = CrossEntropy(logits, targets) + DiceLoss(logits, targets)
```

---

## Metrics

Evaluated at the pixel level after training or during validation:

| Metric | Description |
|---|---|
| Sensitivity (Recall) | `TP / (TP + FN)` |
| Specificity | `TN / (TN + FP)` |
| Dice / F1 | `2·TP / (2·TP + FP + FN)` |
| IoU (Jaccard) | `TP / (TP + FP + FN)` |
| AUC-ROC | Pixel-level area under the ROC curve (requires soft probability scores) |

Dataset-level mean ± std is computed and saved to a JSON summary file.

---

## Outputs

| File | Generated When | Description |
|---|---|---|
| `model_weights.pth` | After training | Saved PyTorch model state dict |
| `history_<name>.json` | After training | Per-epoch train/val loss, IoU, and Dice |
| `Summary_<name>.json` | After validation | Mean ± std of all segmentation metrics |
| `<name>_validation.png` | After validation | Grid plot of input images, ground-truth masks, and predictions |
