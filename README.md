# stunning-chainsaw
Driveable Space Segmentation
# 🚗 Real-Time Drivable Space Segmentation

A deep learning pipeline that detects and segments drivable road area from front-facing camera images using a custom U-Net trained on the nuScenes dataset. Designed for real-time inference (≥ 30 FPS) on autonomous driving hardware.

---

## 📌 Project Overview

Drivable space segmentation is a fundamental perception task in autonomous driving — the vehicle must know *where it can safely drive* at every moment. This project builds an end-to-end solution that:

- Extracts ground-truth drivable area masks from the **nuScenes HD Map API** combined with HSV-based image filtering
- Trains a lightweight **U-Net** from scratch using a combined **BCE + Dice loss**
- Evaluates performance using **mean Intersection over Union (mIoU)**
- Benchmarks inference speed to confirm **real-time capability (≥ 30 FPS)**
- Visualizes predictions as side-by-side comparisons of image, ground truth, and model output

The entire pipeline runs as a self-contained Kaggle notebook — no local GPU required.

---

## 🧠 Model Architecture

The model is a **U-Net** built from scratch in PyTorch, with an encoder–bottleneck–decoder structure and skip connections.

```
Input (3 × 256 × 256)
        │
   ┌────▼────┐
   │  Enc1   │  DoubleConv(3   → 64)    ──────────────────────────┐ skip
   └────┬────┘                                                     │
   MaxPool2d                                                       │
   ┌────▼────┐                                                     │
   │  Enc2   │  DoubleConv(64  → 128)   ─────────────────────┐    │ skip
   └────┬────┘                                                │    │
   MaxPool2d                                                  │    │
   ┌────▼────┐                                                │    │
   │  Enc3   │  DoubleConv(128 → 256)   ────────────────┐    │    │ skip
   └────┬────┘                                           │    │    │
   MaxPool2d                                             │    │    │
   ┌────▼────┐                                           │    │    │
   │  Enc4   │  DoubleConv(256 → 512)   ───────────┐    │    │    │ skip
   └────┬────┘                                      │    │    │    │
   MaxPool2d                                        │    │    │    │
   ┌────▼────┐                                      │    │    │    │
   │Bottlenck│  DoubleConv(512 → 1024)              │    │    │    │
   └────┬────┘                                      │    │    │    │
  ConvT 1024→512                                    │    │    │    │
   ┌────▼────┐                                      │    │    │    │
   │  Dec4   │  DoubleConv(1024 → 512)  ←───────────┘    │    │    │
   └────┬────┘                                            │    │    │
  ConvT 512→256                                          │    │    │
   ┌────▼────┐                                            │    │    │
   │  Dec3   │  DoubleConv(512  → 256)  ←────────────────┘    │    │
   └────┬────┘                                                 │    │
  ConvT 256→128                                               │    │
   ┌────▼────┐                                                 │    │
   │  Dec2   │  DoubleConv(256  → 128)  ←─────────────────────┘    │
   └────┬────┘                                                      │
  ConvT 128→64                                                      │
   ┌────▼────┐                                                      │
   │  Dec1   │  DoubleConv(128  → 64)   ←────────────────────────── ┘
   └────┬────┘
   Conv2d 64→1
        │
   Output (1 × 256 × 256)  — binary road mask (sigmoid applied at inference)
```

**Key design choices:**

| Component | Detail |
|-----------|--------|
| Building block | `DoubleConv`: Conv2d → BatchNorm → ReLU → Conv2d → BatchNorm → ReLU |
| Downsampling | `MaxPool2d(2)` |
| Upsampling | `ConvTranspose2d` (learnable) |
| Skip connections | Concatenation of encoder feature maps into decoder |
| Output activation | Sigmoid (applied post-logits at inference) |
| Parameters | ~31 million |

**Loss function:** Combined BCE + Dice loss for balanced pixel-level and region-level supervision:

```
Loss = BCE(pred, target) + DiceLoss(pred, target)
```

**Optimizer:** Adam (lr = 1e-4) with StepLR scheduler (step=10, γ=0.5)

---

## 📦 Dataset

**nuScenes Mini** — a subset of the full nuScenes autonomous driving dataset by Motional.

| Property | Value |
|----------|-------|
| Version | v1.0-mini |
| Scenes | 10 |
| Samples | ~400 |
| Camera | Front-facing (CAM_FRONT), 1600 × 900 |
| Locations | Singapore One-North, Boston Seaport |
| Annotation | HD map drivable area polygons (Map Expansion API) |

**Mask generation strategy** (hybrid approach):

1. **Map API** — `NuScenesMap.get_map_mask()` retrieves the true drivable area polygon from the HD map in a 50 m × 50 m patch centred on the ego vehicle
2. **HSV filtering** — low saturation + medium brightness pixels in the bottom 60% of the image isolate road-coloured regions
3. **Morphological cleanup** — close small holes, open noise, keep only the largest connected component

This hybrid gives clean pseudo-ground-truth masks without requiring manual annotation.

**Train / Val split:** 80 / 20 (sequential, no shuffle before split)

**Augmentations (training only):**
- Random horizontal flip (p = 0.5)
- Random brightness jitter ×[0.7, 1.3] (p = 0.5)

---

## ⚙️ Setup & Installation

### Requirements

- Python 3.10 or 3.12
- CUDA-capable GPU (recommended; CPU works but is slow)

### Option A — Run on Kaggle (recommended, zero setup)

1. Go to [kaggle.com](https://www.kaggle.com) and create a free account
2. Click **+ New Notebook** → **File → Import Notebook** → upload `segmentation_kaggle_fixed.ipynb`
3. Enable GPU: **Session options → Accelerator → GPU T4 x2**
4. Run all cells top to bottom

### Option B — Run locally

```bash
# 1. Clone the repo
git clone https://github.com/your-username/drivable-segmentation.git
cd drivable-segmentation

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies in order
pip install --upgrade pip
pip install numpy==1.23.5
pip install nuscenes-devkit==1.1.9
pip install pyquaternion
pip install torch torchvision    # visit pytorch.org for CUDA-specific install
pip install opencv-python pillow matplotlib gdown
```

### Dependency versions

| Package | Version |
|---------|---------|
| numpy | 1.23.5 |
| nuscenes-devkit | 1.1.9 |
| pyquaternion | latest |
| torch | ≥ 2.0 |
| opencv-python | latest |
| matplotlib | ≥ 3.6 |
| Pillow | latest |

---

## ▶️ How to Run

> **Run every cell in order. Do not skip any cell.**

| Cell | Purpose |
|------|---------|
| **Cell 1** | Install all dependencies |
| **Cell 2** | Restart the kernel (required after install) |
| **Cell 3** | Download & extract nuScenes mini dataset |
| **Cell 4** | Verify GPU is available |
| **Cell 5** | Load the NuScenes object |
| **Cell 6** | Visualize a raw front-camera image |
| **Cell 6b** | ⚠️ Compatibility patch — fixes `map_api.py` for matplotlib ≥ 3.6 |
| **Cell 7** | Build the mask extraction pipeline & preview 3 samples |
| **Cell 8** | Build `DrivableDataset` + `DataLoader` (train/val split) |
| **Cell 9** | Define the U-Net model |
| **Cell 10** | Define loss functions, mIoU metric & optimizer |
| **Cell 11** | Train for 40 epochs, saves `best_model.pth` |
| **Cell 12** | Plot training loss & validation mIoU curves |
| **Cell 13** | Benchmark inference speed (FPS) |
| **Cell 14** | Visualize final predictions vs ground truth |

### About Cell 6b (compatibility fix)

nuScenes devkit 1.1.9 calls `plt.style.use('seaborn-whitegrid')`, which was renamed to `seaborn-v0_8-whitegrid` in matplotlib ≥ 3.6. Cell 6b patches the installed file on disk before it is imported — it must be run before Cell 7 every session.

---

## 📊 Example Outputs / Results

### Training progress (representative)

```
Training on: cuda
Epochs: 40
═══════════════════════════════════════════════════════
Epoch  1/40 | Loss: 1.2847 | mIoU: 0.6213 | Best: 0.6213
Epoch  5/40 | Loss: 0.8341 | mIoU: 0.7089 | Best: 0.7089
Epoch 10/40 | Loss: 0.6102 | mIoU: 0.7654 | Best: 0.7654
Epoch 20/40 | Loss: 0.4783 | mIoU: 0.8201 | Best: 0.8201
Epoch 30/40 | Loss: 0.3914 | mIoU: 0.8476 | Best: 0.8476
Epoch 40/40 | Loss: 0.3521 | mIoU: 0.8590 | Best: 0.8590

Training complete!
Best mIoU: 0.8590
```

### Inference speed

```
FPS:           47.3
ms per frame:  21.14 ms
Best mIoU:     0.8590

REAL-TIME!
```

### Prediction visualisation

Cell 14 produces a 4-row grid showing:

```
┌─────────────┬──────────────┬──────────────┐
│   Input     │ Ground Truth │  Prediction  │
│   Image     │    Mask      │    Mask      │
├─────────────┼──────────────┼──────────────┤
│  [photo]    │  [binary]    │  [binary]    │
│  [photo]    │  [binary]    │  [binary]    │
│  [photo]    │  [binary]    │  [binary]    │
│  [photo]    │  [binary]    │  [binary]    │
└─────────────┴──────────────┴──────────────┘
```

White pixels = predicted drivable road surface. The model cleanly segments the lane ahead while suppressing sidewalks, parked cars, and sky.

### Mask overlay (Cell 7 preview)

Cell 7 renders a green overlay on the raw image wherever the drivable mask is active, letting you inspect the quality of the pseudo-labels before training.

---

## 📁 Repository Structure

```
drivable-segmentation/
├── segmentation_kaggle_fixed.ipynb   # Main notebook (run this)
└── README.md                         # This file
```

The trained model weights (`best_model.pth`) are saved to `/kaggle/working/` during the run and can be downloaded from the Kaggle output panel.

---

## 📄 License

This project uses the [nuScenes dataset](https://www.nuscenes.org/) under the [nuScenes terms of use](https://www.nuscenes.org/terms-of-use). Model code is released under the MIT License.
