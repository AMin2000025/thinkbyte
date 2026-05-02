# Face Attractiveness Scoring — README

## Quick Start

### 1. Install dependencies
```
pip install torch torchvision timm scikit-learn pandas Pillow
```

### 2. Get the dataset
Clone SCUT-FBP5500 (5,500 face images, rated by 60 humans, score 1–5):
```
git clone https://github.com/HCIILAB/SCUT-FBP5500_v2_Release
```
Then prepare it:
```
python prepare_dataset.py
```

### 3. Train
```
python train.py
```
Training runs for 30 epochs with EfficientNet-B3 + OneCycleLR.
Best model is auto-saved to `face_attractiveness_model.pth`.

### 4. Predict
```python
from train import load_model, predict_prettiness

model = load_model("face_attractiveness_model.pth")
score = predict_prettiness("path/to/face.jpg", model)
print(f"Prettiness: {score:.1f}%")
```

---

## Architecture
- **Backbone**: EfficientNet-B3 (pretrained on ImageNet)
- **Head**: Dropout → Linear(256) → GELU → Linear(1) → Sigmoid
- **Loss**: Huber Loss (robust to noisy human annotations)
- **Optimizer**: AdamW + OneCycleLR scheduler
- **Output**: float in [0, 100]%

---

## Expected Performance (SCUT-FBP5500)
| Metric | Typical result |
|---|---|
| Val MAE (1–5 scale) | ~0.24–0.30 |
| Val R² | ~0.87–0.92 |
| Pearson r | ~0.93–0.96 |

---

## Score Interpretation
| Score | Label |
|---|---|
| 0–30% | Below average |
| 30–50% | Average |
| 50–70% | Attractive |
| 70–85% | Very attractive |
| 85–100% | Exceptionally attractive |

---

## Important Notes
- The model learns from **human crowd-sourced ratings**, so it reflects
  statistical averages across raters — not an objective truth.
- The SCUT-FBP5500 dataset includes both Asian and Caucasian subjects,
  male and female, giving reasonable demographic coverage.
- Always apply **face detection + crop** before running inference for
  best results (e.g. with MTCNN or RetinaFace).
- For production use, consider adding face alignment (landmark-based)
  before the model input.

## requirements.txt
```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
scikit-learn>=1.2.0
pandas>=1.5.0
Pillow>=9.0.0
numpy>=1.23.0
```
