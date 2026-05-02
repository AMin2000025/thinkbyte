"""
predict.py — Use this for ALL predictions after training is done.

Usage:
    python predict.py path/to/face.jpg
    python predict.py path/to/folder/
"""

import os, sys
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models

# ──────────────────────────────────────────────
MODEL_PATH = "c:/Users/amino/OneDrive/Desktop/insta v1/maram/face_model.pth"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────
# MODEL  (must match train.py)
# ──────────────────────────────────────────────
def build_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(512, 1),
        nn.Sigmoid()
    )
    return model


# ──────────────────────────────────────────────
# LOAD  (call once)
# ──────────────────────────────────────────────
def load_model(path=MODEL_PATH):
    checkpoint = torch.load(path, map_location=DEVICE)
    model = build_model().to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"✓ Model loaded  (val MAE during training: {checkpoint.get('val_mae', '?'):.3f})")
    return model


# ──────────────────────────────────────────────
# TRANSFORM
# ──────────────────────────────────────────────
_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ──────────────────────────────────────────────
# PREDICT
# ──────────────────────────────────────────────
def predict_prettiness(image_input, model):
    """
    Returns a dict:
        percentage  → 0–100
        score_1_5   → 1.0–5.0
        label       → human-readable category
    """
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        image = Image.fromarray(image_input[..., ::-1])
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        raise TypeError("Pass a file path, PIL.Image, or numpy array")

    tensor = _transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        raw = model(tensor).squeeze().item()   # [0, 1]

    percentage = round(raw * 100, 1)
    score_1_5  = round(raw * 4.0 + 1.0, 2)

    if percentage >= 85:   label = "Exceptionally attractive"
    elif percentage >= 70: label = "Very attractive"
    elif percentage >= 50: label = "Attractive"
    elif percentage >= 30: label = "Average"
    else:                  label = "Below average"

    return {"percentage": percentage, "score_1_5": score_1_5, "label": label}


def predict_folder(folder, model, exts=(".jpg", ".jpeg", ".png", ".webp")):
    results = []
    for fname in os.listdir(folder):
        if fname.lower().endswith(exts):
            try:
                r = predict_prettiness(os.path.join(folder, fname), model)
                r["file"] = fname
                results.append(r)
            except Exception as e:
                print(f"  ⚠ Skipped {fname}: {e}")
    return sorted(results, key=lambda x: x["percentage"], reverse=True)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_or_folder>")
        sys.exit(1)

    target = sys.argv[1]
    model  = load_model()

    if os.path.isdir(target):
        results = predict_folder(target, model)
        print(f"\n{'File':<35} {'Score':>7}  Label")
        print("─" * 65)
        for r in results:
            print(f"{r['file']:<35} {r['percentage']:>6.1f}%  {r['label']}")
    else:
        r = predict_prettiness(target, model)
        
        y=r["score_1_5"]*3.3
        print(y)
       # print(f"\n{'─'*40}")
        #print(f"  Prettiness : {r['percentage']}%")
        #print(f"  Score (1–5): {r['score_1_5']}")
        #print(f"  Category   : {r['label']}")
        #print(f"{'─'*40}")