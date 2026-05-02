"""
train.py — Simple face attractiveness trainer using ResNet18.
Run ONCE, then use predict.py for all predictions.
"""

import os, json
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
IMAGE_DIR  = "archive (1)/Images/Images/"
LABELS_TXT = "archive (1)/labels.txt"
MODEL_SAVE = "face_model.pth"

BATCH_SIZE = 4
EPOCHS     = 5
LR         = 1e-4
SEED       = 42

torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ──────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────
def load_labels(path):
    records = []
    with open(path) as f:
        for line in f:
            parts = line.strip().replace(",", " ").split()
            if len(parts) >= 2:
                records.append({"image_path": parts[0], "score": float(parts[1])})
    return pd.DataFrame(records)


class FaceDataset(Dataset):
    def __init__(self, df, image_dir, augment=False):
        self.df        = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip() if augment else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(os.path.join(self.image_dir, row["image_path"])).convert("RGB")
        score = (row["score"] - 1.0) / 4.0      # [1,5] → [0,1]
        return self.transform(img), torch.tensor(score, dtype=torch.float32)


# ──────────────────────────────────────────────
# MODEL  (ResNet18 — simple & reliable)
# ──────────────────────────────────────────────
def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Linear(512, 1),
        nn.Sigmoid()
    )
    return model


# ──────────────────────────────────────────────
# TRAIN
# ──────────────────────────────────────────────
def train():
    df = load_labels(LABELS_TXT)
    print(f"Loaded {len(df)} samples | scores: {df.score.min():.2f}–{df.score.max():.2f}")

    train_df, val_df = train_test_split(df, test_size=0.15, random_state=SEED)

    train_loader = DataLoader(FaceDataset(train_df, IMAGE_DIR, augment=True),
                              batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(FaceDataset(val_df,   IMAGE_DIR, augment=False),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = build_model().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    best_mae = float("inf")
    history  = []

    for epoch in range(1, EPOCHS + 1):

        # ── Train ──
        model.train()
        for images, scores in train_loader:
            images, scores = images.to(DEVICE), scores.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images).squeeze(), scores)
            loss.backward()
            optimizer.step()

        # ── Validate ──
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for images, scores in val_loader:
                images = images.to(DEVICE)
                out = model(images).squeeze().cpu().numpy()
                preds.extend(out if out.ndim > 0 else [out.item()])
                targets.extend(scores.numpy())

        mae = mean_absolute_error(targets, preds) * 4   # back to 1–5 scale
        r2  = r2_score(targets, preds)
        history.append({"epoch": epoch, "val_mae": mae, "val_r2": r2})
        print(f"Epoch {epoch:02d}/{EPOCHS}  val_MAE={mae:.3f}  val_R²={r2:.3f}")

        if mae < best_mae:
            best_mae = mae
            torch.save({"model_state": model.state_dict(), "val_mae": mae}, MODEL_SAVE)
            print(f"  ✓ Saved → {MODEL_SAVE}")

        scheduler.step()

    with open("training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✅ Done. Best val MAE = {best_mae:.3f}  |  Model saved to {MODEL_SAVE}")
    print("Now use predict.py for all predictions.")


if __name__ == "__main__":
    train()