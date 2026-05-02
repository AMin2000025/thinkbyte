"""
Dataset Preparation Script — SCUT-FBP5500
==========================================
This script prepares the SCUT-FBP5500 dataset into the CSV format
expected by train.py.

Steps:
1. Download dataset from GitHub (link below)
2. Run this script to generate scut_fbp5500_labels.csv

Dataset source:
  https://github.com/HCIILAB/SCUT-FBP5500_v2_Release
  
After cloning, your folder should look like:
  SCUT-FBP5500_v2/
    Images/
      CF*.jpg   (Caucasian Female)
      CM*.jpg   (Caucasian Male)
      AF*.jpg   (Asian Female)
      AM*.jpg   (Asian Male)
    train_test_files/
      All_labels.txt
"""

import os
import pandas as pd
import shutil
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG — adjust these paths
# ─────────────────────────────────────────────
SCUT_ROOT  = "./SCUT-FBP5500_v2"          # where you cloned/extracted the dataset
OUTPUT_CSV = "scut_fbp5500_labels.csv"
OUTPUT_IMG = "images/"                     # flat folder all images will be copied to


def prepare_scut_fbp5500():
    labels_file = os.path.join(SCUT_ROOT, "train_test_files", "All_labels.txt")
    images_dir  = os.path.join(SCUT_ROOT, "Images")

    if not os.path.exists(labels_file):
        raise FileNotFoundError(
            f"Labels not found at {labels_file}.\n"
            "Download from: https://github.com/HCIILAB/SCUT-FBP5500_v2_Release"
        )

    # ── Parse labels ──
    records = []
    with open(labels_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                filename, score = parts[0], float(parts[1])
                records.append({"image_path": filename, "score": score})

    df = pd.DataFrame(records)

    # ── Infer gender column (optional, useful for analysis) ──
    def get_gender(fname):
        fname = fname.upper()
        if "CF" in fname or "AF" in fname:
            return "female"
        elif "CM" in fname or "AM" in fname:
            return "male"
        return "unknown"

    df["gender"] = df["image_path"].apply(get_gender)

    # ── Copy images to flat output folder ──
    os.makedirs(OUTPUT_IMG, exist_ok=True)
    missing = 0
    for fname in df["image_path"]:
        src = os.path.join(images_dir, fname)
        dst = os.path.join(OUTPUT_IMG, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            print(f"  ⚠ Missing: {src}")
            missing += 1

    if missing:
        print(f"\n{missing} images not found — check SCUT_ROOT path.")

    # ── Save CSV ──
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Saved {len(df)} records to {OUTPUT_CSV}")
    print(f"  Male:   {(df.gender == 'male').sum()}")
    print(f"  Female: {(df.gender == 'female').sum()}")
    print(f"\nScore statistics:")
    print(df["score"].describe().round(3))

    return df


# ─────────────────────────────────────────────
# ALTERNATIVE: Use CelebA (binary attractiveness)
# ─────────────────────────────────────────────
def prepare_celeba_attractiveness(celeba_root="./celeba"):
    """
    CelebA has a binary 'Attractive' attribute.
    We map it to a continuous score using multiple correlated attributes
    to create a richer pseudo-regression target.
    """
    attr_file = os.path.join(celeba_root, "list_attr_celeba.txt")
    df = pd.read_csv(attr_file, sep=r"\s+", header=1)

    # CelebA uses -1/1, convert to 0/1
    df = (df + 1) / 2

    # Beauty-correlated attributes (research-backed)
    beauty_attrs = [
        "Attractive", "Young", "High_Cheekbones",
        "Oval_Face", "Smiling", "Arched_Eyebrows",
        "Wearing_Lipstick", "Heavy_Makeup",
    ]
    weights = [3.0, 1.0, 1.0, 1.0, 1.5, 0.5, 0.5, 0.5]  # tune as needed

    df["score"] = sum(
        w * df[a] for a, w in zip(beauty_attrs, weights) if a in df.columns
    )
    # Normalize to [1, 5]
    mn, mx = df["score"].min(), df["score"].max()
    df["score"] = (df["score"] - mn) / (mx - mn) * 4 + 1
    df["image_path"] = df.index
    df = df[["image_path", "score"]].reset_index(drop=False)
    df.rename(columns={"index": "image_path"}, inplace=True)

    df.to_csv("celeba_attractiveness.csv", index=False)
    print(f"✓ CelebA proxy labels created ({len(df)} images)")
    return df


if __name__ == "__main__":
    df = prepare_scut_fbp5500()
