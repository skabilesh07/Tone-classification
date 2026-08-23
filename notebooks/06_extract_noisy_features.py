"""Convert every noisy audio file into a Mel spectrogram, grouped by
condition, and save feature arrays (with inherited train/val/test split)."""
import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from audio_utils import file_to_mel

RESULTS = os.path.join(PROJECT_ROOT, "results")
SPEC_NOISY = os.path.join(PROJECT_ROOT, "spectrograms", "noisy")
os.makedirs(SPEC_NOISY, exist_ok=True)

meta = pd.read_csv(os.path.join(RESULTS, "metadata_noisy.csv"))

for condition, group in meta.groupby("condition"):
    X, y, splits = [], [], []
    for _, row in group.iterrows():
        path = os.path.join(PROJECT_ROOT, row["filepath"])
        mel_db = file_to_mel(path)
        X.append(mel_db)
        y.append(row["tone"] - 1)
        splits.append(row["split"])

    X = np.array(X, dtype=np.float32)[..., np.newaxis]
    y = np.array(y, dtype=np.int64)
    splits = np.array(splits)

    out_path = os.path.join(SPEC_NOISY, f"{condition}_features.npz")
    np.savez_compressed(out_path, X=X, y=y, split=splits)
    print(f"{condition}: {X.shape} -> {out_path}")

print("Done.")
