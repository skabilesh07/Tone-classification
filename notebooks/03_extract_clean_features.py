"""Step 6: convert every clean audio file into a Mel spectrogram array and
save train/val/test feature matrices for model training."""
import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from audio_utils import file_to_mel

RESULTS = os.path.join(PROJECT_ROOT, "results")
SPEC_CLEAN = os.path.join(PROJECT_ROOT, "spectrograms", "clean")
os.makedirs(SPEC_CLEAN, exist_ok=True)

meta = pd.read_csv(os.path.join(RESULTS, "metadata_clean.csv"))

X, y, splits = [], [], []
for _, row in meta.iterrows():
    path = os.path.join(PROJECT_ROOT, row["filepath"])
    mel_db = file_to_mel(path)
    X.append(mel_db)
    y.append(row["tone"] - 1)  # 0-indexed: tone1->0 ... tone4->3
    splits.append(row["split"])

X = np.array(X, dtype=np.float32)
X = X[..., np.newaxis]  # add channel dim -> (N, 128, T, 1)
y = np.array(y, dtype=np.int64)
splits = np.array(splits)

print("Feature array shape:", X.shape)

out_path = os.path.join(SPEC_CLEAN, "clean_features.npz")
np.savez_compressed(out_path, X=X, y=y, split=splits)
print("Saved:", out_path)

for s in ["train", "val", "test"]:
    n = (splits == s).sum()
    print(f"{s}: {n} samples")
