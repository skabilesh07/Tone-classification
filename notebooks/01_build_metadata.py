
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_CLEAN = os.path.join(PROJECT_ROOT, "dataset_clean")
RESULTS = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

rows = []
for tone_dir in sorted(os.listdir(DATASET_CLEAN)):
    tone_path = os.path.join(DATASET_CLEAN, tone_dir)
    if not os.path.isdir(tone_path):
        continue
    tone_num = int(tone_dir.replace("tone", ""))
    for fname in sorted(os.listdir(tone_path)):
        if not fname.lower().endswith(".wav"):
            continue
        syllable, tone_str, voice = fname[:-4].split("_")
        rows.append({
            "filepath": os.path.relpath(os.path.join(tone_path, fname), PROJECT_ROOT).replace("\\", "/"),
            "tone": tone_num,
            "syllable": syllable,
            "voice": voice,
            "condition": "clean",
        })

df = pd.DataFrame(rows)

train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df["tone"], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["tone"], random_state=42
)

train_df = train_df.copy(); train_df["split"] = "train"
val_df = val_df.copy(); val_df["split"] = "val"
test_df = test_df.copy(); test_df["split"] = "test"

full = pd.concat([train_df, val_df, test_df]).sort_index()
full.to_csv(os.path.join(RESULTS, "metadata_clean.csv"), index=False)

print("Total clean files:", len(full))
print("\nFiles per tone:")
print(full["tone"].value_counts().sort_index())
print("\nSplit sizes:")
print(full["split"].value_counts())
print("\nTone distribution per split:")
print(pd.crosstab(full["split"], full["tone"]))
print("\nVoice distribution per tone:")
print(pd.crosstab(full["tone"], full["voice"]))
print("\nSaved:", os.path.join(RESULTS, "metadata_clean.csv"))
