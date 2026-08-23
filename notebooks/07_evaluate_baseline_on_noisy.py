"""Step 12 (Experiment 2): test the clean-trained baseline CNN on noisy
test spectrograms for each condition. Compares clean vs noisy accuracy and
tracks Tone2/Tone3 and Tone1/Tone4 confusion specifically."""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(PROJECT_ROOT, "results")
FIGURES = os.path.join(RESULTS, "figures")
MODELS = os.path.join(PROJECT_ROOT, "models")
SPEC_CLEAN = os.path.join(PROJECT_ROOT, "spectrograms", "clean")
SPEC_NOISY = os.path.join(PROJECT_ROOT, "spectrograms", "noisy")

TONE_LABELS = ["Tone 1", "Tone 2", "Tone 3", "Tone 4"]
CONDITIONS = ["white_mild", "white_medium", "white_strong",
              "speed_fast", "speed_slow", "pitch_up", "pitch_down"]


def normalize(X):
    return (X + 80.0) / 80.0


model = load_model(os.path.join(MODELS, "cnn_baseline.keras"))

results_rows = []
confusions = {}

# Clean test accuracy (reference row)
clean = np.load(os.path.join(SPEC_CLEAN, "clean_features.npz"))
X_test_clean = normalize(clean["X"][clean["split"] == "test"])
y_test_clean = clean["y"][clean["split"] == "test"]
pred_clean = np.argmax(model.predict(X_test_clean, verbose=0), axis=1)
acc_clean = accuracy_score(y_test_clean, pred_clean)
results_rows.append({"condition": "clean", "test_accuracy": acc_clean})
confusions["clean"] = confusion_matrix(y_test_clean, pred_clean, labels=[0, 1, 2, 3])

for cond in CONDITIONS:
    data = np.load(os.path.join(SPEC_NOISY, f"{cond}_features.npz"))
    X_test = normalize(data["X"][data["split"] == "test"])
    y_test = data["y"][data["split"] == "test"]
    pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc = accuracy_score(y_test, pred)
    results_rows.append({"condition": cond, "test_accuracy": acc})
    confusions[cond] = confusion_matrix(y_test, pred, labels=[0, 1, 2, 3])
    print(f"{cond}: test accuracy = {acc:.4f}")

acc_table = pd.DataFrame(results_rows)
acc_table.to_csv(os.path.join(RESULTS, "noisy_accuracy_table.csv"), index=False)
print("\n", acc_table)

# Grid of confusion matrices
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
all_conds = ["clean"] + CONDITIONS
for ax, cond in zip(axes.flat, all_conds):
    disp = ConfusionMatrixDisplay(confusions[cond], display_labels=TONE_LABELS)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(cond)
fig.suptitle("Clean-trained CNN: confusion matrices across conditions")
fig.tight_layout()
fig.savefig(os.path.join(FIGURES, "noisy_confusion_matrices_grid.png"), dpi=150)
plt.close(fig)

# Tone-pair confusion analysis: Tone2 vs Tone3 (idx 1,2), Tone1 vs Tone4 (idx 0,3)
pair_rows = []
for cond in all_conds:
    cm = confusions[cond]
    t2_t3 = cm[1, 2] + cm[2, 1]
    t2_t3_total = cm[1, :].sum() + cm[2, :].sum()
    t1_t4 = cm[0, 3] + cm[3, 0]
    t1_t4_total = cm[0, :].sum() + cm[3, :].sum()
    pair_rows.append({
        "condition": cond,
        "tone2_tone3_confusions": int(t2_t3),
        "tone2_tone3_rate": t2_t3 / t2_t3_total if t2_t3_total else 0.0,
        "tone1_tone4_confusions": int(t1_t4),
        "tone1_tone4_rate": t1_t4 / t1_t4_total if t1_t4_total else 0.0,
    })
pair_df = pd.DataFrame(pair_rows)
pair_df.to_csv(os.path.join(RESULTS, "tone_pair_confusion_analysis.csv"), index=False)
print("\nTone-pair confusion analysis:\n", pair_df)

# Accuracy comparison bar chart
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(acc_table["condition"], acc_table["test_accuracy"], color="#4C72B0")
ax.set_ylabel("Test accuracy")
ax.set_title("Clean-trained CNN: accuracy across clean & noisy conditions")
ax.set_ylim(0, 1.05)
plt.xticks(rotation=30, ha="right")
fig.tight_layout()
fig.savefig(os.path.join(FIGURES, "accuracy_comparison_baseline.png"), dpi=150)
plt.close(fig)

print("\nSaved noisy evaluation results.")
