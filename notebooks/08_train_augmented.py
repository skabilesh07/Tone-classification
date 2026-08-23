"""Step 12 (Experiment 3): train CNN B on clean + noisy training samples
(all 7 conditions), then evaluate both CNN A (baseline) and CNN B on clean
test and on each noisy test condition to compare robustness."""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from model import build_baseline_cnn

RESULTS = os.path.join(PROJECT_ROOT, "results")
FIGURES = os.path.join(RESULTS, "figures")
MODELS = os.path.join(PROJECT_ROOT, "models")
SPEC_CLEAN = os.path.join(PROJECT_ROOT, "spectrograms", "clean")
SPEC_NOISY = os.path.join(PROJECT_ROOT, "spectrograms", "noisy")

CONDITIONS = ["white_mild", "white_medium", "white_strong",
              "speed_fast", "speed_slow", "pitch_up", "pitch_down"]


def normalize(X):
    return (X + 80.0) / 80.0


def load_split(data, split_name):
    X = normalize(data["X"][data["split"] == split_name])
    y = data["y"][data["split"] == split_name]
    return X, y


clean = np.load(os.path.join(SPEC_CLEAN, "clean_features.npz"))
X_train_parts, y_train_parts = [load_split(clean, "train")[0]], [load_split(clean, "train")[1]]
X_val_parts, y_val_parts = [load_split(clean, "val")[0]], [load_split(clean, "val")[1]]

for cond in CONDITIONS:
    data = np.load(os.path.join(SPEC_NOISY, f"{cond}_features.npz"))
    Xtr, ytr = load_split(data, "train")
    Xv, yv = load_split(data, "val")
    X_train_parts.append(Xtr); y_train_parts.append(ytr)
    X_val_parts.append(Xv); y_val_parts.append(yv)

X_train = np.concatenate(X_train_parts, axis=0)
y_train = np.concatenate(y_train_parts, axis=0)
X_val = np.concatenate(X_val_parts, axis=0)
y_val = np.concatenate(y_val_parts, axis=0)

print("Augmented train set:", X_train.shape, " val set:", X_val.shape)

model_b = build_baseline_cnn(input_shape=X_train.shape[1:], num_classes=4)
history = model_b.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=25,
    batch_size=32,
    verbose=2,
)
model_b.save(os.path.join(MODELS, "cnn_augmented.keras"))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(history.history["accuracy"], label="train")
axes[0].plot(history.history["val_accuracy"], label="val")
axes[0].set_title("CNN B Accuracy"); axes[0].legend()
axes[1].plot(history.history["loss"], label="train")
axes[1].plot(history.history["val_loss"], label="val")
axes[1].set_title("CNN B Loss"); axes[1].legend()
fig.tight_layout()
fig.savefig(os.path.join(FIGURES, "training_history_augmented.png"), dpi=150)
plt.close(fig)

# ---- Compare CNN A (clean-only) vs CNN B (clean+noise) ----
model_a = load_model(os.path.join(MODELS, "cnn_baseline.keras"))

X_test_clean, y_test_clean = load_split(clean, "test")

rows = []
for name, model in [("CNN_A_clean_only", model_a), ("CNN_B_clean_plus_noise", model_b)]:
    acc_clean = accuracy_score(y_test_clean, np.argmax(model.predict(X_test_clean, verbose=0), axis=1))
    noisy_accs = []
    for cond in CONDITIONS:
        data = np.load(os.path.join(SPEC_NOISY, f"{cond}_features.npz"))
        Xte, yte = load_split(data, "test")
        acc = accuracy_score(yte, np.argmax(model.predict(Xte, verbose=0), axis=1))
        noisy_accs.append(acc)
        rows.append({"model": name, "condition": cond, "test_accuracy": acc})
    rows.append({"model": name, "condition": "clean", "test_accuracy": acc_clean})
    rows.append({"model": name, "condition": "noisy_average", "test_accuracy": float(np.mean(noisy_accs))})

comparison = pd.DataFrame(rows)
comparison.to_csv(os.path.join(RESULTS, "model_comparison.csv"), index=False)
print("\nModel comparison:\n", comparison.to_string(index=False))

# Pivot table for readability: model x condition -> accuracy
pivot = comparison.pivot(index="condition", columns="model", values="test_accuracy")
pivot = pivot.reindex(["clean"] + CONDITIONS + ["noisy_average"])
pivot.to_csv(os.path.join(RESULTS, "model_comparison_pivot.csv"))
print("\nPivot:\n", pivot)

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(pivot.index))
width = 0.35
ax.bar(x - width / 2, pivot["CNN_A_clean_only"], width, label="CNN A (clean only)")
ax.bar(x + width / 2, pivot["CNN_B_clean_plus_noise"], width, label="CNN B (clean + noise)")
ax.set_xticks(x)
ax.set_xticklabels(pivot.index, rotation=30, ha="right")
ax.set_ylabel("Test accuracy")
ax.set_title("CNN A vs CNN B: accuracy by condition")
ax.legend()
ax.set_ylim(0, 1.05)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES, "model_comparison_bar.png"), dpi=150)
plt.close(fig)

print("\nSaved augmented model, comparison table, and figures.")
