"""Step 8-10 (Experiment 1): train the baseline CNN on clean spectrograms
only and evaluate clean test accuracy, confusion matrix, precision/recall/F1."""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from model import build_baseline_cnn

RESULTS = os.path.join(PROJECT_ROOT, "results")
FIGURES = os.path.join(RESULTS, "figures")
MODELS = os.path.join(PROJECT_ROOT, "models")
os.makedirs(FIGURES, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

TONE_LABELS = ["Tone 1", "Tone 2", "Tone 3", "Tone 4"]


def normalize(X):
    # mel_db is in dB, roughly in [-80, 0] (top_db=80, ref=max) -> scale to [0, 1]
    return (X + 80.0) / 80.0


data = np.load(os.path.join(PROJECT_ROOT, "spectrograms", "clean", "clean_features.npz"))
X, y, split = data["X"], data["y"], data["split"]
X = normalize(X)

X_train, y_train = X[split == "train"], y[split == "train"]
X_val, y_val = X[split == "val"], y[split == "val"]
X_test, y_test = X[split == "test"], y[split == "test"]

print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

model = build_baseline_cnn(input_shape=X_train.shape[1:], num_classes=4)
model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=16,
    verbose=2,
)

model.save(os.path.join(MODELS, "cnn_baseline.keras"))

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(history.history["accuracy"], label="train")
axes[0].plot(history.history["val_accuracy"], label="val")
axes[0].set_title("Accuracy"); axes[0].set_xlabel("epoch"); axes[0].legend()
axes[1].plot(history.history["loss"], label="train")
axes[1].plot(history.history["val_loss"], label="val")
axes[1].set_title("Loss"); axes[1].set_xlabel("epoch"); axes[1].legend()
fig.tight_layout()
fig.savefig(os.path.join(FIGURES, "training_history_baseline.png"), dpi=150)
plt.close(fig)

# Evaluation on clean test set
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

test_acc = accuracy_score(y_test, y_pred)
train_acc = history.history["accuracy"][-1]
val_acc = history.history["val_accuracy"][-1]

print(f"\nTrain accuracy: {train_acc:.4f}")
print(f"Val accuracy:   {val_acc:.4f}")
print(f"Test accuracy:  {test_acc:.4f}")

report = classification_report(y_test, y_pred, target_names=TONE_LABELS, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(os.path.join(RESULTS, "clean_classification_report.csv"))
print(report_df)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=TONE_LABELS)
fig, ax = plt.subplots(figsize=(5, 5))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Baseline CNN - Clean Test Confusion Matrix")
fig.tight_layout()
fig.savefig(os.path.join(FIGURES, "clean_confusion_matrix.png"), dpi=150)
plt.close(fig)

summary = {
    "train_accuracy": float(train_acc),
    "val_accuracy": float(val_acc),
    "test_accuracy": float(test_acc),
}
with open(os.path.join(RESULTS, "baseline_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\nSaved model, figures, and reports.")
