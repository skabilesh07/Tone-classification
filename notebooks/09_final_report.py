
import os
import json
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(PROJECT_ROOT, "results")

meta = pd.read_csv(os.path.join(RESULTS, "metadata_clean.csv"))
dist = pd.crosstab([meta["tone"]], [meta["syllable"]])
dist["total"] = dist.sum(axis=1)
dist.to_csv(os.path.join(RESULTS, "dataset_distribution_table.csv"))

baseline_summary = json.load(open(os.path.join(RESULTS, "baseline_summary.json")))
acc_table = pd.read_csv(os.path.join(RESULTS, "noisy_accuracy_table.csv"))
pair_df = pd.read_csv(os.path.join(RESULTS, "tone_pair_confusion_analysis.csv"))
pivot = pd.read_csv(os.path.join(RESULTS, "model_comparison_pivot.csv"), index_col=0)

report = []
report.append("# Mandarin Tone Classification with Controlled Noise — Results Report\n")
report.append("## 1. Dataset\n")
report.append(
    "Note: the source download was missing all 6 `ang` / Tone 4 recordings "
    "(48 files instead of the expected 54 per tone). To keep the dataset "
    "perfectly balanced, the `ang` syllable was excluded from **all** tones "
    "rather than training on an imbalanced set. Final dataset: **8 syllables "
    "(a, ai, an, ba, da, ma, mi, tai) x 4 tones x 6 voices = 192 clean files, "
    "48 per tone.**\n"
)
report.append(dist.to_markdown())
report.append("\n## 2. Train/Val/Test Split\n")
report.append(meta["split"].value_counts().to_markdown())

report.append("\n## 3. Experiment 1 — Clean Baseline CNN\n")
report.append(f"- Train accuracy: {baseline_summary['train_accuracy']:.4f}\n")
report.append(f"- Val accuracy: {baseline_summary['val_accuracy']:.4f}\n")
report.append(f"- Test accuracy: {baseline_summary['test_accuracy']:.4f}\n")
report.append("See `figures/clean_confusion_matrix.png`, `figures/training_history_baseline.png`.\n")

report.append("\n## 4. Experiment 2 — Clean-trained CNN Tested on Noisy Audio\n")
report.append(acc_table.to_markdown(index=False))
report.append(
    "\n\nWhite noise causes a clear, monotonic accuracy drop "
    "(89.7% mild -> 62.1% medium -> 24.1% strong), while the small speed "
    "(+/-10%) and pitch (+/-0.5 semitone) perturbations barely affect the "
    "clean-trained CNN. Under strong white noise the model collapses toward "
    "predicting Tone 3 for most inputs, and the Tone2/Tone3 pair shows a "
    "50% mutual-confusion rate at that noise level, while Tone1/Tone4 stays "
    "at 0% confusion — supporting the hypothesis that Tone 2 (rising) and "
    "Tone 3 (dipping) are acoustically closer and harder to tell apart once "
    "the pitch contour is degraded, compared to the more distinct Tone 1 "
    "(flat) vs Tone 4 (falling) pair.\n"
)
report.append(pair_df.to_markdown(index=False))

report.append("\n## 5. Experiment 3 — Noise-Augmented Training (CNN A vs CNN B)\n")
report.append(pivot.to_markdown())
report.append(
    "\n\nCNN B (trained on clean + noisy data) matches CNN A's 100% clean "
    "accuracy while dramatically improving robustness: noisy-condition "
    "average accuracy rises from **81.8% (CNN A) to 96.6% (CNN B)**, with "
    "the largest gain under strong white noise (24.1% -> 96.6%). This "
    "confirms that noise augmentation during training improves robustness "
    "without sacrificing clean performance.\n"
)

report.append("\n## 6. Conclusion\n")
report.append(
    "Controlled noise increases tone confusion, especially for the "
    "Tone2/Tone3 pair under strong white noise, while Tone1/Tone4 remains "
    "comparatively robust. Training with noise augmentation substantially "
    "improves robustness to noisy conditions with no cost to clean-audio "
    "accuracy.\n"
)

out_path = os.path.join(RESULTS, "final_report.md")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report))

print("Saved final report:", out_path)
print("Saved dataset distribution table:", os.path.join(RESULTS, "dataset_distribution_table.csv"))
