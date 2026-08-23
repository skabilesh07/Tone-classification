"""Step 5-7: load a sample file, convert to Mel spectrogram, and plot the
same syllable across all four tones to visually motivate the CNN approach."""
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa.display
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from audio_utils import file_to_mel, SR, HOP_LENGTH

RESULTS = os.path.join(PROJECT_ROOT, "results")
FIGURES = os.path.join(RESULTS, "figures")
os.makedirs(FIGURES, exist_ok=True)

meta = pd.read_csv(os.path.join(RESULTS, "metadata_clean.csv"))

# Single-file sanity check
sample = meta.iloc[0]
sample_path = os.path.join(PROJECT_ROOT, sample["filepath"])
mel_db = file_to_mel(sample_path)
print(f"Loaded {sample['filepath']} -> mel shape {mel_db.shape}")

fig, ax = plt.subplots(figsize=(5, 4))
img = librosa.display.specshow(mel_db, sr=SR, hop_length=HOP_LENGTH, x_axis="time", y_axis="mel", ax=ax)
ax.set_title(f"{sample['syllable']} tone{sample['tone']} ({sample['voice']})")
fig.colorbar(img, ax=ax, format="%+2.0f dB")
fig.tight_layout()
fig.savefig(os.path.join(FIGURES, "sample_single_spectrogram.png"), dpi=150)
plt.close(fig)

# Same syllable across all 4 tones (use a syllable/voice available for all tones)
syllable = "ma"
voice = "fv1"
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for i, tone in enumerate([1, 2, 3, 4]):
    row = meta[(meta["syllable"] == syllable) & (meta["tone"] == tone) & (meta["voice"] == voice)].iloc[0]
    path = os.path.join(PROJECT_ROOT, row["filepath"])
    mel_db = file_to_mel(path)
    img = librosa.display.specshow(mel_db, sr=SR, hop_length=HOP_LENGTH, x_axis="time", y_axis="mel", ax=axes[i])
    axes[i].set_title(f"{syllable} - Tone {tone}")
fig.suptitle(f"Mel spectrograms of '{syllable}' across all 4 tones (voice: {voice})")
fig.tight_layout()
fig.savefig(os.path.join(FIGURES, f"example_spectrograms_{syllable}.png"), dpi=150)
plt.close(fig)

print("Saved figures to", FIGURES)
