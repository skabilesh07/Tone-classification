# Mandarin Tone Classification with Controlled Noise

Classifies Mandarin audio into Tone 1-4 using Mel spectrograms and a CNN,
then adds controlled noise/distortion to test which tone pairs become
confused under noise.

Results and full write-up: [results/final_report.md](results/final_report.md)

## Dataset

Source: Tone Perfect (Michigan State University), voices FV1-3/MV1-3.

The download was missing all 6 `ang` / Tone 4 recordings, so `ang` was
dropped from every tone to keep the dataset balanced:

- **8 syllables**: a, ai, an, ba, da, ma, mi, tai
- **4 tones** x **6 voices** = **192 clean audio files** (48 per tone)

## Folder structure

```
dataset_clean/tone{1-4}/          clean wav files, syllable_toneN_voice.wav
dataset_noisy/<condition>/toneN/  noisy wav files, 7 conditions
spectrograms/clean/               extracted Mel spectrogram arrays (.npz)
spectrograms/noisy/               one .npz per noise condition
src/audio_utils.py                mel spectrogram + noise/speed/pitch functions
src/model.py                      baseline CNN architecture
notebooks/01-09_*.py               pipeline scripts, run in order (see below)
models/                           trained CNN A (clean) and CNN B (clean+noise)
results/                          metadata, tables, figures, final_report.md
```

## Pipeline (run in order)

| Script | What it does |
|---|---|
| `01_build_metadata.py` | Scans `dataset_clean/`, builds `results/metadata_clean.csv` with a stratified train/val/test split |
| `02_visualize_spectrograms.py` | Plots the same syllable ("ma") across all 4 tones |
| `03_extract_clean_features.py` | Converts clean audio to Mel spectrograms, saves `spectrograms/clean/clean_features.npz` |
| `04_train_baseline.py` | Trains CNN A on clean data only (Experiment 1) |
| `05_make_noisy.py` | Generates 7 noise/distortion conditions from clean audio |
| `06_extract_noisy_features.py` | Converts noisy audio to Mel spectrograms |
| `07_evaluate_baseline_on_noisy.py` | Tests CNN A on noisy audio, tone-pair confusion analysis (Experiment 2) |
| `08_train_augmented.py` | Trains CNN B on clean+noisy data, compares CNN A vs CNN B (Experiment 3) |
| `09_final_report.py` | Compiles all tables/figures into `results/final_report.md` |

Run from the project root, e.g.:

```
python notebooks/01_build_metadata.py
python notebooks/02_visualize_spectrograms.py
...
python notebooks/09_final_report.py
```

## Headline results

- **CNN A (clean-trained)**: 100% clean test accuracy, but collapses to
  24.1% under strong white noise.
- **CNN B (clean + noise-augmented)**: 100% clean accuracy, and **96.6%**
  average accuracy across all noisy conditions.
- Under strong noise, **Tone 2 / Tone 3** are confused 50% of the time,
  while **Tone 1 / Tone 4** stay at 0% confusion.

Full tables and figures: [results/](results/), especially
[results/final_report.md](results/final_report.md).
