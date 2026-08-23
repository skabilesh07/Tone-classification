# Mandarin Tone Classification with Controlled Noise — Results Report

## 1. Dataset

Note: the source download was missing all 6 `ang` / Tone 4 recordings (48 files instead of the expected 54 per tone). To keep the dataset perfectly balanced, the `ang` syllable was excluded from **all** tones rather than training on an imbalanced set. Final dataset: **8 syllables (a, ai, an, ba, da, ma, mi, tai) x 4 tones x 6 voices = 192 clean files, 48 per tone.**

|   tone |   a |   ai |   an |   ba |   da |   ma |   mi |   tai |   total |
|-------:|----:|-----:|-----:|-----:|-----:|-----:|-----:|------:|--------:|
|      1 |   6 |    6 |    6 |    6 |    6 |    6 |    6 |     6 |      48 |
|      2 |   6 |    6 |    6 |    6 |    6 |    6 |    6 |     6 |      48 |
|      3 |   6 |    6 |    6 |    6 |    6 |    6 |    6 |     6 |      48 |
|      4 |   6 |    6 |    6 |    6 |    6 |    6 |    6 |     6 |      48 |

## 2. Train/Val/Test Split

| split   |   count |
|:--------|--------:|
| train   |     134 |
| val     |      29 |
| test    |      29 |

## 3. Experiment 1 — Clean Baseline CNN

- Train accuracy: 1.0000

- Val accuracy: 1.0000

- Test accuracy: 1.0000

See `figures/clean_confusion_matrix.png`, `figures/training_history_baseline.png`.


## 4. Experiment 2 — Clean-trained CNN Tested on Noisy Audio

| condition    |   test_accuracy |
|:-------------|----------------:|
| clean        |        1        |
| white_mild   |        0.896552 |
| white_medium |        0.62069  |
| white_strong |        0.241379 |
| speed_fast   |        1        |
| speed_slow   |        1        |
| pitch_up     |        0.965517 |
| pitch_down   |        1        |


White noise causes a clear, monotonic accuracy drop (89.7% mild -> 62.1% medium -> 24.1% strong), while the small speed (+/-10%) and pitch (+/-0.5 semitone) perturbations barely affect the clean-trained CNN. Under strong white noise the model collapses toward predicting Tone 3 for most inputs, and the Tone2/Tone3 pair shows a 50% mutual-confusion rate at that noise level, while Tone1/Tone4 stays at 0% confusion — supporting the hypothesis that Tone 2 (rising) and Tone 3 (dipping) are acoustically closer and harder to tell apart once the pitch contour is degraded, compared to the more distinct Tone 1 (flat) vs Tone 4 (falling) pair.

| condition    |   tone2_tone3_confusions |   tone2_tone3_rate |   tone1_tone4_confusions |   tone1_tone4_rate |
|:-------------|-------------------------:|-------------------:|-------------------------:|-------------------:|
| clean        |                        0 |                0   |                        0 |                  0 |
| white_mild   |                        0 |                0   |                        0 |                  0 |
| white_medium |                        0 |                0   |                        0 |                  0 |
| white_strong |                        7 |                0.5 |                        0 |                  0 |
| speed_fast   |                        0 |                0   |                        0 |                  0 |
| speed_slow   |                        0 |                0   |                        0 |                  0 |
| pitch_up     |                        0 |                0   |                        0 |                  0 |
| pitch_down   |                        0 |                0   |                        0 |                  0 |

## 5. Experiment 3 — Noise-Augmented Training (CNN A vs CNN B)

| condition     |   CNN_A_clean_only |   CNN_B_clean_plus_noise |
|:--------------|-------------------:|-------------------------:|
| clean         |           1        |                 1        |
| white_mild    |           0.896552 |                 0.965517 |
| white_medium  |           0.62069  |                 0.965517 |
| white_strong  |           0.241379 |                 0.965517 |
| speed_fast    |           1        |                 0.931034 |
| speed_slow    |           1        |                 1        |
| pitch_up      |           0.965517 |                 0.931034 |
| pitch_down    |           1        |                 1        |
| noisy_average |           0.817734 |                 0.965517 |


CNN B (trained on clean + noisy data) matches CNN A's 100% clean accuracy while dramatically improving robustness: noisy-condition average accuracy rises from **81.8% (CNN A) to 96.6% (CNN B)**, with the largest gain under strong white noise (24.1% -> 96.6%). This confirms that noise augmentation during training improves robustness without sacrificing clean performance.


## 6. Conclusion

Controlled noise increases tone confusion, especially for the Tone2/Tone3 pair under strong white noise, while Tone1/Tone4 remains comparatively robust. Training with noise augmentation substantially improves robustness to noisy conditions with no cost to clean-audio accuracy.
