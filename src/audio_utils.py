"""Shared audio processing utilities for the Mandarin tone classification project."""
import numpy as np
import librosa

SR = 22050
DURATION = 1.0
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
TARGET_LEN = int(SR * DURATION)


def load_fixed(path, sr=SR, duration=DURATION):
    """Load audio, mono, resampled, padded/trimmed to a fixed duration."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y


def audio_to_mel(y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Convert a fixed-length waveform into a log-scaled Mel spectrogram."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db


def file_to_mel(path, sr=SR, duration=DURATION):
    y = load_fixed(path, sr=sr, duration=duration)
    return audio_to_mel(y, sr=sr)


# ---------------- Controlled noise / distortion ----------------

def add_white_noise(y, snr_db):
    """Add white Gaussian noise at a target signal-to-noise ratio (dB)."""
    signal_power = np.mean(y ** 2)
    if signal_power == 0:
        return y
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), size=y.shape)
    return (y + noise).astype(np.float32)


def change_speed(y, rate):
    """Time-stretch audio (rate > 1 = faster, < 1 = slower)."""
    return librosa.effects.time_stretch(y, rate=rate)


def change_pitch(y, sr, n_steps):
    """Shift pitch by n_steps semitones."""
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


# SNR levels (dB) chosen so mild/medium/strong give a clear, increasing
# amount of distortion -- lower dB = more noise relative to signal.
NOISE_CONDITIONS = {
    "white_mild": {"type": "white", "snr_db": 20},
    "white_medium": {"type": "white", "snr_db": 10},
    "white_strong": {"type": "white", "snr_db": 3},
    "speed_fast": {"type": "speed", "rate": 1.1},
    "speed_slow": {"type": "speed", "rate": 0.9},
    "pitch_up": {"type": "pitch", "n_steps": 0.5},
    "pitch_down": {"type": "pitch", "n_steps": -0.5},
}


def apply_condition(y, sr, condition):
    """Apply a named noise/distortion condition to a fixed-length waveform,
    then re-pad/trim to the original fixed length so all outputs stay the
    same duration."""
    cfg = NOISE_CONDITIONS[condition]
    if cfg["type"] == "white":
        out = add_white_noise(y, cfg["snr_db"])
    elif cfg["type"] == "speed":
        out = change_speed(y, cfg["rate"])
    elif cfg["type"] == "pitch":
        out = change_pitch(y, sr, cfg["n_steps"])
    else:
        raise ValueError(f"Unknown condition type: {cfg['type']}")

    target_len = int(sr * DURATION)
    if len(out) < target_len:
        out = np.pad(out, (0, target_len - len(out)))
    else:
        out = out[:target_len]
    return out.astype(np.float32)
