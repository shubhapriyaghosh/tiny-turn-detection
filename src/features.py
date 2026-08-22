from dataclasses import dataclass
import numpy as np
import librosa


@dataclass
class FeatureConfig:
    sample_rate: int = 16000
    window_seconds: float = 2.0
    n_mels: int = 64
    n_fft: int = 400
    hop_length: int = 160
    fmin: int = 20
    fmax: int = 8000


def load_and_pad(path, cfg):
    wav, sr = librosa.load(path, sr=cfg.sample_rate, mono=True)
    target_len = int(cfg.window_seconds * cfg.sample_rate)
    if len(wav) < target_len:
        wav = np.pad(wav, (target_len - len(wav), 0), mode="constant")
    else:
        wav = wav[-target_len:]
    return wav


def log_mel_spectrogram(wav, cfg):
    mel = librosa.feature.melspectrogram(
        y=wav, sr=cfg.sample_rate, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
        n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax, power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
    return log_mel.astype(np.float32)


def prosodic_features(wav, cfg):
    frame_len = int(0.02 * cfg.sample_rate)
    hop = int(0.02 * cfg.sample_rate)
    frames = librosa.util.frame(wav, frame_length=frame_len, hop_length=hop).T
    frame_energy = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1) + 1e-12)
    thresh = max(frame_energy.mean() * 0.15, 1e-4)
    voiced = frame_energy > thresh

    if voiced.any():
        last_voiced_idx = np.where(voiced)[0][-1]
        trailing_silence_frames = len(voiced) - 1 - last_voiced_idx
    else:
        trailing_silence_frames = len(voiced)
    trailing_silence_ms = trailing_silence_frames * (hop / cfg.sample_rate) * 1000
    voiced_ratio = float(voiced.mean())

    f0 = librosa.yin(
        wav, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"),
        sr=cfg.sample_rate, frame_length=frame_len * 4, hop_length=hop,
    )
    f0_valid = f0[(f0 > 0) & np.isfinite(f0)]
    if len(f0_valid) >= 4:
        half = len(f0_valid) // 2
        pitch_slope = float(f0_valid[half:].mean() - f0_valid[:half].mean())
    else:
        pitch_slope = 0.0

    energy_db = 20 * np.log10(frame_energy + 1e-6)
    half = len(energy_db) // 2
    energy_slope = float(energy_db[half:].mean() - energy_db[:half].mean()) if half > 0 else 0.0
    energy_var = float(np.var(energy_db))

    zcr = librosa.feature.zero_crossing_rate(wav, frame_length=frame_len, hop_length=hop)
    zcr_mean = float(zcr.mean())

    return np.array([
        trailing_silence_ms / 1000.0,
        pitch_slope / 100.0,
        energy_slope,
        energy_var / 100.0,
        voiced_ratio,
        zcr_mean,
    ], dtype=np.float32)


def extract(path, cfg):
    wav = load_and_pad(path, cfg)
    return log_mel_spectrogram(wav, cfg), prosodic_features(wav, cfg)
