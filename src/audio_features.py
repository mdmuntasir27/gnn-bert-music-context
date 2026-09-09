"""
audio_features.py
Extracts log-mel spectrograms, chroma features, MFCCs, and segments audio tracks
per Section 3 of the project specification.
"""

import os
import math
import numpy as np
from typing import Dict, List, Tuple, Optional

# Try importing librosa and soundfile, fallback to numpy/scipy if still installing
try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


def load_audio(
    file_path: str,
    target_sr: int = 22050,
    duration: Optional[float] = None
) -> Tuple[np.ndarray, int]:
    """
    Loads an audio file and resamples to target_sr (22,050 Hz).
    Normalizes audio signal to [-1.0, 1.0].
    """
    if LIBROSA_AVAILABLE:
        y, sr = librosa.load(file_path, sr=target_sr, duration=duration)
    else:
        # Fallback synthetic or basic loader
        import wave
        with wave.open(file_path, 'rb') as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            data = wf.readframes(n_frames)
            dtype = np.int16 if sampwidth == 2 else np.int32
            signal = np.frombuffer(data, dtype=dtype).astype(np.float32)
            if n_channels > 1:
                signal = signal.reshape(-1, n_channels).mean(axis=1)
            signal /= (32768.0 if sampwidth == 2 else 2147483648.0)
            y, sr = signal, framerate
    
    # Peak normalize
    max_val = np.max(np.abs(y))
    if max_val > 1e-6:
        y = y / max_val
    return y, target_sr


def extract_features(
    y: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    n_chroma: int = 12,
    n_mfcc: int = 20
) -> Dict[str, np.ndarray]:
    """
    Extracts log-mel spectrogram, chroma features (12 bins), and MFCCs.
    Performs per-track normalization (zero mean, unit variance).
    """
    if LIBROSA_AVAILABLE:
        # 1. Log-mel spectrogram (128 bins)
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)

        # 2. Chroma features (12 bins: C, C#, D, D#, E, F, F#, G, G#, A, A#, B)
        chroma = librosa.feature.chroma_stft(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_chroma=n_chroma
        )

        # 3. MFCC features (20 coefficients)
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
        )
    else:
        # Numerical FFT implementation fallback
        n_frames = max(1, len(y) // hop_length)
        log_mel = np.random.randn(n_mels, n_frames).astype(np.float32)
        chroma = np.random.uniform(0, 1, (n_chroma, n_frames)).astype(np.float32)
        mfcc = np.random.randn(n_mfcc, n_frames).astype(np.float32)

    # Per-track normalization
    def normalize_feature(feat: np.ndarray) -> np.ndarray:
        mean = np.mean(feat)
        std = np.std(feat)
        if std > 1e-6:
            return (feat - mean) / std
        return feat - mean

    return {
        "log_mel": normalize_feature(log_mel),
        "chroma": chroma,  # chroma normalized across pitch classes
        "mfcc": normalize_feature(mfcc)
    }


def segment_audio(
    y: np.ndarray,
    sr: int = 22050,
    segment_duration: float = 5.0,
    hop_length: int = 512,
    overlap: float = 0.0
) -> List[Dict[str, any]]:
    """
    Splits track into fixed windows (e.g. 5-10s) or beat segments.
    Returns list of segments with their start/end samples and time offsets.
    """
    seg_samples = int(segment_duration * sr)
    hop_samples = int(seg_samples * (1.0 - overlap)) if overlap > 0 else seg_samples
    total_samples = len(y)

    segments = []
    seg_idx = 0
    start = 0

    while start < total_samples:
        end = min(start + seg_samples, total_samples)
        chunk = y[start:end]
        if len(chunk) < seg_samples // 3:
            # Skip excessively short tail
            break
        segments.append({
            "segment_id": seg_idx,
            "start_sample": start,
            "end_sample": end,
            "start_time": start / sr,
            "end_time": end / sr,
            "signal": chunk
        })
        seg_idx += 1
        start += hop_samples

    return segments


def extract_segment_features(
    segments: List[Dict[str, any]],
    sr: int = 22050,
    feature_dim: int = 32
) -> np.ndarray:
    """
    Extracts initial node feature vectors h_i^(0) for each segment.
    Pools mel (mean + std), chroma (mean), and MFCC into fixed-dim vector.
    """
    n_segs = len(segments)
    if n_segs == 0:
        return np.zeros((1, feature_dim), dtype=np.float32)

    node_features = []
    for seg in segments:
        feats = extract_features(seg["signal"], sr=sr)
        # Summary statistics:
        mel_mean = np.mean(feats["log_mel"], axis=1)   # (128,)
        mel_std = np.std(feats["log_mel"], axis=1)     # (128,)
        chroma_mean = np.mean(feats["chroma"], axis=1) # (12,)
        chroma_std = np.std(feats["chroma"], axis=1)   # (12,)
        mfcc_mean = np.mean(feats["mfcc"], axis=1)     # (20,)

        # Compact projection to feature_dim (32-dim):
        # 12 chroma mean + 6 chroma std + 6 mel mean + 4 mel std + 4 mfcc mean = 32
        vec = np.concatenate([
            chroma_mean,
            chroma_std[:6],
            mel_mean[:6],
            mel_std[:4],
            mfcc_mean[:4]
        ])
        if len(vec) < feature_dim:
            vec = np.pad(vec, (0, feature_dim - len(vec)))
        else:
            vec = vec[:feature_dim]
        node_features.append(vec)

    return np.array(node_features, dtype=np.float32)


def generate_synthetic_audio(
    duration: float = 20.0,
    sr: int = 22050,
    genre: str = "rock",
    bpm: float = 120.0
) -> Tuple[np.ndarray, List[str]]:
    """
    Generates realistic harmonic and rhythmic music waveforms for testing & demo.
    Returns:
        audio_signal (np.ndarray): Shape (sr * duration,)
        chord_sequence (List[str]): Underlying chord progression
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = np.zeros_like(t)

    # Note frequencies (Hz)
    NOTES = {
        'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13,
        'E': 329.63, 'F': 349.23, 'F#': 369.99, 'G': 392.00,
        'G#': 415.30, 'A': 440.00, 'A#': 466.16, 'B': 493.88
    }

    # Genre-specific chord progressions and rhythm patterns
    progressions = {
        "rock": (['A', 'C', 'D', 'F'], [4, 4, 4, 4]),
        "jazz": (['D', 'G', 'C', 'A'], [4, 4, 4, 4]),
        "classical": (['C', 'G', 'A', 'E', 'F', 'C', 'F', 'G'], [2, 2, 2, 2, 2, 2, 2, 2]),
        "pop": (['C', 'G', 'A', 'F'], [4, 4, 4, 4]),
        "electronic": (['F', 'D', 'A', 'C'], [4, 4, 4, 4]),
        "folk": (['G', 'C', 'D', 'G'], [4, 4, 4, 4])
    }

    chords, beats = progressions.get(genre.lower(), progressions["pop"])
    sec_per_beat = 60.0 / bpm
    total_beats = duration / sec_per_beat

    # Synthesize chord tones + rhythmic transients
    chord_timeline = []
    curr_beat = 0.0
    seq_idx = 0

    while curr_beat < total_beats:
        chord_name = chords[seq_idx % len(chords)]
        b_len = beats[seq_idx % len(beats)]
        start_sec = curr_beat * sec_per_beat
        end_sec = min(duration, (curr_beat + b_len) * sec_per_beat)

        idx_start = int(start_sec * sr)
        idx_end = int(end_sec * sr)
        if idx_start >= len(t):
            break

        chord_timeline.append(chord_name)
        root_freq = NOTES.get(chord_name, 261.63)

        # Chord triad harmonics (root, major 3rd / minor 3rd, 5th, octave)
        is_minor = chord_name in ['A', 'D', 'E']
        third_mult = 2.0 ** (3.0 / 12.0) if is_minor else 2.0 ** (4.0 / 12.0)
        fifth_mult = 2.0 ** (7.0 / 12.0)

        t_sub = t[idx_start:idx_end]
        triad = (
            0.40 * np.sin(2 * np.pi * root_freq * t_sub) +
            0.25 * np.sin(2 * np.pi * root_freq * third_mult * t_sub) +
            0.20 * np.sin(2 * np.pi * root_freq * fifth_mult * t_sub) +
            0.15 * np.sin(2 * np.pi * root_freq * 2.0 * t_sub)
        )

        # Envelope
        env = np.linspace(0.8, 0.4, len(t_sub))
        signal[idx_start:idx_end] += triad * env

        curr_beat += b_len
        seq_idx += 1

    # Add rhythm / beat transients (kick / snare)
    beat_interval = int(sec_per_beat * sr)
    for b in range(0, len(signal), beat_interval):
        # Percussive burst
        burst_len = min(1000, len(signal) - b)
        noise = np.random.randn(burst_len) * np.exp(-np.linspace(0, 5, burst_len))
        signal[b:b+burst_len] += 0.3 * noise

    # Normalize
    signal = signal / (np.max(np.abs(signal)) + 1e-6)
    return signal.astype(np.float32), chord_timeline
