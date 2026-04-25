"""
Realistic audio degradations for the canary robustness study.

We simulate what a closed generative-music model would do to a
canary signal between training and detection:

    1. Lowpass filter — neural codecs (EnCodec, SoundStream, Suno)
       discard high-freq content because human hearing rolls off
       above ~10-12 kHz and the codec budget goes to lower bands.
    2. Downsample round-trip — rate reduction is the cheapest way
       a codec lowers bitrate; we resample down then back up.
    3. Quantization — finite codebook size introduces amplitude
       quantization noise. We simulate with μ-law style nonlinear
       quantization.
    4. Additive noise — codec residual noise is roughly white at
       low levels.
    5. Time-stretch — generative models often render at slightly
       different tempo than training data.
    6. Pitch-shift — small rendering variation.
    7. Combined "neural codec" pipeline — lowpass + downsample +
       light noise. This is the realistic worst-case a canary
       has to survive in production.

Each transform takes (samples, sr) and returns (samples', sr) at
the same sample rate as input (round-trip). This keeps the canary
detector's sample-rate assumption (32 kHz) stable.
"""

from __future__ import annotations

import numpy as np


def lowpass(samples: np.ndarray, sr: int, cutoff_hz: float,
            order: int = 6) -> np.ndarray:
    """Butterworth lowpass. cutoff_hz=10000 simulates a typical neural
    codec; cutoff_hz=4000 is a phone-quality low-bitrate codec."""
    from scipy.signal import butter, sosfiltfilt
    nyq = sr / 2
    sos = butter(order, cutoff_hz / nyq, btype="lowpass", output="sos")
    return sosfiltfilt(sos, samples).astype("float32")


def downsample_roundtrip(samples: np.ndarray, sr: int,
                         target_sr: int) -> np.ndarray:
    """Resample to target_sr and back. Mimics codec rate reduction:
    16 kHz round-trip is aggressive (mid-range codec), 22.05 kHz is
    typical, 8 kHz is phone-grade."""
    import librosa
    down = librosa.resample(samples, orig_sr=sr, target_sr=target_sr)
    up = librosa.resample(down, orig_sr=target_sr, target_sr=sr)
    return up.astype("float32")


def mulaw_quantize(samples: np.ndarray, n_levels: int = 256) -> np.ndarray:
    """μ-law companding then quantization to n_levels. n_levels=256 is
    8-bit (mild loss, like 64 kbps codec); 16 is harsh phone codec."""
    mu = float(n_levels - 1)
    x = np.clip(samples, -1.0, 1.0)
    # Compress
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    # Quantize to n_levels
    q = np.round((y + 1) / 2 * (n_levels - 1)) / (n_levels - 1) * 2 - 1
    # Expand
    out = np.sign(q) * (1.0 / mu) * (np.power(1.0 + mu, np.abs(q)) - 1.0)
    return out.astype("float32")


def add_white_noise(samples: np.ndarray, snr_db: float,
                    seed: int = 0) -> np.ndarray:
    """Add white Gaussian noise at signal-to-noise ratio snr_db.
    snr_db=30 is a clean codec; snr_db=15 is a noisy one."""
    rng = np.random.default_rng(seed)
    rms = float(np.sqrt(np.mean(samples ** 2))) + 1e-12
    noise_rms = rms / (10 ** (snr_db / 20))
    noise = rng.normal(0, noise_rms, len(samples)).astype("float32")
    return (samples + noise).astype("float32")


def time_stretch(samples: np.ndarray, rate: float) -> np.ndarray:
    """librosa.effects.time_stretch. rate=1.02 is +2% tempo
    (subtle but real for generative output)."""
    import librosa
    out = librosa.effects.time_stretch(samples, rate=rate)
    # Pad/truncate to original length so downstream comparisons stay aligned
    if len(out) < len(samples):
        out = np.pad(out, (0, len(samples) - len(out)))
    else:
        out = out[:len(samples)]
    return out.astype("float32")


def pitch_shift(samples: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    """librosa.effects.pitch_shift. n_steps in semitones; 0.1 is
    barely audible, 0.5 is a noticeable detune."""
    import librosa
    return librosa.effects.pitch_shift(samples, sr=sr, n_steps=n_steps).astype("float32")


def gain(samples: np.ndarray, db: float) -> np.ndarray:
    """Apply gain (db can be negative for attenuation)."""
    return (samples * (10 ** (db / 20))).astype("float32")


# Combined pipelines that mimic real neural codec behavior

def neural_codec_light(samples: np.ndarray, sr: int,
                       seed: int = 0) -> np.ndarray:
    """Light degradation, similar to a high-bitrate neural codec
    (e.g. EnCodec @ 24 kbps): lowpass at 12 kHz, downsample to 24 kHz
    round-trip, light noise."""
    out = lowpass(samples, sr, cutoff_hz=12_000)
    out = downsample_roundtrip(out, sr, target_sr=24_000)
    out = add_white_noise(out, snr_db=35, seed=seed)
    return out


def neural_codec_aggressive(samples: np.ndarray, sr: int,
                            seed: int = 0) -> np.ndarray:
    """Harsh degradation: lowpass at 8 kHz, downsample to 16 kHz,
    moderate noise, μ-law 8-bit. Simulates very low bitrate (~32 kbps).
    Realistic worst case for a canary in a heavily compressed model
    output."""
    out = lowpass(samples, sr, cutoff_hz=8_000)
    out = downsample_roundtrip(out, sr, target_sr=16_000)
    out = add_white_noise(out, snr_db=25, seed=seed)
    out = mulaw_quantize(out, n_levels=256)
    return out


# Registry for sweeps — each entry is (name, fn) where fn takes
# (samples, sr, seed) and returns samples. The seed lets us reproduce
# noise patterns across leaked vs pristine clips for fair comparison.

def _wrap_no_seed(fn):
    def w(s, sr, seed):  # noqa: ARG001
        return fn(s, sr) if "sr" in fn.__code__.co_varnames else fn(s)
    return w


TRANSFORMS = [
    ("identity", lambda s, sr, seed: s),

    # Single-mechanism tests
    ("lowpass_12k",  lambda s, sr, seed: lowpass(s, sr, 12_000)),
    ("lowpass_8k",   lambda s, sr, seed: lowpass(s, sr, 8_000)),
    ("lowpass_4k",   lambda s, sr, seed: lowpass(s, sr, 4_000)),

    ("downsample_22k_rt", lambda s, sr, seed: downsample_roundtrip(s, sr, 22_050)),
    ("downsample_16k_rt", lambda s, sr, seed: downsample_roundtrip(s, sr, 16_000)),
    ("downsample_8k_rt",  lambda s, sr, seed: downsample_roundtrip(s, sr, 8_000)),

    ("noise_30db", lambda s, sr, seed: add_white_noise(s, 30, seed)),
    ("noise_20db", lambda s, sr, seed: add_white_noise(s, 20, seed)),
    ("noise_10db", lambda s, sr, seed: add_white_noise(s, 10, seed)),

    ("mulaw_8bit", lambda s, sr, seed: mulaw_quantize(s, 256)),
    ("mulaw_4bit", lambda s, sr, seed: mulaw_quantize(s, 16)),

    ("time_stretch_+2pct", lambda s, sr, seed: time_stretch(s, 1.02)),
    ("time_stretch_-2pct", lambda s, sr, seed: time_stretch(s, 0.98)),
    ("pitch_shift_+0.5st", lambda s, sr, seed: pitch_shift(s, sr, 0.5)),
    ("pitch_shift_-0.5st", lambda s, sr, seed: pitch_shift(s, sr, -0.5)),

    # Combined "what a real neural codec does" pipelines
    ("neural_codec_light",      lambda s, sr, seed: neural_codec_light(s, sr, seed)),
    ("neural_codec_aggressive", lambda s, sr, seed: neural_codec_aggressive(s, sr, seed)),
]
