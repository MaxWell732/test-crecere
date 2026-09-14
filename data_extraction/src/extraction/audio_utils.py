"""Audio decoding and framing helpers shared by E0/E1/E2/E8b."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


def load_audio(path: Path) -> tuple[np.ndarray, int, dict[str, Any]]:
    """Decode to float32 mono (averaging channels if ever stereo). Returns (x, sr, info)."""
    info = sf.info(str(path))
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    meta = {
        "codec": info.subtype,
        "format": info.format,
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "duration_file_s": float(info.frames) / float(info.samplerate),
    }
    mono = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    return np.ascontiguousarray(mono, dtype=np.float32), int(sr), meta


def frames(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    """(n_frames, win) view; frame i covers samples [i*hop, i*hop+win). Trailing partial frame dropped."""
    if len(x) < win:
        return np.empty((0, win), dtype=x.dtype)
    return np.lib.stride_tricks.sliding_window_view(x, win)[::hop]


def hann(win: int) -> np.ndarray:
    return (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(win) / win)).astype(np.float64)


def frame_rms_db(x: np.ndarray, win: int, hop: int) -> tuple[np.ndarray, np.ndarray]:
    """Window-compensated frame RMS (linear power, dBFS)."""
    fr = frames(x.astype(np.float64), win, hop)
    w = hann(win)
    power = np.mean((fr * w) ** 2, axis=1) / np.mean(w**2)
    return power, 10 * np.log10(power + 1e-20)


def stft_power(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    fr = frames(x.astype(np.float64), win, hop)
    return np.abs(np.fft.rfft(fr * hann(win), axis=1)) ** 2


def frame_to_time(i: int, j: int, win: int, hop: int, sr: int, n_samples: int) -> tuple[float, float]:
    """Time span for an inclusive frame run i..j, using frame centers ± hop/2."""
    start = (i * hop + win / 2 - hop / 2) / sr
    end = (j * hop + win / 2 + hop / 2) / sr
    return max(0.0, start), min(n_samples / sr, end)


def dbfs(value: float) -> float | None:
    return None if value <= 0 else float(10 * np.log10(value))
