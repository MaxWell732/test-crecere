"""E1 · Preprocessing (spec §9-E1).

The 8 kHz master in data/raw is never modified. Model copy at 16 kHz:
  1. censor spans -> digital silence
  2. high-pass 70 Hz (Butterworth order 2, zero-phase)
  3. soxr HQ resample to 16 kHz
  4. peak-normalize to -1 dBFS, peak measured outside censor spans
Censor spans are re-zeroed after steps 2–3 because filter/resampler ringing leaks into them (decisions D-007).
No denoising, no trimming.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import soundfile as sf
import soxr
from scipy.signal import butter, sosfiltfilt

from . import paths
from . import spans as sp
from .audio_utils import load_audio
from .cache import StageContext, run_per_call
from .inventory import censor_spans, master_path, usable_calls


def wav_path(call_id: str):
    return paths.interim("audio_16k", f"{call_id}.wav")


def make_model_copy(x: np.ndarray, sr: int, censor: list[tuple[float, float]], a: dict[str, Any]) -> tuple[np.ndarray, float, float]:
    y = x.astype(np.float64).copy()
    y[sp.to_mask(censor, len(y), sr)] = 0.0
    sos = butter(a["highpass_order"], a["highpass_hz"], btype="highpass", fs=sr, output="sos")
    y = sosfiltfilt(sos, y)
    y = soxr.resample(y, sr, a["model_sr"], quality="HQ")
    mask16 = sp.to_mask(censor, len(y), a["model_sr"])
    y[mask16] = 0.0
    outside = np.abs(y[~mask16])
    peak = float(outside.max()) if outside.size else 0.0
    gain = 10 ** (a["peak_dbfs"] / 20) / peak if peak > 0 else 1.0
    y = np.clip(y * gain, -1.0, 1.0)
    return y.astype(np.float32), peak, gain


def process_call(call_id: str, ctx: StageContext) -> dict[str, Any]:
    a = ctx.cfg["audio"]
    x, sr, _ = load_audio(master_path(call_id, ctx.cfg))
    censor = [(s["start"], s["end"]) for s in censor_spans(call_id)]
    y, peak, gain = make_model_copy(x, sr, censor, a)
    out = wav_path(call_id)
    paths.ensure_dir(out.parent)
    tmp = out.with_name(f".{out.name}.part")
    sf.write(str(tmp), y, a["model_sr"], subtype="PCM_16", format="WAV")
    os.replace(tmp, out)
    return {
        "wav": str(out.relative_to(paths.ROOT)),
        "sample_rate": a["model_sr"],
        "n_samples": int(len(y)),
        "duration_s": round(len(y) / a["model_sr"], 3),
        "source_peak_outside_censor": round(peak, 6),
        "gain_db": round(float(20 * np.log10(gain)), 3),
    }


def run(calls: list[str] | None = None, force: bool = False):
    return run_per_call("preprocess", usable_calls(calls), process_call,
                        lambda c: paths.interim("audio_16k", f"{c}.json"), force=force)
