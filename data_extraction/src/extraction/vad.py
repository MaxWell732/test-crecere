"""E2 · VAD + SNR (spec §9-E2).

- Silero VAD on the 16 kHz model copy -> speech segments.
- SNR = 10·log10(mean power of speech frames / mean power of non-speech frames) on the 8 kHz master, time-aligned,
  excluding censor spans and frames that are mostly exact zeros.
- level_dbfs = RMS level of speech frames (spec §10.2), same frames.
- speech_ratio = speech seconds (outside censor) / (duration − censored_s).
"""

from __future__ import annotations

from importlib import metadata
from typing import Any

import numpy as np
import soundfile as sf

from . import paths
from . import spans as sp
from .audio_utils import frames, load_audio
from .cache import StageContext, read_json, run_per_call
from .inventory import master_path, usable_calls
from .preprocess import wav_path


def load_vad(call_id: str) -> dict[str, Any]:
    return read_json(paths.interim("vad", f"{call_id}.json"))


def snr_and_level(x: np.ndarray, sr: int, speech: list[tuple[float, float]], censor: list[tuple[float, float]],
                  frame_ms: float) -> tuple[float | None, float | None, dict[str, int]]:
    win = int(round(frame_ms * sr / 1000))
    fr = frames(x.astype(np.float64), win, win)
    if len(fr) == 0:
        return None, None, {"speech_frames": 0, "nonspeech_frames": 0}
    rate = sr / win
    zero_frac = np.mean(fr == 0.0, axis=1)
    usable = (zero_frac < 0.5) & ~sp.to_mask(censor, len(fr), rate)
    centers = (np.arange(len(fr)) + 0.5) / rate
    is_speech = np.zeros(len(fr), dtype=bool)
    for s, e in speech:
        is_speech |= (centers >= s) & (centers < e)
    power = np.mean(fr**2, axis=1)
    ps, pn = power[usable & is_speech], power[usable & ~is_speech]
    level = float(10 * np.log10(ps.mean())) if ps.size and ps.mean() > 0 else None
    snr = float(10 * np.log10(ps.mean() / pn.mean())) if ps.size and pn.size and pn.mean() > 0 else None
    return snr, level, {"speech_frames": int(ps.size), "nonspeech_frames": int(pn.size)}


def setup(ctx: StageContext) -> None:
    import torch
    from silero_vad import load_silero_vad

    torch.set_num_threads(ctx.cfg["asr"]["cpu_threads"])
    ctx.models["vad"] = load_silero_vad()
    ctx.models["vad_name"] = f"silero-vad {metadata.version('silero-vad')}"


def process_call(call_id: str, ctx: StageContext) -> dict[str, Any]:
    import torch
    from silero_vad import get_speech_timestamps

    v, q = ctx.cfg["vad"], ctx.cfg["quality"]
    y, sr16 = sf.read(str(wav_path(call_id)), dtype="float32")
    ts = get_speech_timestamps(
        torch.from_numpy(y), ctx.models["vad"], threshold=v["threshold"], sampling_rate=sr16,
        min_speech_duration_ms=v["min_speech_ms"], min_silence_duration_ms=v["min_silence_ms"],
        speech_pad_ms=v["speech_pad_ms"], return_seconds=False,
    )
    speech = sp.merge([(t["start"] / sr16, t["end"] / sr16) for t in ts])
    censor_doc = read_json(paths.interim("censor", f"{call_id}.json"))
    censor = [(s["start"], s["end"]) for s in censor_doc["spans"]]
    x, sr8, info = load_audio(master_path(call_id, ctx.cfg))
    snr, level, counts = snr_and_level(x, sr8, speech, censor, v["frame_ms"])
    speech_s = sp.total(sp.subtract(speech, censor))
    denom = info["duration_file_s"] - censor_doc["censored_s"]
    speech_ratio = speech_s / denom if denom > 0 else 0.0
    flags = []
    if snr is not None and snr < q["low_snr_db"]:
        flags.append("low_snr")
    if speech_ratio < q["no_speech_ratio"]:
        flags.append("no_speech")
    return {
        "model": ctx.models["vad_name"],
        "speech": [{"start": round(s, 3), "end": round(e, 3)} for s, e in speech],
        "speech_s": round(speech_s, 3),
        "speech_ratio": round(speech_ratio, 4),
        "snr_db": None if snr is None else round(snr, 2),
        "level_dbfs": None if level is None else round(level, 2),
        "frame_counts": counts,
        "flags": flags,
    }


def run(calls: list[str] | None = None, force: bool = False):
    return run_per_call("vad", usable_calls(calls), process_call, lambda c: paths.interim("vad", f"{c}.json"),
                        force=force, setup=setup)
