"""E8b · Prosody with praat-parselmouth on the 8 kHz master (spec §9-E8b, §10.2 K).

Valid segments per role: that role's speech − other roles' speech (overlap) − censor, each >= min_segment_s.
Pitch: F0 (floor/ceiling from config) -> semitones re st_ref_hz. Intensity (dB) sampled on voiced pitch frames.
No jitter/shimmer/HNR (telephony codec; spec §13).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import paths
from .. import spans as sp
from ..audio_utils import load_audio
from ..cache import StageContext, file_fingerprint, run_per_call
from ..canonical import load_canonical
from ..inventory import master_path
from ..roles import calls_with_turns

ROLES = ("AGENT", "CLIENT")


def valid_segments(doc: dict[str, Any], role: str, min_s: float) -> list[tuple[float, float]]:
    own = [tuple(s) for s in doc["speech_by_role"].get(role, [])]
    others = [tuple(s) for r, v in doc["speech_by_role"].items() if r != role for s in v]
    censor = [(c["start"], c["end"]) for c in doc["censor"]]
    return [(s, e) for s, e in sp.subtract(sp.subtract(own, others), censor) if e - s >= min_s]


def voiced_frames(snd, segments: list[tuple[float, float]], p: dict[str, Any]) -> dict[str, np.ndarray]:
    """Arrays of (time, semitone, intensity_db) for voiced frames across segments."""
    times, st, inten = [], [], []
    for s, e in segments:
        part = snd.extract_part(from_time=s, to_time=e, preserve_times=True)
        pitch = part.to_pitch(time_step=0.01, pitch_floor=p["f0_floor_hz"], pitch_ceiling=p["f0_ceiling_hz"])
        f0 = pitch.selected_array["frequency"]
        ts = pitch.xs()
        voiced = f0 > 0
        if not voiced.any():
            continue
        intensity = part.to_intensity(minimum_pitch=p["f0_floor_hz"], time_step=0.01)
        ix, iv = intensity.xs(), intensity.values[0]
        if len(ix) == 0:
            continue
        times.append(ts[voiced])
        st.append(12 * np.log2(f0[voiced] / p["st_ref_hz"]))
        inten.append(np.interp(ts[voiced], ix, iv))
    if not times:
        return {"t": np.array([]), "st": np.array([]), "db": np.array([])}
    return {"t": np.concatenate(times), "st": np.concatenate(st), "db": np.concatenate(inten)}


def escalation(doc: dict[str, Any], frames: dict[str, np.ndarray], f: dict[str, Any]) -> tuple[float | None, int]:
    conv = doc["conversation"]
    if conv["conv_start"] is None or frames["t"].size < 2 or conv["conv_duration_s"] <= 0:
        return None, 0
    z_st = (frames["st"] - frames["st"].mean()) / (frames["st"].std() or 1.0)
    z_db = (frames["db"] - frames["db"].mean()) / (frames["db"].std() or 1.0)
    combined = (z_st + z_db) / 2
    xs, ys = [], []
    for t in doc["turns"]:
        if t["role"] != "CLIENT" or t["end"] - t["start"] < f["escalation_min_turn_s"]:
            continue
        sel = (frames["t"] >= t["start"]) & (frames["t"] < t["end"])
        if not sel.any():
            continue
        mid = (t["start"] + t["end"]) / 2
        xs.append((mid - conv["conv_start"]) / conv["conv_duration_s"])
        ys.append(float(combined[sel].mean()))
    if len(xs) < f["escalation_min_turns"]:
        return None, len(xs)
    slope = float(np.polyfit(np.array(xs), np.array(ys), 1)[0])
    return round(slope, 4), len(xs)


def process_call(call_id: str, ctx: StageContext) -> dict[str, Any]:
    import parselmouth

    p, f = ctx.cfg["prosody"], ctx.cfg["features"]
    doc = load_canonical(call_id)
    x, sr, _ = load_audio(master_path(call_id, ctx.cfg))
    snd = parselmouth.Sound(x.astype(np.float64), sampling_frequency=sr)
    out: dict[str, Any] = {}
    client_frames = None
    for role in ROLES:
        segs = valid_segments(doc, role, p["min_segment_s"])
        fr = voiced_frames(snd, segs, p)
        n = int(fr["st"].size)
        out[role.lower()] = {
            "n_segments": len(segs), "seconds": round(sum(e - s for s, e in segs), 3), "n_voiced_frames": n,
            "pitch_range_st": round(float(np.percentile(fr["st"], 90) - np.percentile(fr["st"], 10)), 3) if n else None,
            "f0_median_st": round(float(np.median(fr["st"])), 3) if n else None,
            "intensity_std_db": round(float(np.std(fr["db"])), 3) if n else None,
        }
        if role == "CLIENT":
            client_frames = fr
    esc, n_esc = escalation(doc, client_frames, f)
    return {
        "agent_pitch_range_st": out["agent"]["pitch_range_st"], "client_pitch_range_st": out["client"]["pitch_range_st"],
        "agent_intensity_std_db": out["agent"]["intensity_std_db"], "client_escalation": esc,
        "n_escalation_turns": n_esc, "by_role": out, "parselmouth_version": parselmouth.VERSION,
    }


def run(calls: list[str] | None = None, force: bool = False):
    ids = [c for c in calls_with_turns(calls) if paths.interim("canonical", f"{c}.json").exists()]
    return run_per_call("prosody", ids, process_call, lambda c: paths.interim("features", "prosody", f"{c}.json"),
                        force=force, extra_key=lambda c: file_fingerprint(paths.interim("canonical", f"{c}.json")))
