"""E0 · Inventory, QC and censor detection (spec §9-E0, §9.0).

- Anonymous IDs: the file list (sorted by uuid) is shuffled with `seed` -> C001…C100, stored in data/private/id_map.csv.
  The map is created once; later runs verify the raw file set is unchanged instead of re-shuffling.
- Per call: `interim/censor/<call_id>.json` with censor spans + QC fields. No group information anywhere outside
  data/private/.
- `interim/manifest.csv` is rebuilt from all per-call JSONs at the end of every run.
"""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Any

import numpy as np

from . import paths
from . import spans as sp
from .audio_utils import frame_rms_db, frame_to_time, load_audio, stft_power
from .cache import StageContext, read_json, run_per_call
from .config import load_config, sha256_file

MANIFEST_COLUMNS = [
    "call_id", "sha256", "duration_file_s", "codec", "sample_rate", "channels", "clipping_pct", "level_dbfs",
    "censored_s", "n_censor_segments", "flags", "exclude", "exclusion_reason",
]
CENSOR_PRIORITY = ["zeros", "tone", "noise"]


# --------------------------------------------------------------------------- anonymous IDs (private)
def read_id_map() -> list[dict[str, str]]:
    with open(paths.ID_MAP, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def assign_ids(cfg: dict[str, Any]) -> list[dict[str, str]]:
    rows = []
    for gkey, g in cfg["paths"]["raw_groups"].items():
        for f in sorted((paths.RAW / gkey).glob("*.wav")):
            rows.append({"uuid": f.stem, "group": g["group"]})
    if not rows:
        raise FileNotFoundError("no WAV files under data/raw/ — run `extract download` first")
    uuids = [r["uuid"] for r in rows]
    if len(set(uuids)) != len(uuids):
        raise RuntimeError("duplicate uuid across group folders")
    if paths.ID_MAP.exists():
        existing = read_id_map()
        if {(r["uuid"], r["group"]) for r in existing} != {(r["uuid"], r["group"]) for r in rows}:
            raise RuntimeError(
                "raw file set differs from data/private/id_map.csv; refusing to re-assign call_ids "
                "(delete the id map deliberately to re-shuffle)"
            )
        return existing
    rows.sort(key=lambda r: r["uuid"])
    random.Random(cfg["seed"]).shuffle(rows)
    for i, r in enumerate(rows, start=1):
        r["call_id"] = f"C{i:03d}"
    paths.ensure_dir(paths.PRIVATE)
    with open(paths.ID_MAP, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["uuid", "group", "call_id"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["call_id"]))
    return read_id_map()


def master_path(call_id: str, cfg: dict[str, Any] | None = None) -> Path:
    """8 kHz master for a call. Code-level lookup only; never print its return value during annotation."""
    cfg = cfg or load_config()
    key_by_group = {g["group"]: k for k, g in cfg["paths"]["raw_groups"].items()}
    for r in read_id_map():
        if r["call_id"] == call_id:
            return paths.RAW / key_by_group[r["group"]] / f"{r['uuid']}.wav"
    raise KeyError(call_id)


# --------------------------------------------------------------------------- censor detection (pure)
def refine_edges(x: np.ndarray, sr: int, start: float, end: float, search_s: float, ctx_s: float = 0.02,
                 step_s: float = 0.002) -> tuple[float, float]:
    """Move each boundary (within ±search_s) to the strongest power change point (ctx_s windows on either side).

    Spectral smoothing + 32 ms frames bias noise-span edges outward by up to ~20 ms; this re-anchors them on the
    sample-level energy step (decisions D-004).
    """
    csum = np.concatenate(([0.0], np.cumsum(x.astype(np.float64) ** 2)))
    n, ctx = len(x), int(round(ctx_s * sr))

    def pw(a: int, b: int) -> float:
        a, b = max(0, a), min(n, b)
        return (csum[b] - csum[a]) / (b - a) if b > a else 0.0

    def best(t0: float, rising: bool) -> float:
        cands = np.arange(t0 - search_s, t0 + search_s + 1e-9, step_s)
        scores = []
        for t in cands:
            i = int(round(t * sr))
            inside, outside = (pw(i, i + ctx), pw(i - ctx, i)) if rising else (pw(i - ctx, i), pw(i, i + ctx))
            scores.append(inside / (outside + 1e-12))
        return float(cands[int(np.argmax(scores))])

    s, e = best(start, True), best(end, False)
    s, e = max(0.0, s), min(n / sr, e)
    return (s, e) if e > s else (start, end)


def _runs_to_spans(mask: np.ndarray, win: int, hop: int, sr: int, n: int) -> list[tuple[float, float]]:
    return [frame_to_time(i, j - 1, win, hop, sr, n) for i, j in sp.runs(mask)]


def detect_censor(x: np.ndarray, sr: int, c: dict[str, Any]) -> list[dict[str, Any]]:
    """All candidate types (zeros / tone / noise) with overlaps resolved by priority zeros > tone > noise."""
    return resolve_overlaps(detect_candidates(x, sr, c), c["merge_gap_ms"] / 1000)


def select_censor(spans: list[dict[str, Any]], c: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split candidates into censor spans (active types; tones inside an allowed band and frequency-stable) and
    diagnostics (everything else)."""
    active = set(c.get("active_types", CENSOR_PRIORITY))
    keep, diagnostics = [], []
    for s in spans:
        ok = s["type"] in active
        if ok and s["type"] == "tone":
            freqs = c.get("tone_freq_hz")
            if freqs:
                ok = any(abs(s["freq_hz"] - f) <= c["tone_freq_tolerance_hz"] for f in freqs)
            if ok and c.get("tone_max_freq_std_hz") is not None:
                ok = s["freq_std_hz"] <= c["tone_max_freq_std_hz"]
        (keep if ok else diagnostics).append(s)
    return keep, diagnostics


def resolve_overlaps(spans: list[dict[str, Any]], min_piece: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    taken: list[tuple[float, float]] = []
    for t in CENSOR_PRIORITY:
        typed = [s for s in spans if s["type"] == t]
        for span in typed:
            for s, e in sp.subtract([(span["start"], span["end"])], taken):
                if e - s >= min_piece:
                    out.append({**span, "start": round(s, 3), "end": round(e, 3)})
        taken = sp.merge(taken + [(s["start"], s["end"]) for s in typed])
    return sorted(out, key=lambda s: s["start"])


def detect_candidates(x: np.ndarray, sr: int, c: dict[str, Any]) -> list[dict[str, Any]]:
    """Candidate spans of type zeros / tone / noise (§9-E0 detectors), overlaps not resolved."""
    n = len(x)
    gap = c["merge_gap_ms"] / 1000
    found: dict[str, list[dict[str, Any]]] = {t: [] for t in CENSOR_PRIORITY}

    # zeros: exact-zero sample runs
    min_run = int(round(c["min_zero_run_ms"] * sr / 1000))
    for i, j in sp.runs(x == 0.0):
        if j - i >= min_run:
            found["zeros"].append({"start": i / sr, "end": j / sr})

    win = int(round(c["stft_win_ms"] * sr / 1000))
    hop = int(round(c["stft_hop_ms"] * sr / 1000))
    if n >= win:
        power, db = frame_rms_db(x, win, hop)
        spec = stft_power(x, win, hop)
        n_frames, n_bins = spec.shape
        valid = db > c["energy_floor_dbfs"]
        bin_hz = sr / win

        # tone: strongest peak (±tol bins) holds >= ratio of frame energy, stable bin
        tol = int(c["tone_bin_tolerance"])
        kmin = int(math.ceil(c["tone_min_freq_hz"] / bin_hz))
        peak = np.argmax(spec[:, kmin:], axis=1) + kmin
        csum = np.concatenate([np.zeros((n_frames, 1)), np.cumsum(spec, axis=1)], axis=1)
        rows = np.arange(n_frames)
        lo, hi = np.maximum(peak - tol, 0), np.minimum(peak + tol + 1, n_bins)
        ratio = (csum[rows, hi] - csum[rows, lo]) / (spec.sum(axis=1) + 1e-20)
        cand = valid & (ratio >= c["tone_peak_ratio"])
        stable = np.zeros(n_frames, dtype=bool)
        stable[1:] = np.abs(np.diff(peak)) <= tol
        # parabolic interpolation of the peak on log power -> fine frequency per frame (stability test, D-016)
        km, kp = np.clip(peak - 1, 0, n_bins - 1), np.clip(peak + 1, 0, n_bins - 1)
        la, lb, lc = (np.log(spec[rows, k] + 1e-20) for k in (km, peak, kp))
        denom = la - 2 * lb + lc
        delta = np.where(np.abs(denom) > 1e-12, 0.5 * (la - lc) / np.where(np.abs(denom) > 1e-12, denom, 1.0), 0.0)
        fine_hz = (peak + np.clip(delta, -0.5, 0.5)) * bin_hz
        tone_runs: list[dict[str, Any]] = []
        for i, j in sp.runs(cand):
            seg_start = i
            for k in range(i + 1, j + 1):
                if k == j or not stable[k]:
                    s, e = frame_to_time(seg_start, k - 1, win, hop, sr, n)
                    tone_runs.append({"start": s, "end": e, "freq_hz": float(np.median(peak[seg_start:k]) * bin_hz),
                                      "_f0": seg_start, "_f1": k - 1})
                    seg_start = k
        merged: list[dict[str, Any]] = []
        for r in sorted(tone_runs, key=lambda r: r["start"]):
            if merged and r["start"] - merged[-1]["end"] <= gap and abs(r["freq_hz"] - merged[-1]["freq_hz"]) <= (tol + 0.5) * bin_hz:
                merged[-1]["end"] = max(merged[-1]["end"], r["end"])
                merged[-1]["_f1"] = max(merged[-1]["_f1"], r["_f1"])
            else:
                merged.append(dict(r))
        for r in merged:
            idx = np.arange(r.pop("_f0"), r.pop("_f1") + 1)
            idx = idx[cand[idx]]
            amp = np.sqrt(power[idx])
            r["freq_hz"] = round(float(np.median(fine_hz[idx])), 2)
            r["freq_std_hz"] = round(float(np.std(fine_hz[idx])), 3)
            r["amp_cv"] = round(float(np.std(amp) / (np.mean(amp) + 1e-20)), 4)
        found["tone"] = [r for r in merged if (r["end"] - r["start"]) * 1000 >= c["tone_min_ms"]]

        # noise: high spectral flatness (smoothed spectrum) with energy above the median active-frame energy
        m = int(c["noise_smooth_frames"])
        kernel = np.ones(m) / m
        smooth = np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="same"), 0, spec[:, 1:])
        flat = np.exp(np.mean(np.log(smooth + 1e-20), axis=1)) / (np.mean(smooth, axis=1) + 1e-20)
        ref = float(np.median(power[valid])) if valid.any() else math.inf
        cand_n = valid & (flat >= c["noise_flatness"]) & (power > ref)
        noise_spans = sp.merge(_runs_to_spans(cand_n, win, hop, sr, n), gap=gap)
        noise_spans = [refine_edges(x, sr, s, e, search_s=win / sr) for s, e in noise_spans]
        found["noise"] = [{"start": s, "end": e} for s, e in noise_spans if (e - s) * 1000 >= c["noise_min_ms"]]

    out = [{**s, "start": round(s["start"], 3), "end": round(s["end"], 3), "type": t}
           for t in CENSOR_PRIORITY for s in found[t]]
    return sorted(out, key=lambda s: s["start"])


def summarize_censor(spans: list[dict[str, Any]]) -> dict[str, Any]:
    by_type = {t: round(sum(s["end"] - s["start"] for s in spans if s["type"] == t), 3) for t in CENSOR_PRIORITY}
    return {"censored_s": round(sum(by_type.values()), 3), "n_censor_segments": len(spans), "censored_by_type_s": by_type}


def audio_qc(x: np.ndarray, sr: int, censor_spans: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    """Pre-VAD QC. `level_dbfs_active` = RMS of non-censored frames above -50 dBFS (E2 supersedes with speech frames)."""
    clip = float(np.mean(np.abs(x) >= cfg["audio"]["clipping_threshold"]) * 100) if len(x) else 0.0
    win = int(round(0.032 * sr))
    power, db = frame_rms_db(x, win, win)
    frame_rate = sr / win
    in_censor = sp.to_mask([(s["start"], s["end"]) for s in censor_spans], len(power), frame_rate)
    active = (db > -50) & ~in_censor
    level = float(10 * np.log10(np.mean(power[active]))) if np.any(active) else None
    return {
        "clipping_pct": round(clip, 4),
        "level_dbfs_active": None if level is None else round(level, 2),
        "peak_abs": float(np.max(np.abs(x))) if len(x) else 0.0,
        "n_active_frames": int(np.sum(active)),
    }


# --------------------------------------------------------------------------- stage
def process_call(call_id: str, ctx: StageContext) -> dict[str, Any]:
    cfg = ctx.cfg
    path = master_path(call_id, cfg)
    base: dict[str, Any] = {"sha256": sha256_file(path), "flags": [], "exclude": False, "exclusion_reason": None}
    try:
        x, sr, info = load_audio(path)
    except Exception as exc:  # noqa: BLE001 — corrupt file is data, not a crash
        ctx.bump("corrupt")
        return {**base, "spans": [], "diagnostic_spans": [], "qc": {}, "exclude": True,
                "exclusion_reason": f"corrupt: {type(exc).__name__}", **summarize_censor([])}
    if info["channels"] != 1:
        base["flags"].append("stereo_downmixed")
    if sr != cfg["audio"]["master_sr"]:
        base["flags"].append(f"sample_rate_{sr}")
    c = cfg["censor"]
    active, diagnostics = select_censor(detect_candidates(x, sr, c), c)
    spans = resolve_overlaps(active, c["merge_gap_ms"] / 1000)
    summary = summarize_censor(spans)
    base["diagnostic_spans"] = diagnostics
    for d in diagnostics:
        ctx.bump(f"diagnostic_{d['type']}_spans")
    qc = {**info, **audio_qc(x, sr, spans, cfg)}
    q = cfg["quality"]
    if qc["duration_file_s"] > 0 and summary["censored_s"] / qc["duration_file_s"] * 100 > q["censor_heavy_pct"]:
        base["flags"].append("censor_heavy")
    if qc["clipping_pct"] > q["high_clipping_pct"]:
        base["flags"].append("high_clipping")
    if qc["peak_abs"] < 1e-4 or qc["n_active_frames"] == 0:
        base.update(exclude=True, exclusion_reason="no_audio")
    for t, v in summary["censored_by_type_s"].items():
        if v:
            ctx.bump(f"calls_with_{t}")
    return {**base, "spans": spans, "qc": qc, **summary}


def write_manifest(call_ids: list[str]) -> None:
    rows = []
    for cid in sorted(call_ids):
        p = paths.interim("censor", f"{cid}.json")
        if not p.exists():
            rows.append({"call_id": cid, "exclude": True, "exclusion_reason": "inventory_failed"})
            continue
        d = read_json(p)
        qc = d.get("qc", {})
        rows.append({
            "call_id": cid, "sha256": d["sha256"], "duration_file_s": round(qc.get("duration_file_s", 0.0), 3),
            "codec": qc.get("codec"), "sample_rate": qc.get("sample_rate"), "channels": qc.get("channels"),
            "clipping_pct": qc.get("clipping_pct"), "level_dbfs": qc.get("level_dbfs_active"),
            "censored_s": d["censored_s"], "n_censor_segments": d["n_censor_segments"],
            "flags": ";".join(d["flags"]), "exclude": d["exclude"], "exclusion_reason": d["exclusion_reason"] or "",
        })
    paths.ensure_dir(paths.INTERIM)
    with open(paths.MANIFEST, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def run(calls: list[str] | None = None, force: bool = False):
    cfg = load_config()
    id_rows = assign_ids(cfg)
    all_ids = [r["call_id"] for r in id_rows]
    todo = [c for c in all_ids if not calls or c in calls]
    result = run_per_call("inventory", todo, process_call, lambda c: paths.interim("censor", f"{c}.json"), force=force)
    write_manifest(all_ids)
    return result


# --------------------------------------------------------------------------- helpers for downstream stages
def read_manifest() -> list[dict[str, str]]:
    if not paths.MANIFEST.exists():
        raise FileNotFoundError("data/interim/manifest.csv missing — run `extract inventory` first")
    with open(paths.MANIFEST, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def usable_calls(calls: list[str] | None = None) -> list[str]:
    ids = [r["call_id"] for r in read_manifest() if r["exclude"] != "True"]
    return [c for c in ids if not calls or c in calls]


def censor_spans(call_id: str) -> list[dict[str, Any]]:
    return read_json(paths.interim("censor", f"{call_id}.json"))["spans"]
