"""E3 · ASR with faster-whisper (GPU process 1; spec §9-E3, §5.4).

The model is loaded once. Per call, on CUDA OOM / unsupported-device errors the call is retried with
`fallback_model` on the same device, then `fallback_model` on CPU; every fallback is recorded in the call JSON and
in `_stage.json`. Post-filters are logged in `removed[]`, never silent.
"""

from __future__ import annotations

import gc
import time
from typing import Any

import soundfile as sf

from . import paths
from . import spans as sp
from .cache import StageContext, read_json, run_per_call
from .config import load_yaml
from .inventory import censor_spans, usable_calls
from .preprocess import wav_path
from .textnorm import normalize

_DEVICE_ERRORS = ("cuda", "out of memory", "cudnn", "cublas", "no kernel", "compute type", "device")
NGRAM_MAX_N = 20


def load_asr(call_id: str) -> dict[str, Any]:
    return read_json(paths.interim("asr", f"{call_id}.json"))


# --------------------------------------------------------------------------- pure post-filters
def collapse_repeats(toks: list[str], n_min: int = 3, repeats: int = 3, n_max: int = NGRAM_MAX_N) -> list[bool]:
    """Keep-mask removing all but the first copy of any n-gram (n >= n_min) repeated >= `repeats` times in a row."""
    keep = [True] * len(toks)
    i, length = 0, len(toks)
    while i < length:
        step = 1
        for n in range(min(n_max, (length - i) // repeats), n_min - 1, -1):
            gram = toks[i : i + n]
            k = 1
            while toks[i + k * n : i + (k + 1) * n] == gram:
                k += 1
            if k >= repeats:
                for j in range(i + n, i + k * n):
                    keep[j] = False
                step = k * n
                break
        i += step
    return keep


def post_filter(segments: list[dict[str, Any]], speech: list[tuple[float, float]], censor: list[tuple[float, float]],
                blacklist: set[str], a: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Returns (kept segments with words, removed items with reasons)."""
    removed: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []
    nonspeech_or_censor_ok = sp.subtract(speech, censor)  # time that counts as usable speech
    for seg in segments:
        reason = None
        if seg["compression_ratio"] > a["compression_ratio_threshold"]:
            reason = "compression_ratio"
        elif normalize(seg["text"]) in blacklist:
            reason = "blacklist"
        else:
            items = seg["words"] or [{"start": seg["start"], "end": seg["end"]}]
            dur = sum(max(w["end"] - w["start"], 0.0) for w in items)
            if dur > 0:
                inside = sum(sp.overlap(w["start"], w["end"], nonspeech_or_censor_ok) for w in items)
                if (dur - inside) / dur > a["nonspeech_drop_fraction"]:
                    reason = "nonspeech_or_censor"
        if reason:
            removed.append({"start": seg["start"], "end": seg["end"], "text": seg["text"], "reason": reason})
        else:
            kept.append(seg)

    # n-gram loop collapse on the call's whole word stream
    stream = [(si, wi) for si, seg in enumerate(kept) for wi in range(len(seg["words"]))]
    toks = [normalize(kept[si]["words"][wi]["word"]) for si, wi in stream]
    keep = collapse_repeats(toks, a["ngram_collapse_n"], a["ngram_collapse_repeats"])
    if not all(keep):
        drop = {pos for pos, k in zip(stream, keep) if not k}
        for si, seg in enumerate(kept):
            dropped = [w for wi, w in enumerate(seg["words"]) if (si, wi) in drop]
            if dropped:
                removed.append({"start": dropped[0]["start"], "end": dropped[-1]["end"],
                                "text": " ".join(w["word"] for w in dropped), "reason": "ngram_repeat"})
                seg["words"] = [w for wi, w in enumerate(seg["words"]) if (si, wi) not in drop]
                seg["text"] = " ".join(w["word"] for w in seg["words"])
        kept = [s for s in kept if s["words"] or s["text"]]
    return kept, removed


def confidence(words: list[dict[str, Any]], low: float) -> tuple[float | None, float | None]:
    if not words:
        return None, None
    durs = [max(w["end"] - w["start"], 0.01) for w in words]
    conf = sum(w["probability"] * d for w, d in zip(words, durs)) / sum(durs)
    low_pct = 100 * sum(w["probability"] < low for w in words) / len(words)
    return round(conf, 4), round(low_pct, 2)


# --------------------------------------------------------------------------- model handling
class ModelPool:
    """Keeps at most one WhisperModel in memory (4 GB VRAM)."""

    def __init__(self, a: dict[str, Any]):
        self.a = a
        self.key: tuple[str, str] | None = None
        self.model = None

    def get(self, name: str, device: str):
        if self.key == (name, device):
            return self.model
        self.model = None
        gc.collect()
        from faster_whisper import WhisperModel

        kwargs: dict[str, Any] = {"device": device, "compute_type": self.a["compute_type"]}
        if device == "cpu":
            kwargs["cpu_threads"] = self.a["cpu_threads"]
        self.model = WhisperModel(name, **kwargs)
        self.key = (name, device)
        return self.model


def attempt_chain(a: dict[str, Any]) -> list[tuple[str, str]]:
    """Same model on CPU before switching models, so every call keeps one ASR model where possible (D-019)."""
    chain = [(a["model"], a["device"])]
    if a["device"] != "cpu":
        chain += [(a["model"], "cpu"), (a["fallback_model"], a["device"]), (a["fallback_model"], "cpu")]
    else:
        chain += [(a["fallback_model"], "cpu")]
    return chain


def transcribe(model, audio, a: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    segments, info = model.transcribe(
        audio, language=a["language"], beam_size=a["beam_size"], word_timestamps=a["word_timestamps"],
        vad_filter=a["vad_filter"], condition_on_previous_text=a["condition_on_previous_text"],
        compression_ratio_threshold=a["compression_ratio_threshold"], log_prob_threshold=a["log_prob_threshold"],
        no_speech_threshold=a["no_speech_threshold"], initial_prompt=a["initial_prompt"],
    )
    out = []
    for s in segments:  # generator: decoding happens here
        out.append({
            "id": s.id, "start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip(),
            "avg_logprob": round(s.avg_logprob, 4), "no_speech_prob": round(s.no_speech_prob, 4),
            "compression_ratio": round(s.compression_ratio, 4), "temperature": s.temperature,
            "words": [{"start": round(w.start, 3), "end": round(w.end, 3), "word": w.word.strip(),
                       "probability": round(w.probability, 4)} for w in (s.words or [])],
        })
    meta = {"language": info.language, "language_probability": round(info.language_probability, 4),
            "duration": info.duration, "duration_after_vad": info.duration_after_vad}
    return out, meta


# --------------------------------------------------------------------------- stage
def setup(ctx: StageContext) -> None:
    a = ctx.cfg["asr"]
    ctx.models["pool"] = ModelPool(a)
    ctx.models["blacklist"] = {normalize(t) for t in load_yaml("lexicons.yaml")["hallucination_blacklist"]}
    ctx.models["asr_model"] = a["model"]
    ctx.models["pool"].get(a["model"], a["device"])  # fail fast on setup problems (logged by runner)


def teardown(ctx: StageContext) -> None:
    pool = ctx.models.pop("pool", None)
    ctx.models.pop("blacklist", None)
    if pool is not None:
        pool.model = None
    gc.collect()


def process_call(call_id: str, ctx: StageContext) -> dict[str, Any]:
    a = ctx.cfg["asr"]
    audio, sr = sf.read(str(wav_path(call_id)), dtype="float32")
    assert sr == 16000, sr
    attempts = []
    t0 = time.time()
    for name, device in attempt_chain(a):
        try:
            model = ctx.models["pool"].get(name, device)
            segments, info = transcribe(model, audio, a)
            attempts.append({"model": name, "device": device, "ok": True})
            break
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            attempts.append({"model": name, "device": device, "ok": False, "error": str(exc)[:300]})
            if not any(k in msg for k in _DEVICE_ERRORS):
                raise
            ctx.logger.warning("%s: %s on %s failed (%s); trying next fallback", call_id, name, device, str(exc)[:120])
            ctx.models["pool"].key = None
            ctx.models["pool"].model = None
            gc.collect()
    else:
        raise RuntimeError(f"all ASR attempts failed: {attempts}")
    elapsed = time.time() - t0
    fallback_used = len(attempts) > 1
    if fallback_used:
        ctx.bump(f"fallback_{name}_{device}")

    vad = read_json(paths.interim("vad", f"{call_id}.json"))
    speech = [(s["start"], s["end"]) for s in vad["speech"]]
    censor = [(s["start"], s["end"]) for s in censor_spans(call_id)]
    kept, removed = post_filter(segments, speech, censor, ctx.models["blacklist"], a)
    words = [{**w, "segment": seg["id"]} for seg in kept for w in seg["words"]]
    conf, low_pct = confidence(words, a["low_conf_word_prob"])
    flags = []
    if removed:
        flags.append("hallucination_removed")
        ctx.bump("calls_with_removals")
    if conf is not None and conf < ctx.cfg["quality"]["low_asr_conf"]:
        flags.append("low_asr_conf")
    duration = len(audio) / sr
    return {
        "model": name, "device": device, "compute_type": a["compute_type"], "fallback_used": fallback_used,
        "attempts": attempts, "info": info,
        "segments": [{k: v for k, v in s.items() if k != "words"} for s in kept],
        "words": words, "removed": removed,
        "asr_confidence": conf, "asr_low_conf_pct": low_pct, "n_words": len(words), "flags": flags,
        "elapsed_s": round(elapsed, 2), "rtf": round(elapsed / duration, 4) if duration else None,
    }


def run(calls: list[str] | None = None, force: bool = False, device: str | None = None):
    """`device` is a runtime override (e.g. cpu while the GPU runs diarization); recorded in `_stage.json` and in each
    call's `device`, not part of the config hash."""

    def _setup(ctx: StageContext) -> None:
        if device:
            ctx.cfg = {**ctx.cfg, "asr": {**ctx.cfg["asr"], "device": device}}
            ctx.stats["device_override"] = device
        setup(ctx)

    return run_per_call("asr", usable_calls(calls), process_call, lambda c: paths.interim("asr", f"{c}.json"),
                        force=force, setup=_setup, teardown=teardown)
