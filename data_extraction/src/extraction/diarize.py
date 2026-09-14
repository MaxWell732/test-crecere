"""E4 · Diarization with pyannote (GPU process 2; spec §9-E4).

Audio is always passed in memory ({"waveform": (1, n) tensor, "sample_rate": 16000}).
`exclusive` comes from the pipeline when provided (community-1); otherwise it is derived: overlapped time goes to the
speaker whose covering segment is longer. Labels are normalized to SPK_00.. by first appearance.
"""

from __future__ import annotations

import gc
import time
from typing import Any

import soundfile as sf

from . import paths
from . import spans as sp
from .cache import StageContext, read_json, run_per_call
from .inventory import censor_spans, usable_calls
from .preprocess import wav_path


def load_diarization(call_id: str) -> dict[str, Any]:
    return read_json(paths.interim("diarization", f"{call_id}.json"))


# --------------------------------------------------------------------------- pure helpers
def derive_exclusive(regular: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One speaker at a time; overlapped time -> speaker whose covering segment is longer."""
    bounds = sorted({b for seg in regular for b in (seg["start"], seg["end"])})
    pieces: list[dict[str, Any]] = []
    for s, e in zip(bounds, bounds[1:]):
        if e <= s:
            continue
        covering = [seg for seg in regular if seg["start"] <= s and seg["end"] >= e]
        if not covering:
            continue
        winner = max(covering, key=lambda seg: (seg["end"] - seg["start"], -seg["start"]))
        if pieces and pieces[-1]["speaker"] == winner["speaker"] and abs(pieces[-1]["end"] - s) < 1e-9:
            pieces[-1]["end"] = e
        else:
            pieces.append({"start": s, "end": e, "speaker": winner["speaker"]})
    return pieces


def normalize_labels(regular: list[dict[str, Any]], exclusive: list[dict[str, Any]]) -> tuple[list, list, dict]:
    order: dict[str, str] = {}
    for seg in sorted(regular + exclusive, key=lambda s: s["start"]):
        if seg["speaker"] not in order:
            order[seg["speaker"]] = f"SPK_{len(order):02d}"

    def relabel(items):
        return [{"start": round(s["start"], 3), "end": round(s["end"], 3), "speaker": order[s["speaker"]]}
                for s in sorted(items, key=lambda s: (s["start"], s["end"]))]

    return relabel(regular), relabel(exclusive), order


def annotation_to_segments(annotation) -> list[dict[str, Any]]:
    return [{"start": float(t.start), "end": float(t.end), "speaker": str(label)}
            for t, _, label in annotation.itertracks(yield_label=True)]


# --------------------------------------------------------------------------- stage
def cuda_kernels_ok() -> bool:
    """GPU usable for torch models: device capability kernels present (6.1 runs sm_60/sm_61) and a cuDNN op executes."""
    import torch

    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(0)
    archs = torch.cuda.get_arch_list()
    if f"sm_{major}{minor}" not in archs and not any(a.startswith(f"sm_{major}") and int(a[3:]) <= major * 10 + minor
                                                     for a in archs if a[3:].isdigit()):
        return False
    try:
        conv = torch.nn.Conv1d(1, 4, 9).to("cuda")
        conv(torch.randn(1, 1, 64, device="cuda"))
        torch.cuda.synchronize()
        return True
    except Exception:  # noqa: BLE001
        return False


def _device(cfg_device: str):
    import torch

    if cfg_device == "cuda" and cuda_kernels_ok():
        return torch.device("cuda")
    return torch.device("cpu")


def setup(ctx: StageContext) -> None:
    from pyannote.audio import Pipeline

    d = ctx.cfg["diarization"]
    errors = []
    for name in (d["pipeline"], d["fallback_pipeline"]):
        try:
            pipe = Pipeline.from_pretrained(name)
            if pipe is None:
                raise RuntimeError("from_pretrained returned None (gated model not accessible? see H2)")
            dev = _device(d["device"])
            pipe.to(dev)
            ctx.models.update(pipeline=pipe, pipeline_name=name, device=str(dev))
            if name != d["pipeline"]:
                ctx.bump("fallback_pipeline")
            ctx.logger.info("diarization pipeline %s on %s", name, dev)
            return
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {type(exc).__name__}: {str(exc)[:300]}")
            ctx.logger.warning("could not load %s: %s", name, str(exc)[:300])
    raise RuntimeError("no diarization pipeline could be loaded (H2: HF token + accepted terms?) | " + " | ".join(errors))


def teardown(ctx: StageContext) -> None:
    ctx.models.pop("pipeline", None)
    ctx.models.pop("extra_pipelines", None)
    gc.collect()


def load_overrides() -> dict[str, dict[str, Any]]:
    """Per-call diarization overrides (configs/diarization_overrides.yaml: {call_id: {pipeline?, num_speakers?, reason}})."""
    p = paths.CONFIGS / "diarization_overrides.yaml"
    if not p.exists():
        return {}
    import yaml

    return (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("overrides", {}) or {}


def override_key(call_id: str) -> str:
    ov = load_overrides().get(call_id)
    if not ov:
        return ""
    import hashlib
    import json

    return "ov" + hashlib.sha256(json.dumps(ov, sort_keys=True).encode()).hexdigest()[:10]


def _get_pipeline(ctx: StageContext, name: str):
    """Default pipeline comes from setup; override pipelines are loaded once, lazily."""
    if name == ctx.models["pipeline_name"]:
        return ctx.models["pipeline"]
    extra = ctx.models.setdefault("extra_pipelines", {})
    if name not in extra:
        from pyannote.audio import Pipeline

        extra[name] = Pipeline.from_pretrained(name).to(_device(ctx.cfg["diarization"]["device"]))
        ctx.logger.info("loaded override pipeline %s", name)
    return extra[name]


def _run_pipeline(ctx: StageContext, pipeline, waveform, kwargs: dict[str, Any], d: dict[str, Any]):
    import torch

    try:
        return pipeline({"waveform": waveform, "sample_rate": 16000}, **kwargs)
    except torch.cuda.OutOfMemoryError:
        ctx.logger.warning("CUDA OOM in diarization; retrying this call on CPU")
        ctx.bump("cpu_retry")
        pipeline.to(torch.device("cpu"))
        torch.cuda.empty_cache()
        try:
            return pipeline({"waveform": waveform, "sample_rate": 16000}, **kwargs)
        finally:
            pipeline.to(_device(d["device"]))


def process_call(call_id: str, ctx: StageContext) -> dict[str, Any]:
    import torch

    d = ctx.cfg["diarization"]
    override = load_overrides().get(call_id) or {}
    pipeline_name = override.get("pipeline", ctx.models["pipeline_name"])
    pipeline = _get_pipeline(ctx, pipeline_name)
    if override.get("num_speakers"):
        kwargs: dict[str, Any] = {"num_speakers": int(override["num_speakers"])}
    else:
        kwargs = {"min_speakers": override.get("min_speakers", d["min_speakers"]),
                  "max_speakers": override.get("max_speakers", d["max_speakers"])}
    if override:
        ctx.bump("overridden_calls")
    y, sr = sf.read(str(wav_path(call_id)), dtype="float32")
    waveform = torch.from_numpy(y).unsqueeze(0)
    t0 = time.time()
    output = _run_pipeline(ctx, pipeline, waveform, kwargs, d)
    elapsed = time.time() - t0

    if hasattr(output, "speaker_diarization"):  # pyannote.audio 4.x DiarizeOutput
        regular = annotation_to_segments(output.speaker_diarization)
        excl_ann = getattr(output, "exclusive_speaker_diarization", None)
        exclusive = annotation_to_segments(excl_ann) if excl_ann is not None else derive_exclusive(regular)
        derived = excl_ann is None
    else:  # pyannote.audio 3.x Annotation
        regular = annotation_to_segments(output)
        exclusive, derived = derive_exclusive(regular), True
    regular, exclusive, label_map = normalize_labels(regular, exclusive)

    vad = read_json(paths.interim("vad", f"{call_id}.json"))
    speech = [(s["start"], s["end"]) for s in vad["speech"]]
    censor = [(s["start"], s["end"]) for s in censor_spans(call_id)]
    speakers = {}
    for spk in sorted({s["speaker"] for s in regular}):
        own = [(s["start"], s["end"]) for s in regular if s["speaker"] == spk]
        speakers[spk] = {"regular_s": round(sp.total(own), 3),
                         "speech_s": round(sp.total(sp.subtract(sp.intersect(own, speech), censor)), 3)}
    n = len(speakers)
    flags = ["one_speaker"] if n == 1 else ["three_speakers"] if n >= 3 else []
    return {
        "pipeline": pipeline_name, "device": ctx.models["device"], "exclusive_derived": derived,
        "speaker_args": kwargs, "override": override or None,
        "regular": regular, "exclusive": exclusive, "speakers": speakers, "n_speakers": n,
        "raw_label_map": label_map, "flags": flags,
        "elapsed_s": round(elapsed, 2), "rtf": round(elapsed / (len(y) / sr), 4) if len(y) else None,
    }


def run(calls: list[str] | None = None, force: bool = False):
    return run_per_call("diarize", usable_calls(calls), process_call,
                        lambda c: paths.interim("diarization", f"{c}.json"), force=force, setup=setup, teardown=teardown,
                        extra_key=override_key)
