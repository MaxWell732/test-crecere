"""Compare diarization variants on calls where one speaker holds almost all speech (decisions D-023).

For each call: run each variant on the GPU, save segments to data/interim/diarization_variants/<variant>/<call>.json,
and score speaker separation with a transparent text heuristic:
  sentences (ASR words split at . ? ! … or pauses > 1 s) get the majority speaker of their words (exclusive segments);
  sentences are tagged `agent` / `client` by marker phrases (below); score = share of agent sentences on the most
  agent-heavy speaker − share of client sentences on that same speaker (1 = perfect split, 0 = no split).
The heuristic only ranks variants; roles are still annotated from the rendered turns and reviewed by a person (H5).

Usage:  source scripts/env.sh && uv run python scripts/diarization_variants.py C020 C024 ...
"""

from __future__ import annotations

import json
import re
import sys

from extraction import _env

_env.confine_caches()

import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from pyannote.audio import Pipeline  # noqa: E402

from extraction import paths  # noqa: E402
from extraction.cache import write_json  # noqa: E402
from extraction.diarize import _device, annotation_to_segments, derive_exclusive, normalize_labels  # noqa: E402
from extraction.textnorm import normalize  # noqa: E402
from extraction.turns import assign_speakers  # noqa: E402

AGENT = re.compile(r"area de embargos|judicializaciones|alivios financieros|le hablo|le llamo|lo llamo|me comunico|"
                   r"nos comunicamos|estamos llamando|proceso legal|confidencialidad|descuento|propuesta|queremos ofrecerle|"
                   r"le informo|dueñ[oa]s? de|duen[oa]s? de|compr(o|amos) la cartera|centrales de riesgo|me permite|"
                   r"podria confirmarme|me confirma|acuerdo de pago|medios de pago|paz y salvo|pasisalvo|monto total|"
                   r"cuotas? de|tiempo limitado|que tenga (un|muy|buen)")
CLIENT = re.compile(r"\bcon (el|ella) habla\b|\bno tengo\b|\bno puedo\b|\bme pagan\b|desemplead|\bno soy\b|"
                    r"de parte de quien|quien (la|lo|me) (llama|necesita)|\bmi (mama|hermano|esposo|esposa|hija|hijo)\b|"
                    r"\bno me (alcanza|queda|sirve|da)\b|\byo (no|ya|pague|tengo|estoy|quiero)\b|\bestoy (trabajando|ocupad)|"
                    r"\bse me (olvido|paso)\b|\bno he podido\b|"
                    r"^(si|no),? (senor|senora|senorita)$|^(digame|cuenteme|digame digame)$|^bien,? gracias|^muy bien,? gracias")
VARIANTS = {
    "c1_default": ("pyannote/speaker-diarization-community-1", {"min_speakers": 1, "max_speakers": 3}),
    "c1_num2": ("pyannote/speaker-diarization-community-1", {"num_speakers": 2}),
    "v31_num2": ("pyannote/speaker-diarization-3.1", {"num_speakers": 2}),
}


def sentences(words):
    out, cur = [], []
    for i, w in enumerate(words):
        if cur and (words[i - 1]["word"].rstrip().endswith((".", "?", "!", "…")) or w["start"] - words[i - 1]["end"] > 1.0):
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return out


def separation_score(words, exclusive):
    spk = assign_speakers(words, exclusive, 0.5)
    tagged = []
    for idx in sentences(words):
        text = normalize(" ".join(words[i]["word"] for i in idx))
        labels = [spk[i] for i in idx]
        major = max(set(labels), key=labels.count)
        kind = "agent" if AGENT.search(text) else "client" if CLIENT.search(text) else None
        if kind:
            tagged.append((major, kind))
    n_agent = sum(k == "agent" for _, k in tagged)
    n_client = sum(k == "client" for _, k in tagged)
    if not n_agent or not n_client:
        return None, n_agent, n_client
    speakers = {s for s, _ in tagged}
    best = max(speakers, key=lambda s: sum(1 for x, k in tagged if x == s and k == "agent"))
    a_share = sum(1 for x, k in tagged if x == best and k == "agent") / n_agent
    c_share = sum(1 for x, k in tagged if x == best and k == "client") / n_client
    return round(a_share - c_share, 3), n_agent, n_client


def main(calls):
    dev = _device("cuda")
    pipes = {name: Pipeline.from_pretrained(name).to(dev) for name in {name for name, _ in VARIANTS.values()}}
    summary = {}
    for cid in calls:
        y, sr = sf.read(str(paths.interim("audio_16k", f"{cid}.wav")), dtype="float32")
        wav = {"waveform": torch.from_numpy(y).unsqueeze(0), "sample_rate": sr}
        words = json.load(open(paths.interim("asr", f"{cid}.json")))["words"]
        row = {}
        for variant, (name, kwargs) in VARIANTS.items():
            out = pipes[name](wav, **kwargs)
            ann = out.speaker_diarization if hasattr(out, "speaker_diarization") else out
            excl = getattr(out, "exclusive_speaker_diarization", None)
            reg = annotation_to_segments(ann)
            ex = annotation_to_segments(excl) if excl is not None else derive_exclusive(reg)
            reg, ex, _ = normalize_labels(reg, ex)
            score, n_a, n_c = separation_score(words, ex)
            write_json(paths.interim("diarization_variants", variant, f"{cid}.json"),
                       {"call_id": cid, "pipeline": name, "kwargs": kwargs, "regular": reg, "exclusive": ex,
                        "separation_score": score, "agent_sentences": n_a, "client_sentences": n_c})
            row[variant] = score
        valid = {k: v for k, v in row.items() if v is not None}
        best = max(valid, key=valid.get) if valid else None
        summary[cid] = {"scores": row, "best": best}
        print(f"{cid}: {row} -> best {best}", flush=True)
    write_json(paths.interim("diarization_variants", "summary.json"), summary)
    print("DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
