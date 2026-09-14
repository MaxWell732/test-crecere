"""E5 · Fusion & turns (spec §9-E5).

1. Word -> speaker: max overlap with exclusive segments; else nearest within `word_to_segment_max_gap_s`; else UNKNOWN.
2. Speech stretches per speaker = regular diarization ∩ VAD speech − censor; each word attaches to its speaker's
   stretch with max overlap (or nearest within the gap, extending it; else a new stretch). Consecutive stretches of
   the same speaker with no other speaker's stretch starting in between form one *unit* ("speech stretch" of §9-E5).
3. Backchannel: a unit of B while A holds the floor with duration <= backchannel_max_s and <= backchannel_max_words.
   With `backchannel_requires_overlap: false` (literal spec, default) any such unit inside A's floor qualifies;
   with `true` it must start before A's speech so far has ended.
4. Turn: consecutive speech of one speaker until another speaker produces non-backchannel speech.
5. Interruption: B's non-backchannel unit starts while A is speaking, overlap >= interruption_min_overlap_s, and A's
   speech ends before B's does. Latency on floor transfers without overlap: B.start − A.end.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from typing import Any

from . import paths
from . import spans as sp
from .config import load_yaml
from .textnorm import normalize
from .asr import load_asr
from .cache import StageContext, file_fingerprint, read_json, run_per_call
from .diarize import load_diarization
from .inventory import censor_spans, usable_calls

UNKNOWN = "UNKNOWN"


def load_turns(call_id: str) -> dict[str, Any]:
    return read_json(paths.interim("turns", f"{call_id}.json"))


def assign_speakers(words: list[dict[str, Any]], exclusive: list[dict[str, Any]], max_gap: float) -> list[str]:
    ex = sorted(exclusive, key=lambda s: s["start"])
    ends = [s["end"] for s in ex]
    out = []
    for w in words:
        ws, we = w["start"], max(w["end"], w["start"] + 0.01)
        j = bisect_right(ends, ws)
        best, best_ov = None, 0.0
        k = j
        while k < len(ex) and ex[k]["start"] < we:
            ov = min(we, ex[k]["end"]) - max(ws, ex[k]["start"])
            if ov > best_ov:
                best, best_ov = ex[k]["speaker"], ov
            k += 1
        if best is None:
            cands = []
            if j - 1 >= 0:
                cands.append((ws - ex[j - 1]["end"], ex[j - 1]["speaker"]))
            if j < len(ex):
                cands.append((ex[j]["start"] - we, ex[j]["speaker"]))
            cands = [c for c in cands if 0 <= c[0] <= max_gap]
            best = min(cands)[1] if cands else UNKNOWN
        out.append(best)
    return out


_SENTENCE_END = re.compile(r"[.?!…]+[\"')\]]*$")


def smooth_sentence_speakers(words: list[dict[str, Any]], speakers: list[str], t: dict[str, Any],
                             backchannel_tokens: set[str]) -> tuple[list[str], list[dict[str, Any]]]:
    """Reassign short minority-speaker runs inside an ASR sentence to the sentence's majority speaker (D-021).

    Sentences end at . ? ! … or at a pause > smoothing_max_word_gap_s. A run is reassigned only if it lasts
    <= smoothing_max_run_s, holds <= smoothing_max_share of the sentence's words, the sentence has
    >= smoothing_min_words, and the run is not made only of backchannel tokens. Returns (speakers, changes).
    """
    out = list(speakers)
    changes: list[dict[str, Any]] = []
    if not t.get("sentence_smoothing") or not words:
        return out, changes
    sentences, cur = [], [0]
    for i in range(1, len(words)):
        if _SENTENCE_END.search(words[i - 1]["word"]) or words[i]["start"] - words[i - 1]["end"] > t["smoothing_max_word_gap_s"]:
            sentences.append(cur)
            cur = []
        cur.append(i)
    sentences.append(cur)
    for idx in sentences:
        if len(idx) < t["smoothing_min_words"]:
            continue
        labels = [out[i] for i in idx]
        counts = {k: labels.count(k) for k in set(labels)}
        top = sorted(counts.items(), key=lambda kv: -kv[1])
        if len(top) < 2 or top[0][1] == top[1][1] or top[0][0] == UNKNOWN:
            continue
        major = top[0][0]
        j = 0
        while j < len(idx):
            if out[idx[j]] == major:
                j += 1
                continue
            k = j
            while k + 1 < len(idx) and out[idx[k + 1]] != major:
                k += 1
            run = idx[j : k + 1]
            dur = words[run[-1]]["end"] - words[run[0]]["start"]
            only_bc = all(normalize(words[i]["word"]) in backchannel_tokens for i in run)
            if dur <= t["smoothing_max_run_s"] and len(run) / len(idx) <= t["smoothing_max_share"] and not only_bc:
                for i in run:
                    changes.append({"start": words[i]["start"], "end": words[i]["end"], "word": words[i]["word"],
                                    "from": out[i], "to": major})
                    out[i] = major
            j = k + 1
    return out, changes


def build_stretches(words: list[dict[str, Any]], speakers: list[str], regular: list[dict[str, Any]],
                    speech: list[tuple[float, float]], censor: list[tuple[float, float]], max_gap: float,
                    removed: dict[str, list[tuple[float, float]]] | None = None) -> list[dict]:
    """`removed`: per-speaker time spans taken away from that speaker's diarized speech (smoothing / overrides)."""
    stretches: dict[str, list[dict[str, Any]]] = {}
    removed = removed or {}
    for spk in sorted({s["speaker"] for s in regular}):
        own = [(s["start"], s["end"]) for s in regular if s["speaker"] == spk]
        stretches[spk] = [{"speaker": spk, "start": s, "end": e, "words": []}
                          for s, e in sp.subtract(sp.subtract(sp.intersect(own, speech), censor), removed.get(spk, []))]
    unknown: list[dict[str, Any]] = []
    for w, spk in zip(words, speakers):
        if spk == UNKNOWN:
            if unknown and w["start"] - unknown[-1]["end"] <= max_gap:
                unknown[-1]["words"].append(w)
                unknown[-1]["end"] = max(unknown[-1]["end"], w["end"])
            else:
                unknown.append({"speaker": UNKNOWN, "start": w["start"], "end": w["end"], "words": [w]})
            continue
        own = stretches.setdefault(spk, [])
        ws, we = w["start"], max(w["end"], w["start"] + 0.01)
        best, best_ov = None, 0.0
        for st in own:
            ov = min(we, st["end"]) - max(ws, st["start"])
            if ov > best_ov:
                best, best_ov = st, ov
        if best is None:
            near = [(max(st["start"] - we, ws - st["end"]), i) for i, st in enumerate(own)]
            near = [n for n in near if n[0] <= max_gap]
            if near:
                best = own[min(near)[1]]
                best["start"], best["end"] = min(best["start"], w["start"]), max(best["end"], w["end"])
            else:
                best = {"speaker": spk, "start": w["start"], "end": w["end"], "words": []}
                own.append(best)
        best["words"].append(w)
    allst = [st for lst in stretches.values() for st in lst] + unknown
    for st in allst:
        st["words"].sort(key=lambda w: w["start"])
    return sorted(allst, key=lambda s: (s["start"], -(s["end"] - s["start"])))


def group_units(stretches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for st in stretches:
        if units and units[-1]["speaker"] == st["speaker"]:
            u = units[-1]
            u["end"] = max(u["end"], st["end"])
            u["words"].extend(st["words"])
            u["stretches"].append((st["start"], st["end"]))
        else:
            units.append({"speaker": st["speaker"], "start": st["start"], "end": st["end"], "words": list(st["words"]),
                          "stretches": [(st["start"], st["end"])]})
    for u in units:
        u["words"].sort(key=lambda w: w["start"])
    return units


def _text(words: list[dict[str, Any]], censor: list[dict[str, Any]]) -> str:
    items = [(w["start"], w["word"]) for w in words] + [(c["start"], "[CENSURADO]") for c in censor]
    return " ".join(t for _, t in sorted(items, key=lambda x: x[0])).strip()


def build_turns(units: list[dict[str, Any]], censor: list[dict[str, Any]], t: dict[str, Any]) -> dict[str, Any]:
    bc_s, bc_w = t["backchannel_max_s"], t["backchannel_max_words"]
    min_ov, max_gap = t["interruption_min_overlap_s"], t["word_to_segment_max_gap_s"]
    requires_overlap = t.get("backchannel_requires_overlap", False)
    turns: list[dict[str, Any]] = []
    interruptions: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None

    def open_turn(u, is_int=False, latency=None):
        return {"speaker": u["speaker"], "start": u["start"], "end": u["end"], "words": list(u["words"]),
                "stretches": list(u["stretches"]), "is_interruption": is_int, "latency_s": latency,
                "backchannels": [], "censor": []}

    for u in units:
        if cur is None:
            cur = open_turn(u)
            continue
        if u["speaker"] == cur["speaker"]:
            cur["end"] = max(cur["end"], u["end"])
            cur["words"].extend(u["words"])
            cur["stretches"].extend(u["stretches"])
            continue
        short = (u["end"] - u["start"]) <= bc_s and len(u["words"]) <= bc_w
        if short and (not requires_overlap or u["start"] < cur["end"]):
            cur["backchannels"].append({"speaker": u["speaker"], "start": u["start"], "end": u["end"],
                                        "words": u["words"], "text": _text(u["words"], [])})
            continue
        a_end = cur["end"]
        overlap = max(0.0, min(a_end, u["end"]) - u["start"])
        a_speaking = any(s <= u["start"] < e for s, e in cur["stretches"])
        is_int = a_speaking and overlap >= min_ov and a_end < u["end"]
        latency = round(u["start"] - a_end, 3) if u["start"] >= a_end else None
        turns.append(cur)
        cur = open_turn(u, is_int, latency)
        if is_int:
            interruptions.append({"interrupter": u["speaker"], "interrupted": turns[-1]["speaker"],
                                  "start": round(u["start"], 3), "overlap_s": round(overlap, 3),
                                  "interrupted_end": round(a_end, 3), "yield_time_s": round(a_end - u["start"], 3)})
    if cur is not None:
        turns.append(cur)

    # ids + censor placement
    turns.sort(key=lambda x: x["start"])
    int_iter = iter(interruptions)
    for i, tr in enumerate(turns, start=1):
        tr["turn_id"] = f"T{i:03d}"
        tr["words"].sort(key=lambda w: w["start"])
    by_start = {(tr["speaker"], round(tr["start"], 3)): tr["turn_id"] for tr in turns}
    for it in int_iter:
        it["turn_id"] = by_start.get((it["interrupter"], it["start"]))
    unassigned = []
    for c in censor:
        best, best_key = None, None
        for tr in turns:
            ov = min(c["end"], tr["end"]) - max(c["start"], tr["start"])
            dist = 0.0 if ov > 0 else max(tr["start"] - c["end"], c["start"] - tr["end"])
            key = (dist, -ov, tr["start"] > c["start"])
            if dist <= max_gap and (best_key is None or key < best_key):
                best, best_key = tr, key
        (best["censor"] if best else unassigned).append(c)

    out_turns = []
    for tr in turns:
        out_turns.append({
            "turn_id": tr["turn_id"], "speaker": tr["speaker"], "start": round(tr["start"], 3), "end": round(tr["end"], 3),
            "words": [{"w": w["word"], "start": w["start"], "end": w["end"], "prob": w["probability"]} for w in tr["words"]],
            "text": _text(tr["words"], tr["censor"]), "is_interruption": tr["is_interruption"], "latency_s": tr["latency_s"],
            "backchannels": [{"speaker": b["speaker"], "start": round(b["start"], 3), "end": round(b["end"], 3),
                              "text": b["text"], "n_words": len(b["words"]),
                              "words": [{"w": w["word"], "start": w["start"], "end": w["end"], "prob": w["probability"]}
                                        for w in b["words"]]} for b in tr["backchannels"]],
            "censor": tr["censor"],
            "source_turn_ids": sorted({w["src_turn"] for w in tr["words"] if "src_turn" in w}),
        })
    return {"turns": out_turns, "interruptions": interruptions, "unassigned_censor": unassigned}


def find_overlaps(stretches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = sorted(((s["start"], s["end"], s["speaker"]) for s in stretches), key=lambda x: x[0])
    out = []
    for i, (s1, e1, k1) in enumerate(items):
        for s2, e2, k2 in items[i + 1:]:
            if s2 >= e1:
                break
            if k1 != k2 and min(e1, e2) > s2:
                out.append({"start": round(s2, 3), "end": round(min(e1, e2), 3), "speakers": sorted([k1, k2])})
    return out


def fuse(asr_words, exclusive, regular, speech, censor_list, t, backchannel_tokens=None) -> dict[str, Any]:
    censor = [(c["start"], c["end"]) for c in censor_list]
    speakers = assign_speakers(asr_words, exclusive, t["word_to_segment_max_gap_s"])
    if backchannel_tokens is None:
        backchannel_tokens = {normalize(x) for x in load_yaml("lexicons.yaml").get("backchannel_tokens", [])}
    speakers, smoothing = smooth_sentence_speakers(asr_words, speakers, t, backchannel_tokens)
    removed: dict[str, list[tuple[float, float]]] = {}
    for ch in smoothing:  # 0.1 s margin so no sliver of the old speaker's diarized speech is left behind
        removed.setdefault(ch["from"], []).append((ch["start"] - 0.1, ch["end"] + 0.1))
    stretches = build_stretches(asr_words, speakers, regular, speech, censor, t["word_to_segment_max_gap_s"],
                                removed={k: sp.merge(v, gap=0.3) for k, v in removed.items()})
    result = build_turns(group_units(stretches), censor_list, t)
    overlaps = find_overlaps(stretches)
    result["overlaps"] = overlaps
    result["overlap_s"] = round(sp.total([(o["start"], o["end"]) for o in overlaps]), 3)
    result["stats"] = {"n_words": len(asr_words), "n_unknown_words": sum(s == UNKNOWN for s in speakers),
                       "n_stretches": len(stretches), "n_turns": len(result["turns"]),
                       "n_wordless_turns": sum(not tr["words"] for tr in result["turns"]),
                       "n_backchannels": sum(len(tr["backchannels"]) for tr in result["turns"]),
                       "n_smoothed_words": len(smoothing)}
    result["smoothing"] = smoothing
    return result


def process_call(call_id: str, ctx: StageContext) -> dict[str, Any]:
    asr = load_asr(call_id)
    diar = load_diarization(call_id)
    vad = read_json(paths.interim("vad", f"{call_id}.json"))
    speech = [(s["start"], s["end"]) for s in vad["speech"]]
    return fuse(asr["words"], diar["exclusive"], diar["regular"], speech, censor_spans(call_id), ctx.cfg["turns"])


def run(calls: list[str] | None = None, force: bool = False):
    return run_per_call("turns", usable_calls(calls), process_call, lambda c: paths.interim("turns", f"{c}.json"),
                        force=force, extra_key=lambda c: file_fingerprint(
                            *(paths.interim(d, f"{c}.json") for d in ("asr", "diarization", "vad", "censor"))))
