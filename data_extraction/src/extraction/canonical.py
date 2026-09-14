"""E7 · Canonical JSON + rendered transcripts (spec §9-E7).

Refuses any call whose role review (H5) is incomplete. Turns are rebuilt at *role* level with the E5 algorithm: speakers
mapped to the same role are fused (e.g. one agent split into two diarization clusters), and accepted suspect-turn
reassignments move that turn's words to the proposed role (decisions D-011). E5 turn ids are kept in
`source_turn_ids`. No group field anywhere.
"""

from __future__ import annotations

import re
from typing import Any

from . import paths
from . import spans as sp
from .asr import load_asr
from .cache import StageContext, file_fingerprint, read_json, run_per_call
from .config import load_yaml
from .diarize import load_diarization
from . import roles
from .render import NOTATION, fmt_ts, render_line, turn_text
from .turns import build_stretches, build_turns, find_overlaps, group_units, load_turns

NON_CONVERSATION = {"SYSTEM"}


def canonical_path(call_id: str):
    return paths.interim("canonical", f"{call_id}.json")


def transcript_path(call_id: str):
    return paths.LLM_INPUTS / "calls" / f"{call_id}.txt"


def load_canonical(call_id: str) -> dict[str, Any]:
    return read_json(canonical_path(call_id))


def canonical_turn_text(turn: dict[str, Any], cfg: dict[str, Any]) -> str:
    return turn_text(turn, lambda r: r, cfg["asr"]["low_conf_word_prob"], mask=turn["role"] == "AGENT",
                     bc_speaker_key="role")


def role_fusion(tdoc: dict[str, Any], diar: dict[str, Any], speech: list[tuple[float, float]],
                censor_list: list[dict[str, Any]], mapping: dict[str, Any], t: dict[str, Any]):
    speaker_map, overrides = mapping["speaker_map"], mapping["overrides"]

    def role_of(spk: str) -> str:
        return speaker_map.get(spk, "UNKNOWN")

    tagged: list[tuple[dict[str, Any], str]] = []
    removed_from: dict[str, list[tuple[float, float]]] = {}
    for tr in tdoc["turns"]:
        role = overrides.get(tr["turn_id"], role_of(tr["speaker"]))
        if tr["turn_id"] in overrides:
            removed_from.setdefault(role_of(tr["speaker"]), []).append((tr["start"], tr["end"]))
        for w in tr["words"]:
            tagged.append(({"word": w["w"], "start": w["start"], "end": w["end"], "probability": w["prob"],
                            "src_turn": tr["turn_id"]}, role))
        for b in tr["backchannels"]:
            for w in b.get("words", []):
                tagged.append(({"word": w["w"], "start": w["start"], "end": w["end"], "probability": w["prob"],
                                "src_turn": tr["turn_id"]}, role_of(b["speaker"])))
    for ch in tdoc.get("smoothing", []):  # E5 sentence smoothing moved these words away from that speaker (D-021)
        if role_of(ch["from"]) != role_of(ch["to"]):
            removed_from.setdefault(role_of(ch["from"]), []).append((ch["start"] - 0.1, ch["end"] + 0.1))
    removed_from = {r: sp.merge(v, gap=0.3) for r, v in removed_from.items()}
    tagged.sort(key=lambda x: x[0]["start"])
    words, word_roles = [w for w, _ in tagged], [r for _, r in tagged]

    per_role: dict[str, list[tuple[float, float]]] = {}
    for seg in diar["regular"]:
        per_role.setdefault(role_of(seg["speaker"]), []).append((seg["start"], seg["end"]))
    regular = [{"start": s, "end": e, "speaker": role}
               for role, lst in per_role.items()
               for s, e in sp.subtract(sp.merge(lst), removed_from.get(role, []))]
    censor = [(c["start"], c["end"]) for c in censor_list]
    stretches = build_stretches(words, word_roles, regular, speech, censor, t["word_to_segment_max_gap_s"])
    result = build_turns(group_units(stretches), censor_list, t)
    speech_by_role: dict[str, list[tuple[float, float]]] = {}
    for st in stretches:
        speech_by_role.setdefault(st["speaker"], []).append((st["start"], st["end"]))
    speech_by_role = {r: [[round(s, 3), round(e, 3)] for s, e in sp.merge(v)] for r, v in sorted(speech_by_role.items())}
    return result, find_overlaps(stretches), speech_by_role


def render_transcript(call_id: str, doc: dict[str, Any], cfg: dict[str, Any]) -> str:
    conv = doc["conversation"]
    per_role: dict[str, float] = {}
    for t in doc["turns"]:
        per_role[t["role"]] = per_role.get(t["role"], 0.0) + (t["end"] - t["start"])
    present = ", ".join(f"{r} ({s:.1f} s in turns)" for r, s in sorted(per_role.items())) or "none"
    lines = [
        f"# call_id: {call_id}",
        f"# conversation duration: {fmt_ts(conv['conv_duration_s'] or 0.0)} (first to last non-SYSTEM speech)",
        f"# speakers present: {present}",
        NOTATION,
        "",
    ]
    lines += [render_line(t, t["role"], canonical_turn_text(t, cfg)) for t in doc["turns"]]
    return "\n".join(lines) + "\n"


def pii_scan(doc: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, str]]:
    pats = {k: re.compile(v) for k, v in load_yaml("lexicons.yaml")["pii_patterns"].items()}
    hits = []
    for t in doc["turns"]:
        text = canonical_turn_text(t, cfg)
        for kind, pat in pats.items():
            for m in pat.finditer(text):
                hits.append({"turn_id": t["turn_id"], "kind": kind, "match": m.group(0)})
    return hits


def process_call(call_id: str, ctx: StageContext) -> dict[str, Any]:
    cfg = ctx.cfg
    ann_path = roles.annotation_path(call_id)
    if not ann_path.exists():
        raise roles.NotApproved(f"{call_id}: no role annotation")
    ann = read_json(ann_path)
    errs = roles.validate_doc(ann, call_id, cfg)
    if errs:
        raise ValueError(f"{call_id}: role annotation invalid: {errs[:3]}")
    mapping = roles.approved_mapping(call_id, ann)

    tdoc, diar, asr = load_turns(call_id), load_diarization(call_id), load_asr(call_id)
    vad = read_json(paths.interim("vad", f"{call_id}.json"))
    cen = read_json(paths.interim("censor", f"{call_id}.json"))
    speech = [(s["start"], s["end"]) for s in vad["speech"]]
    result, overlaps, speech_by_role = role_fusion(tdoc, diar, speech, cen["spans"], mapping, cfg["turns"])

    turns = [{
        "turn_id": tr["turn_id"], "role": tr["speaker"], "start": tr["start"], "end": tr["end"], "text": tr["text"],
        "words": tr["words"],
        "backchannels": [{"role": b["speaker"], "start": b["start"], "end": b["end"], "text": b["text"],
                          "n_words": b["n_words"], "words": b["words"]} for b in tr["backchannels"]],
        "is_interruption": tr["is_interruption"], "latency_s": tr["latency_s"], "censor": tr["censor"],
        "source_turn_ids": tr["source_turn_ids"],
    } for tr in result["turns"]]
    conv_turns = [t for t in turns if t["role"] not in NON_CONVERSATION]
    conv_start = min((t["start"] for t in conv_turns), default=None)
    conv_end = max((t["end"] for t in conv_turns), default=None)
    duration = cen["qc"]["duration_file_s"]
    censor = [(c["start"], c["end"]) for c in cen["spans"]]
    nonspeech = sp.subtract(sp.complement(speech, 0.0, duration), censor)

    flags = set(cen["flags"]) | set(vad["flags"]) | set(asr["flags"]) | set(diar["flags"]) | set(mapping["flags"])
    if conv_turns and conv_turns[0]["role"] == "AGENT":
        flags.add("agent_spoke_first")

    doc: dict[str, Any] = {
        "audio": {
            "duration_file_s": duration, "snr_db": vad["snr_db"], "clipping_pct": cen["qc"]["clipping_pct"],
            "level_dbfs": vad["level_dbfs"], "speech_ratio": vad["speech_ratio"], "censored_s": cen["censored_s"],
            "n_censor_segments": cen["n_censor_segments"],
        },
        "asr": {"model": asr["model"], "device": asr["device"], "fallback_used": asr["fallback_used"],
                "asr_confidence": asr["asr_confidence"], "asr_low_conf_pct": asr["asr_low_conf_pct"],
                "n_words": asr["n_words"]},
        "diarization": {"pipeline": diar["pipeline"], "n_speakers_detected": diar["n_speakers"],
                        "exclusive_derived": diar["exclusive_derived"]},
        "speaker_map": mapping["speaker_map"],
        "conversation": {"conv_start": conv_start, "conv_end": conv_end,
                         "conv_duration_s": None if conv_start is None else round(conv_end - conv_start, 3)},
        "turns": turns,
        "speech_by_role": speech_by_role,
        "overlaps": overlaps,
        "overlap_s": round(sp.total([(o["start"], o["end"]) for o in overlaps]), 3),
        "interruptions": result["interruptions"],
        "censor": cen["spans"],
        "unassigned_censor": result["unassigned_censor"],
        "nonspeech": [{"start": round(s, 3), "end": round(e, 3)} for s, e in nonspeech],
        "edits": mapping["edits"],
        "flags": sorted(flags),
        "versions": {
            "pipeline_version": cfg["version"],
            "stage_keys": {name: read_json(p).get("_cache_key") for name, p in [
                ("inventory", paths.interim("censor", f"{call_id}.json")), ("vad", paths.interim("vad", f"{call_id}.json")),
                ("asr", paths.interim("asr", f"{call_id}.json")),
                ("diarize", paths.interim("diarization", f"{call_id}.json")),
                ("turns", paths.interim("turns", f"{call_id}.json"))]},
            "vad_model": vad["model"], "asr_model": asr["model"], "diarization_pipeline": diar["pipeline"],
            "roles_codebook_sha256": ann["_meta"]["codebook_sha256"], "roles_model": ann["_meta"]["model"],
        },
    }
    out = transcript_path(call_id)
    paths.ensure_dir(out.parent)
    out.write_text(render_transcript(call_id, doc, cfg), encoding="utf-8")
    doc["pii_hits"] = pii_scan(doc, cfg)
    return doc


def write_pii_log() -> None:
    lines = []
    for p in sorted(paths.interim("canonical").glob("C*.json")):
        for h in read_json(p).get("pii_hits", []):
            lines.append(f"{p.stem}\t{h['turn_id']}\t{h['kind']}\t{h['match']}")
    paths.ensure_dir(paths.LOGS)
    (paths.LOGS / "pii_scan.log").write_text("call_id\tturn_id\tkind\tmatch\n" + "\n".join(lines) + "\n", encoding="utf-8")


def run(calls: list[str] | None = None, force: bool = False):
    ids = roles.calls_with_turns(calls)
    result = run_per_call("canonical", ids, process_call, canonical_path, force=force,
                          extra_key=lambda c: roles.review_fingerprint(c) + file_fingerprint(
                              *(paths.interim(d, f"{c}.json") for d in ("turns", "diarization", "asr", "vad", "censor"))))
    write_pii_log()
    return result
