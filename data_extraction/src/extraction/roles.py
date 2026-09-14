"""E6 · Role assignment (spec §9-E6).

render            -> data/interim/llm_inputs/roles/<call_id>.txt  (what Claude Code reads)
validate          -> schema, speakers covered, turn_ids exist, quotes verified; progress.csv (pass=roles)
build_review_sheet-> data/interim/review/roles_review.csv for H5 (reviewer columns are never overwritten)
approved_mapping  -> used by E7; refuses calls whose review rows are incomplete or stale

Masking is applied to all speakers' turns in role inputs (roles are not known yet); see decisions D-010.
Review-sheet extension: `accept_suspect` = "all" or turn_ids (space/comma separated) of this speaker's suspect turns
whose proposed_role the reviewer accepts -> explicit turn-level edits + flag diar_corrected.
"""

from __future__ import annotations

import csv
import json
import re
from typing import Any

from . import paths
from .annot_utils import schema_errors, update_progress, verify_quote
from .cache import StageContext, file_fingerprint, read_json, run_per_call
from .config import load_config, sha256_bytes
from .diarize import load_diarization
from .inventory import usable_calls
from .render import NOTATION, fmt_ts, render_line, turn_text
from .turns import load_turns

ROLES = ("AGENT", "CLIENT", "THIRD_PARTY", "SYSTEM", "UNKNOWN")
REVIEWER_COLUMNS = ("approved", "corrected_role", "accept_suspect", "note")
REVIEW_COLUMNS = ["call_id", "speaker", "proposed_role", "confidence", "seconds", "n_turns", "sample_1", "sample_2",
                  "sample_3", "suspect_turns", *REVIEWER_COLUMNS]


class NotApproved(RuntimeError):
    pass


def input_path(call_id: str):
    return paths.LLM_INPUTS / "roles" / f"{call_id}.txt"


def annotation_path(call_id: str):
    return paths.ANNOTATIONS / "roles" / f"{call_id}.json"


def calls_with_turns(calls: list[str] | None) -> list[str]:
    return [c for c in usable_calls(calls) if paths.interim("turns", f"{c}.json").exists()]


def role_turn_text(turn: dict[str, Any], cfg: dict[str, Any]) -> str:
    return turn_text(turn, lambda s: s, cfg["asr"]["low_conf_word_prob"], mask=True)


def _speakers(tdoc: dict[str, Any]) -> list[str]:
    return sorted({t["speaker"] for t in tdoc["turns"]} | {b["speaker"] for t in tdoc["turns"] for b in t["backchannels"]})


# --------------------------------------------------------------------------- render
def render_call(call_id: str, cfg: dict[str, Any]) -> tuple[str, list[str]]:
    tdoc, diar = load_turns(call_id), load_diarization(call_id)
    duration = read_json(paths.interim("audio_16k", f"{call_id}.json"))["duration_s"]
    turns, ann = tdoc["turns"], cfg["annotation"]
    shown = [t for i, t in enumerate(turns) if t["start"] < ann["roles_context_s"] or i < ann["roles_context_turns"]]
    speakers = _speakers(tdoc)
    lines = [f"# call_id: {call_id}", f"# audio length: {fmt_ts(duration)} | turns in call: {len(turns)}",
             "# speaker summary (whole call):"]
    for spk in speakers:
        secs = diar["speakers"].get(spk, {}).get("speech_s")
        if secs is None:
            secs = sum(t["end"] - t["start"] for t in turns if t["speaker"] == spk)
        n_t = sum(t["speaker"] == spk for t in turns)
        n_b = sum(b["speaker"] == spk for t in turns for b in t["backchannels"])
        lines.append(f"#   {spk}: speech {secs:.1f} s | {n_t} turns | {n_b} backchannels")
    lines += [f"# turns shown: starting before {ann['roles_context_s']:.0f} s or among the first "
              f"{ann['roles_context_turns']} turns ({len(shown)} of {len(turns)})", NOTATION, ""]
    lines += [render_line(t, t["speaker"], role_turn_text(t, cfg)) for t in shown]
    shown_ids = {t["turn_id"] for t in shown}
    if ann.get("roles_render_full_call"):
        rest = [t for t in turns if t["turn_id"] not in shown_ids]
        if rest:
            lines += ["", "# rest of the call (for suspect_turns: text that clearly belongs to another speaker):"]
            lines += [render_line(t, t["speaker"], role_turn_text(t, cfg)) for t in rest]
        return "\n".join(lines) + "\n", [t["turn_id"] for t in turns]
    present = {t["speaker"] for t in shown}
    extra = [t for spk in speakers if spk not in present for t in [x for x in turns if x["speaker"] == spk][:3]]
    if extra:
        lines += ["", "# additional turns (first turns of speakers not present above):"]
        lines += [render_line(t, t["speaker"], role_turn_text(t, cfg)) for t in sorted(extra, key=lambda t: t["start"])]
    return "\n".join(lines) + "\n", sorted(shown_ids | {t["turn_id"] for t in extra})


def _render_one(call_id: str, ctx: StageContext) -> dict[str, Any]:
    text, shown = render_call(call_id, ctx.cfg)
    out = input_path(call_id)
    paths.ensure_dir(out.parent)
    out.write_text(text, encoding="utf-8")
    return {"input": str(out.relative_to(paths.ROOT)), "turn_ids_shown": shown}


def render(calls: list[str] | None = None, force: bool = False):
    return run_per_call("roles", calls_with_turns(calls), _render_one,
                        lambda c: paths.interim("roles_render", f"{c}.json"), force=force,
                        extra_key=lambda c: file_fingerprint(paths.interim("turns", f"{c}.json"),
                                                             paths.interim("diarization", f"{c}.json")))


# --------------------------------------------------------------------------- validate
def validate_doc(doc: Any, call_id: str, cfg: dict[str, Any]) -> list[str]:
    errors = schema_errors(doc, "roles.schema.json")
    if errors:
        return errors
    if doc["call_id"] != call_id or doc["_meta"]["call_id"] != call_id:
        errors.append("call_id mismatch between file name, call_id and _meta.call_id")
    tdoc = load_turns(call_id)
    by_id = {t["turn_id"]: t for t in tdoc["turns"]}
    speakers = set(_speakers(tdoc))
    for spk in sorted(speakers - set(doc["speaker_map"])):
        errors.append(f"speaker_map missing {spk}")
    for spk in sorted(set(doc["speaker_map"]) - speakers):
        errors.append(f"speaker_map has unknown speaker {spk}")
    min_ratio = cfg["annotation"]["quote_match_min"]
    for i, ev in enumerate(doc["evidence"]):
        t = by_id.get(ev["turn_id"])
        if t is None:
            errors.append(f"evidence[{i}]: turn_id {ev['turn_id']} does not exist")
            continue
        if ev["speaker"] != t["speaker"] and ev["speaker"] not in {b["speaker"] for b in t["backchannels"]}:
            errors.append(f"evidence[{i}]: {ev['turn_id']} is spoken by {t['speaker']}, not {ev['speaker']}")
        if not verify_quote(ev["quote"], role_turn_text(t, cfg), min_ratio):
            errors.append(f"evidence[{i}]: quote not found in {ev['turn_id']}")
    for i, s in enumerate(doc["suspect_turns"]):
        if s["turn_id"] not in by_id:
            errors.append(f"suspect_turns[{i}]: turn_id {s['turn_id']} does not exist")
    return errors


def validate(calls: list[str] | None = None, force: bool = False) -> int:
    cfg = load_config()
    counts = {"valid": 0, "invalid": 0, "todo": 0}
    updates = []
    for cid in calls_with_turns(calls):
        p = annotation_path(cid)
        if not p.exists():
            counts["todo"] += 1
            updates.append({"call_id": cid, "pass": "roles", "status": "todo", "errors": []})
            continue
        try:
            errs = validate_doc(read_json(p), cid, cfg)
        except json.JSONDecodeError as exc:
            errs = [f"invalid JSON: {exc}"]
        status = "valid" if not errs else "invalid"
        counts[status] += 1
        updates.append({"call_id": cid, "pass": "roles", "status": status, "errors": errs})
        for e in errs:
            print(f"{cid}: {e}")
    update_progress(updates)
    print(f"roles annotations: {counts}")
    return 1 if counts["invalid"] else 0


# --------------------------------------------------------------------------- review sheet (H5)
def read_review() -> list[dict[str, str]]:
    if not paths.ROLES_REVIEW.exists():
        return []
    with open(paths.ROLES_REVIEW, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def build_review_sheet(calls: list[str] | None = None, force: bool = False) -> int:
    cfg = load_config()
    existing = {(r["call_id"], r["speaker"]): r for r in read_review()}
    scope = calls_with_turns(calls)
    rows = [r for (cid, _), r in existing.items() if cid not in scope]
    skipped: list[str] = []
    for cid in scope:
        p = annotation_path(cid)
        if not p.exists():
            skipped.append(f"{cid} (not annotated)")
            continue
        doc = read_json(p)
        if validate_doc(doc, cid, cfg):
            skipped.append(f"{cid} (invalid annotation)")
            continue
        tdoc, diar = load_turns(cid), load_diarization(cid)
        by_id = {t["turn_id"]: t for t in tdoc["turns"]}
        for spk, role in sorted(doc["speaker_map"].items()):
            spk_turns = [t for t in tdoc["turns"] if t["speaker"] == spk]
            samples = sorted(sorted(spk_turns, key=lambda t: -len(t["words"]))[:3], key=lambda t: t["start"])
            lines = [render_line(t, spk, role_turn_text(t, cfg))[:260] for t in samples] + ["", "", ""]
            suspects = [s for s in doc["suspect_turns"] if by_id.get(s["turn_id"], {}).get("speaker") == spk]
            prev = existing.get((cid, spk), {})
            row = {
                "call_id": cid, "speaker": spk, "proposed_role": role, "confidence": doc["confidence"],
                "seconds": diar["speakers"].get(spk, {}).get("speech_s", ""), "n_turns": len(spk_turns),
                "sample_1": lines[0], "sample_2": lines[1], "sample_3": lines[2],
                "suspect_turns": "; ".join(f"{s['turn_id']}->{s['proposed_role']}: {s['reason']}" for s in suspects),
                **{c: prev.get(c, "") for c in REVIEWER_COLUMNS},
            }
            if prev and prev.get("proposed_role") != role and prev.get("approved"):
                row["approved"] = ""
                row["note"] = (prev.get("note", "") + f" [re-review: proposed role changed from {prev['proposed_role']}]").strip()
            rows.append(row)
    paths.ensure_dir(paths.REVIEW)
    with open(paths.ROLES_REVIEW, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=REVIEW_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["call_id"], r["speaker"])))
    pending = sum(1 for r in rows if r.get("approved", "").strip().upper() not in ("Y", "N"))
    print(f"roles_review.csv: {len(rows)} rows ({pending} awaiting review) -> {paths.ROLES_REVIEW.relative_to(paths.ROOT)}")
    if skipped:
        print("not included: " + ", ".join(skipped))
    return 0


def review_fingerprint(call_id: str) -> str:
    p = annotation_path(call_id)
    rows = sorted((r for r in read_review() if r["call_id"] == call_id), key=lambda r: r["speaker"])
    payload = (p.read_bytes() if p.exists() else b"missing") + json.dumps(rows, sort_keys=True).encode()
    return sha256_bytes(payload)[:12]


def approved_mapping(call_id: str, doc: dict[str, Any]) -> dict[str, Any]:
    rows = {r["speaker"]: r for r in read_review() if r["call_id"] == call_id}
    if not rows:
        raise NotApproved(f"{call_id}: no rows in roles_review.csv (run roles-review-sheet, then H5)")
    tdoc = load_turns(call_id)
    turn_speaker = {t["turn_id"]: t["speaker"] for t in tdoc["turns"]}
    speaker_map: dict[str, str] = {}
    overrides: dict[str, str] = {}
    edits: list[dict[str, Any]] = []
    flags: set[str] = set()
    for spk, proposed in doc["speaker_map"].items():
        r = rows.get(spk)
        if r is None:
            raise NotApproved(f"{call_id}/{spk}: speaker missing from roles_review.csv (rebuild the sheet)")
        if r["proposed_role"] != proposed:
            raise NotApproved(f"{call_id}/{spk}: review row is stale (annotation changed); rebuild sheet and re-review")
        approved = r["approved"].strip().upper()
        corrected = r["corrected_role"].strip().upper()
        if approved not in ("Y", "N"):
            raise NotApproved(f"{call_id}/{spk}: `approved` not filled")
        if approved == "N":
            if corrected not in ROLES:
                raise NotApproved(f"{call_id}/{spk}: approved=N requires corrected_role in {ROLES}")
            speaker_map[spk] = corrected
            flags.add("roles_manual_fix")
            edits.append({"type": "speaker_role_correction", "speaker": spk, "from_role": proposed, "to_role": corrected,
                          "note": r.get("note", ""), "source": "roles_review"})
        else:
            if corrected and corrected != proposed:
                raise NotApproved(f"{call_id}/{spk}: approved=Y but corrected_role={corrected}; use approved=N")
            speaker_map[spk] = proposed
        own = {s["turn_id"]: s for s in doc["suspect_turns"] if turn_speaker.get(s["turn_id"]) == spk}
        acc = r.get("accept_suspect", "").strip()
        chosen = list(own) if acc.lower() == "all" else [x for x in re.split(r"[\s,;]+", acc) if x]
        for tid in chosen:
            if tid not in own:
                raise NotApproved(f"{call_id}/{spk}: accept_suspect lists {tid}, which is not a suspect turn of {spk}")
            overrides[tid] = own[tid]["proposed_role"]
            flags.add("diar_corrected")
            edits.append({"type": "turn_role_override", "turn_id": tid, "speaker": spk, "to_role": own[tid]["proposed_role"],
                          "reason": own[tid]["reason"], "source": "roles_review"})
    return {"speaker_map": speaker_map, "overrides": overrides, "edits": edits, "flags": sorted(flags)}
