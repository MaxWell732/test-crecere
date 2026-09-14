"""E11 · Tables (spec §9-E11, §10).

`group` enters tables here for the first time (from data/private/id_map.csv). Every column is validated against
configs/variables.yaml. Annotation files that fail `annotate-validate` are treated as missing (flag annotation_missing).
"""

from __future__ import annotations

import csv
from typing import Any

import numpy as np
import pandas as pd

from . import paths
from .annot_utils import codebook_sha256
from .annotation import annotatable_calls, annotation_path, validate_doc
from .cache import read_json
from .canonical import load_canonical
from .config import load_config, load_yaml
from .derived import SCRIPT_ITEMS, final_outcome_rank, objection_counts, ptp_strength, script_checklist_score
from .features.embeddings import discourse_similarity
from .inventory import read_id_map, read_manifest

RUBRIC = ("clarity", "empathy", "active_listening", "objection_handling", "control_focus", "professionalism")


# --------------------------------------------------------------------------- spec helpers
def table_specs() -> dict[str, list[dict[str, Any]]]:
    return load_yaml("variables.yaml")["tables"]


def tri(v: Any) -> str | None:
    return None if v is None else "na" if v == "na" else "true" if v is True else "false"


def split_censored(v: Any) -> tuple[float | None, bool]:
    if v == "censored":
        return None, True
    return (None if v is None else float(v)), False


def _feature(family: str, cid: str) -> dict[str, Any]:
    p = paths.interim("features", family, f"{cid}.json")
    return read_json(p) if p.exists() else {}


def _valid_annotation(pass_name: str, cid: str, cfg, cb) -> dict[str, Any] | None:
    p = annotation_path(pass_name, cid)
    if not p.exists():
        return None
    doc = read_json(p)
    return doc if not validate_doc(doc, pass_name, cid, cfg, cb) else None


# --------------------------------------------------------------------------- builders
def call_row(cid: str, group: str, cfg, cb, similarity: dict[str, float | None]) -> tuple[dict, list, list, list]:
    canon = load_canonical(cid)
    audio, conv = canon["audio"], canon["conversation"]
    dyn, pros, lex = _feature("dynamics", cid), _feature("prosody", cid), _feature("lexical", cid)
    pa, pb = _valid_annotation("a", cid, cfg, cb), _valid_annotation("b", cid, cfg, cb)
    turns = {t["turn_id"]: t for t in canon["turns"]}
    flags = set(canon["flags"])
    if pa is None or pb is None:
        flags.add("annotation_missing")
    for doc in (pa, pb):
        if doc and "quote_verification_failed" in doc.get("flags", []):
            flags.add("quote_verification_failed")

    def rel_time(turn_id):
        if turn_id is None or turn_id not in turns or conv["conv_start"] is None:
            return None
        return round(turns[turn_id]["start"] - conv["conv_start"], 3)

    row: dict[str, Any] = {
        "call_id": cid, "group": group, "duration_file_s": audio["duration_file_s"], "snr_db": audio["snr_db"],
        "clipping_pct": audio["clipping_pct"], "level_dbfs": audio["level_dbfs"], "speech_ratio": audio["speech_ratio"],
        "censored_s": audio["censored_s"], "n_censor_segments": audio["n_censor_segments"],
        "asr_confidence": canon["asr"]["asr_confidence"], "asr_low_conf_pct": canon["asr"]["asr_low_conf_pct"],
        "n_speakers_detected": canon["diarization"]["n_speakers_detected"],
    }
    for k in ("conv_duration_s", "agent_speech_s", "client_speech_s", "agent_talk_share", "silence_pct", "n_turns",
              "turns_per_min", "mean_turn_s_agent", "mean_turn_s_client", "max_agent_monologue_s", "n_monologues_30s",
              "agent_latency_median_s", "agent_latency_p90_s", "client_latency_median_s", "agent_spoke_first",
              "time_to_first_agent_voice_s", "dead_air_count", "dead_air_s", "interruptions_per_min_agent",
              "interruptions_per_min_client", "overlap_s", "yield_time_s", "backchannels_agent_per_min",
              "words_per_min_agent", "words_per_min_client", "last_turn_role"):
        row[k] = dyn.get(k)
    for k in ("agent_pitch_range_st", "client_pitch_range_st", "agent_intensity_std_db", "client_escalation"):
        row[k] = pros.get(k)
    for k in ("repeat_requests", "repeat_requests_agent", "repeat_requests_client", "mechanical_repetition",
              "sentiment_start", "sentiment_end", "sentiment_delta", "sentiment_min", "agent_questions",
              "politeness_markers", "lexical_richness_mattr", "fillers_per_min"):
        row[k] = lex.get(k)
    row["agent_discourse_similarity"] = similarity.get(cid)

    objections_rows, offers_rows = [], []
    if pa is not None:
        c, d, o, n = pa["contact"], pa["debt_context"], pa["outcome"], pa["negotiation"]
        row.update({
            "contact_type": c["contact_type"], "identity_validated": tri(c["identity_validated"]), "product": d["product"],
            "debt_amount": split_censored(d["debt_amount"])[0], "overdue_amount": split_censored(d["overdue_amount"])[0],
            "non_payment_reasons": list(d["non_payment_reasons"]),
            "final_outcome": o["final_outcome"], "final_outcome_rank": final_outcome_rank(o["final_outcome"]),
            "ptp": o["ptp"], "ptp_type_commitment": o["ptp_type_commitment"],
            "ptp_amount": split_censored(o["ptp_amount"])[0], "ptp_date": o["ptp_date"],
            "ptp_type": o["ptp_type"], "ptp_confirmed_by_client": tri(o["ptp_confirmed_by_client"]),
            "agent_recapped": tri(o["agent_recapped"]), "callback_scheduled": o["callback_scheduled"],
            "ptp_strength": ptp_strength(o),
            "had_negotiation": n["had_negotiation"], "first_anchor": n["first_anchor"],
            "n_offers_agent": sum(x["proposed_by"] == "agent" for x in n["offers"]),
            "n_counteroffers_client": sum(x["proposed_by"] == "client" for x in n["offers"]),
            **objection_counts(pa["objections"]),
            "time_to_identification_s": rel_time(pa["milestones"]["identification_turn_id"]),
            "time_to_debt_mention_s": rel_time(pa["milestones"]["debt_mention_turn_id"]),
            "time_to_first_offer_s": rel_time(pa["milestones"]["first_offer_turn_id"]),
            "time_to_ptp_s": rel_time(pa["milestones"]["ptp_turn_id"]),
            "closing_by": pa["failures"]["closing_by"],
            "agent_misunderstood": len(pa["failures"]["agent_misunderstood_turn_ids"]),
            "incoherent_responses": len(pa["failures"]["incoherent_response_turn_ids"]),
            "client_expresses_annoyance": pa["failures"]["client_expresses_annoyance"]["value"],
            **{item: tri(pa["compliance"][item]["value"]) for item in SCRIPT_ITEMS},
            "script_checklist_score": script_checklist_score(pa["compliance"], c["contact_type"]),
            "debt_disclosed_to_third_party": tri(pa["compliance"]["debt_disclosed_to_third_party"]["value"]),
            "coercive_language": tri(pa["compliance"]["coercive_language"]["value"]),
            "blindness_guess": pa["blindness"]["guess_agent_type"],
        })
        for i, ob in enumerate(sorted(pa["objections"], key=lambda x: turns[x["turn_id"]]["start"]), start=1):
            objections_rows.append({"call_id": cid, "group": group, "objection_idx": i, "turn_id": ob["turn_id"],
                                    "time_s": rel_time(ob["turn_id"]), "type": ob["type"],
                                    "agent_technique": ob["agent_technique"], "response_turn_id": ob["response_turn_id"],
                                    "resolved": ob["resolved"]})
        for i, of in enumerate(sorted(n["offers"], key=lambda x: turns[x["turn_id"]]["start"]), start=1):
            a, a_c = split_censored(of["amount"])
            offers_rows.append({"call_id": cid, "group": group, "offer_idx": i, "turn_id": of["turn_id"],
                                "time_s": rel_time(of["turn_id"]), "proposed_by": of["proposed_by"], "type": of["type"],
                                "amount": a, "amount_censored": a_c, "date_text": of["date_text"]})
    if pb is not None:
        for k in RUBRIC:
            s = pb["scores"][k]["score"]
            row[k] = None if s == "NA" else int(s)
    row["flags"] = sorted(flags)

    sentiment = lex.get("turn_sentiment", {})
    turn_rows = []
    for t in canon["turns"]:
        durs = [max(w["end"] - w["start"], 0.01) for w in t["words"]]
        conf = sum(w["prob"] * d for w, d in zip(t["words"], durs)) / sum(durs) if durs else None
        turn_rows.append({
            "call_id": cid, "group": group, "turn_id": t["turn_id"], "role": t["role"], "start_s": t["start"],
            "end_s": t["end"], "duration_s": round(t["end"] - t["start"], 3), "n_words": len(t["words"]),
            "is_interruption": t["is_interruption"], "latency_s": t["latency_s"],
            "n_backchannels_received": len(t["backchannels"]), "asr_confidence": None if conf is None else round(conf, 4),
            "sentiment": sentiment.get(t["turn_id"]) if t["role"] == "CLIENT" else None,
            "contains_censor": bool(t["censor"]),
        })
    return row, turn_rows, objections_rows, offers_rows


# --------------------------------------------------------------------------- coercion & validation
def coerce(df: pd.DataFrame, spec: list[dict[str, Any]]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for v in spec:
        name = v["name"]
        s = df[name] if name in df.columns else pd.Series([None] * len(df), index=df.index, dtype=object)
        t = v["type"]
        if t == "int":
            out[name] = pd.array([None if x is None or (isinstance(x, float) and np.isnan(x)) else x for x in s], dtype="Int64")
        elif t == "float":
            out[name] = pd.to_numeric(s, errors="coerce").astype("float64")
        elif t == "bool":
            vals = [None if x is None or (isinstance(x, float) and np.isnan(x)) else x for x in s]
            out[name] = pd.array(vals, dtype="boolean")
        elif t == "list":
            out[name] = [list(x) if isinstance(x, (list, tuple, np.ndarray)) else [] for x in s]
        else:
            out[name] = pd.array([None if x is None or (isinstance(x, float) and np.isnan(x)) else x for x in s], dtype=object)
    for extra in df.columns:
        if extra not in out.columns:
            out[extra] = df[extra]
    return out


def validate_table(df: pd.DataFrame, spec: list[dict[str, Any]], table: str) -> list[str]:
    errors = []
    names = [v["name"] for v in spec]
    for c in df.columns:
        if c not in names:
            errors.append(f"{table}: column `{c}` not in variables.yaml")
    for v in spec:
        name, t = v["name"], v["type"]
        if name not in df.columns:
            errors.append(f"{table}: missing column `{name}`")
            continue
        s = df[name]
        nulls = s.isna() if t != "list" else pd.Series([x is None for x in s])
        if not v.get("nullable", True) and bool(nulls.any()):
            errors.append(f"{table}.{name}: {int(nulls.sum())} null values in non-nullable column")
        vals = [x for x, n in zip(s, nulls) if not n]
        if t == "int":
            bad = [x for x in vals if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer))]
        elif t == "float":
            bad = [x for x in vals if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, float, np.integer, np.floating))]
        elif t == "bool":
            bad = [x for x in vals if not isinstance(x, (bool, np.bool_))]
        elif t == "list":
            bad = [x for x in vals if not isinstance(x, list) or any(not isinstance(i, str) for i in x)]
        else:
            bad = [x for x in vals if not isinstance(x, str)]
        if bad:
            errors.append(f"{table}.{name}: {len(bad)} values of wrong type for {t} (e.g. {bad[0]!r})")
            continue
        if "allowed" in v:
            allowed = set(map(str, v["allowed"]))
            items = [i for x in vals for i in x] if t == "list" else vals
            outside = sorted({str(i) for i in items} - allowed)
            if outside:
                errors.append(f"{table}.{name}: values outside allowed set: {outside[:5]}")
        if t in ("int", "float") and vals:
            if "min" in v and min(vals) < v["min"]:
                errors.append(f"{table}.{name}: value below min {v['min']}")
            if "max" in v and max(vals) > v["max"]:
                errors.append(f"{table}.{name}: value above max {v['max']}")
    return errors


def reliability_decisions() -> dict[str, dict[str, str]]:
    p = paths.VALIDATION / "variable_reliability.csv"
    if not p.exists():
        return {}
    with open(p, newline="", encoding="utf-8") as fh:
        return {r["variable"]: r for r in csv.DictReader(fh)}


def _csv_ready(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].map(lambda x: isinstance(x, list)).any():
            out[c] = out[c].map(lambda x: ";".join(x) if isinstance(x, list) else x)
    return out


# --------------------------------------------------------------------------- stage
def run() -> int:
    cfg = load_config()
    cb = codebook_sha256()
    specs = table_specs()
    groups = {r["call_id"]: r["group"] for r in read_id_map()}
    manifest = {r["call_id"]: r for r in read_manifest()}
    usable = [c for c, r in manifest.items() if r["exclude"] != "True"]
    ids = annotatable_calls()
    missing = sorted(set(usable) - set(ids))
    if missing:
        print(f"WARNING: {len(missing)} usable calls have no canonical JSON yet: {missing[:10]}")
    centroids = {c: _feature("embeddings", c).get("centroid") for c in ids}
    similarity = discourse_similarity(centroids, groups)

    calls, turns, objections, offers = [], [], [], []
    for cid in ids:
        r, t, o, f = call_row(cid, groups[cid], cfg, cb, similarity)
        calls.append(r)
        turns += t
        objections += o
        offers += f
    frames = {"calls": pd.DataFrame(calls), "turns": pd.DataFrame(turns), "objections": pd.DataFrame(objections),
              "offers": pd.DataFrame(offers)}

    decisions = reliability_decisions()
    if not decisions:
        print("WARNING: data/validation/variable_reliability.csv missing — no reliability decisions applied (run `validate`)")
    dropped = sorted(v for v, d in decisions.items() if d.get("decision") == "drop" and v in {x["name"] for x in specs["calls"]})
    errors = []
    out_specs = dict(specs)
    for table, df in frames.items():
        spec = specs[table]
        df = coerce(df if len(df.columns) else pd.DataFrame(columns=[v["name"] for v in spec]), spec)
        if table == "calls" and dropped:
            if cfg["tables"]["keep_dropped"]:
                df = df.rename(columns={v: f"{v}_unreliable" for v in dropped})
                spec = [({**v, "name": f"{v['name']}_unreliable"} if v["name"] in dropped else v) for v in spec]
            else:
                df = df.drop(columns=dropped)
                spec = [v for v in spec if v["name"] not in dropped]
            out_specs[table] = spec
        errors += validate_table(df, spec, table)
        frames[table] = df
    if errors:
        for e in errors:
            print("ERROR", e)
        print(f"{len(errors)} validation error(s); tables NOT written")
        return 1

    paths.ensure_dir(paths.PROCESSED)
    for table, df in frames.items():
        df.to_parquet(paths.PROCESSED / f"{table}.parquet", index=False)
        if table != "turns":
            _csv_ready(df).to_csv(paths.PROCESSED / f"{table}.csv", index=False)

    dict_rows = []
    for table, spec in specs.items():
        for v in spec:
            dec = decisions.get(v["name"], {}).get("decision") if table == "calls" else None
            if dec is None:
                dec = "keep" if v["source"] in ("D", "A") else "not_validated"
            dict_rows.append({"variable": v["name"], "table": table, "description": v["description"], "type": v["type"],
                              "allowed_values": ";".join(map(str, v.get("allowed", []))), "unit": v.get("unit", ""),
                              "source": v["source"], "reliability_decision": dec})
    pd.DataFrame(dict_rows).to_csv(paths.PROCESSED / "data_dictionary.csv", index=False)

    n_calls = len(frames["calls"])
    print(f"tables written: calls={n_calls} turns={len(frames['turns'])} objections={len(frames['objections'])} "
          f"offers={len(frames['offers'])}; dropped from calls: {dropped or 'none'}")
    if n_calls != len(usable):
        print(f"ROW CHECK FAILED: calls has {n_calls} rows, expected {len(usable)} usable audios")
        return 1
    return 0
