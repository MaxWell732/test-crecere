"""E10 · Validation (spec §9-E10).

select : seeded, stratified, disjoint dev (3+3) / gold (10+10) / WER (2+2 ⊂ gold) selection + templates for H6/H7.
run    : role accuracy (H5), agreement with the human gold (κ, weighted κ, exact match, F1, Spearman), self-consistency κ,
         blindness-probe accuracy, WER (+ number/date token accuracy, filler check) — every metric overall and by group —
         then one reliability decision per variable -> variable_reliability.csv + validation_report.md.
Group labels are read here (code only) to compute by-group metrics; output files list call_ids only.
"""

from __future__ import annotations

import csv
import random
import re
from statistics import mean
from typing import Any, Callable

from . import paths
from .annot_utils import codebook_sha256
from .annotation import annotatable_calls, annotation_path, validate_doc
from .asr import load_asr
from .cache import read_json
from .config import load_config, load_yaml
from .derived import FINAL_OUTCOME_ORDER, SCRIPT_ITEMS
from .inventory import master_path, read_id_map, usable_calls
from .roles import read_review
from .textnorm import normalize

GOLD_COLUMNS = ["call_id", "roles_ok", "contact_type", "identity_validated", "final_outcome", "ptp", "ptp_amount",
                "ptp_date_text", "ptp_confirmed_by_client", "agent_recapped", "n_objections", "objection_types",
                "n_offers_agent", "first_anchor", *SCRIPT_ITEMS, "debt_disclosed_to_third_party", "clarity", "empathy",
                "active_listening"]
RANK = {"keep": 2, "flag": 1, "drop": 0}


def tri(v: Any) -> str | None:
    return None if v is None else "na" if v == "na" else "true" if v is True else "false" if v is False else str(v)


# Claude-side values per variable (pass A = a, pass B = b)
EXTRACT: dict[str, Callable[[dict, dict | None], Any]] = {
    "contact_type": lambda a, b: a["contact"]["contact_type"],
    "identity_validated": lambda a, b: tri(a["contact"]["identity_validated"]),
    "product": lambda a, b: a["debt_context"]["product"],
    "debt_amount": lambda a, b: a["debt_context"]["debt_amount"],
    "overdue_amount": lambda a, b: a["debt_context"]["overdue_amount"],
    "non_payment_reasons": lambda a, b: a["debt_context"]["non_payment_reasons"],
    "final_outcome": lambda a, b: a["outcome"]["final_outcome"],
    "ptp": lambda a, b: tri(a["outcome"]["ptp"]),
    "ptp_type_commitment": lambda a, b: a["outcome"]["ptp_type_commitment"],
    "ptp_amount": lambda a, b: a["outcome"]["ptp_amount"],
    "ptp_date": lambda a, b: a["outcome"]["ptp_date"],
    "ptp_type": lambda a, b: a["outcome"]["ptp_type"],
    "ptp_confirmed_by_client": lambda a, b: tri(a["outcome"]["ptp_confirmed_by_client"]),
    "agent_recapped": lambda a, b: tri(a["outcome"]["agent_recapped"]),
    "callback_scheduled": lambda a, b: tri(a["outcome"]["callback_scheduled"]),
    "had_negotiation": lambda a, b: tri(a["negotiation"]["had_negotiation"]),
    "first_anchor": lambda a, b: a["negotiation"]["first_anchor"],
    "n_offers_agent": lambda a, b: sum(o["proposed_by"] == "agent" for o in a["negotiation"]["offers"]),
    "n_counteroffers_client": lambda a, b: sum(o["proposed_by"] == "client" for o in a["negotiation"]["offers"]),
    "n_objections": lambda a, b: len(a["objections"]),
    "n_objections_resolved": lambda a, b: sum(o["resolved"] == "yes" for o in a["objections"]),
    "objection_types": lambda a, b: sorted({o["type"] for o in a["objections"]}),
    "closing_by": lambda a, b: a["failures"]["closing_by"],
    "agent_misunderstood": lambda a, b: len(a["failures"]["agent_misunderstood_turn_ids"]),
    "incoherent_responses": lambda a, b: len(a["failures"]["incoherent_response_turn_ids"]),
    "client_expresses_annoyance": lambda a, b: tri(a["failures"]["client_expresses_annoyance"]["value"]),
    "blindness_guess": lambda a, b: a["blindness"]["guess_agent_type"],
    **{item: (lambda a, b, item=item: tri(a["compliance"][item]["value"])) for item in
       (*SCRIPT_ITEMS, "debt_disclosed_to_third_party", "coercive_language")},
    **{k: (lambda a, b, k=k: None if b is None else b["scores"][k]["score"]) for k in
       ("clarity", "empathy", "active_listening", "objection_handling", "control_focus", "professionalism")},
    **{f"milestone_{k}": (lambda a, b, k=k: a["milestones"][f"{k}_turn_id"]) for k in
       ("identification", "debt_mention", "first_offer", "ptp")},
}
SELF_CONSISTENCY_METRIC = {  # variables judged on rerun-vs-first-run agreement
    **{k: "kappa" for k in ("contact_type", "identity_validated", "product", "ptp", "ptp_type_commitment", "ptp_type",
                            "ptp_confirmed_by_client", "agent_recapped", "callback_scheduled", "had_negotiation",
                            "first_anchor", "closing_by", "client_expresses_annoyance", "blindness_guess",
                            *SCRIPT_ITEMS, "debt_disclosed_to_third_party", "coercive_language")},
    "final_outcome": "wkappa",
    "non_payment_reasons": "f1",
    **{k: "wkappa" for k in ("clarity", "empathy", "active_listening", "objection_handling", "control_focus",
                             "professionalism")},
    **{k: "exact" for k in ("debt_amount", "overdue_amount", "ptp_amount", "ptp_date", "milestone_identification",
                            "milestone_debt_mention", "milestone_first_offer", "milestone_ptp")},
    **{k: "spearman" for k in ("n_offers_agent", "n_counteroffers_client", "n_objections", "n_objections_resolved",
                               "agent_misunderstood", "incoherent_responses")},
}
DERIVED_FROM = {  # D-over-L variables inherit the worst decision of their inputs
    "final_outcome_rank": ["final_outcome"],
    "ptp_strength": ["ptp_amount", "ptp_date", "ptp_confirmed_by_client"],
    "script_checklist_score": list(SCRIPT_ITEMS) + ["contact_type"],
    "pct_objections_resolved": ["n_objections", "n_objections_resolved"],
    "n_objections_partial": ["n_objections"],
    "time_to_identification_s": ["milestone_identification"],
    "time_to_debt_mention_s": ["milestone_debt_mention"],
    "time_to_first_offer_s": ["milestone_first_offer"],
    "time_to_ptp_s": ["milestone_ptp"],
}


# --------------------------------------------------------------------------- selection + templates
def _write_ids(name: str, ids: list[str]) -> None:
    with open(paths.VALIDATION / name, "w", newline="", encoding="utf-8") as fh:
        fh.write("call_id\n" + "".join(f"{c}\n" for c in sorted(ids)))


def read_ids(name: str) -> list[str]:
    p = paths.VALIDATION / name
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as fh:
        return [r["call_id"] for r in csv.DictReader(fh)]


def select(force: bool = False) -> int:
    cfg = load_config()
    v = cfg["validation"]
    paths.ensure_dir(paths.VALIDATION)
    if all((paths.VALIDATION / n).exists() for n in ("dev_ids.csv", "gold_ids.csv", "wer_ids.csv")) and not force:
        print("selection already exists (use --force only deliberately: it invalidates gold work)")
    else:
        groups = {r["call_id"]: r["group"] for r in read_id_map()}
        dev, gold, wer = [], [], []
        for g in sorted(set(groups.values())):
            pool = sorted(c for c in usable_calls() if groups[c] == g)
            random.Random(f"{cfg['seed']}:validation:{g}").shuffle(pool)
            need = v["dev_per_group"] + v["gold_per_group"]
            if len(pool) < need:
                raise RuntimeError(f"not enough usable calls in one group for dev+gold ({len(pool)} < {need})")
            dev += pool[: v["dev_per_group"]]
            g_ids = pool[v["dev_per_group"] : need]
            gold += g_ids
            wer += g_ids[: v["wer_per_group"]]
        _write_ids("dev_ids.csv", dev)
        _write_ids("gold_ids.csv", gold)
        _write_ids("wer_ids.csv", wer)
        print(f"selected dev={len(dev)} gold={len(gold)} wer={len(wer)} (call_ids only; stratified by group)")
    write_templates()
    return 0


def write_templates() -> None:
    gold, wer = read_ids("gold_ids.csv"), read_ids("wer_ids.csv")
    gpath = paths.VALIDATION / "gold_annotations.csv"
    if not gpath.exists():
        with open(gpath, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=GOLD_COLUMNS)
            w.writeheader()
            w.writerows({"call_id": c} for c in gold)
        print(f"template: {gpath.relative_to(paths.ROOT)}")
    enums = load_yaml("variables.yaml")["tables"]
    allowed = {x["name"]: x.get("allowed") for x in enums["calls"]}
    readme = ["# Gold annotation (H6) — instructions", "",
              "Annotate each call in `gold_annotations.csv` from the rendered transcript and the audio, following",
              "`configs/codebook.md`, **without opening Claude's annotation files for these calls**.", "",
              "Blank = null (not mentioned). Use `censored` when mentioned but censored, `na` when not applicable.",
              "Booleans: true / false. Lists (`objection_types`): values separated by `;`. Amounts: plain numbers (COP).",
              "Rubric scores: 1–5 or NA. `roles_ok`: true if every turn role in the rendered transcript is right.", "",
              "| column | allowed |", "|---|---|"]
    for col in GOLD_COLUMNS[1:]:
        name = {"ptp_date_text": "ptp_date", "objection_types": None}.get(col, col)
        vals = allowed.get(name) if name else ["no_money", "already_paid", "call_later", "doesnt_recognize_debt",
                                               "wrong_amount", "wants_discount", "distrust_who_is_calling",
                                               "annoyed_by_calls", "asks_for_human", "other"]
        readme.append(f"| {col} | {'; '.join(map(str, vals)) if vals else 'number / text / true-false'} |")
    (paths.VALIDATION / "README_gold.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    refs = paths.ensure_dir(paths.VALIDATION / "wer_refs")
    (refs / "README.md").write_text(
        "# WER references (H7)\n\nFor each `<call_id>.whisper.txt`: listen to the call, correct the text so it is exactly "
        "what was said (both speakers, in order; keep [CENSURADO] where audio is censored), and save it as "
        "`<call_id>.txt`. Do not add punctuation rules of your own; numbers may be written as digits or words.\n"
        "Audio paths for these calls: data/private/review_audio.csv.\n", encoding="utf-8")
    for cid in wer:
        hyp = refs / f"{cid}.whisper.txt"
        asr_p = paths.interim("asr", f"{cid}.json")
        if not hyp.exists() and asr_p.exists():
            hyp.write_text("\n".join(s["text"] for s in read_json(asr_p)["segments"]) + "\n", encoding="utf-8")
    rows = [{"call_id": c, "purpose": p, "audio_path": str(master_path(c).relative_to(paths.ROOT))}
            for p, ids in (("gold", gold), ("wer", wer), ("dev", read_ids("dev_ids.csv"))) for c in ids]
    paths.ensure_dir(paths.PRIVATE)
    with open(paths.PRIVATE / "review_audio.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["call_id", "purpose", "audio_path"])
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- metric primitives
def kappa(a: list, b: list, weights: str | None = None, labels: list | None = None) -> float | None:
    from sklearn.metrics import cohen_kappa_score

    if len(a) < 2:
        return None
    labs = labels or sorted(set(a) | set(b), key=str)
    if len(set(a) | set(b)) < 2:
        return None
    return round(float(cohen_kappa_score(a, b, labels=labs, weights=weights)), 4)


def spearman(a: list, b: list) -> float | None:
    from scipy.stats import spearmanr

    if len(a) < 3 or len(set(a)) < 2 or len(set(b)) < 2:
        return None
    return round(float(spearmanr(a, b).statistic), 4)


def set_f1(pairs: list[tuple[list, list]]) -> float | None:
    tp = fp = fn = 0
    for gold, pred in pairs:
        g, p = set(gold), set(pred)
        tp, fp, fn = tp + len(g & p), fp + len(p - g), fn + len(g - p)
    return None if tp + fp + fn == 0 else round(2 * tp / (2 * tp + fp + fn), 4)


def parse_number(v: Any) -> Any:
    if v is None or v == "censored" or isinstance(v, (int, float)):
        return v
    s = re.sub(r"[^\d]", "", str(v))
    return float(s) if s else None


def parse_gold(value: str | None, metric: str, var: str) -> Any:
    v = (value or "").strip()
    if v == "":
        return None
    low = v.casefold()
    if low in ("censored", "censurado"):
        return "censored"
    if metric == "kappa":
        return {"y": "true", "yes": "true", "si": "true", "sí": "true", "n": "false", "no": "false"}.get(low, low)
    if metric == "wkappa":
        return "NA" if low == "na" else low if var == "final_outcome" else int(float(low))
    if metric == "spearman":
        return int(float(low))
    if metric == "f1":
        return sorted({x.strip() for x in low.split(";") if x.strip()})
    return v


def agreement(metric: str, var: str, pairs: list[tuple[Any, Any]]) -> tuple[float | None, int, str]:
    """pairs of (reference, prediction). Returns (value, n, note)."""
    if metric == "kappa":
        pairs = [(str(g), str(p)) for g, p in pairs if g is not None and p is not None]
        return kappa([g for g, _ in pairs], [p for _, p in pairs]), len(pairs), ""
    if metric == "wkappa":
        pairs = [(g, p) for g, p in pairs if g not in (None, "NA") and p not in (None, "NA")]
        if var == "final_outcome":
            conv = [(FINAL_OUTCOME_ORDER.index(g), FINAL_OUTCOME_ORDER.index(p)) for g, p in pairs
                    if g in FINAL_OUTCOME_ORDER and p in FINAL_OUTCOME_ORDER]
            return kappa([g for g, _ in conv], [p for _, p in conv], "quadratic", list(range(8))), len(conv), ""
        conv = [(int(g), int(p)) for g, p in pairs]
        return kappa([g for g, _ in conv], [p for _, p in conv], "quadratic", [1, 2, 3, 4, 5]), len(conv), ""
    if metric == "spearman":
        pairs = [(g, p) for g, p in pairs if g is not None and p is not None]
        return spearman([g for g, _ in pairs], [p for _, p in pairs]), len(pairs), ""
    if metric == "f1":
        pairs = [(g or [], p or []) for g, p in pairs]
        return set_f1(pairs), len(pairs), ""
    if metric == "exact":
        censored = sum(1 for g, p in pairs if "censored" in (g, p))
        numeric = var in ("ptp_amount", "debt_amount", "overdue_amount")
        norm = (lambda x: parse_number(x)) if numeric else (lambda x: None if x is None else normalize(str(x)))
        kept = [(norm(g), norm(p)) for g, p in pairs if "censored" not in (g, p) and (g is not None or p is not None)]
        if not kept:
            return None, 0, f"censored rows: {censored}"
        return round(sum(g == p for g, p in kept) / len(kept), 4), len(kept), f"censored rows excluded: {censored}"
    raise ValueError(metric)


def by_group(fn: Callable[[list[str]], tuple[float | None, int, str]], ids: list[str], groups: dict[str, str]) -> dict:
    overall = fn(ids)
    h = fn([c for c in ids if groups.get(c) == "human"])
    a = fn([c for c in ids if groups.get(c) == "ai"])
    return {"value": overall[0], "n": overall[1], "note": overall[2], "value_human": h[0], "value_ai": a[0],
            "n_human": h[1], "n_ai": a[1]}


# --------------------------------------------------------------------------- WER
NUM_WORDS = set("""cero uno una un dos tres cuatro cinco seis siete ocho nueve diez once doce trece catorce quince dieciseis
diecisiete dieciocho diecinueve veinte veintiuno veintiun veintidos veintitres veinticuatro veinticinco veintiseis veintisiete
veintiocho veintinueve treinta cuarenta cincuenta sesenta setenta ochenta noventa cien ciento doscientos doscientas
trescientos trescientas cuatrocientos cuatrocientas quinientos quinientas seiscientos seiscientas setecientos setecientas
ochocientos ochocientas novecientos novecientas mil millon millones enero febrero marzo abril mayo junio julio agosto
septiembre setiembre octubre noviembre diciembre lunes martes miercoles jueves viernes sabado domingo""".split())
FILLERS = {"eh", "em", "mm", "ehh", "emm", "pues", "bueno"}


def normalize_wer(text: str) -> str:
    t = re.sub(r"\[censurado\]", " ", text, flags=re.I)
    t = re.sub(r"(?<=\d)[.,](?=\d{3}\b)", "", t)  # 400.000 -> 400000 (both sides)
    return normalize(t)


def wer_stats(ids: list[str]) -> dict[str, Any]:
    import jiwer

    refs, hyps, per_call = [], [], {}
    num_total = num_ok = fill_ref = fill_hyp = 0
    for cid in ids:
        ref_p = paths.VALIDATION / "wer_refs" / f"{cid}.txt"
        if not ref_p.exists():
            continue
        ref = normalize_wer(ref_p.read_text(encoding="utf-8"))
        hyp = normalize_wer(" ".join(w["word"] for w in load_asr(cid)["words"]))
        if not ref:
            continue
        out = jiwer.process_words(ref, hyp)
        ref_words = out.references[0]
        for chunk in out.alignments[0]:
            for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                if ref_words[i].isdigit() or ref_words[i] in NUM_WORDS:
                    num_total += 1
                    num_ok += chunk.type == "equal"
        fill_ref += sum(w in FILLERS for w in ref.split()) + ref.count("o sea")
        fill_hyp += sum(w in FILLERS for w in hyp.split()) + hyp.count("o sea")
        refs.append(ref)
        hyps.append(hyp)
        per_call[cid] = round(out.wer, 4)
    if not refs:
        return {"wer": None, "n": 0}
    return {"wer": round(jiwer.wer(refs, hyps), 4), "n": len(refs), "per_call": per_call,
            "number_token_accuracy": round(num_ok / num_total, 4) if num_total else None, "number_tokens": num_total,
            "filler_ref": fill_ref, "filler_hyp": fill_hyp}


# --------------------------------------------------------------------------- run
def _valid(pass_name: str, cid: str, cfg, cb) -> dict | None:
    p = annotation_path(pass_name, cid)
    if not p.exists():
        return None
    doc = read_json(p)
    return doc if not validate_doc(doc, pass_name, cid, cfg, cb) else None


def decide(metric: str, value: float | None, vh: float | None, va: float | None, v: dict[str, Any],
           cap_flag: bool = False, n: int | None = None) -> tuple[str, str]:
    if value is None:
        return "flag", "metric undefined (too few rows or no variance)"
    min_n = v.get("min_rows_decision", 0)
    if n is not None and n < min_n:
        return "flag", f"too few rows to decide (n={n} < {min_n})"
    if metric == "exact":
        dec = "keep" if value >= v["exact_match_keep"] else "flag" if value >= v["exact_match_flag"] else "drop"
    else:
        dec = "keep" if value >= v["kappa_keep"] else "flag" if value >= v["kappa_flag"] else "drop"
    notes = []
    if dec == "keep" and vh is not None and va is not None and abs(vh - va) > v["group_gap_flag"]:
        dec, notes = "flag", [f"group gap {abs(vh - va):.2f}"]
    if cap_flag and dec == "keep":
        dec, notes = "flag", notes + ["self-consistency only (no gold)"]
    return dec, "; ".join(notes)


def run() -> int:
    cfg = load_config()
    v = cfg["validation"]
    cb = codebook_sha256()
    groups = {r["call_id"]: r["group"] for r in read_id_map()}
    ids = annotatable_calls()
    gold_ids = read_ids("gold_ids.csv")
    gold_path = paths.VALIDATION / "gold_annotations.csv"
    gold_rows = {}
    if gold_path.exists():
        with open(gold_path, newline="", encoding="utf-8-sig") as fh:
            gold_rows = {r["call_id"]: r for r in csv.DictReader(fh)
                         if any((r.get(c) or "").strip() for c in GOLD_COLUMNS[1:])}
    pa = {c: _valid("a", c, cfg, cb) for c in ids}
    pb = {c: _valid("b", c, cfg, cb) for c in ids}
    rerun = {c: _valid("rerun_a", c, cfg, cb) for c in gold_ids if c in ids}
    specs = load_yaml("variables.yaml")["tables"]
    rows: list[dict[str, Any]] = []
    report: list[str] = ["# Validation report", "", f"Codebook sha256: `{cb}`", "",
                         f"Calls with valid pass A: {sum(x is not None for x in pa.values())} / {len(ids)}; "
                         f"pass B: {sum(x is not None for x in pb.values())}; gold rows filled: {len(gold_rows)}; "
                         f"reruns: {sum(x is not None for x in rerun.values())}", ""]

    # --- role accuracy (H5)
    review = [r for r in read_review() if r.get("approved", "").strip().upper() in ("Y", "N")]

    def role_acc(cids: list[str]) -> tuple[float | None, int, str]:
        rs = [r for r in review if r["call_id"] in set(cids)]
        return (round(sum(r["approved"].strip().upper() == "Y" for r in rs) / len(rs), 4) if rs else None), len(rs), ""

    def role_acc_sec(cids: list[str]) -> tuple[float | None, int, str]:
        rs = [r for r in review if r["call_id"] in set(cids) and r.get("seconds")]
        tot = sum(float(r["seconds"]) for r in rs)
        ok = sum(float(r["seconds"]) for r in rs if r["approved"].strip().upper() == "Y")
        return (round(ok / tot, 4) if tot else None), len(rs), ""

    all_ids = sorted(groups)
    role_rows = {"role_accuracy_speakers": by_group(role_acc, all_ids, groups),
                 "role_accuracy_seconds": by_group(role_acc_sec, all_ids, groups)}
    gold_roles = by_group(lambda cs: ((round(mean(gold_rows[c]["roles_ok"].strip().casefold() in ("true", "y", "yes", "si", "sí")
                                                  for c in cs), 4) if cs else None), len(cs), ""),
                          [c for c in gold_rows if (gold_rows[c].get("roles_ok") or "").strip()], groups)
    role_rows["roles_ok_gold"] = gold_roles
    role_gap = None
    ra = role_rows["role_accuracy_seconds"]
    if ra["value_human"] is not None and ra["value_ai"] is not None:
        role_gap = abs(ra["value_human"] - ra["value_ai"])

    # --- WER
    wer_ids = read_ids("wer_ids.csv")
    w_all = wer_stats(wer_ids)
    w_h = wer_stats([c for c in wer_ids if groups.get(c) == "human"])
    w_a = wer_stats([c for c in wer_ids if groups.get(c) == "ai"])
    wer_gap = abs(w_h["wer"] - w_a["wer"]) if w_h["wer"] is not None and w_a["wer"] is not None else None

    # --- blindness probe
    def blind(cids):
        rs = [(pa[c]["blindness"]["guess_agent_type"], groups[c]) for c in cids if pa.get(c)]
        return (round(sum(gss == g for gss, g in rs) / len(rs), 4) if rs else None), len(rs), \
            f"unsure: {sum(gss == 'unsure' for gss, _ in rs)}"
    blind_row = by_group(blind, ids, groups)

    # --- per-variable reliability
    decisions: dict[str, tuple[str, str, str, dict]] = {}
    gold_var_ids = [c for c in gold_ids if c in gold_rows and pa.get(c)]
    for table in ("calls", "objections"):
        for var in specs[table]:
            vw = var.get("validate_with")
            if not vw:
                continue
            name = "objection_types" if table == "objections" else var["name"]
            metric, gcol = vw["metric"], vw["gold"]
            extract_key = {"ptp_date": "ptp_date"}.get(name, name)

            def fn(cs, gcol=gcol, metric=metric, extract_key=extract_key, name=name):
                pairs = []
                for c in cs:
                    if extract_key in ("clarity", "empathy", "active_listening") and pb.get(c) is None:
                        continue
                    pred = EXTRACT[extract_key](pa[c], pb.get(c))
                    pairs.append((parse_gold(gold_rows[c].get(gcol), metric, name), pred))
                return agreement(metric, name, pairs)

            res = by_group(fn, gold_var_ids, groups)
            dec, note = decide(metric, res["value"], res["value_human"], res["value_ai"], v, n=res.get("n"))
            key = var["name"] if table == "calls" else "objections.type"
            decisions[key] = (metric, dec, note, res)

    # self-consistency (rerun vs first run)
    rerun_ids = [c for c, d in rerun.items() if d is not None and pa.get(c)]
    self_rows = {}
    for name, metric in SELF_CONSISTENCY_METRIC.items():
        def fn(cs, name=name, metric=metric):
            if name in ("clarity", "empathy", "active_listening", "objection_handling", "control_focus", "professionalism"):
                return None, 0, "pass B has no rerun"
            return agreement(metric, name, [(EXTRACT[name](rerun[c], None), EXTRACT[name](pa[c], None)) for c in cs])
        res = by_group(fn, rerun_ids, groups)
        self_rows[name] = (metric, res)
        if name not in decisions and res["n"]:
            dec, note = decide(metric, res["value"], res["value_human"], res["value_ai"], v, cap_flag=True, n=res.get("n"))
            decisions[name] = (f"self_consistency_{metric}", dec, note, res)

    for derived, inputs in DERIVED_FROM.items():
        found = [decisions[i] for i in inputs if i in decisions]
        if found:
            worst = min(found, key=lambda d: RANK[d[1]])
            decisions[derived] = ("inherited", worst[1], f"worst of {inputs}", {"value": None, "n": None})

    fill_note = ""
    for var in specs["calls"]:
        name = var["name"]
        if name in decisions:
            continue
        if var["source"] in ("D", "A"):
            dec, notes = "keep", []
            if "roles" in var.get("depends", []) and role_gap is not None and role_gap > v["role_accuracy_gap_flag"]:
                dec, notes = "flag", notes + [f"role accuracy gap {role_gap:.3f}"]
            if "asr" in var.get("depends", []) and wer_gap is not None and wer_gap > v["wer_gap_flag"]:
                dec, notes = "flag", notes + [f"WER gap {wer_gap:.3f}"]
            if name == "fillers_per_min":
                lo, hi = v["filler_ratio_keep"]
                if not w_all.get("n") or w_all.get("filler_ref", 0) < v["filler_min_ref_count"]:
                    dec, notes = "drop", ["not validated: too few fillers in WER references (or refs missing)"]
                else:
                    ratio = w_all["filler_hyp"] / w_all["filler_ref"]
                    fill_note = f"filler hyp/ref ratio {ratio:.2f}"
                    if not lo <= ratio <= hi:
                        dec, notes = "drop", [fill_note]
                    else:
                        notes.append(fill_note)
            decisions[name] = ("deterministic", dec, "; ".join(notes), {"value": None, "n": None})
        elif var["source"] in ("L", "D+L", "L+D", "D over L"):
            decisions[name] = ("none", "flag", "no gold and no self-consistency data", {"value": None, "n": None})

    for name, (metric, dec, note, res) in sorted(decisions.items()):
        rows.append({"variable": name, "metric": metric, "value": res.get("value"), "value_human": res.get("value_human"),
                     "value_ai": res.get("value_ai"), "n": res.get("n"), "decision": dec, "note": note})
    paths.ensure_dir(paths.VALIDATION)
    with open(paths.VALIDATION / "variable_reliability.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["variable", "metric", "value", "value_human", "value_ai", "n", "decision", "note"])
        w.writeheader()
        w.writerows(rows)

    def fmt(x):
        return "" if x is None else f"{x:.3f}" if isinstance(x, float) else str(x)

    report += ["## Roles (H5)", "", "| metric | overall | human | ai | n |", "|---|---|---|---|---|"]
    for k, r in role_rows.items():
        report.append(f"| {k} | {fmt(r['value'])} | {fmt(r['value_human'])} | {fmt(r['value_ai'])} | {r['n']} |")
    report += ["", "## ASR (H7)", "", "| metric | overall | human | ai | n |", "|---|---|---|---|---|",
               f"| WER | {fmt(w_all.get('wer'))} | {fmt(w_h.get('wer'))} | {fmt(w_a.get('wer'))} | {w_all.get('n')} |",
               f"| number/date token accuracy | {fmt(w_all.get('number_token_accuracy'))} | "
               f"{fmt(w_h.get('number_token_accuracy'))} | {fmt(w_a.get('number_token_accuracy'))} | "
               f"{w_all.get('number_tokens', 0)} tokens |",
               f"| fillers ref / hyp | {w_all.get('filler_ref', '')} / {w_all.get('filler_hyp', '')} | | | |", "",
               "## Blindness probe", "", "| overall | human | ai | n | note |", "|---|---|---|---|---|",
               f"| {fmt(blind_row['value'])} | {fmt(blind_row['value_human'])} | {fmt(blind_row['value_ai'])} | "
               f"{blind_row['n']} | {blind_row['note']} |", "",
               "## Agreement with gold and self-consistency", "",
               "| variable | metric | value | human | ai | n | self-consistency | decision | note |",
               "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        sc = self_rows.get(r["variable"])
        sc_txt = fmt(sc[1]["value"]) if sc else ""
        report.append(f"| {r['variable']} | {r['metric']} | {fmt(r['value'])} | {fmt(r['value_human'])} | "
                      f"{fmt(r['value_ai'])} | {r['n'] if r['n'] is not None else ''} | {sc_txt} | {r['decision']} | {r['note']} |")
    (paths.VALIDATION / "validation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    counts = {d: sum(r["decision"] == d for r in rows) for d in ("keep", "flag", "drop")}
    print(f"validation: {counts} -> data/validation/validation_report.md, variable_reliability.csv")
    return 0
