"""End-to-end E5 → E11 on two synthetic calls in a temporary data root (no real audio, no ML models).

C001 (human): account holder, objection, agent offer, firm promise with recap.
C002 (ai): third party; the agent's self-description must be masked; one 1 kHz censor beep inside an agent turn.
"""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from extraction import annot_utils, paths
from extraction import config as cfgmod
from extraction.cache import write_json
from extraction.derived import SCRIPT_ITEMS
from extraction.diarize import derive_exclusive
from extraction.inventory import summarize_censor, write_manifest

REAL_DATA = paths.DATA
DUR = 30.0
SPEECH = {
    "C001": [
        ("SPK_00", 1.0, 6.0, "Buenos días, le habla Laura del Banco Andino. ¿Hablo con el señor Pérez?"),
        ("SPK_01", 6.5, 8.0, "Sí, con él habla."),
        ("SPK_00", 8.5, 14.0, "Le llamo por su tarjeta de crédito en mora por cuatrocientos mil pesos."),
        ("SPK_01", 14.5, 18.0, "No tengo plata ahora, me quedé sin trabajo."),
        ("SPK_00", 18.5, 23.0, "Entiendo. ¿Puede abonar cien mil el viernes por PSE?"),
        ("SPK_01", 23.5, 25.5, "Sí, el viernes pago cien mil."),
        ("SPK_00", 26.0, 29.0, "Perfecto, le confirmo cien mil el viernes por PSE. Gracias, feliz día."),
    ],
    "C002": [
        ("SPK_00", 0.5, 5.0, "Hola, soy su asistente virtual de cobranza del Banco Andino, gracias por atender."),
        ("SPK_01", 5.5, 7.0, "¿Aló? ¿Quién habla?"),
        ("SPK_00", 7.5, 12.0, "Llamo por una obligación pendiente. ¿Hablo con la señora Gómez?"),
        ("SPK_01", 12.5, 15.0, "No, ella no está, soy la hija."),
        ("SPK_00", 15.5, 19.0, "Gracias, por favor dígale que se comunique con nosotros."),
    ],
}
GROUPS = {"C001": ("human", "humanos", "uuid-aaa"), "C002": ("ai", "ia", "uuid-bbb")}
BEEP = {"C002": (3.0, 3.4)}


def _patch_paths(tmp: Path, monkeypatch) -> None:
    data = tmp / "data"
    for name, value in {
        "ROOT": tmp, "DATA": data, "RAW": data / "raw", "PRIVATE": data / "private", "ID_MAP": data / "private" / "id_map.csv",
        "INTERIM": data / "interim", "MANIFEST": data / "interim" / "manifest.csv", "LLM_INPUTS": data / "interim" / "llm_inputs",
        "ANNOTATIONS": data / "interim" / "annotations", "REVIEW": data / "interim" / "review",
        "ROLES_REVIEW": data / "interim" / "review" / "roles_review.csv", "VALIDATION": data / "validation",
        "PROCESSED": data / "processed", "LOGS": data / "logs",
    }.items():
        monkeypatch.setattr(paths, name, value)
    for stage, spec in list(cfgmod.STAGES.items()):
        monkeypatch.setitem(cfgmod.STAGES, stage, {**spec, "dir": data / Path(spec["dir"]).relative_to(REAL_DATA)})
    monkeypatch.setattr(annot_utils, "PROGRESS", data / "interim" / "annotations" / "progress.csv")


def _words(text: str, start: float, end: float, avoid=None) -> list[dict]:
    toks = text.split()
    step = (end - start) / len(toks)
    out = [{"start": round(start + i * step, 3), "end": round(start + (i + 0.85) * step, 3), "word": t, "probability": 0.93}
           for i, t in enumerate(toks)]
    if avoid:
        out = [w for w in out if w["end"] <= avoid[0] or w["start"] >= avoid[1]]
    return out


def _write_call(cid: str, rng) -> None:
    group, key, uuid = GROUPS[cid]
    segs, beep = SPEECH[cid], BEEP.get(cid)
    sr = 8000
    t = np.arange(int(DUR * sr)) / sr
    x = 0.001 * rng.standard_normal(len(t))
    for spk, s, e, _ in segs:
        idx = (t >= s) & (t < e)
        f0 = (120 if spk == "SPK_00" else 210) * (1 + 0.08 * np.sin(2 * np.pi * 0.7 * t[idx]))
        phase = 2 * np.pi * np.cumsum(f0) / sr
        x[idx] += 0.1 * sum((0.6 / k) * np.sin(k * phase) for k in range(1, 6)) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t[idx]) ** 2)
    if beep:
        idx = (t >= beep[0]) & (t < beep[1])
        x[idx] = 0.3 * np.sin(2 * np.pi * 1000 * t[idx])
    raw = paths.ensure_dir(paths.RAW / key)
    sf.write(raw / f"{uuid}.wav", x.astype(np.float32), sr, subtype="PCM_16")

    spans = [{"start": beep[0], "end": beep[1], "type": "tone", "freq_hz": 1000.0, "freq_std_hz": 0.0, "amp_cv": 0.0}] if beep else []
    qc = {"codec": "PCM_16", "format": "WAV", "sample_rate": sr, "channels": 1, "frames": len(t), "duration_file_s": DUR,
          "clipping_pct": 0.0, "level_dbfs_active": -20.0, "peak_abs": 0.3, "n_active_frames": 500}
    write_json(paths.interim("censor", f"{cid}.json"), {"call_id": cid, "sha256": "0" * 64, "flags": [], "exclude": False,
               "exclusion_reason": None, "diagnostic_spans": [], "spans": spans, "qc": qc, **summarize_censor(spans)})
    write_json(paths.interim("audio_16k", f"{cid}.json"), {"call_id": cid, "duration_s": DUR})
    speech = [{"start": s - 0.1, "end": e + 0.1} for _, s, e, _ in segs]
    write_json(paths.interim("vad", f"{cid}.json"), {"call_id": cid, "speech": speech, "speech_s": 20.0, "speech_ratio": 0.7,
               "snr_db": 15.0, "level_dbfs": -20.0, "flags": [], "model": "silero-vad test"})
    words = [w for _, s, e, text in segs for w in _words(text, s, e, beep)]
    write_json(paths.interim("asr", f"{cid}.json"), {
        "call_id": cid, "model": "large-v3", "device": "cuda", "fallback_used": False, "attempts": [], "info": {},
        "segments": [{"id": i, "start": s, "end": e, "text": text} for i, (_, s, e, text) in enumerate(segs)],
        "words": words, "removed": [], "asr_confidence": 0.93, "asr_low_conf_pct": 0.0, "n_words": len(words), "flags": []})
    regular = [{"start": s, "end": e, "speaker": spk} for spk, s, e, _ in segs]
    write_json(paths.interim("diarization", f"{cid}.json"), {
        "call_id": cid, "pipeline": "test", "device": "cpu", "exclusive_derived": True, "regular": regular,
        "exclusive": derive_exclusive(regular), "n_speakers": 2, "flags": [],
        "speakers": {spk: {"regular_s": 10.0, "speech_s": 10.0} for spk in ("SPK_00", "SPK_01")}})


class FakeAnalyzer:
    def predict(self, texts):
        return [SimpleNamespace(probas={"POS": 0.2, "NEG": 0.5, "NEU": 0.3}) for _ in texts]


class FakeST:
    def encode(self, texts, **_):
        emb = np.ones((len(texts), 8)) + np.arange(len(texts))[:, None] * 0.01
        return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def _meta(cid: str, cb: str) -> dict:
    return {"call_id": cid, "annotator": "claude-code", "model": "test", "claude_code_version": "test",
            "codebook_sha256": cb, "annotated_at": "2026-09-12T12:00:00Z"}


@pytest.fixture()
def synthetic(tmp_path, monkeypatch):
    _patch_paths(tmp_path, monkeypatch)
    rng = np.random.default_rng(7)
    paths.ensure_dir(paths.PRIVATE)
    with open(paths.ID_MAP, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["uuid", "group", "call_id"])
        w.writeheader()
        for cid, (group, _, uuid) in GROUPS.items():
            w.writerow({"uuid": uuid, "group": group, "call_id": cid})
    for cid in SPEECH:
        _write_call(cid, rng)
    write_manifest(list(SPEECH))
    return tmp_path


def test_pipeline_e5_to_e11(synthetic, monkeypatch):
    from extraction import annotation, canonical, roles, tables, turns, validation
    from extraction.config import load_config
    from extraction.features import dynamics, embeddings, lexical, prosody

    cfg = load_config()
    cb = annot_utils.codebook_sha256()
    calls = list(SPEECH)

    # E5
    assert not turns.run().failed
    assert [t["speaker"] for t in turns.load_turns("C001")["turns"]] == ["SPK_00", "SPK_01"] * 3 + ["SPK_00"]

    # E6
    assert not roles.render().failed
    assert "[T001 | SPK_00 |" in roles.input_path("C001").read_text(encoding="utf-8")
    assert "[AGENTE]" in roles.input_path("C002").read_text(encoding="utf-8")
    for cid in calls:
        tdoc = turns.load_turns(cid)
        evidence = []
        for spk in ("SPK_00", "SPK_01"):
            t = next(t for t in tdoc["turns"] if t["speaker"] == spk)
            evidence.append({"speaker": spk, "turn_id": t["turn_id"],
                             "quote": " ".join(roles.role_turn_text(t, cfg).split()[:4])})
        write_json(roles.annotation_path(cid), {
            "_meta": _meta(cid, cb), "call_id": cid, "evidence": evidence, "suspect_turns": [], "confidence": "high",
            "speaker_map": {"SPK_00": "AGENT", "SPK_01": "CLIENT" if cid == "C001" else "THIRD_PARTY"}})
    assert roles.validate() == 0
    assert roles.build_review_sheet() == 0

    # E7 refuses before H5, runs after
    assert set(canonical.run().failed) == set(calls)
    rows = roles.read_review()
    for r in rows:
        r["approved"] = "Y"
    with open(paths.ROLES_REVIEW, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=roles.REVIEW_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    assert not canonical.run().failed
    transcript = canonical.transcript_path("C002").read_text(encoding="utf-8")
    assert "[AGENTE]" in transcript and "[CENSURADO]" in transcript and "asistente virtual" not in transcript

    # E8
    monkeypatch.setattr(lexical, "setup", lambda ctx: ctx.models.update(sentiment=FakeAnalyzer(), sentiment_model="fake"))
    monkeypatch.setattr(embeddings, "setup", lambda ctx: ctx.models.update(st=FakeST()))
    for mod in (dynamics, prosody, lexical, embeddings):
        assert not mod.run().failed, mod.__name__
    dyn = canonical.read_json(paths.interim("features", "dynamics", "C001.json"))
    assert dyn["n_turns"] == 7 and dyn["agent_spoke_first"] is True
    assert dyn["agent_latency_median_s"] == pytest.approx(0.5)

    # E9
    canon = {cid: canonical.load_canonical(cid) for cid in calls}

    def q(cid, tid):
        turn = next(t for t in canon[cid]["turns"] if t["turn_id"] == tid)
        return " ".join(canonical.canonical_turn_text(turn, cfg).split()[:4])

    def item(cid, value, tid=None):
        return {"value": value, "turn_id": tid, "quote": q(cid, tid) if tid else None}

    flags = {"client_expresses_annoyance": {"value": False, "turn_id": None, "quote": None}}
    comp1 = {k: item("C001", False) for k in (*SCRIPT_ITEMS, "coercive_language")}
    comp1.update(greets=item("C001", True, "T001"), identifies_self=item("C001", True, "T001"),
                 validates_identity_before_disclosing=item("C001", True, "T002"), states_amount=item("C001", True, "T003"),
                 offers_payment_channels=item("C001", True, "T005"), recaps_agreement=item("C001", True, "T007"),
                 polite_closing=item("C001", True, "T007"), debt_disclosed_to_third_party=item("C001", "na"))
    pa1 = {
        "_meta": _meta("C001", cb),
        "contact": {"contact_type": "account_holder", "identity_validated": True,
                    "evidence": [{"turn_id": "T002", "quote": q("C001", "T002")}]},
        "debt_context": {"product": "credit_card", "debt_amount": None, "overdue_amount": 400000,
                         "non_payment_reasons": ["unemployment_income_loss"],
                         "evidence": [{"turn_id": "T004", "quote": q("C001", "T004")}]},
        "outcome": {"final_outcome": "firm_promise", "ptp": True, "ptp_type_commitment": "firm", "ptp_amount": 100000,
                    "ptp_date": "el viernes", "ptp_type": "partial",
                    "ptp_confirmed_by_client": True, "agent_recapped": True, "callback_scheduled": False,
                    "evidence": [{"turn_id": "T006", "quote": q("C001", "T006")}]},
        "negotiation": {"had_negotiation": True, "first_anchor": "agent", "offers": [
            {"turn_id": "T005", "proposed_by": "agent", "type": "partial_payment", "amount": 100000,
             "date_text": "el viernes", "quote": q("C001", "T005")}]},
        "objections": [{"turn_id": "T004", "type": "no_money", "quote": q("C001", "T004"), "response_turn_id": "T005",
                        "agent_technique": "alternative_offer", "resolved": "yes"}],
        "compliance": comp1,
        "milestones": {"identification_turn_id": "T002", "debt_mention_turn_id": "T003", "first_offer_turn_id": "T005",
                       "ptp_turn_id": "T006"},
        "failures": {"agent_misunderstood_turn_ids": [], "incoherent_response_turn_ids": [], **flags, "closing_by": "agent"},
        "blindness": {"guess_agent_type": "unsure", "reason": "synthetic call"},
    }
    comp2 = {k: item("C002", "na") for k in SCRIPT_ITEMS}
    comp2.update(greets=item("C002", True, "T001"), identifies_self=item("C002", False),
                 debt_disclosed_to_third_party=item("C002", True, "T003"), coercive_language=item("C002", False))
    pa2 = {
        "_meta": _meta("C002", cb),
        "contact": {"contact_type": "third_party", "identity_validated": False,
                    "evidence": [{"turn_id": "T004", "quote": q("C002", "T004")}]},
        "debt_context": {"product": "unknown", "debt_amount": None, "overdue_amount": None,
                         "non_payment_reasons": ["not_stated"], "evidence": []},
        "outcome": {"final_outcome": "third_party", "ptp": False, "ptp_type_commitment": "na", "ptp_amount": None,
                    "ptp_date": None, "ptp_type": None,
                    "ptp_confirmed_by_client": "na", "agent_recapped": "na", "callback_scheduled": False, "evidence": []},
        "negotiation": {"had_negotiation": False, "first_anchor": "none", "offers": []},
        "objections": [],
        "compliance": comp2,
        "milestones": {"identification_turn_id": None, "debt_mention_turn_id": "T003", "first_offer_turn_id": None,
                       "ptp_turn_id": None},
        "failures": {"agent_misunderstood_turn_ids": [], "incoherent_response_turn_ids": [], **flags, "closing_by": "agent"},
        "blindness": {"guess_agent_type": "unsure", "reason": "synthetic call"},
    }
    for cid, doc in (("C001", pa1), ("C002", pa2)):
        write_json(annotation.annotation_path("a", cid), doc)
        scores = {k: {"score": 3, "justification": "respuesta correcta pero genérica", "turn_ids": ["T001"]}
                  for k in ("clarity", "empathy", "active_listening", "objection_handling", "control_focus", "professionalism")}
        if cid == "C002":
            scores["objection_handling"] = {"score": "NA", "justification": "no hubo objeciones", "turn_ids": []}
        write_json(annotation.annotation_path("b", cid), {"_meta": _meta(cid, cb), "scores": scores})
    assert annotation.validate("a") == 0
    assert annotation.validate("b") == 0

    # E10 (no gold yet) + E11
    assert validation.run() == 0
    assert (paths.VALIDATION / "variable_reliability.csv").exists()
    assert tables.run() == 0
    calls_df = pd.read_parquet(paths.PROCESSED / "calls.parquet").set_index("call_id")
    assert list(calls_df.index) == calls
    assert calls_df.loc["C001", "ptp_strength"] == 3
    assert calls_df.loc["C001", "script_checklist_score"] == pytest.approx(70.0)
    assert pd.isna(calls_df.loc["C002", "script_checklist_score"])
    assert calls_df.loc["C002", "group"] == "ai" and calls_df.loc["C002", "censored_s"] == pytest.approx(0.4)
    assert calls_df.loc["C001", "pct_objections_resolved"] == pytest.approx(100.0)
    assert list(calls_df.loc["C002", "non_payment_reasons"]) == ["not_stated"]
    assert len(pd.read_parquet(paths.PROCESSED / "objections.parquet")) == 1
    assert len(pd.read_parquet(paths.PROCESSED / "offers.parquet")) == 1
    turns_df = pd.read_parquet(paths.PROCESSED / "turns.parquet")
    assert set(turns_df["role"]) == {"AGENT", "CLIENT", "THIRD_PARTY"}
    assert turns_df.loc[(turns_df.call_id == "C002") & (turns_df.turn_id == "T001"), "contains_censor"].item()
    assert (paths.PROCESSED / "data_dictionary.csv").exists()
