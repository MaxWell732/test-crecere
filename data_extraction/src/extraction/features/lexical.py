"""E8c · Lexical features: lexicons, mechanical repetition, sentiment (spec §9-E8c, §10.2 H/K).

Lexicons match on normalized text (textnorm.normalize). Utterances = turns and backchannels, attributed to their role.
Sentiment: pysentimiento `sentiment` (es) on CLIENT turns with >= sentiment_min_words; score = P(POS) − P(NEG).
The `asks_if_robot` / `asks_for_human` lexicons in lexicons.yaml are unused since D-029 (kept: that file is part of the
ASR cache key).
"""

from __future__ import annotations

import re
from functools import lru_cache
from itertools import combinations
from typing import Any

import numpy as np
from rapidfuzz import fuzz

from .. import paths
from .. import spans as sp
from ..cache import StageContext, file_fingerprint, run_per_call
from ..canonical import load_canonical
from ..config import load_yaml
from ..roles import calls_with_turns
from ..textnorm import normalize

AGENT, CLIENT = "AGENT", "CLIENT"


@lru_cache(maxsize=1)
def lexicons() -> dict[str, Any]:
    lx = load_yaml("lexicons.yaml")
    comp = lambda pats: [re.compile(p) for p in pats]  # noqa: E731
    rr = lx["repeat_requests"]
    return {
        "rr_full": set(rr["full_utterance"]), "rr_any": comp(rr["anywhere"]), "rr_alo": comp(rr["alo_after_s"]),
        "polite": comp(lx["politeness"]), "fillers": comp(lx["fillers"]),
    }


def utterances(doc: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for t in doc["turns"]:
        out.append({"role": t["role"], "turn_id": t["turn_id"], "start": t["start"], "end": t["end"],
                    "raw": " ".join(w["w"] for w in t["words"]), "kind": "turn"})
        for b in t["backchannels"]:
            out.append({"role": b["role"], "turn_id": t["turn_id"], "start": b["start"], "end": b["end"],
                        "raw": " ".join(w["w"] for w in b["words"]) or b["text"], "kind": "backchannel"})
    for u in out:
        u["norm"] = normalize(u["raw"])
    return out


def count_matches(patterns: list[re.Pattern], text: str) -> int:
    return sum(len(p.findall(text)) for p in patterns)


def repeat_requests(utts: list[dict[str, Any]], conv_start: float | None, alo_after_s: float) -> dict[str, int]:
    lx = lexicons()
    counts = {AGENT: 0, CLIENT: 0, "other": 0}
    for u in utts:
        n = int(u["norm"] in lx["rr_full"]) + count_matches(lx["rr_any"], u["norm"])
        if conv_start is not None and u["start"] > conv_start + alo_after_s:
            n += int(any(p.search(u["norm"]) for p in lx["rr_alo"]))
        counts[u["role"] if u["role"] in (AGENT, CLIENT) else "other"] += n
    return counts


def mechanical_repetition(agent_turn_texts: list[str], ratio: float, min_words: int) -> tuple[bool, int, float | None]:
    texts = [normalize(t) for t in agent_turn_texts]
    texts = [t for t in texts if len(t.split()) >= min_words]
    best, n_pairs = None, 0
    for a, b in combinations(texts, 2):
        r = fuzz.token_set_ratio(a, b)
        best = r if best is None else max(best, r)
        n_pairs += r >= ratio
    return n_pairs > 0, n_pairs, best


def mattr(tokens: list[str], window: int) -> float | None:
    if not tokens:
        return None
    if len(tokens) <= window:
        return round(len(set(tokens)) / len(tokens), 4)
    ratios = [len(set(tokens[i : i + window])) / window for i in range(len(tokens) - window + 1)]
    return round(float(np.mean(ratios)), 4)


def setup(ctx: StageContext) -> None:
    import torch
    from pysentimiento import create_analyzer

    torch.set_num_threads(ctx.cfg["asr"]["cpu_threads"])
    ctx.models["sentiment"] = create_analyzer(task="sentiment", lang="es")
    ctx.models["sentiment_model"] = ctx.cfg["features"]["sentiment_model"]


def sentiment_scores(analyzer, texts: list[str]) -> list[float]:
    if not texts:
        return []
    preds = analyzer.predict(texts)
    preds = preds if isinstance(preds, list) else [preds]
    return [round(float(p.probas.get("POS", 0.0) - p.probas.get("NEG", 0.0)), 4) for p in preds]


def process_call(call_id: str, ctx: StageContext) -> dict[str, Any]:
    f = ctx.cfg["features"]
    lx = lexicons()
    doc = load_canonical(call_id)
    conv_start = doc["conversation"]["conv_start"]
    utts = utterances(doc)
    rr = repeat_requests(utts, conv_start, f["repeat_request_alo_after_s"])
    agent_turns = [u for u in utts if u["role"] == AGENT and u["kind"] == "turn"]
    mech, mech_pairs, mech_best = mechanical_repetition([u["raw"] for u in agent_turns],
                                                        f["mechanical_repetition_ratio"], f["mechanical_repetition_min_words"])
    agent_norm = " ".join(u["norm"] for u in agent_turns)
    agent_speech_s = sp.total([tuple(s) for s in doc["speech_by_role"].get(AGENT, [])])

    sent_turns = [t for t in doc["turns"] if t["role"] == CLIENT and len(t["words"]) >= f["sentiment_min_words"]]
    scores = sentiment_scores(ctx.models["sentiment"], [" ".join(w["w"] for w in t["words"]) for t in sent_turns])
    k = f["sentiment_edge_turns"]
    s_start = round(float(np.mean(scores[:k])), 4) if scores else None
    s_end = round(float(np.mean(scores[-k:])), 4) if scores else None
    return {
        "repeat_requests": rr[AGENT] + rr[CLIENT],
        "repeat_requests_agent": rr[AGENT],
        "repeat_requests_client": rr[CLIENT],
        "mechanical_repetition": mech,
        "mechanical_repetition_pairs": mech_pairs,
        "mechanical_repetition_max_ratio": mech_best,
        "agent_questions": sum(u["raw"].count("?") for u in agent_turns),
        "politeness_markers": count_matches(lx["polite"], agent_norm),
        "lexical_richness_mattr": mattr(agent_norm.split(), f["mattr_window"]),
        "n_agent_tokens": len(agent_norm.split()),
        "fillers_per_min": round(count_matches(lx["fillers"], agent_norm) / (agent_speech_s / 60), 3) if agent_speech_s > 0 else None,
        "sentiment_start": s_start,
        "sentiment_end": s_end,
        "sentiment_delta": None if s_start is None else round(s_end - s_start, 4),
        "sentiment_min": min(scores) if scores else None,
        "n_sentiment_turns": len(scores),
        "turn_sentiment": {t["turn_id"]: s for t, s in zip(sent_turns, scores)},
        "sentiment_model": ctx.models["sentiment_model"],
    }


def run(calls: list[str] | None = None, force: bool = False):
    ids = [c for c in calls_with_turns(calls) if paths.interim("canonical", f"{c}.json").exists()]
    return run_per_call("lexical", ids, process_call, lambda c: paths.interim("features", "lexical", f"{c}.json"),
                        force=force, setup=setup,
                        extra_key=lambda c: file_fingerprint(paths.interim("canonical", f"{c}.json")))
