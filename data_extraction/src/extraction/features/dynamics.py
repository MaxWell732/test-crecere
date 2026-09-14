"""E8a · Conversation dynamics from canonical JSON only (spec §10.2 G).

Times are seconds from conv_start. Rates are per conversation minute. `time_to_*` milestones are D-over-L and are
computed in E11 from pass A. Shares are fractions (0–1); `*_pct` are percentages (0–100).
"""

from __future__ import annotations

import statistics
from typing import Any

import numpy as np

from .. import paths
from .. import spans as sp
from ..cache import StageContext, file_fingerprint, run_per_call
from ..canonical import load_canonical
from ..roles import calls_with_turns

AGENT, CLIENT, SYSTEM = "AGENT", "CLIENT", "SYSTEM"


def _median(xs: list[float]) -> float | None:
    return round(float(statistics.median(xs)), 3) if xs else None


def _p90(xs: list[float]) -> float | None:
    return round(float(np.percentile(xs, 90)), 3) if xs else None


def compute(doc: dict[str, Any], f: dict[str, Any]) -> dict[str, Any]:
    conv = doc["conversation"]
    start, end = conv["conv_start"], conv["conv_end"]
    turns = [t for t in doc["turns"] if t["role"] != SYSTEM]
    censor = [(c["start"], c["end"]) for c in doc["censor"]]
    speech = {r: [tuple(s) for s in v] for r, v in doc["speech_by_role"].items()}
    out: dict[str, Any] = {"conv_start": start, "conv_end": end}
    if start is None:
        return {**out, "conv_duration_s": 0.0, "n_turns": 0, "agent_spoke_first": False, "empty_conversation": True}

    dur = end - start
    minutes = dur / 60 if dur > 0 else None
    window = [(start, end)]
    agent_s = sp.total(sp.subtract(sp.intersect(speech.get(AGENT, []), window), censor))
    client_s = sp.total(sp.subtract(sp.intersect(speech.get(CLIENT, []), window), censor))
    all_speech = sp.merge([s for v in speech.values() for s in v])
    censored_in = sp.total(sp.intersect(censor, window))
    silence = sp.subtract(sp.subtract(window, all_speech), censor)
    gaps = [(s, e) for s, e in silence if e - s >= f["dead_air_min_s"]]

    def role_turns(role):
        return [t for t in turns if t["role"] == role]

    def per_min(n):
        return round(n / minutes, 4) if minutes else None

    agent_turn_durs = [t["end"] - t["start"] for t in role_turns(AGENT)]
    client_turn_durs = [t["end"] - t["start"] for t in role_turns(CLIENT)]
    agent_lat, client_lat = [], []
    for prev, cur in zip(turns, turns[1:]):
        if cur["latency_s"] is None:
            continue
        if prev["role"] == CLIENT and cur["role"] == AGENT:
            agent_lat.append(cur["latency_s"])
        elif prev["role"] == AGENT and cur["role"] == CLIENT:
            client_lat.append(cur["latency_s"])

    ints = doc["interruptions"]
    int_agent = [i for i in ints if i["interrupter"] == AGENT and i["interrupted"] == CLIENT]
    int_client = [i for i in ints if i["interrupter"] == CLIENT and i["interrupted"] == AGENT]

    def n_words(role):
        return sum(len(t["words"]) for t in doc["turns"] if t["role"] == role) + \
            sum(len(b["words"]) for t in doc["turns"] for b in t["backchannels"] if b["role"] == role)

    first_agent = next((t for t in turns if t["role"] == AGENT), None)
    first_client = next((t for t in turns if t["role"] == CLIENT), None)
    agent_first = bool(turns) and turns[0]["role"] == AGENT
    ttfav = None
    if not agent_first and first_agent is not None and first_client is not None:
        ttfav = round(first_agent["start"] - first_client["end"], 3)
    last = doc["turns"][-1] if doc["turns"] else None

    out.update({
        "conv_duration_s": round(dur, 3),
        "agent_speech_s": round(agent_s, 3),
        "client_speech_s": round(client_s, 3),
        "agent_talk_share": round(agent_s / (agent_s + client_s), 4) if agent_s + client_s > 0 else None,
        "silence_pct": round(100 * sp.total(silence) / (dur - censored_in), 3) if dur - censored_in > 0 else None,
        "n_turns": len(turns),
        "turns_per_min": per_min(len(turns)),
        "mean_turn_s_agent": round(float(np.mean(agent_turn_durs)), 3) if agent_turn_durs else None,
        "mean_turn_s_client": round(float(np.mean(client_turn_durs)), 3) if client_turn_durs else None,
        "max_agent_monologue_s": round(max(agent_turn_durs), 3) if agent_turn_durs else None,
        "n_monologues_30s": sum(d >= f["monologue_min_s"] for d in agent_turn_durs),
        "agent_latency_median_s": _median(agent_lat),
        "agent_latency_p90_s": _p90(agent_lat),
        "client_latency_median_s": _median(client_lat),
        "n_agent_latency_obs": len(agent_lat),
        "agent_spoke_first": agent_first,
        "time_to_first_agent_voice_s": ttfav,
        "dead_air_count": len(gaps),
        "dead_air_s": round(sum(e - s for s, e in gaps), 3),
        "interruptions_per_min_agent": per_min(len(int_agent)),
        "interruptions_per_min_client": per_min(len(int_client)),
        "overlap_s": doc["overlap_s"],
        "yield_time_s": _median([i["yield_time_s"] for i in int_client]),
        "backchannels_agent_per_min": per_min(sum(b["role"] == AGENT for t in doc["turns"] for b in t["backchannels"])),
        "words_per_min_agent": round(n_words(AGENT) / (agent_s / 60), 2) if agent_s > 0 else None,
        "words_per_min_client": round(n_words(CLIENT) / (client_s / 60), 2) if client_s > 0 else None,
        "last_turn_role": last["role"] if last else None,
        "roles_present": sorted({t["role"] for t in doc["turns"]}),
    })
    return out


def process_call(call_id: str, ctx: StageContext) -> dict[str, Any]:
    return compute(load_canonical(call_id), ctx.cfg["features"])


def run(calls: list[str] | None = None, force: bool = False):
    ids = [c for c in calls_with_turns(calls) if paths.interim("canonical", f"{c}.json").exists()]
    return run_per_call("dynamics", ids, process_call, lambda c: paths.interim("features", "dynamics", f"{c}.json"),
                        force=force, extra_key=lambda c: file_fingerprint(paths.interim("canonical", f"{c}.json")))
