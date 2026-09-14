import pytest

from extraction.config import load_config
from extraction.features.dynamics import compute
from extraction.features.lexical import mattr, mechanical_repetition


def words(n):
    return [{"w": "x", "start": 0, "end": 0, "prob": 1.0}] * n


def turn(tid, role, start, end, n, latency=None, backchannels=()):
    return {"turn_id": tid, "role": role, "start": start, "end": end, "words": words(n), "latency_s": latency,
            "is_interruption": False, "backchannels": list(backchannels), "censor": []}


@pytest.fixture()
def doc():
    return {
        "conversation": {"conv_start": 0.0, "conv_end": 60.0, "conv_duration_s": 60.0},
        "turns": [
            turn("T001", "AGENT", 0, 35, 70, backchannels=[{"role": "CLIENT", "start": 10, "end": 10.4, "text": "ajá",
                                                            "words": words(1)}]),
            turn("T002", "CLIENT", 36, 40, 8, latency=1.0),
            turn("T003", "AGENT", 44, 60, 30, latency=4.0),
        ],
        "speech_by_role": {"AGENT": [[0, 35], [44, 60]], "CLIENT": [[10, 10.4], [36, 40]]},
        "censor": [],
        "interruptions": [{"interrupter": "CLIENT", "interrupted": "AGENT", "yield_time_s": 1.5}],
        "overlap_s": 0.4,
    }


def test_dynamics_known_timeline(doc):
    f = compute(doc, load_config()["features"])
    assert f["agent_speech_s"] == pytest.approx(51.0)
    assert f["client_speech_s"] == pytest.approx(4.4)
    assert f["agent_talk_share"] == pytest.approx(51 / 55.4, abs=1e-4)
    assert f["dead_air_count"] == 1 and f["dead_air_s"] == pytest.approx(4.0)
    assert f["silence_pct"] == pytest.approx(100 * 5 / 60, abs=1e-3)
    assert f["max_agent_monologue_s"] == pytest.approx(35) and f["n_monologues_30s"] == 1
    assert f["agent_latency_median_s"] == pytest.approx(4.0)
    assert f["client_latency_median_s"] == pytest.approx(1.0)
    assert f["turns_per_min"] == pytest.approx(3.0)
    assert f["interruptions_per_min_client"] == pytest.approx(1.0)
    assert f["yield_time_s"] == pytest.approx(1.5)
    assert f["backchannels_agent_per_min"] == 0
    assert f["agent_spoke_first"] is True and f["time_to_first_agent_voice_s"] is None


def test_censor_excluded_from_dead_air(doc):
    doc["censor"] = [{"start": 41.0, "end": 42.0, "type": "tone"}]
    f = compute(doc, load_config()["features"])
    assert f["dead_air_count"] == 0  # 40–44 gap split by the censor span into 1 s + 2 s


def test_mechanical_repetition_and_mattr():
    rep, pairs, best = mechanical_repetition(
        ["le recuerdo que su obligación se encuentra en mora", "le recuerdo que su obligación está en mora hoy", "sí"],
        80, 5)
    assert rep and pairs == 1 and best >= 80
    assert mattr(["a", "b", "a", "b"], 50) == 0.5
