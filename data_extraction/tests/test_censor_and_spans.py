import numpy as np
import pytest

from extraction import spans as sp
from extraction.config import load_config
from extraction.inventory import detect_censor, summarize_censor

SR = 8000


def test_span_algebra():
    assert sp.merge([(3, 4), (0, 1), (0.5, 2)]) == [(0, 2), (3, 4)]
    assert sp.merge([(0, 1), (1.1, 2)], gap=0.2) == [(0, 2)]
    assert sp.intersect([(0, 5)], [(1, 2), (4, 6)]) == [(1, 2), (4, 5)]
    assert sp.subtract([(0, 5)], [(1, 2), (4, 6)]) == [(0, 1), (2, 4)]
    assert sp.total([(0, 1), (0.5, 2)]) == pytest.approx(2.0)
    assert sp.runs(np.array([0, 1, 1, 0, 1], dtype=bool)) == [(1, 3), (4, 5)]


def _speechlike(n, rng):
    """Harmonic 'voiced' signal with slowly varying f0 and amplitude, plus a low noise floor."""
    t = np.arange(n) / SR
    f0 = 140 + 30 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    x = sum((0.6 / k) * np.sin(k * phase) for k in range(1, 12))
    env = 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * 2.5 * t))
    return 0.08 * x * env + 0.002 * rng.standard_normal(n)


@pytest.fixture()
def synthetic():
    rng = np.random.default_rng(0)
    x = _speechlike(10 * SR, rng)
    truth = {"tone": (2.00, 2.60), "zeros": (4.00, 4.30), "noise": (6.00, 6.50)}
    s, e = (int(v * SR) for v in truth["tone"])
    x[s:e] = 0.3 * np.sin(2 * np.pi * 1000 * np.arange(e - s) / SR)
    s, e = (int(v * SR) for v in truth["zeros"])
    x[s:e] = 0.0
    s, e = (int(v * SR) for v in truth["noise"])
    x[s:e] = 0.3 * rng.standard_normal(e - s)
    return x.astype(np.float32), truth


def test_censor_detector_finds_each_type_within_20ms(synthetic):
    x, truth = synthetic
    c = load_config()["censor"]
    found = detect_censor(x, SR, c)
    for ctype, (ts, te) in truth.items():
        matches = [s for s in found if s["type"] == ctype and s["start"] < te and s["end"] > ts]
        assert len(matches) == 1, (ctype, found)
        assert abs(matches[0]["start"] - ts) <= 0.02, (ctype, matches[0])
        assert abs(matches[0]["end"] - te) <= 0.02, (ctype, matches[0])
    tone = next(s for s in found if s["type"] == "tone")
    assert abs(tone["freq_hz"] - 1000) < 40
    assert len(found) == 3, found


def test_select_censor_keeps_stable_beep_and_rejects_voice_like_tone():
    rng = np.random.default_rng(2)
    t = np.arange(int(1.0 * SR)) / SR
    x = 0.002 * rng.standard_normal(4 * SR)
    x[SR : 2 * SR] = 0.3 * np.sin(2 * np.pi * 1000 * t)                                        # digital beep
    f_inst = 250 + 12 * np.sin(2 * np.pi * 5 * t)                                               # vibrato ~ voice
    x[3 * SR - SR // 2 : 3 * SR + SR // 2] = 0.3 * np.sin(2 * np.pi * np.cumsum(f_inst) / SR)
    c = load_config()["censor"]
    from extraction.inventory import detect_candidates, select_censor

    cands = [s for s in detect_candidates(x.astype(np.float32), SR, c) if s["type"] == "tone"]
    keep, diag = select_censor(cands, c)
    assert len(keep) == 1 and abs(keep[0]["freq_hz"] - 1000) < 5 and keep[0]["freq_std_hz"] < 1
    assert diag and all(abs(d["freq_hz"] - 1000) > 50 or d["freq_std_hz"] > c["tone_max_freq_std_hz"] for d in diag)


def test_no_false_positives_on_speechlike_signal():
    rng = np.random.default_rng(1)
    x = _speechlike(8 * SR, rng).astype(np.float32)
    assert detect_censor(x, SR, load_config()["censor"]) == []


def test_summary_totals():
    spans = [{"start": 0, "end": 1, "type": "tone"}, {"start": 2, "end": 2.1, "type": "zeros"}]
    s = summarize_censor(spans)
    assert s["n_censor_segments"] == 2 and s["censored_s"] == pytest.approx(1.1)
    assert s["censored_by_type_s"] == {"zeros": 0.1, "tone": 1.0, "noise": 0.0}
    assert summarize_censor([]) == {"censored_s": 0, "n_censor_segments": 0,
                                    "censored_by_type_s": {"zeros": 0, "tone": 0, "noise": 0}}
