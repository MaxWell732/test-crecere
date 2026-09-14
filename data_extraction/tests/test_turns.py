import pytest

from extraction.config import load_config
from extraction.diarize import derive_exclusive
from extraction.turns import fuse

T_CFG = load_config()["turns"]
T = {**T_CFG, "sentence_smoothing": False}  # floor/backchannel logic tests; smoothing has its own tests below
T_SMOOTH = {**T_CFG, "sentence_smoothing": True}


def words_in(start, end, n, prefix="w"):
    step = (end - start) / n
    return [{"start": round(start + i * step, 3), "end": round(start + (i + 0.8) * step, 3), "word": f"{prefix}{i}",
             "probability": 0.9} for i in range(n)]


def run(segments, word_spans, censor=(), cfg=T):
    regular = [{"start": s, "end": e, "speaker": k} for s, e, k in segments]
    exclusive = derive_exclusive(regular)
    words = sorted((w for s, e, n, p in word_spans for w in words_in(s, e, n, p)), key=lambda w: w["start"])
    speech = [(0.0, 1000.0)]
    return fuse(words, exclusive, regular, speech, list(censor), cfg)


def test_backchannel_does_not_take_floor_and_latency():
    out = run(
        [(0, 5, "A"), (2.0, 2.4, "B"), (5.5, 8, "A"), (8.6, 12, "B")],
        [(0, 2.0, 4, "a"), (2.0, 2.4, 1, "bc"), (2.4, 5, 5, "a"), (5.5, 8, 5, "a"), (8.6, 12, 8, "b")],
    )
    turns = out["turns"]
    assert [t["speaker"] for t in turns] == ["A", "B"]
    assert len(turns[0]["backchannels"]) == 1
    assert turns[1]["latency_s"] == pytest.approx(0.6)
    assert not turns[1]["is_interruption"]


def test_interruption_with_yield():
    out = run([(0, 6, "A"), (5.0, 9, "B")], [(0, 5.0, 10, "a"), (5.0, 9, 6, "b")])
    assert [t["speaker"] for t in out["turns"]] == ["A", "B"]
    assert out["turns"][1]["is_interruption"]
    assert out["turns"][1]["latency_s"] is None
    (it,) = out["interruptions"]
    assert it["yield_time_s"] == pytest.approx(1.0)
    assert it["interrupter"] == "B" and it["turn_id"] == "T002"
    assert out["overlap_s"] == pytest.approx(1.0)


def test_overlap_without_yield_is_not_interruption():
    out = run([(0, 10, "A"), (4, 6, "B")], [(0, 4, 8, "a"), (4, 6, 5, "b"), (6, 10, 8, "a")])
    assert not any(t["is_interruption"] for t in out["turns"])
    assert out["interruptions"] == []
    assert out["overlap_s"] == pytest.approx(2.0)


def test_short_unit_followed_by_same_speaker_takes_floor():
    out = run([(0, 3, "A"), (3.5, 3.8, "B"), (5.0, 8.0, "B")], [(0, 3, 6, "a"), (3.5, 3.8, 1, "si"), (5, 8, 6, "b")])
    turns = out["turns"]
    assert [t["speaker"] for t in turns] == ["A", "B"]
    assert turns[1]["start"] == pytest.approx(3.5)
    assert turns[1]["latency_s"] == pytest.approx(0.5)
    assert turns[0]["backchannels"] == []


def test_censor_inside_turn_rendered():
    out = run([(0, 4, "A"), (5, 8, "B")], [(0, 1.5, 3, "a"), (2.0, 4, 3, "a"), (5, 8, 5, "b")],
              censor=[{"start": 1.5, "end": 2.0, "type": "tone"}])
    assert "[CENSURADO]" in out["turns"][0]["text"]
    assert out["turns"][0]["censor"][0]["type"] == "tone"
    assert out["unassigned_censor"] == []


def test_word_without_segment_is_unknown_beyond_gap():
    regular = [{"start": 0, "end": 1, "speaker": "A"}]
    words = [{"start": 0.1, "end": 0.3, "word": "hola", "probability": 1.0},
             {"start": 3.0, "end": 3.2, "word": "lejos", "probability": 1.0}]
    out = fuse(words, regular, regular, [(0, 10)], [], T)
    assert out["stats"]["n_unknown_words"] == 1


def _w(tokens, start, step=0.3):
    return [{"start": round(start + i * step, 2), "end": round(start + i * step + 0.25, 2), "word": tok, "probability": 0.9}
            for i, tok in enumerate(tokens)]


BC = {"si", "aja", "claro"}


def test_smoothing_fixes_late_boundary_inside_sentence():
    from extraction.turns import smooth_sentence_speakers

    words = _w("Me alegra que lleguemos a este acuerdo de pago y que pueda aprovechar el descuento.".split(), 116.4)
    speakers = ["B"] * 4 + ["A"] * 11
    out, changes = smooth_sentence_speakers(words, speakers, T_SMOOTH, BC)
    assert out == ["A"] * 15 and len(changes) == 4


def test_smoothing_keeps_short_answer_in_own_sentence_and_backchannels():
    from extraction.turns import smooth_sentence_speakers

    words = _w("¿Hablo con el señor Pérez? Sí, con él.".split(), 10.0)
    speakers = ["A"] * 5 + ["B"] * 3
    out, _ = smooth_sentence_speakers(words, speakers, T_SMOOTH, BC)
    assert out == speakers  # second sentence has < min_words and is entirely B anyway
    words = _w("le ofrecemos un descuento ajá del siete por ciento hoy.".split(), 20.0)
    speakers = ["A"] * 4 + ["B"] + ["A"] * 5
    out, _ = smooth_sentence_speakers(words, speakers, T_SMOOTH, BC)
    assert out == speakers  # backchannel-only run untouched


def test_backchannel_tokens_from_lexicon_protect_no():
    from extraction.config import load_yaml
    from extraction.textnorm import normalize
    from extraction.turns import smooth_sentence_speakers

    tokens = {normalize(x) for x in load_yaml("lexicons.yaml")["backchannel_tokens"]}
    assert "no" in tokens and "" not in tokens
    words = _w("el acuerdo de pago el documento no ok señor entonces pues".split(), 70.0)
    speakers = ["A"] * 6 + ["B"] + ["A"] * 4
    out, _ = smooth_sentence_speakers(words, speakers, T_SMOOTH, tokens)
    assert out == speakers


def test_smoothing_leaves_no_sliver_of_old_speaker():
    regular = [{"start": 0.0, "end": 1.2, "speaker": "B"}, {"start": 1.2, "end": 6.0, "speaker": "A"}]
    words = _w("Me alegra que lleguemos a este acuerdo de pago y que pueda aprovechar.".split(), 0.1, step=0.4)
    out = fuse(words, derive_exclusive(regular), regular, [(0.0, 10.0)], [], T_SMOOTH)
    assert out["stats"]["n_smoothed_words"] == 3
    assert [t["speaker"] for t in out["turns"]] == ["A"]
    assert out["stats"]["n_wordless_turns"] == 0


def test_smoothing_leaves_long_minority_run():
    from extraction.turns import smooth_sentence_speakers

    words = _w("uno dos tres cuatro cinco seis siete ocho nueve diez".split(), 0.0, step=0.5)
    speakers = ["A"] * 6 + ["B"] * 4  # 4 words over 1.75 s > 1.5 s
    out, changes = smooth_sentence_speakers(words, speakers, T_SMOOTH, BC)
    assert out == speakers and not changes


def test_derive_exclusive_longer_segment_wins():
    ex = derive_exclusive([{"start": 0, "end": 10, "speaker": "A"}, {"start": 8, "end": 9, "speaker": "B"}])
    assert ex == [{"start": 0, "end": 10, "speaker": "A"}]
    ex = derive_exclusive([{"start": 0, "end": 2, "speaker": "A"}, {"start": 1, "end": 6, "speaker": "B"}])
    assert ex == [{"start": 0, "end": 1, "speaker": "A"}, {"start": 1, "end": 6, "speaker": "B"}]
