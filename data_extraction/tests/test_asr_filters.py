from extraction.asr import attempt_chain, collapse_repeats, confidence, post_filter
from extraction.config import load_config
from extraction.textnorm import normalize

A = load_config()["asr"]


def test_attempt_chain_keeps_primary_model_before_fallback_model():
    chain = attempt_chain({"model": "large-v3", "fallback_model": "large-v3-turbo", "device": "cuda"})
    assert chain == [("large-v3", "cuda"), ("large-v3", "cpu"), ("large-v3-turbo", "cuda"), ("large-v3-turbo", "cpu")]


def seg(i, start, end, text, cr=1.2, probs=None):
    toks = text.split()
    step = (end - start) / max(len(toks), 1)
    words = [{"start": start + k * step, "end": start + (k + 0.9) * step, "word": t,
              "probability": (probs or [0.9] * len(toks))[k]} for k, t in enumerate(toks)]
    return {"id": i, "start": start, "end": end, "text": text, "compression_ratio": cr, "words": words}


def test_collapse_repeats():
    toks = "hola a b c a b c a b c fin".split()
    keep = collapse_repeats(toks)
    assert [t for t, k in zip(toks, keep) if k] == "hola a b c fin".split()
    assert all(collapse_repeats("a b c a b c fin".split()))  # only 2 repeats


def test_post_filter_reasons():
    speech = [(0.0, 10.0)]
    censor = [(20.0, 30.0)]
    blacklist = {normalize("¡Gracias por ver el video!")}
    segs = [
        seg(0, 0, 2, "buenos días señor"),
        seg(1, 2, 4, "gracias por ver el video"),
        seg(2, 4, 6, "texto repetido raro", cr=3.1),
        seg(3, 21, 25, "palabras en la censura"),
        seg(4, 12, 14, "fuera de voz"),
    ]
    kept, removed = post_filter(segs, speech, censor, blacklist, A)
    assert [s["id"] for s in kept] == [0]
    assert {r["reason"] for r in removed} == {"blacklist", "compression_ratio", "nonspeech_or_censor"}


def test_confidence_weighted():
    words = [{"start": 0, "end": 1, "probability": 1.0}, {"start": 1, "end": 4, "probability": 0.2}]
    conf, low = confidence(words, 0.5)
    assert conf == round((1.0 * 1 + 0.2 * 3) / 4, 4)
    assert low == 50.0
