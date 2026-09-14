"""Deterministic transcript rendering for annotation inputs (E6 roles, E7 calls).

Quote verification re-renders turns with these functions, so quotes are checked against exactly what the annotator saw.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Callable

from .config import load_yaml
from .textnorm import strip_accents

CENSOR_TOKEN = "[CENSURADO]"
MASK_TOKEN = "[AGENTE]"
NOTATION = ("# notation: word[?] = low-confidence ASR word · [CENSURADO] = censored audio · "
            "[AGENTE] = masked agent self-description · (ROLE: text) = backchannel")


def fmt_ts(t: float) -> str:
    t = round(max(t, 0.0), 1)
    m = int(t // 60)
    return f"{m:02d}:{t - 60 * m:04.1f}"


@lru_cache(maxsize=1)
def masking_patterns() -> tuple[re.Pattern, ...]:
    return tuple(re.compile(p) for p in load_yaml("lexicons.yaml")["agent_masking"])


def _fold(text: str) -> str:
    """Lowercase + accent strip, one output char per input char (index-aligned with the input)."""
    out = []
    for ch in text:
        base = strip_accents(ch).lower()
        out.append(base[0] if base else " ")
    return "".join(out)


def mask_hits(tokens: list[str], patterns: tuple[re.Pattern, ...]) -> list[bool]:
    if not tokens or not patterns:
        return [False] * len(tokens)
    offsets, pos = [], 0
    for tok in tokens:
        offsets.append((pos, pos + len(tok)))
        pos += len(tok) + 1
    folded = _fold(" ".join(tokens))
    hit = [False] * len(tokens)
    for pat in patterns:
        for m in pat.finditer(folded):
            for i, (s, e) in enumerate(offsets):
                if s < m.end() and e > m.start():
                    hit[i] = True
    return hit


def mask_tokens(tokens: list[str], patterns: tuple[re.Pattern, ...]) -> list[str]:
    out: list[str] = []
    prev = False
    for tok, h in zip(tokens, mask_hits(tokens, patterns)):
        if not h:
            out.append(tok)
        elif not prev:
            out.append(MASK_TOKEN)
        prev = h
    return out


def turn_text(turn: dict[str, Any], label_of: Callable[[str], str], low_conf: float, mask: bool,
              bc_speaker_key: str = "speaker") -> str:
    """Words (+[?]) with [CENSURADO] and inline backchannels `(LABEL: text)` in time order; optional masking."""
    words = turn["words"]
    hits = mask_hits([w["w"] for w in words], masking_patterns()) if mask else [False] * len(words)
    items: list[tuple[float, int, str]] = []
    prev = False
    for w, h in zip(words, hits):
        if not h:
            items.append((w["start"], 0, w["w"] + ("[?]" if w["prob"] < low_conf else "")))
        elif not prev:
            items.append((w["start"], 0, MASK_TOKEN))
        prev = h
    for c in turn.get("censor", []):
        items.append((c["start"], 1, CENSOR_TOKEN))
    for b in turn.get("backchannels", []):
        items.append((b["start"], 2, f"({label_of(b[bc_speaker_key])}: {b['text']})"))
    return " ".join(t for _, _, t in sorted(items, key=lambda x: (x[0], x[1])))


def render_line(turn: dict[str, Any], label: str, text: str) -> str:
    return f"[{turn['turn_id']} | {label} | {fmt_ts(turn['start'])}–{fmt_ts(turn['end'])}] {text}"
