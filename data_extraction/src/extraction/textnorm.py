"""Text normalization shared by E3 blacklist, E8c lexicons and quote verification."""

from __future__ import annotations

import re
import unicodedata

_LOWCONF = re.compile(r"\[\?\]")
_NON_WORD = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SPACES = re.compile(r"\s+")


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize(text: str, accents: bool = True) -> str:
    """casefold, drop `[?]` markers, (optionally) strip accents, punctuation -> space, collapse whitespace."""
    t = _LOWCONF.sub(" ", text or "").casefold()
    if accents:
        t = strip_accents(t)
    t = _NON_WORD.sub(" ", t).replace("_", " ")
    return _SPACES.sub(" ", t).strip()


def tokens(text: str) -> list[str]:
    return normalize(text).split()
