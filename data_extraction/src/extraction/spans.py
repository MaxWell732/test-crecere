"""Interval algebra on [start, end) spans in seconds. Inputs need not be sorted; outputs are sorted and disjoint."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

Span = tuple[float, float]


def merge(spans: Iterable[Sequence[float]], gap: float = 0.0) -> list[Span]:
    """Union of spans; spans separated by <= gap are joined."""
    items = sorted((float(s), float(e)) for s, e, *_ in spans if e > s)
    out: list[list[float]] = []
    for s, e in items:
        if out and s <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def total(spans: Iterable[Sequence[float]]) -> float:
    return float(sum(e - s for s, e in merge(spans)))


def intersect(a: Iterable[Sequence[float]], b: Iterable[Sequence[float]]) -> list[Span]:
    a_m, b_m = merge(a), merge(b)
    out: list[Span] = []
    i = j = 0
    while i < len(a_m) and j < len(b_m):
        s, e = max(a_m[i][0], b_m[j][0]), min(a_m[i][1], b_m[j][1])
        if e > s:
            out.append((s, e))
        if a_m[i][1] < b_m[j][1]:
            i += 1
        else:
            j += 1
    return out


def subtract(a: Iterable[Sequence[float]], b: Iterable[Sequence[float]]) -> list[Span]:
    b_m = merge(b)
    out: list[Span] = []
    for s, e in merge(a):
        cur = s
        for bs, be in b_m:
            if be <= cur:
                continue
            if bs >= e:
                break
            if bs > cur:
                out.append((cur, bs))
            cur = max(cur, be)
            if cur >= e:
                break
        if cur < e:
            out.append((cur, e))
    return out


def overlap(start: float, end: float, spans: Iterable[Sequence[float]]) -> float:
    return float(sum(max(0.0, min(end, e) - max(start, s)) for s, e, *_ in spans))


def complement(spans: Iterable[Sequence[float]], start: float, end: float) -> list[Span]:
    return subtract([(start, end)], spans)


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Index runs [i, j) where mask is True."""
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return []
    d = np.diff(np.concatenate(([0], m.view(np.int8), [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def to_mask(spans: Iterable[Sequence[float]], n: int, rate: float) -> np.ndarray:
    """Boolean mask of length n (samples or frames at `rate` per second) covering the spans."""
    m = np.zeros(n, dtype=bool)
    for s, e, *_ in spans:
        i, j = max(0, int(np.floor(s * rate))), min(n, int(np.ceil(e * rate)))
        if j > i:
            m[i:j] = True
    return m
