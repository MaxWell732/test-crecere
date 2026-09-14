"""Shared helpers for Claude-Code annotation files: schemas, quote verification, progress.csv."""

from __future__ import annotations

import csv
import json
from functools import lru_cache
from typing import Any

from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from rapidfuzz import fuzz

from . import paths
from .cache import now_iso
from .config import sha256_file
from .textnorm import normalize

PROGRESS = paths.ANNOTATIONS / "progress.csv"
PROGRESS_COLUMNS = ["call_id", "pass", "status", "errors", "checked_at"]


@lru_cache(maxsize=1)
def _registry() -> Registry:
    resources = []
    for p in paths.SCHEMAS.glob("*.schema.json"):
        doc = json.loads(p.read_text(encoding="utf-8"))
        resources.append((p.name, Resource.from_contents(doc)))
    return Registry().with_resources(resources)


@lru_cache(maxsize=8)
def validator(schema_name: str) -> Draft202012Validator:
    schema = json.loads((paths.SCHEMAS / schema_name).read_text(encoding="utf-8"))
    return Draft202012Validator(schema, registry=_registry())


def schema_errors(doc: Any, schema_name: str) -> list[str]:
    errs = sorted(validator(schema_name).iter_errors(doc), key=lambda e: list(e.absolute_path))
    return [f"schema: /{'/'.join(map(str, e.absolute_path))}: {e.message[:200]}" for e in errs]


def normalize_quote(text: str) -> str:
    """§9-E9 quote rule: casefold, strip punctuation and [?], collapse spaces (accents kept)."""
    return normalize(text, accents=False)


def quote_score(quote: str, turn_text: str) -> float:
    q, t = normalize_quote(quote), normalize_quote(turn_text)
    if not q or not t:
        return 0.0
    return float(fuzz.partial_ratio(q, t))


def verify_quote(quote: str | None, turn_text: str | None, min_ratio: float) -> bool:
    if quote is None or turn_text is None:
        return False
    return quote_score(quote, turn_text) >= min_ratio


def codebook_sha256() -> str | None:
    p = paths.CONFIGS / "codebook.md"
    return sha256_file(p) if p.exists() else None


def read_progress() -> list[dict[str, str]]:
    if not PROGRESS.exists():
        return []
    with open(PROGRESS, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def update_progress(updates: list[dict[str, Any]]) -> None:
    rows = {(r["call_id"], r["pass"]): r for r in read_progress()}
    for u in updates:
        rows[(u["call_id"], u["pass"])] = {**u, "errors": " | ".join(u.get("errors") or [])[:2000], "checked_at": now_iso()}
    paths.ensure_dir(PROGRESS.parent)
    with open(PROGRESS, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=PROGRESS_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(rows.values(), key=lambda r: (r["pass"], r["call_id"])))
