"""Filesystem layout (spec §6). Every path the pipeline writes is under ROOT."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
SCHEMAS = CONFIGS / "schemas"
DOCS = ROOT / "docs"
CACHE = ROOT / ".cache"

DATA = ROOT / "data"
RAW = DATA / "raw"
PRIVATE = DATA / "private"
ID_MAP = PRIVATE / "id_map.csv"
INTERIM = DATA / "interim"
MANIFEST = INTERIM / "manifest.csv"
LLM_INPUTS = INTERIM / "llm_inputs"
ANNOTATIONS = INTERIM / "annotations"
REVIEW = INTERIM / "review"
ROLES_REVIEW = REVIEW / "roles_review.csv"
VALIDATION = DATA / "validation"
PROCESSED = DATA / "processed"
LOGS = DATA / "logs"


def interim(*parts: str) -> Path:
    return INTERIM.joinpath(*parts)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
