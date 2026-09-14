"""configs/pipeline.yaml loading and per-stage config hashes.

A stage's hash covers the config keys it reads, the content of any config files it reads, and the hashes
of its upstream stages — so changing a censor threshold invalidates E0 and everything downstream.
`version` is deliberately not hashed: it is a human-facing changelog counter recorded in `_stage.json`.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from . import paths

STAGES: dict[str, dict[str, Any]] = {
    "download": {"dir": paths.RAW, "keys": ["paths.raw_groups"], "upstream": []},
    "inventory": {
        "dir": paths.interim("censor"),
        "keys": ["seed", "audio.master_sr", "audio.clipping_threshold", "censor", "quality"],
        "upstream": ["download"],
    },
    "preprocess": {"dir": paths.interim("audio_16k"), "keys": ["audio"], "upstream": ["inventory"]},
    "vad": {"dir": paths.interim("vad"), "keys": ["vad"], "upstream": ["preprocess"]},
    "asr": {"dir": paths.interim("asr"), "keys": ["asr"], "files": ["lexicons.yaml"], "upstream": ["vad"]},
    "diarize": {"dir": paths.interim("diarization"), "keys": ["diarization"], "upstream": ["vad"]},
    "turns": {"dir": paths.interim("turns"), "keys": ["turns"], "upstream": ["asr", "diarize"]},
    "roles": {
        "dir": paths.interim("roles_render"),
        "keys": ["annotation.roles_context_s", "annotation.roles_context_turns", "annotation.roles_render_full_call"],
        "upstream": ["turns"],
    },
    "canonical": {
        "dir": paths.interim("canonical"),
        "keys": ["asr.low_conf_word_prob", "quality"],
        "files": ["lexicons.yaml"],
        "upstream": ["turns"],
    },
    "dynamics": {"dir": paths.interim("features", "dynamics"), "keys": ["features"], "upstream": ["canonical"]},
    "prosody": {
        "dir": paths.interim("features", "prosody"),
        "keys": ["prosody", "features.escalation_min_turn_s", "features.escalation_min_turns"],
        "upstream": ["canonical"],
    },
    "lexical": {
        "dir": paths.interim("features", "lexical"),
        "keys": ["features"],
        "files": ["lexicons.yaml"],
        "upstream": ["canonical"],
    },
    "embeddings": {
        "dir": paths.interim("features", "embeddings"),
        "keys": ["features.embedding_model"],
        "upstream": ["canonical"],
    },
}


@lru_cache(maxsize=4)
def _load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_config(path: Path | None = None) -> dict[str, Any]:
    return _load(str(path or paths.CONFIGS / "pipeline.yaml"))


def load_yaml(name: str) -> dict[str, Any]:
    return _load(str(paths.CONFIGS / name))


def get(cfg: dict[str, Any], dotted: str) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        cur = cur[part]
    return cur


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def stage_hash(stage: str, cfg: dict[str, Any] | None = None) -> str:
    cfg = cfg if cfg is not None else load_config()
    spec = STAGES[stage]
    payload = {
        "stage": stage,
        "keys": {k: get(cfg, k) for k in spec["keys"]},
        "files": {f: sha256_file(paths.CONFIGS / f) for f in spec.get("files", [])},
        "upstream": {u: stage_hash(u, cfg) for u in spec["upstream"]},
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode()
    return sha256_bytes(blob)[:16]
