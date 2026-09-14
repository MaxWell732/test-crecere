"""Atomic JSON I/O and the resumable per-call stage runner (spec §1.9, §6 stage metadata).

Each per-call stage writes one primary JSON per call *last* (after any binary side outputs), carrying
`_cache_key`. A call is skipped when that JSON exists with the current key, unless --force.
One failing call is logged and the batch continues; `_stage.json` summarizes the run.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm

from . import config as cfgmod
from .logging_utils import get_logger

TRACKED_PACKAGES = [
    "numpy", "scipy", "pandas", "soundfile", "soxr", "torch", "torchaudio", "torchcodec", "ctranslate2",
    "faster-whisper", "pyannote.audio", "silero-vad", "praat-parselmouth", "pysentimiento", "transformers",
    "sentence-transformers", "rapidfuzz", "jsonschema",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1, default=_json_default)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _json_default(o: Any) -> Any:
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def file_fingerprint(*files: Path) -> str:
    """Identity of upstream outputs (mtime + size) so a regenerated upstream file makes downstream calls stale (D-020)."""
    parts = []
    for f in files:
        try:
            st = f.stat()
            parts.append(f"{f.name}:{st.st_mtime_ns}:{st.st_size}")
        except FileNotFoundError:
            parts.append(f"{f.name}:missing")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def is_fresh(path: Path, key: str) -> bool:
    if not path.exists():
        return False
    try:
        return read_json(path).get("_cache_key") == key
    except (json.JSONDecodeError, OSError, AttributeError):
        return False


def package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


@dataclass
class StageContext:
    stage: str
    cfg: dict[str, Any]
    key: str
    logger: Any
    stats: dict[str, Any] = field(default_factory=dict)
    models: dict[str, Any] = field(default_factory=dict)

    def bump(self, name: str, n: int = 1) -> None:
        self.stats[name] = self.stats.get(name, 0) + n


@dataclass
class StageResult:
    ok: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


def run_per_call(
    stage: str,
    call_ids: list[str],
    process: Callable[[str, StageContext], dict[str, Any]],
    output_path: Callable[[str], Path],
    *,
    force: bool = False,
    setup: Callable[[StageContext], None] | None = None,
    teardown: Callable[[StageContext], None] | None = None,
    extra_key: Callable[[str], str] | None = None,
) -> StageResult:
    """Run `process(call_id, ctx)` for each call; it returns the primary JSON (written atomically here)."""
    cfg = cfgmod.load_config()
    key = cfgmod.stage_hash(stage, cfg)
    logger = get_logger(stage)
    ctx = StageContext(stage=stage, cfg=cfg, key=key, logger=logger)
    result = StageResult()
    started = now_iso()
    t0 = time.time()

    def call_key(cid: str) -> str:
        extra = extra_key(cid) if extra_key is not None else ""
        return f"{key}:{extra}" if extra else key

    todo = [c for c in call_ids if force or not is_fresh(output_path(c), call_key(c))]
    result.skipped = [c for c in call_ids if c not in set(todo)]
    logger.info("stage=%s key=%s calls=%d todo=%d skipped(cached)=%d", stage, key, len(call_ids), len(todo),
                len(result.skipped))

    if todo and setup is not None:
        setup(ctx)
    try:
        for cid in tqdm(todo, desc=stage, unit="call", disable=not todo):
            t_call = time.time()
            try:
                out = process(cid, ctx)
                out = {"call_id": cid, **out, "_cache_key": call_key(cid), "_stage": stage, "_written_at": now_iso()}
                write_json(output_path(cid), out)
                result.ok.append(cid)
                logger.debug("%s ok (%.1fs)", cid, time.time() - t_call)
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 — one call must not stop the batch
                result.failed[cid] = f"{type(exc).__name__}: {exc}"
                logger.error("%s FAILED: %s\n%s", cid, exc, traceback.format_exc())
    finally:
        if teardown is not None and todo:
            teardown(ctx)
        write_stage_meta(stage, ctx, result, started, time.time() - t0)

    logger.info("stage=%s done: ok=%d skipped=%d failed=%d", stage, len(result.ok), len(result.skipped),
                len(result.failed))
    for cid, err in result.failed.items():
        logger.warning("  failed %s -> %s", cid, err)
    return result


def write_stage_meta(stage: str, ctx: StageContext, result: StageResult, started: str, elapsed_s: float) -> None:
    meta_path = Path(cfgmod.STAGES[stage]["dir"]) / "_stage.json"
    try:
        previous = read_json(meta_path) if meta_path.exists() else {}
    except (json.JSONDecodeError, OSError):
        previous = {}
    # ctx.models may hold loaded model objects (Silero, analyzers, pipelines): record only JSON-safe metadata
    safe_models = {k: v for k, v in ctx.models.items() if isinstance(v, (str, int, float, bool, type(None)))}
    runs = previous.get("runs", [])[-19:]
    run = {
        "stage": stage,
        "config_hash": ctx.key,
        "pipeline_version": ctx.cfg.get("version"),
        "started_at": started,
        "ended_at": now_iso(),
        "elapsed_s": round(elapsed_s, 1),
        "n_ok": len(result.ok),
        "n_skipped": len(result.skipped),
        "n_failed": len(result.failed),
        "failures": result.failed,
        "models": safe_models,
        "stats": ctx.stats,
        "packages": package_versions(),
    }
    write_json(meta_path, {**run, "runs": runs + [{k: v for k, v in run.items() if k != "packages"}]})
